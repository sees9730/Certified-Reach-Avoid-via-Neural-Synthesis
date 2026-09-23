"""Effort-budget geometry, warm start, generator, and training workflow checks."""
from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from examples.xv15_uncertain.model import DEG, load_config
from examples.xv15_uncertain.neural_certified_nominal_drift.main import configure_reproducibility, make_hyperparameters
from examples.xv15_uncertain.neural_certified_uncertain_param.main import build_problem
from examples.xv15_uncertain.neural_certified_uncertain_param.main_constraint_energy import (
    ALPHA, ETA, GROUPS, REFINEMENT_FLAGS, SAT_KEYS, augment_baseline_cells,
    build_energy_regions, build_networks, energy_budget, energy_cell_edges,
    lift_baseline_value_state, load_sat_baseline, make_energy_parameters,
    parse_args, pretrain_value_network, refinement_overrides,
)
from src.training_utils import GENERATOR_MARGIN
from src.discretization import (
    compute_rectangular_partition_outside_goal, compute_rectangular_partition_outside_goal_and_unsafe,
)
from src.phi_module import compute_GV_autograd
from src.training_utils import refine_failing_cells

ENERGY_MAX, ENERGY_MARGIN = 20.0, 1.0
BUDGET = ENERGY_MAX - ENERGY_MARGIN


@pytest.fixture
def baseline(tmp_path):
    """Small synthetic bundle with SAT metadata to exercise the loading contract.

    The random networks are not themselves asserted to be a SAT certificate.
    The actual saved seed0 SAT bundle is checked separately during integration.
    """
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    configure_reproducibility(0)
    config = load_config()
    config['controller_hidden_dim'] = 8
    params = make_hyperparameters(config, 0, tmp_path)
    params.network.n_hidden_1 = params.network.n_hidden_2 = 8
    value, generator, controller, dynamics, regions, arrays = build_problem(config, params)
    rects = dict(init=[regions.init], goal=[regions.goal], unsafe=regions.unsafe.components,
                 outside=compute_rectangular_partition_outside_goal(regions.full, regions.goal),
                 generator=compute_rectangular_partition_outside_goal_and_unsafe(regions.full, regions.goal, regions.unsafe))
    cells = {k: [box.to_torch() for box in boxes] for k, boxes in rects.items()}
    bundle = dict(V_state_dict=deepcopy(value.state_dict()), GV_state_dict=deepcopy(generator.state_dict()),
                  control_state_dict=deepcopy(controller.state_dict()), hyperparameters=params.to_dict(),
                  final_results={key: True for key in SAT_KEYS}, region_cells=cells, regions=regions.to_dict())
    outputs = tmp_path / 'baseline' / 'outputs'
    outputs.mkdir(parents=True)
    checkpoint = outputs / 'eval_bundle.pth'
    torch.save(bundle, checkpoint)
    (outputs / 'run_config.json').write_text(json.dumps(dict(example=config)))
    yield config, params, bundle, checkpoint, (value, generator, controller, dynamics, arrays)
    torch.set_num_threads(old)


def setup_energy(baseline, *, freeze=False, extra=()):
    config, _, bundle, checkpoint, _ = baseline
    args = parse_args(['--energy-max', str(ENERGY_MAX), '--energy-margin', str(ENERGY_MARGIN),
                       '--pretrain-epochs', '0', '--no-plots', *extra])
    params = make_energy_parameters(bundle, checkpoint, args)
    value, generator, controller = build_networks(config, params, bundle, freeze)
    return params, value, generator, controller


def covered(point, cells):
    return any(bool(((point >= lo) & (point <= hi)).all()) for lo, hi in cells)


def test_domain_includes_upper_unsafe_band_and_separated_goal(baseline):
    config, _, bundle, _, (_, _, _, _, arrays) = baseline
    budget = energy_budget(ENERGY_MAX, ENERGY_MARGIN)
    regions, boxes = build_energy_regions(config, ENERGY_MAX, ENERGY_MARGIN)
    cells = augment_baseline_cells(bundle['region_cells'], regions, 4)
    init = torch.tensor(arrays['init_range'].mean(axis=1))
    goal = torch.tensor(arrays['goal_range'].mean(axis=1))
    assert budget == np.float32(BUDGET)
    # Match the pendulum: keep the full domain, including the upper band.
    np.testing.assert_array_equal(regions.full.bounds[-1], [0.0, ENERGY_MAX])
    # Initial states start with zero accumulated effort.
    assert regions.init.contains(torch.cat([init, torch.tensor([0.])]))
    assert not regions.init.contains(torch.cat([init, torch.tensor([1.])]))
    goal_max = regions.goal.upper[-1]
    assert goal_max == np.float32(BUDGET - 0.01 * min(ENERGY_MARGIN, BUDGET))
    assert 0 < goal_max < budget < ENERGY_MAX
    np.testing.assert_array_equal(regions.goal.bounds[-1], [0.0, goal_max])
    assert regions.goal.contains(torch.cat([goal, torch.tensor([goal_max])]))
    assert not regions.goal.contains(torch.cat([goal, torch.tensor([budget])]))
    assert not regions.unsafe.contains(torch.cat([goal, torch.tensor([budget / 2])]))
    assert not regions.unsafe.contains(torch.cat([goal, torch.tensor([0.])]))
    # The whole closed upper band is unsafe, including physical goal states.
    assert boxes.shape[0] == len(arrays['unsafe_ranges']) + 1
    np.testing.assert_array_equal(boxes[-1, -1], [budget, ENERGY_MAX])
    np.testing.assert_array_equal(boxes[-1, :-1], arrays['full_range'])
    # Insert both region boundaries into the uniform domain grid.
    edges = energy_cell_edges(regions, 4)
    expected_edges = np.unique(np.concatenate((
        np.linspace(0.0, ENERGY_MAX, 5, dtype=np.float32), [goal_max, budget])))
    np.testing.assert_array_equal(edges, expected_edges)
    # Extrusion never moves a physical endpoint, and init stays degenerate.
    for (lo, hi), (source_lo, source_hi) in zip(cells['init'], bundle['region_cells']['init']):
        assert lo[-1] == hi[-1] == 0
        torch.testing.assert_close(lo[:-1], source_lo, rtol=0, atol=0)
        torch.testing.assert_close(hi[:-1], source_hi, rtol=0, atol=0)
    full = torch.tensor(arrays['full_range'])
    states = full[:, 0] + torch.rand(60, 3) * (full[:, 1] - full[:, 0])
    states = torch.cat([states, goal[None]])
    gap_mid = (float(goal_max) + float(budget)) / 2
    band_mid = (float(budget) + ENERGY_MAX) / 2
    for energy, x in product((0., float(goal_max) / 2, gap_mid, float(budget), band_mid, ENERGY_MAX), states):
        point = torch.cat([x, torch.tensor([energy])])
        # V >= 0 is required everywhere: goal + outside must tile the domain.
        assert covered(point, cells['goal'] + cells['outside'])
        if energy >= float(budget):
            assert covered(point, cells['unsafe'])
        if energy > float(budget):
            assert not covered(point, cells['generator'])
    # Physical goal states join the generator cover only in the gap.
    for energy in (0., float(goal_max) / 2, band_mid, ENERGY_MAX):
        assert not covered(torch.cat([goal, torch.tensor([energy])]), cells['generator'])
    gap_point = torch.cat([goal, torch.tensor([gap_mid])])
    assert covered(gap_point, cells['generator'])
    assert covered(gap_point, cells['outside'])
    assert not covered(gap_point, cells['goal'])
    assert not covered(gap_point, cells['unsafe'])
    # Refinement must preserve the E=0 initial slice; split only spatial axes.
    refined, count = refine_failing_cells([cells['init'][0]], torch.tensor([True]), 2, N_to_refine=1)
    assert count == 1 and len(refined) == 8
    assert all(lo[-1] == hi[-1] == 0 for lo, hi in refined)


def test_cell_groups_follow_the_pendulum_band_transfers(baseline):
    config, _, bundle, _, _ = baseline
    regions, _ = build_energy_regions(config, ENERGY_MAX, ENERGY_MARGIN)
    counts = {name: len(bundle['region_cells'][name]) for name in GROUPS}
    ceiling = regions.unsafe.components[-1].lower[-1]
    goal_max = regions.goal.upper[-1]
    for energy_cells in (1, 2, 4):
        cells = augment_baseline_cells(bundle['region_cells'], regions, energy_cells)
        edges = energy_cell_edges(regions, energy_cells)
        intervals = list(zip(edges[:-1], edges[1:]))
        slabs = len(intervals)
        goal_slabs = sum(hi <= goal_max for lo, hi in intervals)
        generator_slabs = sum(hi <= ceiling for lo, hi in intervals)
        gap_slabs = sum(lo >= goal_max and hi <= ceiling for lo, hi in intervals)
        unsafe_slabs = sum(lo >= ceiling for lo, hi in intervals)
        assert gap_slabs > 0 and unsafe_slabs > 0
        assert len(cells['init']) == counts['init']
        assert len(cells['goal']) == goal_slabs * counts['goal']
        assert len(cells['outside']) == slabs * counts['outside'] + (slabs - goal_slabs) * counts['goal']
        assert len(cells['generator']) == generator_slabs * counts['generator'] + gap_slabs * counts['goal']
        assert len(cells['unsafe']) == slabs * counts['unsafe'] + unsafe_slabs * (counts['goal'] + counts['outside'])
        assert all(hi[-1] <= ceiling for lo, hi in cells['generator'])
        # Every augmented cell retains the exact physical endpoints of its source.
        source_groups = dict(init=('init',), goal=('goal',),
                             outside=('outside', 'goal'), generator=('generator', 'goal'),
                             unsafe=('unsafe', 'goal', 'outside'))
        def key(lo, hi):
            return tuple(lo.tolist()) + tuple(hi.tolist())
        for name in GROUPS:
            allowed = {key(lo, hi) for source in source_groups[name]
                       for lo, hi in bundle['region_cells'][source]}
            assert all(key(lo[:-1], hi[:-1]) in allowed for lo, hi in cells[name])


def test_lift_preserves_networks_and_adds_energy_input_only_to_certificate(baseline):
    params, value, generator, controller = setup_energy(baseline)
    _, _, bundle, _, (base_value, _, base_controller, _, arrays) = baseline
    assert params.network.n_inputs == 4 and params.include_energy and not params.include_time
    assert generator.include_energy and not generator.include_time
    x = torch.tensor(arrays['init_range'][:, 0]) + torch.rand(12, 3) * torch.tensor(
        arrays['init_range'][:, 1] - arrays['init_range'][:, 0])
    # The controller still sees three inputs and is bit-identical.
    torch.testing.assert_close(controller(x), base_controller(x), rtol=0, atol=0)
    # V(x, E) == V_baseline(x) at every E, so the warm start changes nothing.
    for energy in (0.0, 7.5, BUDGET, ENERGY_MAX):
        xe = torch.cat([x, torch.full((12, 1), energy)], dim=1)
        torch.testing.assert_close(value(xe), base_value(x), rtol=1e-6, atol=1e-6)
    # The appended column is exactly zero, so dV/dE vanishes at the warm start.
    torch.testing.assert_close(value.layer1.weight[:, -1], torch.zeros(value.layer1.weight.shape[0]),
                               rtol=0, atol=0)
    lifted = lift_baseline_value_state(bundle['V_state_dict'], ENERGY_MAX)
    assert float(lifted['input_scale'][-1]) == ENERGY_MAX
    assert float(lifted['input_offset'][-1]) == 0.0
    assert float(params.network.input_scale[-1]) == ENERGY_MAX
    with pytest.raises(ValueError, match="three spatial"):
        lift_baseline_value_state(lifted, ENERGY_MAX)


def test_energy_rate_matches_monte_carlo_effort_and_enters_the_generator(baseline):
    config, _, _, _, (_, _, _, _, arrays) = baseline
    params, value, generator, controller = setup_energy(baseline)
    x = torch.tensor(arrays['init_range'][:, 0]) + torch.rand(16, 3) * torch.tensor(
        arrays['init_range'][:, 1] - arrays['init_range'][:, 0])
    # raw_control reproduces run_mc.py's normalized_effort integrand exactly.
    scale = torch.tensor([config['dynamics']['mass'] * config['dynamics']['gravity'],
                          config['control']['alpha_max_deg'] * DEG,
                          config['control']['delta_max_deg_per_second'] * DEG])
    torch.testing.assert_close(controller.raw_control(x), controller(x) / scale)
    rate = (controller.raw_control(x) ** 2).sum(dim=1, keepdim=True)
    assert bool((rate > 0).all())
    # Give V a genuine energy dependence, then check the analytic generator.
    with torch.no_grad():
        value.layer1.weight[:, -1].normal_(0.0, 0.5)
    xe = torch.cat([x, torch.rand(16, 1) * ENERGY_MAX], dim=1)
    torch.testing.assert_close(generator(xe), compute_GV_autograd(generator, xe), atol=1e-4, rtol=1e-4)
    # The energy term is exactly dV/dE * dE/dt: zeroing the rate removes it.
    probe = xe.clone().requires_grad_(True)
    dVdE = torch.autograd.grad(value(probe).sum(), probe)[0][:, -1:]
    assert bool((dVdE.abs() > 0).any())
    original = controller.raw_control
    try:
        controller.raw_control = lambda state: torch.zeros_like(original(state))
        without_energy = generator(xe)
    finally:
        controller.raw_control = original
    torch.testing.assert_close(generator(xe) - without_energy, dVdE * rate, atol=1e-4, rtol=1e-4)


def test_certificate_pretraining_preserves_frozen_controller(baseline, tmp_path):
    config, _, bundle, _, _ = baseline
    params, value, generator, controller = setup_energy(baseline, freeze=True)
    regions, unsafe_boxes = build_energy_regions(config, ENERGY_MAX, ENERGY_MARGIN)
    before = deepcopy(controller.state_dict())
    params.training.pretrain_epochs, params.training.pretrain_n_samples = 2, len(unsafe_boxes)
    outputs = tmp_path / 'pretrain'
    outputs.mkdir()
    pretrain_value_network(value, generator, controller, regions, unsafe_boxes, params, outputs)
    for key, tensor in controller.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in controller.parameters())
    assert (outputs / 'V_pretrained.pth').is_file()


def test_reject_non_sat_or_inconsistent_source(baseline, tmp_path):
    config, _, bundle, checkpoint, _ = baseline

    def save(mutate, name):
        broken = deepcopy(bundle)
        mutate(broken)
        directory = tmp_path / name / 'outputs'
        directory.mkdir(parents=True)
        path = directory / 'eval_bundle.pth'
        torch.save(broken, path)
        (directory / 'run_config.json').write_text(json.dumps(dict(example=config)))
        return path

    load_sat_baseline(checkpoint)  # the unmodified fixture is accepted
    with pytest.raises(ValueError, match="SAT final_results"):
        load_sat_baseline(save(lambda b: b['final_results'].update(generator_satisfied=False), 'unsat'))
    with pytest.raises(ValueError, match="purely spatial"):
        load_sat_baseline(save(lambda b: b['hyperparameters'].update(include_energy=True), 'already_energy'))
    with pytest.raises(ValueError, match="missing region_cells"):
        load_sat_baseline(save(lambda b: b.update(region_cells={}), 'no_cells'))
    with pytest.raises(ValueError, match="robust density interval"):
        load_sat_baseline(save(
            lambda b: b['GV_state_dict'].update({'f.density_interval': torch.tensor([1.0, 1.4])}), 'density'))


def test_training_cli_saves_energy_results_without_overwriting_source(baseline, tmp_path):
    _, _, _, checkpoint, _ = baseline
    before = checkpoint.read_bytes()
    run_dir = tmp_path / 'energy_run'
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]), MPLBACKEND='Agg')
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / 'examples' / 'xv15_uncertain' /
                             'neural_certified_uncertain_param' / 'main_constraint_energy.py'),
         '--baseline-checkpoint', str(checkpoint), '--energy-max', str(ENERGY_MAX),
         '--energy-margin', str(ENERGY_MARGIN), '--energy-cells', '1', '--epochs', '1',
         '--pretrain-epochs', '0', '--no-plots', '--output-dir', str(run_dir)],
        capture_output=True, text=True, env=environment, timeout=900)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert checkpoint.read_bytes() == before, "the baseline checkpoint must never be rewritten"
    result = json.loads((run_dir / 'outputs' / 'energy_result.json').read_text())
    assert result['energy_budget'] == BUDGET and result['energy_max'] == ENERGY_MAX
    assert result['energy_domain'] == [0.0, ENERGY_MAX]
    assert result['unsafe_energy_band'] == [BUDGET, ENERGY_MAX]
    assert result['goal_energy_max'] < result['energy_ceiling'] == BUDGET
    assert result['alpha'] == ALPHA
    assert result['results']['generator_margin'] == ETA
    assert set(SAT_KEYS) <= set(result['results'])
    metadata = json.loads((run_dir / 'outputs' / 'run_config.json').read_text())
    assert metadata['mode'] == 'effort_constrained_uncertain_parameters'
    assert metadata['coordinate_order'] == ['v', 'gamma', 'beta', 'E']
    assert metadata['hyperparameters']['include_energy'] and not metadata['hyperparameters']['include_time']
    assert metadata['energy_budget'] == BUDGET and metadata['alpha'] == ALPHA
    assert metadata['energy_domain'] == [0.0, ENERGY_MAX]
    assert metadata['unsafe_energy_band'] == [BUDGET, ENERGY_MAX]
    assert metadata['probability_lower_bound'] == 1 - ALPHA / metadata['beta']
    assert (run_dir / 'outputs' / 'eval_bundle.pth').is_file()
    assert (run_dir / 'outputs' / 'initial_region_cells.pth').is_file()
    assert not list((run_dir / 'results').glob('*.pdf'))


def test_reject_degenerate_energy_bounds(baseline):
    config, _, _, _, _ = baseline
    for bad in ((20.0, 0.0), (20.0, 20.0), (20.0, 25.0), (0.0, 1.0),
                (float('inf'), 1.0), (20.0, float('nan'))):
        with pytest.raises(ValueError, match="energy_margin"):
            build_energy_regions(config, *bad)
    regions, _ = build_energy_regions(config, ENERGY_MAX, ENERGY_MARGIN)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="energy-cells"):
            energy_cell_edges(regions, bad)


def test_refinement_flags_override_only_what_is_passed(baseline):
    _, _, bundle, checkpoint, _ = baseline
    saved = bundle['hyperparameters']['refinement']
    groups = ('v_goal', 'v_init', 'v_unsafe', 'v_outside', 'gv_generator')

    def refinement(argv):
        args = parse_args(['--no-plots', *argv])
        return make_energy_parameters(bundle, checkpoint, args).refinement, refinement_overrides(args)

    # A flag left at None inherits the baseline bundle; a flag carrying a
    # concrete default is applied and recorded even when nothing is passed.
    plain, overrides = refinement([])
    defaults = parse_args(['--no-plots'])
    assert overrides == {name: getattr(defaults, name) for name in REFINEMENT_FLAGS
                         if getattr(defaults, name) is not None}
    for name in REFINEMENT_FLAGS:
        assert name in overrides or getattr(defaults, name) is None
    assert plain.gv_generator.merge_relax_margin == saved['gv_generator']['merge_relax_margin']
    assert plain.gv_generator.N_to_refine == saved['gv_generator']['N_to_refine']
    assert plain.gv_generator.max_cells == saved['gv_generator']['max_cells']
    assert plain.v_outside.merge_relax_margin == (
        defaults.outside_merge_margin if defaults.outside_merge_margin is not None
        else saved['v_outside']['merge_relax_margin'])

    # Each flag moves exactly one setting; the other groups are untouched.
    tuned, overrides = refinement(['--outside-merge-margin', '5', '--generator-merge-margin', '-0.5',
                                   '--n-to-refine', '800', '--max-generator-cells', '40000'])
    assert tuned.v_outside.merge_relax_margin == 5.0
    assert tuned.gv_generator.merge_relax_margin == -0.5
    assert tuned.gv_generator.N_to_refine == 800
    assert tuned.gv_generator.max_cells == 40000
    assert set(overrides) == set(REFINEMENT_FLAGS) - {'merge_interval'}
    for name in ('v_goal', 'v_init', 'v_unsafe'):
        assert getattr(tuned, name).merge_relax_margin == saved[name]['merge_relax_margin']
        assert getattr(tuned, name).N_to_refine == saved[name]['N_to_refine']

    # merge-interval is the one flag that applies to every group at once.
    shared, overrides = refinement(['--merge-interval', '251'])
    assert [getattr(shared, name).merge_interval for name in groups] == [251] * 5
    assert overrides['merge_interval'] == 251
    assert 'generator_merge_margin' not in overrides and 'n_to_refine' not in overrides
    # Refinement intervals stay put, so merges keep their out-of-phase cadence.
    assert shared.gv_generator.refine_interval == saved['gv_generator']['refine_interval']


@pytest.mark.parametrize("argv", [
    ['--outside-merge-margin', 'nan'], ['--generator-merge-margin', 'inf'],
    ['--n-to-refine', '0'], ['--max-generator-cells', '-1'], ['--merge-interval', '0'],
])
def test_reject_degenerate_refinement_flags(argv):
    with pytest.raises(SystemExit):
        parse_args(['--no-plots', *argv])


def test_corollary_constants_match_the_shared_trainer():
    # alpha is what compute_loss_init_bounds enforces; eta its generator margin.
    assert ALPHA == 1.0
    assert ETA == GENERATOR_MARGIN == 1e-4
