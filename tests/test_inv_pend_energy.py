"""Regression checks for the robust pendulum energy augmentation."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.inv_pend_adversarial.neural_certified import main_constraint_energy as energy
from src.control_network import InvertControlNN, WrapperConterlNN
from src.discretization import (
    compute_rectangular_partition_outside_goal,
    compute_rectangular_partition_outside_goal_and_unsafe,
)
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.regions import Region


def make_models(freeze=False):
    torch.manual_seed(12)
    config = Hyperparameters.default()
    config.network.n_hidden_1 = config.network.n_hidden_2 = 8
    config.network.input_scale = [2 * np.pi, 20.0]
    nominal = create_V(config.network, input_offset=[0.0, 0.0], output_offset=0.1)
    controller = WrapperConterlNN(InvertControlNN(hidden_dim=energy.CONTROLLER_HIDDEN_DIM))
    bundle = {"V_state_dict": nominal.state_dict(), "control_state_dict": controller.state_dict()}
    params = energy.EnergyHyperparameters.from_dict(config.to_dict())
    params.energy_max, params.energy_margin = 0.6, 0.03
    params.network.n_inputs = 3
    params.network.input_scale = [2 * np.pi, 20.0, 0.6]
    return nominal, params, energy.build_networks(params, bundle, freeze)


def test_lift_preserves_nominal_value_and_freezes_controller():
    nominal, params, (value, generator, controller) = make_models(freeze=True)
    x = torch.randn(16, 2)
    for effort in (0.0, 0.3, 0.6):
        z = torch.cat((x, torch.full((16, 1), effort)), dim=1)
        torch.testing.assert_close(value(z), nominal(x), atol=2e-6, rtol=1e-5)
    assert not any(p.requires_grad for p in controller.parameters())
    assert generator.f.controller is controller
    assert params.to_dict()["include_energy"] is True
    assert params.to_dict()["coordinate_order"] == ["theta", "omega", "E"]
    torch.testing.assert_close(generator.f.f_ol.d, torch.tensor([0.0, energy.DRIFT_MAG]))
    assert params.to_dict()["drift_mag"] == energy.DRIFT_MAG > 0


def test_energy_unsafe_band_and_zero_initial_energy():
    regions, boxes = energy.build_energy_regions(0.6, 0.03)
    np.testing.assert_array_equal(regions.init.bounds[-1], [0.0, 0.0])
    assert boxes.shape == (7, 3, 2)
    assert np.all(boxes[-1, -1] == np.array([0.57, 0.6], dtype=np.float32))
    assert regions.unsafe.contains(np.array([0.0, 0.0, 0.58], dtype=np.float32))
    assert not regions.goal.contains(np.array([0.0, 0.0, 0.58], dtype=np.float32))
    assert regions.goal.contains(np.array([0.0, 0.0, 0.2], dtype=np.float32))
    assert not regions.unsafe.contains(np.array([0.0, 0.0, 0.2], dtype=np.float32))
    gap_point = np.array([0.0, 0.0, (regions.goal.upper[-1] + boxes[-1, -1, 0]) / 2], dtype=np.float32)
    generator_boxes = compute_rectangular_partition_outside_goal_and_unsafe(
        regions.full, regions.goal, regions.unsafe)
    assert any(box.contains(gap_point) for box in generator_boxes)


def test_generator_matches_autograd_and_trains_energy_weights():
    _, _, (value, generator, controller) = make_models()
    # Exercise a nonzero dV/dE; the exact nominal lift starts with zero.
    with torch.no_grad():
        value.layer1.weight[:, -1].normal_()
    value.double()
    generator.double()
    z = torch.tensor([[0.4, -0.2, 0.1], [1.2, 0.8, 0.4]], dtype=torch.float64, requires_grad=True)
    observed_inputs = []
    hook = controller.policy_net.register_forward_pre_hook(
        lambda module, inputs: observed_inputs.append(inputs[0].detach().clone()))
    actual = generator(z)
    hook.remove()
    assert observed_inputs
    for inputs in observed_inputs:
        torch.testing.assert_close(inputs, z[:, :-1])
    grad = torch.autograd.grad(value(z).sum(), z, create_graph=True)[0]
    h_omega = torch.autograd.grad(grad[:, 1].sum(), z, create_graph=True)[0][:, 1:2]
    x = z[:, :-1]
    expected = (grad[:, :-1] * generator.f(x)).sum(dim=1, keepdim=True)
    expected = expected + energy.DRIFT_MAG * grad[:, 1:2].abs()
    rate = controller.raw_control(x).square()
    expected = expected + grad[:, -1:] * rate + 0.5 * energy.DYNAMICS["sigma"]**2 * h_omega
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)
    # Independently check the energy term's gradient reaches the state-only policy.
    energy_grad = torch.autograd.grad((grad[:, -1:] * rate).sum(), controller.policy_net.fc1.weight, retain_graph=True)[0]
    assert energy_grad.abs().max() > 0
    actual.sum().backward()
    assert value.layer1.weight.grad[:, -1].abs().max() > 0
    assert controller.policy_net.fc1.weight.grad.abs().max() > 0


def test_baseline_checkpoint_follows_seed():
    args = energy.parse_args(["--seed", "2"])
    assert args.baseline_checkpoint == energy.HERE / "seed2/outputs/eval_bundle.pth"
    assert energy.HERE.name == "neural_certified"


@pytest.mark.parametrize("freeze", [False, True])
def test_pretraining_updates_only_value_and_restores_controller_flags(tmp_path, freeze):
    _, params, (value, generator, controller) = make_models(freeze=freeze)
    regions, unsafe_boxes = energy.build_energy_regions(0.6, 0.03)
    params.training.pretrain_epochs = 10
    params.training.pretrain_n_samples = 24
    before_control = {key: tensor.clone() for key, tensor in controller.state_dict().items()}
    before_value = {key: tensor.clone() for key, tensor in value.state_dict().items()}
    energy.pretrain_value_network(value, generator, controller, regions, unsafe_boxes, params, tmp_path)
    for key, tensor in controller.state_dict().items():
        torch.testing.assert_close(tensor, before_control[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in controller.parameters())
    assert all(parameter.requires_grad == (not freeze) for parameter in controller.parameters())
    assert any(not torch.equal(tensor, before_value[key]) for key, tensor in value.state_dict().items())
    assert (tmp_path / "V_pretrained.pth").is_file()
    assert not (tmp_path / "controller_pretrained.pth").exists()
    # Joint training can still differentiate through the controller afterward.
    if not freeze:
        generator(torch.tensor([[0.4, -0.2, 0.1]])).sum().backward()
        assert controller.policy_net.fc1.weight.grad.abs().max() > 0


def make_baseline_cells():
    """A spatial partition with uneven splits, as after adaptive refinement."""
    regions, unsafe_boxes = energy.build_energy_regions(0.6, 0.03)
    full, goal = Region(regions.full.bounds[:-1]), Region(regions.goal.bounds[:-1])
    unsafe = Region.union(*(Region(box[:-1]) for box in unsafe_boxes[:-1]))
    boxes = {
        "init": [Region(regions.init.bounds[:-1])],
        "goal": [goal],
        "unsafe": unsafe.components,
        "outside": compute_rectangular_partition_outside_goal(full, goal),
        "generator": compute_rectangular_partition_outside_goal_and_unsafe(full, goal, unsafe),
    }
    result = {}
    for name, components in boxes.items():
        cells = [box.to_torch() for box in components]
        lower, upper = cells.pop(0)
        cut = lower[0] + 0.37 * (upper[0] - lower[0])
        left_upper, right_lower = upper.clone(), lower.clone()
        left_upper[0] = right_lower[0] = cut
        result[name] = [(lower, left_upper), (right_lower, upper), *cells]
    return result, regions


def test_inherited_spatial_cells_are_preserved_exactly():
    source, regions = make_baseline_cells()
    result = energy.augment_baseline_cells(source, regions, energy_cells=3)
    def key(cell):
        return tuple(cell[0].tolist()) + tuple(cell[1].tolist())
    # Extra energy-only constraints transfer existing goal/outside cells.
    allowed_sources = {"init": ["init"], "goal": ["goal"],
                       "unsafe": ["unsafe", "goal", "outside"],
                       "outside": ["outside", "goal"], "generator": ["generator", "goal"]}
    for name, cells in result.items():
        inherited = {key(cell) for group in allowed_sources[name] for cell in source[group]}
        projected = {key((lower[:-1], upper[:-1])) for lower, upper in cells}
        assert projected <= inherited
        assert {key(cell) for cell in source[name]} <= projected
    assert len(result["init"]) == len(source["init"])
    assert all(lower[-1] == upper[-1] == 0 for lower, upper in result["init"])
    edges = energy.energy_cell_edges(regions, 3)
    goal_max, ceiling = regions.goal.upper[-1], regions.unsafe.components[-1].lower[-1]
    assert goal_max in edges and ceiling in edges
    n_goal = sum(edges[1:] <= goal_max)
    n_safe = sum(edges[1:] <= ceiling)
    assert len(result["goal"]) == n_goal * len(source["goal"])
    assert len(result["generator"]) == n_safe * len(source["generator"]) + (n_safe - n_goal) * len(source["goal"])
    original = source["init"][0][0].clone()
    result["init"][0][0][1] += 1
    torch.testing.assert_close(source["init"][0][0], original, rtol=0, atol=0)


def test_extrusion_covers_energy_unsafe_band_and_goal_gap():
    source, regions = make_baseline_cells()
    result = energy.augment_baseline_cells(source, regions, energy_cells=1)
    def covered(name, point):
        point = torch.tensor(point)
        return any(bool(((lower <= point) & (point <= upper)).all()) for lower, upper in result[name])
    # New unsafe band covers both the old goal and states outside that goal.
    for theta, omega in [(0.0, 0.0), (3.0, 0.0), (-6.0, -15.0), (1.0, 10.0)]:
        assert covered("unsafe", [theta, omega, 0.58])
    gap_energy = float((regions.goal.upper[-1] + regions.unsafe.components[-1].lower[-1]) / 2)
    assert covered("generator", [0.0, 0.0, gap_energy])
    assert covered("outside", [0.0, 0.0, gap_energy])
    assert not covered("goal", [0.0, 0.0, gap_energy])
    assert not covered("generator", [0.0, 0.0, 0.58])
    assert covered("goal", [0.0, 0.0, 0.1])
    assert not covered("generator", [0.0, 0.0, 0.1])


def test_missing_or_augmented_baseline_cells_are_rejected():
    source, regions = make_baseline_cells()
    with pytest.raises(ValueError, match="final region_cells"):
        energy.augment_baseline_cells(None, regions)
    source["generator"] = [(torch.zeros(3), torch.ones(3))]
    with pytest.raises(ValueError, match="2D baseline cells"):
        energy.augment_baseline_cells(source, regions)


def test_mc_loads_energy_controller_and_uses_same_energy(tmp_path):
    from examples.inv_pend_adversarial import run_mc

    _, _, (_, generator, controller) = make_models()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    checkpoint = outputs / "eval_bundle.pth"
    torch.save({"control_state_dict": controller.state_dict()}, checkpoint)
    args = run_mc.parse_args(["--energy-controller-dir", str(tmp_path)])
    assert run_mc.discover_seed_checkpoints(args.energy_controller_dir, "outputs/eval_bundle.pth") == [(None, checkpoint)]
    loaded = run_mc.load_control_net(checkpoint)
    dt = 0.005
    result = run_mc.rollout_mc(loaded, n_mc=3, T_max=0.02, dt=dt, mode="zero", seed=7, n_paths=3)
    for path, controls, accumulated in zip(result["paths"], result["uraw_paths"], result["energies"]):
        # MC records pre-step u and the corresponding pre-step physical state.
        states = torch.tensor(path[:len(controls)], dtype=torch.float32)
        with torch.no_grad():
            rates = generator._compute_energy_rate(states).squeeze(-1).numpy()
        np.testing.assert_allclose(rates, np.square(controls), rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose(float(rates.sum()) * dt, accumulated, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize("summary", [False, True])
def test_energy_plots_keep_physical_coordinates_before_energy(tmp_path, monkeypatch, summary):
    from functools import partial
    from matplotlib.figure import Figure
    from matplotlib import rcParams
    from src import visualization

    _, _, (value, generator, _) = make_models()
    regions, _ = energy.build_energy_regions(0.6, 0.03)
    # Normalization must not truncate the plotted physical domain.
    value.input_scale[0] = 1.0
    figures, grids = [], []
    monkeypatch.setitem(rcParams, "font.family", ["DejaVu Sans"])
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: figures.append(fig))
    for name in ("visualize_value_function", "visualize_generator", "plot_constraint_regions"):
        monkeypatch.setattr(visualization, name,
                            partial(getattr(visualization, name), resolution=8, figsize=(4, 3)))
    hook = value.register_forward_pre_hook(lambda module, inputs: grids.append(inputs[0].detach().numpy()))
    state_before = {key: tensor.clone() for key, tensor in generator.state_dict().items()}
    rng_before = torch.get_rng_state()
    if summary:
        visualization.create_summary_plots(value, generator, regions, {}, beta_ra=2.0, output_dir=tmp_path)
    else:
        visualization.visualize_training_progress(value, generator, regions, {}, epoch=0, output_dir=tmp_path)
    hook.remove()

    assert len(figures) == (3 if summary else 2)
    for fig in figures:
        assert [(ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes[:2]] == [("x₁", "x₂"), ("x₂", "E")]
        fig.canvas.draw()
    for physical_grid, energy_grid in zip(grids[::2], grids[1::2]):
        np.testing.assert_allclose(physical_grid[:, :2].min(axis=0), regions.full.lower[:2])
        np.testing.assert_allclose(physical_grid[:, :2].max(axis=0), regions.full.upper[:2])
        np.testing.assert_allclose(physical_grid[:, -1], 0.3)
        np.testing.assert_allclose(energy_grid[:, 0], 0.0)
        np.testing.assert_allclose(energy_grid[:, 1:].min(axis=0), regions.full.lower[1:])
        np.testing.assert_allclose(energy_grid[:, 1:].max(axis=0), regions.full.upper[1:])
    for key, tensor in generator.state_dict().items():
        torch.testing.assert_close(tensor, state_before[key], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_mc_pair_comparison_selects_only_requested_runs(tmp_path, monkeypatch):
    from examples.inv_pend_adversarial import run_mc

    baseline, constrained = tmp_path / "baseline/seed4", tmp_path / "energy/run"
    for directory in (baseline, constrained):
        (directory / "outputs").mkdir(parents=True)
        (directory / "outputs/eval_bundle.pth").touch()
    calls = []

    def evaluate(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        res = dict(outcomes=["success", "fail"], energies=[0.1, 0.2], hit_times=[1.0, 2.0],
                   paths=[], uraw_paths=[])
        return {mode: dict(res=res, stats=run_mc.compute_stats(res), viz_paths=[], viz_success=[])
                for mode in run_mc.EVAL_MODES}

    monkeypatch.setattr(run_mc, "evaluate_controller_seed", evaluate)
    plot_dirs = []
    for name in ("plot_phase_trajectories", "plot_energy_vs_thit", "plot_u_raw_trajectories",
                 "plot_failure_trajectories", "plot_success_rate_summary", "plot_energy_distribution"):
        monkeypatch.setattr(run_mc, name, lambda *args, **kwargs: plot_dirs.append(kwargs["save_dir"]))
    output_dir = tmp_path / "comparison"
    run_mc.main(["--baseline-controller-dir", str(baseline), "--energy-controller-dir", str(constrained),
                 "--n-mc", "2", "--output-dir", str(output_dir)])
    assert [path for path, _ in calls] == [directory / "outputs/eval_bundle.pth"
                                         for directory in (baseline, constrained)]
    assert all(kwargs == dict(pretrained=False, n_mc=2, t_max=30.0, dt=0.005, mc_seed=42, n_paths=5)
               for _, kwargs in calls)
    cache = torch.load(output_dir / "mc_cache.pth", weights_only=False)
    assert cache["controller_labels"] == ["Cert.", "Cert. (energy)"]
    assert cache["meta"]["seed_counts"] == {"Cert.": 1, "Cert. (energy)": 1}
    assert cache["meta"]["n_mc"] == 2
    assert set(cache["results"]) == set(run_mc.EVAL_MODES)
    assert cache["meta"]["figures"] == [6, 7, 8]
    assert plot_dirs == [output_dir] * 3


def test_mc_energy_boxplots_group_modes_and_include_all_outcomes(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from examples.inv_pend_adversarial import run_mc

    figures, plotted_data = [], []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: figures.append(fig))
    original_boxplot = Axes.boxplot

    def capture_boxplot(ax, data, **kwargs):
        plotted_data.append(data)
        return original_boxplot(ax, data, **kwargs)

    monkeypatch.setattr(Axes, "boxplot", capture_boxplot)
    labels = ["Cert.", "Cert. (energy)"]
    results = {}
    for index, mode in enumerate(run_mc.EVAL_MODES):
        res = dict(outcomes=["success", "fail", "timeout"],
                   energies=[index + 0.5, 100.0, 200.0], hit_times=[1.0, 2.0, 30.0])
        other = dict(outcomes=["timeout"], energies=[0.2], hit_times=[30.0])
        entries = [(labels[0], res, run_mc.compute_stats(res)),
                   (labels[1], other, run_mc.compute_stats(other))]
        # Pool by controller identity even if an input list has a different order.
        results[mode] = entries if index % 2 else entries[::-1]
    run_mc.plot_energy_distribution(results, run_mc.build_controller_styles(labels), save_dir=tmp_path)
    assert len(plotted_data) == 4
    # Modes are fast, lookahead, nearest_unsafe, velocity, uniform, zero.
    expected_groups = [list(range(6)), [5], [4], [0, 1, 2, 3]]
    for data, indices in zip(plotted_data, expected_groups):
        assert len(data) == 2
        np.testing.assert_allclose(data[0], [value for index in indices
                                            for value in (index + 0.5, 100.0, 200.0)])
        np.testing.assert_allclose(data[1], [0.2] * len(indices))
    assert len(figures) == 1
    fig = figures[0]
    assert len(fig.axes) == 4
    assert [ax.get_title() for ax in fig.axes] == ["Overall", "Zero disturbance", "Uniform disturbance", "Adversarial"]
    for index, ax in enumerate(fig.axes):
        assert len(ax.patches) == 2
        assert ax.get_subplotspec().rowspan.start == 0
        assert ax.get_subplotspec().colspan.start == index
        assert [tick.get_text() for tick in ax.get_xticklabels()] == labels
        expected_means = [np.mean(values) for values in plotted_data[index]]
        assert [text.get_text() for text in ax.texts] == [f"Mean: {mean:.4f}" for mean in expected_means]
        mean_markers = [line for line in ax.lines if line.get_marker() == "D"]
        assert len(mean_markers) == 2
        for position, (annotation, marker, mean) in enumerate(zip(ax.texts, mean_markers, expected_means), start=1):
            np.testing.assert_allclose(annotation.xy, [position, mean])
            np.testing.assert_allclose(marker.get_ydata(), [mean])
    fig.canvas.draw()
