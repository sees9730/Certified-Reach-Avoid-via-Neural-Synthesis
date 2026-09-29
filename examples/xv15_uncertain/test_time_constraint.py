"""Clock augmentation checks; no pretraining or bound-training runs."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from examples.xv15_uncertain.neural_certified_uncertain_param import main_constraint_energy as energy
from examples.xv15_uncertain.neural_certified_uncertain_param import main_constraint_time as clock
from examples.xv15_uncertain.test_energy_constraint import baseline
from src.crown_bounds import SymbolicCROWNCache_Phi


def setup_clock(baseline, freeze=False):
    config, _, bundle, checkpoint, _ = baseline
    args = clock.parse_args(['--energy-max', '20', '--energy-margin', '1', '--no-plots'])
    params = clock.make_energy_parameters(bundle, checkpoint, args)
    return params, clock.build_networks(config, params, bundle, freeze)


@pytest.mark.parametrize('freeze', [False, True])
def test_clock_preserves_warm_start_and_training_contract(baseline, freeze):
    config, _, bundle, _, (original, _, original_controller, _, arrays) = baseline
    params, (value, generator, controller) = setup_clock(baseline, freeze)
    x = torch.tensor(arrays['init_range'].mean(axis=1)).expand(4, -1)
    z = torch.cat((x, torch.tensor([[0.], [5.], [19.], [20.]])), dim=1)
    torch.testing.assert_close(value(z), original(x))
    torch.testing.assert_close(controller(x), original_controller(x), rtol=0, atol=0)
    assert all(p.requires_grad == (not freeze) for p in controller.parameters())
    assert generator.f.controller is controller
    assert generator.include_energy and not generator.include_time
    metadata = clock.TimeHyperparameters.from_dict(params.to_dict()).to_dict()
    assert metadata['clock_rate'] == 1.0
    assert metadata['energy_definition'] == 'integral(1 dt) = elapsed time in seconds'
    assert metadata['energy_budget'] == 19.0
    assert metadata['coordinate_order'] == ['v', 'gamma', 'beta', 'E']
    regions, boxes = clock.build_energy_regions(config, 20., 1.)
    old_regions, old_boxes = energy.build_energy_regions(config, 20., 1.)
    np.testing.assert_array_equal(boxes, old_boxes)
    cells = clock.augment_baseline_cells(bundle['region_cells'], regions, 1)
    old_cells = energy.augment_baseline_cells(bundle['region_cells'], old_regions, 1)
    for name in clock.GROUPS:
        assert len(cells[name]) == len(old_cells[name])
        for cell, old_cell in zip(cells[name], old_cells[name]):
            for bound, old_bound in zip(cell, old_cell):
                torch.testing.assert_close(bound, old_bound, atol=0, rtol=0)


def test_clock_generator_matches_autograd_and_effort_difference(baseline):
    config, _, bundle, _, (_, _, _, _, arrays) = baseline
    params, (value, generator, controller) = setup_clock(baseline)
    _, effort, _ = energy.build_networks(config, params, bundle)
    # The initial lift has dV/dE=0; a nonzero column is essential to this check.
    with torch.no_grad():
        value.layer1.weight[:, -1].normal_(0., 0.5)
    effort.V_net.load_state_dict(value.state_dict())
    generator.double()
    effort.double()
    x = torch.tensor(arrays['init_range'], dtype=torch.float64)
    x = x[:, 0] + torch.rand(8, 3, dtype=torch.float64) * (x[:, 1] - x[:, 0])
    z = torch.cat((x, torch.linspace(0, 20, 8, dtype=torch.float64)[:, None]), dim=1)
    observed = []
    hook = controller.register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0]))
    actual = generator(z)
    hook.remove()
    assert observed and all(state.shape[-1] == 3 for state in observed)
    torch.testing.assert_close(actual, clock.compute_clock_GV_autograd(generator, z), atol=1e-9, rtol=1e-7)
    probe = z.clone().requires_grad_(True)
    dVdE = torch.autograd.grad(value(probe).sum(), probe)[0][:, -1:]
    assert dVdE.abs().max() > 1e-8
    effort_rate = controller.raw_control(x).square().sum(dim=1, keepdim=True)
    torch.testing.assert_close(actual - effort(z), dVdE * (1 - effort_rate), atol=1e-9, rtol=1e-7)
    actual.sum().backward()
    assert value.layer1.weight.grad[:, -1].abs().max() > 0
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in controller.parameters())


def test_clock_rate_does_not_call_controller(baseline, monkeypatch):
    _, (_, generator, controller) = setup_clock(baseline)

    def forbidden(*args):
        raise AssertionError('The clock rate must not evaluate the policy')

    monkeypatch.setattr(controller, 'raw_control', forbidden)
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(5, 3, dtype=dtype)
        torch.testing.assert_close(generator._compute_energy_rate(x), torch.ones(5, 1, dtype=dtype))


def test_clock_supports_shared_bound_evaluation_without_training(baseline):
    _, (value, generator, _) = setup_clock(baseline)
    with torch.no_grad():
        value.layer1.weight[:, -1].normal_(0., 0.5)
    before = deepcopy(generator.state_dict())
    arrays = baseline[-1][-1]
    x = torch.tensor(arrays['init_range'].mean(axis=1)).expand(2, -1)
    z = torch.cat((x, torch.tensor([[1.], [10.]])), dim=1)
    cache = SymbolicCROWNCache_Phi(generator, num_cells=2, input_dim=4, device='cpu')
    radius = torch.tensor([1e-3, 1e-5, 1e-5, 1e-3])
    upper = cache.compute_bounds(z - radius, z + radius)
    assert upper.shape == (2,) and torch.isfinite(upper).all()
    center = generator(z).squeeze(-1)
    assert (center <= upper + 1e-5).all()
    upper.sum().backward()
    assert value.layer1.weight.grad is not None
    assert torch.isfinite(value.layer1.weight.grad).all()
    for name, state in generator.state_dict().items():
        torch.testing.assert_close(state, before[name], atol=0, rtol=0)
