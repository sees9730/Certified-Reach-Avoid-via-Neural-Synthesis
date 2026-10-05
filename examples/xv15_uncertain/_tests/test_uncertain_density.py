"""Check robust density support and neural bounds without running training."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from examples.xv15_uncertain.model import NominalClosedLoopDrift, XV15Aero, load_config
from examples.xv15_uncertain.neural_certified_nominal_drift.main import (
    configure_reproducibility, make_hyperparameters,
)
from examples.xv15_uncertain.neural_certified_uncertain_param.main import build_problem
from examples.xv15_uncertain.uncertain_density import DensityIntervalDrift, load_density_interval
from src.crown_bounds import SymbolicCROWNCache_Phi
from src.dynamics import Dynamics
from src.phi_module import compute_GV_autograd, create_GV


@pytest.fixture
def problem(tmp_path):
    configure_reproducibility(0)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_config()
    # Keep these density-only regression checks at nominal mass.
    config["uncertainty"]["mass_kg"] = [config["dynamics"]["mass"]] * 2
    params = make_hyperparameters(config, 0, tmp_path)
    yield config, params, build_problem(config, params)
    torch.set_num_threads(old_threads)


def fixed_density_drift(config, controller, density):
    fixed = deepcopy(config)
    fixed["dynamics"]["density"] = density
    return NominalClosedLoopDrift(XV15Aero(fixed), controller)


def sample_states(arrays, count=32):
    full = torch.tensor(arrays["full_range"])
    states = full[:, 0] + torch.rand(count, 3) * (full[:, 1] - full[:, 0])
    return torch.cat([states, torch.cartesian_prod(*[full[i] for i in range(3)])])


@pytest.mark.parametrize("interval", [
    [1.16375, 1.28625], [1.1, 1.4], [1.225, 1.225], [1.3, 1.3],
])
def test_support_equals_maximum_of_physical_endpoints(problem, interval):
    config, _, (_, _, controller, dynamics, _, arrays) = problem
    drift = DensityIntervalDrift(dynamics.f.aero, controller, interval)
    states = sample_states(arrays)
    direction = torch.randn_like(states)
    endpoint_dots = [(fixed_density_drift(config, controller, rho)(states) * direction).sum(1, keepdim=True)
                     for rho in interval]
    expected = torch.maximum(*endpoint_dots)
    actual = drift.support(states, direction)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    # Interior densities are bounded by the same support, with one controller.
    for fraction in (0.25, 0.5, 0.75):
        rho = interval[0] + fraction * (interval[1] - interval[0])
        dot = (fixed_density_drift(config, controller, rho)(states) * direction).sum(1, keepdim=True)
        assert (dot <= actual + 2e-5).all()


def test_shared_density_preserves_lift_drag_correlation(problem):
    _, _, (_, _, controller, dynamics, _, arrays) = problem
    states = torch.tensor(arrays["init_range"].mean(axis=1))[None]
    u = controller(states)
    lift, drag = dynamics.f.aero.forces(states[:, 0], u[:, 1], states[:, 2])
    # Choose a direction orthogonal to df/drho: uncertainty must cancel.
    direction = torch.stack([lift, drag * states[:, 0], torch.zeros_like(lift)], dim=1)
    direction = direction / direction.norm(dim=1, keepdim=True)
    nominal_dot = (dynamics.f(states) * direction).sum(1, keepdim=True)
    torch.testing.assert_close(dynamics.f.support(states, direction), nominal_dot, atol=1e-6, rtol=1e-5)


def endpoint_generators(config, params, value, controller, diffusion):
    generators = []
    for rho in load_density_interval(config):
        fixed = Dynamics.dynamics(f=fixed_density_drift(config, controller, rho), g=diffusion, state_dim=3)
        generators.append(create_GV(value, fixed, params.network,
                                    input_offset=value.input_offset, verify=False))
    return generators


def test_robust_generator_matches_endpoint_autograd_including_diffusion(problem):
    config, params, (value, generator, controller, dynamics, _, arrays) = problem
    states = sample_states(arrays)
    endpoints = endpoint_generators(config, params, value, controller, dynamics.g)
    expected = torch.maximum(*[compute_GV_autograd(gv, states) for gv in endpoints])
    torch.testing.assert_close(generator(states), expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(generator(states), compute_GV_autograd(generator, states), rtol=1e-4, atol=1e-5)
    assert not generator.include_time and not generator.include_energy
    assert value.config.n_inputs == 3


def test_robust_bounds_cover_endpoints_and_have_training_gradients(problem):
    config, params, (value, generator, controller, dynamics, _, arrays) = problem
    boxes = torch.tensor(np.stack([arrays["init_range"], arrays["goal_range"]]))
    lower, upper = boxes[:, :, 0], boxes[:, :, 1]
    cache = SymbolicCROWNCache_Phi(generator, 2, input_dim=3)
    bounds = cache.compute_bounds(lower, upper)
    assert torch.isfinite(bounds).all()
    endpoints = endpoint_generators(config, params, value, controller, dynamics.g)
    for _ in range(8):
        states = lower + torch.rand_like(lower) * (upper - lower)
        for gv in endpoints:
            assert (gv(states).flatten() <= bounds + 1e-5).all()
    bounds.sum().backward()
    for model in (value, controller):
        # A constant output bias has zero state derivatives and is absent
        # from the generator graph. All other weights must receive gradients.
        gradients = [p.grad for name, p in model.named_parameters()
                     if p.requires_grad and not (model is value and name == "output.bias")]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


@pytest.mark.parametrize("interval", [[0, 1.3], [-1, 1], [1.3, 1.1], [1, float("inf")], [float("nan"), 1], [1]])
def test_invalid_density_intervals_are_rejected(interval):
    with pytest.raises(ValueError):
        load_density_interval({"uncertainty": {"air_density_kg_m3": interval}})
