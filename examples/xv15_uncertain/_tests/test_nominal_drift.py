"""Physics, generator and differentiable-bound checks; no training runs."""
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.synthesis.xv15aircraft_syn.main import (
    ClosedLoopDrift as OriginalClosedLoopDrift,
    XV15KLinearAeroTorch as OriginalAero,
)
from examples.xv15_uncertain.model import DEG, DiagonalDiffusion, load_config, load_region_arrays
from examples.xv15_uncertain.neural_certified_nominal_drift.main import (
    build_problem, configure_reproducibility, make_hyperparameters,
)
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi
from src.phi_module import compute_GV_autograd


@pytest.fixture
def problem(tmp_path):
    configure_reproducibility(0)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_config()
    params = make_hyperparameters(config, 0, tmp_path)
    yield build_problem(config, params)
    torch.set_num_threads(old_threads)


def test_problem_units_and_boundary_coverage():
    config = load_config()
    arrays = load_region_arrays(config)
    scale = np.array([1.0, DEG, DEG])[:, None]
    np.testing.assert_allclose(arrays["full_range"], np.array([[0.5, 100], [-20, 20], [0, 90]]) * scale)
    np.testing.assert_allclose(arrays["init_range"], np.array([[28, 32], [8.5, 10.5], [58, 62]]) * scale)
    np.testing.assert_allclose(arrays["goal_range"], np.array([[65, 85], [-2, 10], [25, 35]]) * scale)
    domain = arrays["full_range"]
    for axis in range(3):
        for side in range(2):
            assert any(box[axis, side] == domain[axis, side] and
                       all(np.array_equal(box[j], domain[j]) for j in range(3) if j != axis)
                       for box in arrays["unsafe_ranges"])
    sigma = DiagonalDiffusion(config)(torch.zeros(7, 3))
    torch.testing.assert_close(sigma, torch.tensor([0.5, 0.1 * DEG, 0.1 * DEG]).expand(7, 3))
    assert config["beta_ra"] == 5.0


def test_drift_matches_original_aircraft(problem):
    _, _, controller, dynamics, _, arrays = problem
    full = torch.tensor(arrays["full_range"])
    states = full[:, 0] + torch.rand(128, 3) * (full[:, 1] - full[:, 0])
    states = torch.cat([states, torch.cartesian_prod(*[full[i] for i in range(3)])])
    reference = OriginalClosedLoopDrift(OriginalAero(), controller)
    torch.testing.assert_close(dynamics.f(states), reference(states), rtol=2e-5, atol=2e-5)


def test_controller_bounds_trim_and_gradients(problem):
    _, _, controller, dynamics, _, arrays = problem
    # Include extreme inputs to exercise saturation as well as the physical domain.
    x = torch.randn(256, 3) * torch.tensor([1000.0, 10.0, 10.0])
    u = controller(x)
    assert torch.isfinite(u).all()
    assert (u[:, 0] >= controller.T_min).all() and (u[:, 0] <= controller.T_max).all()
    assert (u[:, 1].abs() <= controller.alpha_max).all()
    assert (u[:, 2].abs() <= controller.delta_max).all()
    goal = torch.tensor(arrays["goal_range"])
    assert (controller.x_eq > goal[:, 0]).all() and (controller.x_eq < goal[:, 1]).all()
    torch.testing.assert_close(controller(controller.x_eq[None])[0], controller.u_eq)
    torch.testing.assert_close(dynamics.f(controller.x_eq[None]), torch.zeros(1, 3), atol=1e-5, rtol=0)
    initial = torch.tensor(arrays["init_range"].mean(axis=1))[None].requires_grad_(True)
    torch.autograd.grad(controller(initial).sum(), initial)[0].isfinite().all().item()


def test_generator_matches_autograd_with_diffusion(problem):
    value, generator, _, _, _, arrays = problem
    full = torch.tensor(arrays["full_range"])
    x = full[:, 0] + torch.rand(32, 3) * (full[:, 1] - full[:, 0])
    x = torch.cat([x, torch.cartesian_prod(*[full[i] for i in range(3)])])
    actual = generator(x)
    expected = compute_GV_autograd(generator, x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert value.config.n_inputs == 3
    assert not generator.include_time and not generator.include_energy


def test_bounds_enclose_samples_and_backpropagate(problem):
    value, generator, controller, _, _, arrays = problem
    boxes = torch.tensor(np.stack([arrays["init_range"], arrays["goal_range"]]))
    lower, upper = boxes[:, :, 0], boxes[:, :, 1]
    value_cache = SymbolicCROWNCache(value, 2, input_dim=3)
    generator_cache = SymbolicCROWNCache_Phi(generator, 2, input_dim=3)
    vl, vu = value_cache.compute_bounds(lower, upper)
    gu = generator_cache.compute_bounds(lower, upper)
    assert torch.isfinite(torch.cat([vl, vu, gu])).all()
    for _ in range(8):
        points = lower + torch.rand_like(lower) * (upper - lower)
        with torch.no_grad():
            v, gv = value(points).flatten(), generator(points).flatten()
        assert (v >= vl - 1e-5).all() and (v <= vu + 1e-5).all()
        assert (gv <= gu + 1e-5).all()
    (gu.sum() + vu.sum()).backward()
    for model in (value, controller):
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
