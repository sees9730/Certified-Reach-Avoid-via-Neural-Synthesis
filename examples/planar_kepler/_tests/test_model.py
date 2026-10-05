"""Physics, four-dimensional coverage, generator, and differentiable-bound checks."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from examples.planar_kepler.model import (
    AccelerationDiffusion, PlanarKepler, load_config, load_region_arrays,
)
from examples.planar_kepler.neural_certified_nominal_drift.main import (
    build_problem, configure_reproducibility, make_hyperparameters,
)
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi
from src.discretization import discretize_regions
from src.phi_module import compute_GV_autograd


@pytest.fixture
def problem(tmp_path):
    configure_reproducibility(0)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_config()
    params = make_hyperparameters(config, 0, tmp_path)
    yield config, params, build_problem(config, params)
    torch.set_num_threads(old_threads)


def test_polar_drift_matches_cartesian_inverse_square_gravity_and_thrust():
    physics = PlanarKepler(load_config())
    x = torch.tensor([[0.8, 0.3, 0.2, 0.4], [1.5, -0.4, -0.3, -0.2]], dtype=torch.float64)
    u = torch.tensor([[0.1, -0.2], [-0.15, 0.3]], dtype=torch.float64)
    drift = physics(x, u)
    r, theta, v, omega = x.unbind(dim=1)
    er = torch.stack([theta.cos(), theta.sin()], dim=1)
    et = torch.stack([-theta.sin(), theta.cos()], dim=1)
    # Convert polar coordinate accelerations into inertial Cartesian acceleration.
    actual = (drift[:, 2] - r * omega ** 2)[:, None] * er
    actual += (r * drift[:, 3] + 2 * v * omega)[:, None] * et
    expected = (-physics.mu / r ** 2 + u[:, 0])[:, None] * er + u[:, 1, None] * et
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(drift[:, :2], x[:, 2:])


def test_uncontrolled_kepler_conserves_energy_and_angular_momentum():
    physics = PlanarKepler(load_config())
    x = torch.tensor([[0.8, 0.3, 0.2, 0.4], [1.5, -0.4, -0.3, -0.2]],
                     dtype=torch.float64, requires_grad=True)
    r, _, v, omega = x.unbind(dim=1)
    energy = (v ** 2 + r ** 2 * omega ** 2) / 2 - physics.mu / r
    momentum = r ** 2 * omega
    drift = physics(x, torch.zeros(2, 2, dtype=x.dtype))
    for conserved in (energy, momentum):
        gradient = torch.autograd.grad(conserved.sum(), x, retain_graph=True)[0]
        torch.testing.assert_close((gradient * drift).sum(dim=1), torch.zeros(2, dtype=x.dtype),
                                   rtol=0, atol=1e-12)


def test_circular_orbit_requires_no_control():
    config = load_config()
    r = 1.2
    omega = np.sqrt(config["dynamics"]["mu"] / r ** 3)
    x = torch.tensor([[r, 0.2, 0, omega]], dtype=torch.float64)
    drift = PlanarKepler(config)(x, torch.zeros(1, 2, dtype=x.dtype))
    torch.testing.assert_close(drift, torch.tensor([[0, omega, 0, 0]], dtype=x.dtype), rtol=0, atol=1e-12)


def test_diffusion_is_acceleration_noise_with_tangential_radius_conversion():
    config = load_config()
    x = torch.tensor([[0.8, 0, 0, 0], [1.5, 0, 0, 0]])
    actual = AccelerationDiffusion(config)(x)
    sigma_r = config["dynamics"]["radial_acceleration_noise"]
    sigma_t = config["dynamics"]["tangential_acceleration_noise"]
    torch.testing.assert_close(actual, torch.tensor([[0, 0, sigma_r, sigma_t / 0.8],
                                                    [0, 0, sigma_r, sigma_t / 1.5]]))
    zero_noise = deepcopy(config)
    zero_noise["dynamics"].update(radial_acceleration_noise=0, tangential_acceleration_noise=0)
    assert not AccelerationDiffusion(zero_noise)(x).any()


def test_controller_bounds_goal_equilibrium_and_generator(problem):
    _, _, (value, generator, controller, dynamics, _, arrays) = problem
    x = torch.randn(256, 4) * 1000
    u = controller(x)
    assert torch.isfinite(u).all()
    assert (u >= controller.control_low).all() and (u <= controller.control_high).all()
    torch.testing.assert_close(controller(controller.x_eq[None])[0], controller.u_eq)
    torch.testing.assert_close(dynamics.f(controller.x_eq[None]), torch.zeros(1, 4), rtol=0, atol=1e-6)
    assert value.config.n_inputs == 4 and dynamics.state_dim == 4
    full = torch.tensor(arrays["full_range"])
    points = full[:, 0] + torch.rand(32, 4) * (full[:, 1] - full[:, 0])
    points = torch.cat([points, torch.cartesian_prod(*full)])
    torch.testing.assert_close(generator(points), compute_GV_autograd(generator, points), rtol=1e-4, atol=1e-5)
    # Zero inputs used while tracing must be finite, although outside the physical domain.
    assert torch.isfinite(generator(torch.zeros(2, 4))).all()


def test_bounds_enclose_samples_and_train_controller(problem):
    _, _, (value, generator, controller, _, _, arrays) = problem
    boxes = torch.tensor(np.stack([arrays["init_range"], arrays["goal_range"]]))
    low, high = boxes[:, :, 0], boxes[:, :, 1]
    value_cache = SymbolicCROWNCache(value, 2, input_dim=4)
    generator_cache = SymbolicCROWNCache_Phi(generator, 2, input_dim=4)
    vl, vu = value_cache.compute_bounds(low, high)
    gu = generator_cache.compute_bounds(low, high)
    assert torch.isfinite(torch.cat([vl, vu, gu])).all()
    for _ in range(12):
        points = low + torch.rand_like(low) * (high - low)
        with torch.no_grad():
            v, gv = value(points).flatten(), generator(points).flatten()
        assert (v >= vl - 1e-5).all() and (v <= vu + 1e-5).all()
        assert (gv <= gu + 1e-5).all()
    gu.sum().backward()
    for parameter in controller.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_4d_discretization_covers_safe_interior_minus_goal(problem):
    config, params, (_, _, _, _, regions, arrays) = problem
    assert arrays["unsafe_ranges"].shape == (8, 4, 2)
    cells = discretize_regions(regions, params.discretization, use_radial_generator=False)
    full = arrays["full_range"].astype(np.float64)
    width = np.asarray(config["regions"]["unsafe_boundary_width"])
    interior_volume = np.prod(full[:, 1] - full[:, 0] - 2 * width)
    goal_volume = np.prod(np.diff(arrays["goal_range"].astype(np.float64), axis=1))
    actual_volume = sum(float(torch.prod((hi - lo).double())) for lo, hi in cells["generator"])
    assert actual_volume == pytest.approx(interior_volume - goal_volume, rel=1e-6)
    for axis in range(4):
        for side in range(2):
            box = arrays["unsafe_ranges"][2 * axis + side]
            assert box[axis, side] == arrays["full_range"][axis, side]
            assert all(np.array_equal(box[j], arrays["full_range"][j]) for j in range(4) if j != axis)


@pytest.mark.parametrize("invalid", ["zero_radius", "negative_noise", "goal_velocity", "bad_trim", "wrapped_angle"])
def test_reject_invalid_physical_problem(invalid, tmp_path):
    config = load_config()
    if invalid == "zero_radius":
        config["regions"]["full_range"][0][0] = 0
    elif invalid == "negative_noise":
        config["dynamics"]["radial_acceleration_noise"] = -0.1
    elif invalid == "goal_velocity":
        config["regions"]["goal_range"][3] = [0.1, 0.2]
    elif invalid == "bad_trim":
        config["control"]["radial_acceleration"] = [-0.1, 0.1]
    else:
        config["regions"]["full_range"][1] = [-4, 4]
    with pytest.raises(ValueError):
        params = make_hyperparameters(config, 0, tmp_path)
        build_problem(config, params)
