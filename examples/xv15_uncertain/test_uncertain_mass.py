"""Physical corner, generator, and differentiable-bound checks for mass uncertainty."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch

from examples.xv15_uncertain.model import NominalClosedLoopDrift, XV15Aero, load_config
from examples.xv15_uncertain.neural_certified_nominal_drift.main import (
    configure_reproducibility, make_hyperparameters,
)
from examples.xv15_uncertain.neural_certified_uncertain_param.main import build_problem
from examples.xv15_uncertain.uncertain_density import DensityIntervalDrift, load_density_interval
from examples.xv15_uncertain.uncertain_parameters import DensityMassIntervalDrift, load_mass_interval
from src.crown_bounds import SymbolicCROWNCache_Phi
from src.dynamics import Dynamics
from src.phi_module import compute_GV_autograd, create_GV


@pytest.fixture
def problem(tmp_path):
    configure_reproducibility(0)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_config()
    # Exercise +/-1% independently of the user's current experiment settings.
    config["uncertainty"]["mass_kg"] = [0.99 * config["dynamics"]["mass"],
                                        1.01 * config["dynamics"]["mass"]]
    params = make_hyperparameters(config, 0, tmp_path)
    yield config, params, build_problem(config, params)
    torch.set_num_threads(old_threads)


def physical_drift(config, controller, density, mass):
    fixed = deepcopy(config)
    fixed["dynamics"].update(density=density, mass=mass)
    # Reuse the SAME controller so actual mass cannot alter thrust limits.
    return NominalClosedLoopDrift(XV15Aero(fixed), controller)


def sample_states(arrays):
    full = torch.tensor(arrays["full_range"])
    states = full[:, 0] + torch.rand(64, 3) * (full[:, 1] - full[:, 0])
    return torch.cat([states, torch.cartesian_prod(*full)])


def test_mass_interval_and_legacy_config():
    config = load_config()
    nominal = config["dynamics"]["mass"]
    config["uncertainty"]["mass_kg"] = [0.99 * nominal, 1.01 * nominal]
    assert load_mass_interval(config) == pytest.approx((0.99 * nominal, 1.01 * nominal))
    del config["uncertainty"]["mass_kg"]
    assert load_mass_interval(config) == (nominal, nominal)


@pytest.mark.parametrize("density,mass", [
    ([1.16375, 1.28625], [5841, 5959]),
    ([1.1, 1.4], [5600, 6100]),
    ([1.225, 1.225], [5841, 5959]),
    ([1.16375, 1.28625], [5900, 5900]),
    ([1.225, 1.225], [5900, 5900]),
    ([1.3, 1.3], [6000, 6000]),
])
def test_support_matches_physical_corners_and_encloses_interior(problem, density, mass):
    config, _, (_, _, controller, dynamics, _, arrays) = problem
    drift = DensityMassIntervalDrift(dynamics.f.aero, controller, density, mass)
    states = sample_states(arrays)
    # Include opposite directions to exercise both mass endpoints.
    states = states.repeat(2, 1)
    p = torch.randn_like(states[:len(states) // 2])
    p = torch.cat([p, -p])
    dots = [(physical_drift(config, controller, rho, m)(states) * p).sum(1, keepdim=True)
            for rho, m in product(density, mass)]
    expected = torch.stack(dots).amax(dim=0)
    actual = drift.support(states, p)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)
    for support in drift.support_forms(states, p):
        torch.testing.assert_close(support, expected, rtol=2e-5, atol=3e-5)
    for a, b in product((0.25, 0.5, 0.75), repeat=2):
        rho = density[0] + a * (density[1] - density[0])
        m = mass[0] + b * (mass[1] - mass[0])
        dot = (physical_drift(config, controller, rho, m)(states) * p).sum(1, keepdim=True)
        assert (dot <= actual + 3e-5).all()


def corner_generators(config, params, value, controller, diffusion):
    return [create_GV(value, Dynamics.dynamics(
        f=physical_drift(config, controller, rho, mass), g=diffusion, state_dim=3),
        params.network, input_offset=value.input_offset, verify=False)
        for rho, mass in product(load_density_interval(config), load_mass_interval(config))]


def test_training_generator_matches_corner_autograd_with_diffusion(problem):
    config, params, (value, generator, controller, dynamics, _, arrays) = problem
    states = sample_states(arrays)
    corners = corner_generators(config, params, value, controller, dynamics.g)
    expected = torch.stack([compute_GV_autograd(gv, states) for gv in corners]).amax(dim=0)
    torch.testing.assert_close(generator(states), expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(generator(states), compute_GV_autograd(generator, states), rtol=1e-4, atol=1e-5)
    density_only = DensityIntervalDrift(dynamics.f.aero, controller, load_density_interval(config))
    torch.testing.assert_close(dynamics.f(states), density_only(states))
    p = torch.randn_like(states)
    correction = dynamics.f.support(states, p) - density_only.support(states, p)
    assert (correction >= -3e-5).all()
    assert correction.max() > 1e-3
    assert value.config.n_inputs == 3
    assert not generator.include_time and not generator.include_energy


def test_bounds_enclose_all_parameter_corners_and_backpropagate(problem):
    config, params, (value, generator, controller, dynamics, _, arrays) = problem
    boxes = torch.tensor(np.stack([arrays["init_range"], arrays["goal_range"]]))
    lower, upper = boxes[:, :, 0], boxes[:, :, 1]
    cache = SymbolicCROWNCache_Phi(generator, 2, input_dim=3)
    bounds = cache.compute_bounds(lower, upper)
    assert torch.isfinite(bounds).all()
    corners = corner_generators(config, params, value, controller, dynamics.g)
    for _ in range(8):
        states = lower + torch.rand_like(lower) * (upper - lower)
        for gv in corners:
            assert (gv(states).flatten() <= bounds + 1e-5).all()
    bounds.sum().backward()
    for model in (value, controller):
        gradients = [p.grad for name, p in model.named_parameters()
                     if p.requires_grad and not (model is value and name == "output.bias")]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


class SupportFormDrift(DensityMassIntervalDrift):
    """Trace one equivalent formula to compare its propagated upper bound."""
    def __init__(self, *args, form):
        super().__init__(*args)
        self.form = form

    def support(self, x, p):
        return self.support_forms(x, p)[self.form]


def test_combined_ibp_selects_tightest_form_on_each_cell(problem):
    config, params, (value, generator, controller, dynamics, _, arrays) = problem
    full = torch.tensor(arrays["full_range"])
    width = (full[:, 1] - full[:, 0]) * 0.01
    lower = full[:, 0] + torch.rand(12, 3) * (full[:, 1] - full[:, 0] - width)
    upper = lower + width
    # Include the same broad boxes used to identify the original regression.
    boxes = torch.tensor(np.stack([arrays["init_range"], arrays["goal_range"]]))
    lower = torch.cat([boxes[:, :, 0], lower])
    upper = torch.cat([boxes[:, :, 1], upper])
    cache = SymbolicCROWNCache_Phi(generator, len(lower), input_dim=3)
    bounds = cache.compute_bounds(lower, upper)
    form_bounds = []
    for form in range(3):
        drift = SupportFormDrift(dynamics.f.aero, controller, load_density_interval(config),
                                 load_mass_interval(config), form=form)
        gv = create_GV(value, Dynamics.dynamics(f=drift, g=dynamics.g, state_dim=3),
                       params.network, input_offset=value.input_offset, verify=False)
        form_cache = SymbolicCROWNCache_Phi(gv, len(lower), input_dim=3)
        form_bounds.append(form_cache.compute_bounds(lower, upper))
    expected = torch.stack(form_bounds).amin(dim=0)
    torch.testing.assert_close(bounds, expected, atol=1e-6, rtol=1e-5)
    assert (bounds <= form_bounds[0] + 1e-6).all()
    assert bounds[0] < 0.8 * form_bounds[0][0]
    # Zero-width state boxes should recover exact robust point evaluations.
    mid = (lower + upper) / 2
    torch.testing.assert_close(cache.compute_bounds(mid, mid), generator(mid).flatten(),
                               atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("interval", [[0, 5900], [-1, 5900], [6000, 5900],
                                     [5900, float("inf")], [float("nan"), 5900], [5900]])
def test_invalid_mass_intervals_are_rejected(interval):
    config = load_config()
    config["uncertainty"]["mass_kg"] = interval
    with pytest.raises(ValueError):
        load_mass_interval(config)
