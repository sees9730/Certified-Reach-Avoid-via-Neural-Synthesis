"""Regression checks for the nominal double-integrator example (not training)."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.asteroid_landing_uncertain.model import (
    DoubleIntegratorDrift, NeuralControl, DiagonalDiffusion, load_config,
    load_region_arrays, validate_config,
)
from examples.asteroid_landing_uncertain.neural_certified_nominal_drift.main import build_problem, make_hyperparameters
from src.crown_bounds import SymbolicCROWNCache_Phi
from src.phi_module import verify_GV


def test_drift_at_origin_and_arbitrary_positions():
    x = torch.tensor([[0., 0., 0., 0.], [7., -3., 0.2, -0.4]])
    u = torch.tensor([[0.3, -0.5], [-0.8, 0.6]])
    actual = DoubleIntegratorDrift()(x, u)
    torch.testing.assert_close(actual, torch.tensor([[0., 0., 0.3, -0.5], [0.2, -0.4, -0.8, 0.6]]))


def test_control_limits_and_velocity_covariance():
    config = load_config()
    config['control']['u_max'] = [0.3, 0.8]
    controller = NeuralControl(config, [0.] * 4, [1.] * 4)
    with torch.no_grad():
        controller.fc2.weight.zero_()
        controller.fc2.bias.copy_(torch.tensor([100., -100.]))
    torch.testing.assert_close(controller(torch.zeros(1, 4)), torch.tensor([[0.3, -0.8]]))
    sigma = DiagonalDiffusion(config)(torch.zeros(3, 4))
    expected = torch.tensor([0., 0., *config['dynamics']['velocity_diffusion_rate']])
    torch.testing.assert_close(sigma.square(), expected.expand(3, -1))


def test_boxes_and_legacy_config_rejection():
    config = load_config()
    arrays = load_region_arrays(config)
    assert arrays['unsafe_ranges'].shape == (8, 4, 2)
    assert np.all(arrays['goal_range'][:, 0] < 0) and np.all(arrays['goal_range'][:, 1] > 0)
    for axis in range(4):
        assert arrays['unsafe_ranges'][2 * axis, axis, 0] == arrays['full_range'][axis, 0]
        assert arrays['unsafe_ranges'][2 * axis + 1, axis, 1] == arrays['full_range'][axis, 1]
    old = deepcopy(config)
    old.pop('problem')
    with pytest.raises(ValueError, match='incompatible'):
        validate_config(old)
    touching = deepcopy(config)
    touching['initial']['px'] = [0.92, 0.95]
    with pytest.raises(ValueError, match='unsafe'):
        validate_config(touching)


def test_generator_matches_autograd_and_bounds_cross_origin(tmp_path):
    config = load_config()
    params = make_hyperparameters(config, 0, tmp_path, 8)
    value, generator, _, dynamics, _, _ = build_problem(config, params)
    points = torch.tensor([[0., 0., 0., 0.], [0.5, 0.2, 0.02, -0.01], [-0.5, -0.2, 0.1, 0.2]])
    assert verify_GV(generator, dynamics=dynamics, x=points, tol=1e-4, verbose=False)
    # The old gravitational drift was singular at the origin. The new graph
    # must give finite sound bounds on a box containing it.
    cache = SymbolicCROWNCache_Phi(generator, 1, 4, 'cpu')
    lo, hi = torch.full((1, 4), -0.01), torch.full((1, 4), 0.01)
    upper = cache.compute_bounds(lo, hi)
    assert torch.isfinite(upper).all()
    assert (generator(torch.zeros(1, 4)).reshape(-1) <= upper + 1e-5).all()
