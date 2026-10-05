"""Regression checks for loss scaling and safety-first generator updates."""
import pytest
import torch

from examples.planar_kepler.model import load_config
from examples.planar_kepler.neural_certified_nominal_drift.main import build_problem, make_hyperparameters
from examples.planar_kepler.neural_certified_nominal_drift.training_schedule import SafetyFirstGeneratorSchedule
from src.crown_bounds import SymbolicCROWNCache
from src.hyperparameters import Hyperparameters
from src.training_utils import compute_total_loss_bounds


def safety_bounds(unsafe=5.1):
    return dict(unsafe=(torch.tensor([unsafe]), torch.tensor([6.])),
                init=(torch.tensor([.1]), torch.tensor([.9])),
                goal=(torch.tensor([.05]), torch.tensor([.2])),
                outside=(torch.tensor([.05]), torch.tensor([6.])))


def test_generator_waits_for_safety_then_ramps_and_backs_off():
    schedule = SafetyFirstGeneratorSchedule(5., warmup_epochs=2, ramp_epochs=4)
    assert schedule(0, safety_bounds()) == 0
    assert schedule(2, safety_bounds(unsafe=.1)) == 0
    assert schedule(3, safety_bounds()) == .25
    assert schedule(4, safety_bounds()) == .5
    assert schedule(5, safety_bounds(unsafe=4.)) == .25
    assert schedule(6, safety_bounds()) == .5
    assert schedule(7, safety_bounds()) == .75
    assert schedule(8, safety_bounds()) == 1.
    assert schedule(9, safety_bounds()) == 1.


def test_safety_gate_uses_worst_cell_and_rejects_nonfinite_bounds():
    schedule = SafetyFirstGeneratorSchedule(5., warmup_epochs=0, ramp_epochs=4)
    bounds = safety_bounds()
    bounds['unsafe'] = (torch.cat([torch.full((1000,), 5.2), torch.tensor([.1])]), torch.tensor([6.]))
    assert schedule(0, bounds) == 0
    assert schedule(1, safety_bounds(unsafe=float('nan'))) == 0
    assert schedule(2, safety_bounds()) == .25
    assert schedule(3, safety_bounds(unsafe=float('-inf'))) == .125


def test_mean_losses_do_not_inflate_when_cells_are_duplicated_and_sat_is_unchanged():
    kwargs = dict(beta_ra=5., V_goal_lower=torch.tensor([-.1, .2]),
                  V_unsafe_lower=torch.tensor([4., 5.2]), V_init_upper=torch.tensor([1.2, .3]),
                  V_outside_lower=torch.tensor([-.5, .3]), Phi_upper=torch.tensor([.2, -.1]),
                  V_generator_lower=torch.tensor([.1, .1]), generator_weight=.5)
    loss, parts, sat = compute_total_loss_bounds(**kwargs, loss_reduction='mean')
    doubled = {k: v.repeat(2) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    loss2, parts2, sat2 = compute_total_loss_bounds(**doubled, loss_reduction='mean')
    torch.testing.assert_close(loss, loss2)
    assert parts == parts2 and sat == sat2
    _, _, sum_sat = compute_total_loss_bounds(**kwargs)
    assert sat == sum_sat and sat['unsafe'] is False and sat['generator'] is False
    sum_loss, _, _ = compute_total_loss_bounds(**kwargs)
    sum_loss2, _, _ = compute_total_loss_bounds(**doubled)
    torch.testing.assert_close(sum_loss2, 2 * sum_loss)


def test_training_controls_survive_hyperparameter_round_trip(tmp_path):
    params = make_hyperparameters(load_config(), 0, tmp_path)
    saved = Hyperparameters.from_dict(params.to_dict())
    assert saved.training.loss_reduction == 'mean'
    assert saved.training.loss_weights['unsafe'] == 5.
    assert saved.training.max_grad_norm == 1.
    assert Hyperparameters.default().training.loss_reduction == 'sum'


def test_initial_certificate_separates_unsafe_faces_from_initial_set(tmp_path):
    config = load_config()
    params = make_hyperparameters(config, 0, tmp_path)
    value, _, _, _, _, arrays = build_problem(config, params)
    boxes = torch.tensor(arrays['unsafe_ranges'])
    cache = SymbolicCROWNCache(value, 8, input_dim=4)
    lower, _ = cache.compute_bounds(boxes[:, :, 0], boxes[:, :, 1])
    assert float(lower.min()) > 4.0
    init = torch.tensor(arrays['init_range'])
    init_cache = SymbolicCROWNCache(value, 1, input_dim=4)
    _, upper = init_cache.compute_bounds(init[:, 0][None], init[:, 1][None])
    assert float(upper.max()) < 1.


@pytest.mark.parametrize('kwargs', [dict(warmup_epochs=-1), dict(ramp_epochs=0), dict(target_weight=float('nan'))])
def test_invalid_schedule_parameters(kwargs):
    with pytest.raises(ValueError):
        SafetyFirstGeneratorSchedule(5., **kwargs)
