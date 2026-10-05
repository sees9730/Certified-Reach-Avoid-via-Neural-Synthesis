"""Four-state event logic and reproducible empirical rollouts."""
import torch

from examples.planar_kepler.model import KeplerEqMLPControl, find_goal_equilibrium, load_config, load_region_arrays
from examples.planar_kepler.run_mc import classify, rollout_mc


def test_events_check_velocity_even_at_target_position():
    arrays = {key: torch.tensor(value) for key, value in load_region_arrays(load_config()).items()}
    x = torch.tensor([[1., 0., 0., 0.], [1., 0., .2, 0.], [1., 0., 0., .58],
                      [.5, 0., 0., 0.], [float('nan'), 0., 0., 0.]])
    assert classify(x, arrays).tolist() == [1, 0, 2, 3, 4]


def test_mc_repeatability_and_kinematics():
    config = load_config()
    x_eq, u_eq = find_goal_equilibrium(config)
    controller = KeplerEqMLPControl(config, x_eq, u_eq, [1.] * 4)
    kwargs = dict(n_mc=4, n_paths=4, dt=.01, t_max=.03, seed=7)
    first = rollout_mc(controller, config, **kwargs)
    second = rollout_mc(controller, config, **kwargs)
    torch.testing.assert_close(first["final_states"], second["final_states"], rtol=0, atol=0)
    step = first["path_states"][1] - first["initial_states"]
    # Position increments are deterministic conditional on the starting velocity.
    torch.testing.assert_close(step[:, :2], .01 * first["initial_states"][:, 2:], rtol=1e-10, atol=1e-12)
    deterministic = rollout_mc(controller, config, stochastic=False, **kwargs)
    assert not torch.equal(first["final_states"][:, 2:], deterministic["final_states"][:, 2:])


def test_mc_accepts_no_saved_paths():
    config = load_config()
    x_eq, u_eq = find_goal_equilibrium(config)
    controller = KeplerEqMLPControl(config, x_eq, u_eq, [1.] * 4)
    result = rollout_mc(controller, config, n_mc=2, n_paths=0, dt=.01, t_max=.01)
    assert result["path_states"].shape == (2, 0, 4)
