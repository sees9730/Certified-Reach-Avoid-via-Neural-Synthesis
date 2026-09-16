"""Animate a trained or baseline double-integrator SDE rollout as standalone HTML."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from examples.asteroid_landing_uncertain.model import (
    validate_config, load_config, load_region_arrays, NominalClosedLoopDrift, DiagonalDiffusion,
)
from src.dynamics import Dynamics

SAT_KEYS = ("goal_satisfied", "unsafe_satisfied", "init_satisfied", "outside_satisfied", "generator_satisfied")


class LinearFeedbackControl(torch.nn.Module):
    """Position/velocity feedback, clipped to the configured axis limits."""
    def __init__(self, config, goal_center, kp=2.0, kd=3.0):
        super().__init__()
        if not np.isfinite([kp, kd]).all() or min(kp, kd) <= 0:
            raise ValueError("kp and kd must be positive and finite")
        self.kp, self.kd = kp, kd
        self.register_buffer('goal_center', torch.as_tensor(goal_center, dtype=torch.float32))
        self.register_buffer('limits', torch.as_tensor(config['control']['u_max'], dtype=torch.float32))

    def forward(self, x):
        u = -self.kp * (x[:, :2] - self.goal_center[:2]) - self.kd * (x[:, 2:] - self.goal_center[2:])
        return torch.maximum(torch.minimum(u, self.limits), -self.limits)


def load_baseline(config, kp=2.0, kd=3.0):
    arrays = load_region_arrays(config)
    controller = LinearFeedbackControl(config, arrays['goal_range'].mean(axis=1), kp, kd)
    dynamics = Dynamics.dynamics(f=NominalClosedLoopDrift(controller), g=DiagonalDiffusion(config), state_dim=4)
    return dict(value=None, generator=None, controller=controller.eval(), dynamics=dynamics,
                arrays=arrays, certified=False, mode='baseline')


def load_trained(seed_dir):
    # Baseline rollouts need neither training artifacts nor the bound library.
    from examples.asteroid_landing_uncertain.neural_certified_nominal_drift.main import build_problem
    from src.hyperparameters import Hyperparameters
    outputs = Path(seed_dir) / "outputs"
    config = validate_config(json.loads((outputs / "run_config.json").read_text())["example"])
    # Only load trusted local training artifacts. Validate the problem before
    # attempting to reconstruct a historical checkpoint.
    bundle = torch.load(outputs / "eval_bundle.pth", map_location="cpu", weights_only=False)
    params = Hyperparameters.from_dict(bundle["hyperparameters"])
    params.training.device = "cpu"
    value, generator, controller, dynamics, _, arrays = build_problem(config, params)
    value.load_state_dict(bundle["V_state_dict"])
    controller.load_state_dict(bundle["control_state_dict"])
    value.eval()
    controller.eval()
    generator.eval()
    results = bundle.get("final_results") or {}
    return dict(value=value, generator=generator, controller=controller, dynamics=dynamics,
                arrays=arrays, certified=all(bool(results.get(k, False)) for k in SAT_KEYS), mode='trained')


def inside(x, box):
    return bool(np.all((x >= box[:, 0]) & (x <= box[:, 1])))


@torch.no_grad()
def simulate_rollout(run, horizon=20., dt=0.005, frame_dt=0.1, seed=0, stochastic=True):
    if not np.isfinite([horizon, dt, frame_dt]).all() or min(horizon, dt, frame_dt) <= 0:
        raise ValueError("horizon, dt and frame_dt must be positive and finite")
    rng = np.random.default_rng(seed)
    arrays = run["arrays"]
    x = rng.uniform(arrays["init_range"][:, 0], arrays["init_range"][:, 1])
    rows, time, next_frame, outcome = [], 0., 0., "time_limit"
    while True:
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite simulated state; reduce the integration step")
        if not inside(x, arrays["full_range"]):
            outcome = "domain_exit"
        elif any(inside(x, box) for box in arrays["unsafe_ranges"]):
            outcome = "unsafe"
        elif inside(x, arrays["goal_range"]):
            outcome = "goal"
        terminal = outcome != "time_limit" or time >= horizon
        xt = torch.as_tensor(x, dtype=torch.float32)[None]
        if time >= next_frame or terminal:
            rows.append(dict(t=time, x=x.tolist(), u=run["controller"](xt)[0].tolist(),
                             V=float(run["value"](xt).item()) if run['value'] is not None else None,
                             GV=float(run["generator"](xt).item()) if run['generator'] is not None else None))
            next_frame = time + frame_dt
        if terminal:
            break
        h = min(dt, horizon - time)
        drift = run["dynamics"].f(xt)[0].numpy()
        sigma = run["dynamics"].g(xt)[0].numpy() if stochastic else np.zeros(4)
        x = x + h * drift + np.sqrt(h) * sigma * rng.standard_normal(4)
        time = min(horizon, time + h)
    return dict(samples=rows, outcome=outcome, stochastic=stochastic, seed=seed, dt=dt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("trained", "baseline"), default="trained")
    parser.add_argument("--config", type=Path, default=HERE / 'config.json', help="Baseline problem configuration")
    parser.add_argument("--kp", type=float, default=2.0, help="Baseline position feedback gain")
    parser.add_argument("--kd", type=float, default=3.0, help="Baseline velocity feedback gain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--frame-dt", type=float, default=0.1)
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    seed_dir = args.run_dir or HERE / "neural_certified_nominal_drift" / f"seed{args.seed}_double_integrator"
    run = (load_baseline(load_config(args.config), args.kp, args.kd)
           if args.controller == 'baseline' else load_trained(seed_dir))
    rollout = simulate_rollout(run, args.seconds, args.dt, args.frame_dt, args.sim_seed, not args.deterministic)
    payload = dict(rollout=rollout, certified=run["certified"], mode=run['mode'],
                   regions={name: box.tolist() for name, box in run["arrays"].items()})
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = (HERE / "rollout_template.html").read_text()
    out_dir = HERE / 'results' if args.controller == 'baseline' else seed_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("double_integrator_baseline.html" if args.controller == 'baseline' else "double_integrator_rollout.html")
    path.write_text(template.replace("__ROLLOUT__", encoded))
    print(f"Animation: {path}; rollout outcome: {rollout['outcome']}")


if __name__ == "__main__":
    main()
