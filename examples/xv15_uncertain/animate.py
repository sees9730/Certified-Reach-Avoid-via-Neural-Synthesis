"""Standalone HTML rollout animation for a trained XV-15 nominal-drift run.

Run as: python examples/xv15_uncertain/animate.py --seed 0

Loads a seed's outputs/eval_bundle.pth (the trained controller and
certificate weights) and outputs/run_config.json (the exact config/
hyperparameters that run used), rebuilds the aircraft dynamics, controller,
and certificate from this example's own model.py, then simulates one
closed-loop SDE rollout with Euler-Maruyama and renders it into a
self-contained animate.html (no server, no external assets) using
flight_view_template.html.

This module only reads a saved eval bundle; it does not modify training,
bound-checking, or plotting code, and does not reuse any nova_3d_xv15_syn code.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.xv15_uncertain.model import (
    DEG, DiagonalDiffusion, NominalClosedLoopDrift, XV15Aero,
    XV15EqMLPControl, find_goal_equilibrium, load_region_arrays,
)
from src.dynamics import Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV, verify_GV

SAT_KEYS = ("goal_satisfied", "unsafe_satisfied", "init_satisfied", "outside_satisfied", "generator_satisfied")


def load_run(seed_dir: Path):
    """Rebuild the trained controller/certificate/generator from a seed's saved outputs."""
    output_dir = seed_dir / "outputs"
    run_config = json.loads((output_dir / "run_config.json").read_text())
    bundle = torch.load(output_dir / "eval_bundle.pth", map_location="cpu")

    config = run_config["example"]
    params = Hyperparameters.from_dict(bundle["hyperparameters"])
    arrays = load_region_arrays(config)

    aero = XV15Aero(config)
    x_eq, u_eq = find_goal_equilibrium(config, aero)

    controller = XV15EqMLPControl(config, x_eq, u_eq, params.network.input_scale)
    controller.load_state_dict(bundle["control_state_dict"])
    controller.eval()

    value = create_V(params.network, input_offset=x_eq.tolist(), output_offset=np.float32(0.1))
    value.load_state_dict(bundle["V_state_dict"])
    value.eval()

    diffusion = DiagonalDiffusion(config)
    drift = NominalClosedLoopDrift(aero, controller)
    dynamics = Dynamics.dynamics(f=drift, g=diffusion, state_dim=3)
    generator = create_GV(
        V_net=value, dynamics=dynamics, network_config=params.network,
        input_offset=x_eq.tolist(), include_time=False, include_energy=False, verify=False,
    )
    if not verify_GV(generator, dynamics=dynamics, x=x_eq[None], tol=1e-3):
        raise RuntimeError("Reconstructed generator disagrees with autograd at trim; refusing to animate")

    results = bundle.get("final_results") or {}
    overall_sat = bool(results) and all(bool(results.get(key, False)) for key in SAT_KEYS)
    beta_ra = float(params.constraints.beta_ra)
    return dict(aero=aero, controller=controller, value=value, generator=generator, diffusion=diffusion,
                arrays=arrays, beta_ra=beta_ra, overall_sat=overall_sat, config=config)


def _inside(x, box):
    return bool(np.all(x >= box[:, 0]) and np.all(x <= box[:, 1]))


def _inside_any(x, boxes):
    return any(_inside(x, box) for box in boxes)


def simulate_rollout(run, *, seconds=20.0, dt=0.01, frame_dt=0.05, seed=0, stochastic=True):
    """One closed-loop Euler-Maruyama rollout, recorded at a fixed display cadence.

    Returns a dict with 'columns' naming each recorded field and 'samples',
    a list of rows in that column order (JSON-friendly, no numpy scalars).
    """
    if not np.isfinite([seconds, dt, frame_dt]).all() or min(seconds, dt, frame_dt) <= 0:
        raise ValueError("seconds, dt, and frame_dt must be positive and finite")

    arrays = run["arrays"]
    domain, init_range, goal_range = arrays["full_range"], arrays["init_range"], arrays["goal_range"]
    unsafe_boxes = arrays["unsafe_ranges"]
    sigma = run["diffusion"].sigma.numpy() if stochastic else np.zeros(3)
    rng = np.random.default_rng(seed)

    x = rng.uniform(init_range[:, 0], init_range[:, 1])
    if not _inside(x, init_range):
        raise ValueError("Sampled initial state fell outside the init region")
    position = np.zeros(2)
    time, next_frame, outcome, rows = 0.0, 0.0, "time_limit", []

    with torch.no_grad():
        while True:
            if not np.isfinite(x).all():
                raise FloatingPointError("Rollout diverged; reduce --dt")
            if x[0] <= 1e-3:
                raise FloatingPointError("Rollout crossed near-zero airspeed; reduce --dt")
            if not _inside(x, domain):
                outcome = "domain_exit"
            elif _inside_any(x, unsafe_boxes):
                outcome = "unsafe"
            elif _inside(x, goal_range):
                outcome = "goal"
            terminal = outcome != "time_limit" or time >= seconds

            xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            u = run["controller"](xt).squeeze(0).numpy()
            f = run["aero"](xt, torch.as_tensor(u, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            v_value = float(run["value"](xt).squeeze())
            gv_value = float(run["generator"](xt).squeeze())

            if time >= next_frame or terminal:
                rows.append([
                    round(time, 4), float(x[0]), float(x[1] / DEG), float(x[2] / DEG),
                    float(position[0]), float(position[1]),
                    float(u[0]), float(u[1] / DEG), float(u[2] / DEG),
                    v_value, gv_value,
                ])
                next_frame = time + frame_dt
            if terminal:
                break

            h = min(dt, seconds - time)
            position = position + h * x[0] * np.array([np.cos(x[1]), np.sin(x[1])])
            x = x + h * f + np.sqrt(h) * sigma * rng.standard_normal(3)
            time = min(seconds, time + h)

    return dict(
        outcome=outcome, seed=int(seed), dt=dt, stochastic=bool(stochastic),
        method="Euler-Maruyama; stop events checked at integration steps; illustrative, not a verification proof",
        columns=["time_s", "airspeed_m_s", "gamma_deg", "tilt_deg", "distance_m", "relative_altitude_m",
                 "thrust_N", "alpha_deg", "tilt_rate_deg_s", "V", "GV"],
        samples=rows,
    )


def render_html(seed_dir: Path, run, rollout, out_name="animate.html"):
    weight = run["config"]["dynamics"]["mass"] * run["config"]["dynamics"]["gravity"]
    thrust_min, thrust_max = (float(f) * weight for f in run["config"]["control"]["thrust_over_weight"])
    problem = dict(
        beta_ra=run["beta_ra"],
        domain=run["arrays"]["full_range"].tolist(),
        initial=run["arrays"]["init_range"].tolist(),
        goal=run["arrays"]["goal_range"].tolist(),
        unsafe=run["arrays"]["unsafe_ranges"].tolist(),
        mass=run["config"]["dynamics"]["mass"],
        gravity=run["config"]["dynamics"]["gravity"],
        thrust_min=thrust_min,
        thrust_max=thrust_max,
        alpha_max_deg=float(run["config"]["control"]["alpha_max_deg"]),
        delta_max_deg_s=float(run["config"]["control"]["delta_max_deg_per_second"]),
    )
    payload = dict(rollout=rollout, problem=problem, certified=run["overall_sat"])
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")

    template = (HERE / "flight_view_template.html").read_text()
    out_dir = seed_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    out_path.write_text(template.replace("__XV15_ROLLOUT__", encoded))
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Trained seed to load (seed{N}/ subdirectory)")
    parser.add_argument("--run-dir", type=Path, default=None,
                         help="Seed directory holding outputs/ (default: neural_certified_nominal_drift/seed{seed})")
    parser.add_argument("--seconds", type=float, default=20.0, help="Simulated horizon")
    parser.add_argument("--dt", type=float, default=0.01, help="Euler-Maruyama integration step")
    parser.add_argument("--frame-dt", type=float, default=0.05, help="Recorded-sample spacing for the animation")
    parser.add_argument("--sim-seed", type=int, default=0, help="RNG seed for the rollout, independent of --seed")
    parser.add_argument("--deterministic", action="store_true", help="Disable SDE noise (drift only)")
    parser.add_argument("--out-name", default="animate.html", help="Output file name under <seed_dir>/results/")
    args = parser.parse_args(argv)

    seed_dir = args.run_dir or (HERE / "neural_certified_nominal_drift" / f"seed{args.seed}")
    run = load_run(seed_dir)
    print("Simulating the trained XV-15 controller for animation...", flush=True)
    rollout = simulate_rollout(run, seconds=args.seconds, dt=args.dt, frame_dt=args.frame_dt,
                                seed=args.sim_seed, stochastic=not args.deterministic)
    out_path = render_html(seed_dir, run, rollout, out_name=args.out_name)
    last_row = rollout["samples"][-1]
    print(f"Animation: {out_path} ({rollout['outcome']} at {last_row[0]:.2f} s)", flush=True)


if __name__ == "__main__":
    main()
