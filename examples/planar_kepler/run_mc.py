"""Empirical Euler-Maruyama rollouts of a saved planar Kepler controller.

All event decisions use the full four-dimensional state. Trajectory plots
are illustrations; certification is reported by the synthesis runner.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from examples.planar_kepler.model import (
    AccelerationDiffusion, KeplerEqMLPControl, PlanarKepler,
    load_region_arrays, validate_config,
)

REASONS = ("time_limit", "goal", "unsafe", "domain_exit", "nonfinite")


def load_controller(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "eval_bundle.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = json.loads((checkpoint.parent / "run_config.json").read_text())["example"]
    validate_config(config)
    state = payload["control_state_dict"]
    controller = KeplerEqMLPControl(config, state["x_eq"], state["u_eq"], state["input_scale"])
    controller.load_state_dict(state)
    if not all(torch.isfinite(value).all() for value in controller.state_dict().values()):
        raise ValueError("Controller checkpoint contains nonfinite tensors")
    return controller.eval(), config


def inside(x, box):
    return ((x >= box[:, 0]) & (x <= box[:, 1])).all(dim=-1)


def classify(x, arrays):
    """Safety failures take precedence over goal entry and use all four axes."""
    reasons = torch.zeros(len(x), dtype=torch.long, device=x.device)
    reasons[inside(x, arrays["goal_range"])] = 1
    unsafe = ((x[:, None] >= arrays["unsafe_ranges"][None, :, :, 0]) &
              (x[:, None] <= arrays["unsafe_ranges"][None, :, :, 1])).all(dim=2).any(dim=1)
    reasons[unsafe] = 2
    reasons[~inside(x, arrays["full_range"])] = 3
    reasons[~torch.isfinite(x).all(dim=1)] = 4
    return reasons


def rollout_mc(controller, config, *, n_mc=256, t_max=10.0, dt=0.01,
               seed=42, n_paths=10, stochastic=True):
    if n_mc < 1 or n_paths < 0 or seed < 0:
        raise ValueError("n_mc must be positive; n_paths and seed must be nonnegative")
    if not all(math.isfinite(v) and v > 0 for v in (dt, t_max)):
        raise ValueError("dt and t_max must be positive and finite")
    arrays = {key: torch.tensor(value, dtype=torch.float64)
              for key, value in load_region_arrays(config).items()}
    physics, diffusion = PlanarKepler(config), AccelerationDiffusion(config)
    initial_rng, noise_rng = [np.random.default_rng([seed, stream]) for stream in range(2)]
    init = arrays["init_range"].numpy()
    x = torch.from_numpy(initial_rng.uniform(init[:, 0], init[:, 1], size=(n_mc, 4)))
    initial = x.clone()
    reasons = classify(x, arrays)
    stop_times = torch.zeros(n_mc, dtype=x.dtype)
    n_paths = min(n_paths, n_mc)
    trace_states, trace_times = [x[:n_paths].clone()], [0.0]
    controller.eval()
    with torch.no_grad():
        for step in range(math.ceil(t_max / dt)):
            active = torch.where(reasons == 0)[0]
            if not len(active):
                break
            t, next_t = step * dt, min((step + 1) * dt, t_max)
            h = next_t - t
            noise = torch.from_numpy(noise_rng.standard_normal((n_mc, 4)))
            current = x[active]
            control = controller(current.float()).double()
            x[active] = current + h * physics(current, control)
            if stochastic:
                x[active] += math.sqrt(h) * diffusion(current) * noise[active]
            reasons[active] = classify(x[active], arrays)
            stop_times[active] = next_t
            trace_states.append(x[:n_paths].clone())
            trace_times.append(next_t)
    return dict(initial_states=initial, final_states=x, reasons=reasons, stop_times=stop_times,
                path_states=torch.stack(trace_states), path_times=torch.tensor(trace_times),
                seed=seed, n_mc=n_mc, dt=dt, t_max=t_max, stochastic=stochastic)


def plot_rollouts(result, config, output):
    states, times = result["path_states"].numpy(), result["path_times"].numpy()
    goal = load_region_arrays(config)["goal_range"]
    labels = ("Radius r", "Angle theta (rad)", "Radial velocity r_dot", "Angular velocity theta_dot (rad / time)")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plane_fig, plane_ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for path in range(states.shape[1]):
        valid = times <= float(result["stop_times"][path]) + 1e-6
        trace = states[valid, path]
        for axis, ax in enumerate(axes.flat):
            ax.plot(times[valid], trace[:, axis], alpha=0.75, linewidth=1)
        plane_ax.plot(trace[:, 0] * np.cos(trace[:, 1]), trace[:, 0] * np.sin(trace[:, 1]),
                      alpha=0.75, linewidth=1)
    for axis, ax in enumerate(axes.flat):
        ax.axhspan(goal[axis, 0], goal[axis, 1], color="#149b85", alpha=0.15, label="Goal interval")
        ax.set(xlabel="Time", ylabel=labels[axis])
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    fig.suptitle("Planar Kepler rollouts · reaching the goal requires all four coordinates")
    plane_ax.scatter([0], [0], color="#da8b24", marker="*", s=100, label="Gravity center")
    center = goal[:2].mean(axis=1)
    plane_ax.scatter([center[0] * np.cos(center[1])], [center[0] * np.sin(center[1])],
                     color="#149b85", marker="x", s=80, label="Target position center")
    plane_ax.set(xlabel="Cartesian position X = r cos(theta)", ylabel="Cartesian position Y = r sin(theta)",
                 title="Position trajectories in the orbital plane")
    plane_ax.set_aspect("equal", adjustable="datalim")
    plane_ax.grid(alpha=0.15)
    plane_ax.legend()
    for name, figure in (("state_trajectories", fig), ("orbital_plane", plane_fig)):
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"{name}.{suffix}", dpi=160)
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=HERE / "neural_certified_nominal_drift/seed0/outputs/eval_bundle.pth")
    parser.add_argument("--output-dir", type=Path, default=HERE / "run_mc_results")
    parser.add_argument("--n-mc", type=int, default=256)
    parser.add_argument("--n-paths", type=int, default=10)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Disable acceleration noise")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    controller, config = load_controller(args.checkpoint)
    result = rollout_mc(controller, config, n_mc=args.n_mc, n_paths=args.n_paths,
                        t_max=args.t_max, dt=args.dt, seed=args.seed, stochastic=not args.deterministic)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: int((result["reasons"] == index).sum()) for index, name in enumerate(REASONS)}
    summary = dict(checkpoint=str(args.checkpoint.resolve()), n_mc=args.n_mc, seed=args.seed,
                   dt=args.dt, t_max=args.t_max, stochastic=not args.deterministic,
                   counts=counts, empirical_success_rate=counts["goal"] / args.n_mc,
                   interpretation="Empirical Euler-Maruyama outcomes; goal and safety decisions use all four states.")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(dict(result=result, config=config), args.output_dir / "mc_cache.pth")
    if not args.no_plots:
        plot_rollouts(result, config, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
