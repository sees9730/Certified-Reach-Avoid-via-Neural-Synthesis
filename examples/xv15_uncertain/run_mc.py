"""Compare XV-15 controllers over density-mode x mass-mode Monte Carlo scenarios.

Zero means nominal parameters. Uniform draws are independent for density and
mass. Adversarial endpoint selection is a causal one-step heuristic, not a
worst-case reach-avoid proof. All controllers share initial conditions and
Brownian increments. Saved results can be plotted again with postprocess_mc.py.
"""
import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.xv15_uncertain.model import (
    DEG, DiagonalDiffusion, XV15Aero, XV15EqMLPControl, load_config, load_region_arrays,
)
from examples.xv15_uncertain.uncertain_density import load_density_interval
from examples.xv15_uncertain.uncertain_parameters import load_mass_interval

MODES = ("zero", "uniform", "adversarial")
ATTACKS = ("lookahead", "nearest_unsafe")
OUTPUT_DIR = HERE / "run_mc_results"
DEFAULT_CONTROLLERS = (
    ("Cert. (uncertain parameters)", HERE / "neural_certified_uncertain_param"),
    ("Cert. (nominal drift)", HERE / "neural_certified_nominal_drift"),
    ("RL (SB3 PPO)", HERE / "rl_sb3_ppo"),
)
REASONS = ("time_limit", "goal", "unsafe", "domain_exit", "nonfinite")


class ExportedPPOControl(nn.Module):
    """Deterministic xv15_sb3_ppo_actor_v1 export; no SB3 dependency needed."""
    def __init__(self, payload):
        super().__init__()
        layers, previous = [], 3
        for width in payload["hidden_dims"]:
            layers.extend([nn.Linear(previous, width), nn.Tanh()])
            previous = width
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(previous, 3)
        for name in ("obs_center", "obs_scale", "action_low", "action_high"):
            self.register_buffer(name, torch.empty(3))
        self.load_state_dict(payload["control_state_dict"])
        if not (self.obs_scale > 0).all() or not (self.action_high > self.action_low).all():
            raise ValueError("Invalid PPO observation scales or action bounds")

    def forward(self, x):
        action = self.action_net(self.policy_net((x - self.obs_center) / self.obs_scale))
        action = action.clamp(-1.0, 1.0)
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)


def discover_checkpoints(directory):
    """Accept a checkpoint, outputs directory, seed directory, or seed parent."""
    directory = Path(directory)
    if directory.is_file():
        return [(directory.parent.parent.name, directory)]
    for base in (directory / "outputs", directory):
        for filename in ("eval_bundle.pth", "rl_controller.pth"):
            if (base / filename).is_file():
                return [(base.parent.name, base / filename)]
    found = []
    seeds = sorted((p for p in directory.glob("seed*") if re.fullmatch(r"seed\d+", p.name)),
                   key=lambda p: int(p.name[4:]))
    for seed in seeds:
        found.extend(discover_checkpoints(seed))
    return found


def load_controller(checkpoint):
    """Load saved physical controls and normalization, never recompute trim."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") == "xv15_sb3_ppo_actor_v1":
        controller = ExportedPPOControl(payload)
        kind = payload["format"]
    elif "V_state_dict" in payload and payload.get("control_state_dict") is not None:
        state = payload["control_state_dict"]
        saved_config = checkpoint.parent / "run_config.json"
        if not saved_config.is_file():
            raise ValueError(f"Missing training configuration beside {checkpoint}")
        config = json.loads(saved_config.read_text())["example"]
        controller = XV15EqMLPControl(config, state["x_eq"], state["u_eq"], state["input_scale"])
        controller.load_state_dict(state)
        kind = "neural_certificate"
    else:
        raise ValueError(f"Unsupported controller checkpoint: {checkpoint}")
    controller.eval()
    if not all(torch.isfinite(t).all() for t in controller.state_dict().values()):
        raise ValueError(f"Nonfinite controller weights: {checkpoint}")
    return controller, kind


def inside(x, box):
    return ((x >= box[:, 0]) & (x <= box[:, 1])).all(dim=-1)


def classify(x, arrays):
    """Safety failures take precedence over goal entry on overlapping boundaries."""
    reason = torch.zeros(len(x), dtype=torch.long)
    reason[inside(x, arrays["goal_range"])] = 1
    unsafe = ((x[:, None, :] >= arrays["unsafe_ranges"][None, :, :, 0])
              & (x[:, None, :] <= arrays["unsafe_ranges"][None, :, :, 1])).all(dim=2).any(dim=1)
    reason[unsafe] = 2
    reason[~inside(x, arrays["full_range"])] = 3
    reason[~torch.isfinite(x).all(dim=1)] = 4
    return reason


def adversarial_score(x, arrays, method):
    """Common state-based attack objective for certified and RL controllers."""
    goal = arrays["goal_range"]
    scale = arrays["full_range"][:, 1] - arrays["full_range"][:, 0]
    if method == "lookahead":
        score = ((x - goal.mean(dim=1)) / ((goal[:, 1] - goal[:, 0]) / 2)).square().sum(dim=1)
    elif method == "nearest_unsafe":
        boxes = arrays["unsafe_ranges"]
        distances = torch.maximum(boxes[None, :, :, 0] - x[:, None, :],
                                  x[:, None, :] - boxes[None, :, :, 1]).clamp_min(0) / scale
        score = -distances.square().sum(dim=2).min(dim=1).values
    else:
        raise ValueError(f"Unknown attack: {method}")
    reason = classify(x, arrays)
    score = torch.where(reason == 1, -1e6, score)
    score = torch.where(reason >= 2, 1e6, score)
    return score


def select_parameters(aero, x, u, arrays, *, density_mode, mass_mode,
                      density_interval, mass_interval, density_draw, mass_draw, dt, attack):
    """Joint endpoint search for adversarial coordinates; hold other draws fixed."""
    densities = (density_interval if density_mode == "adversarial" else
                 [density_draw if density_mode == "uniform" else aero.density])
    masses = (mass_interval if mass_mode == "adversarial" else
              [mass_draw if mass_mode == "uniform" else aero.mass])
    candidates = [(torch.as_tensor(rho, dtype=x.dtype).expand(len(x)),
                   torch.as_tensor(mass, dtype=x.dtype).expand(len(x)))
                  for rho, mass in product(densities, masses)]
    if len(candidates) == 1:
        return candidates[0]
    scores = [adversarial_score(x + dt * aero(x, u, density=rho, mass=mass), arrays, attack)
              for rho, mass in candidates]
    best = torch.stack(scores, dim=1).argmax(dim=1)
    rows = torch.arange(len(x))
    return tuple(torch.stack([pair[i] for pair in candidates], dim=1)[rows, best] for i in range(2))


def rollout_mc(controller, config, *, n_mc=500, t_max=20.0, dt=0.01, seed=42,
               density_mode="zero", mass_mode="zero", uniform_refresh="step",
               attack="lookahead", stochastic=True, n_paths=5, trace_dt=0.1):
    """Batched Euler-Maruyama with shared, independent RNG streams per quantity.

    Draw for ALL trajectory slots on each step before masking terminated paths.
    Thus early termination and parameter modes cannot change another path's noise.
    Uniform parameters are independent; neither controller observes their values.
    """
    if n_mc < 1 or n_paths < 0 or seed < 0:
        raise ValueError("n_mc must be positive; n_paths and seed must be nonnegative")
    if not all(math.isfinite(v) and v > 0 for v in (dt, t_max, trace_dt)):
        raise ValueError("dt, t_max, and trace_dt must be finite and positive")
    if density_mode not in MODES or mass_mode not in MODES or uniform_refresh not in ("step", "episode"):
        raise ValueError("Invalid parameter realization mode")
    if attack not in ATTACKS:
        raise ValueError("Invalid adversarial method")
    density_interval, mass_interval = load_density_interval(config), load_mass_interval(config)
    arrays = {k: torch.tensor(v, dtype=torch.float64) for k, v in load_region_arrays(config).items()}
    aero = XV15Aero(config)
    sigma = DiagonalDiffusion(config).sigma.double() if stochastic else torch.zeros(3, dtype=torch.float64)
    init_rng, density_rng, mass_rng, noise_rng = [np.random.default_rng([seed, k]) for k in range(4)]
    init = arrays["init_range"].numpy()
    x = torch.from_numpy(init_rng.uniform(init[:, 0], init[:, 1], size=(n_mc, 3)))
    initial_states = x.clone()
    fixed_density = density_rng.uniform(*density_interval, size=n_mc)
    fixed_mass = mass_rng.uniform(*mass_interval, size=n_mc)
    reasons = classify(x, arrays)
    stop_times = torch.zeros(n_mc, dtype=torch.float64)
    effort = torch.zeros_like(stop_times)
    effort_scale = torch.tensor([aero.mass * aero.gravity, config["control"]["alpha_max_deg"] * DEG,
                                config["control"]["delta_max_deg_per_second"] * DEG], dtype=torch.float64)
    n_paths = min(n_paths, n_mc)
    paths = [dict(times=[0.0], states=[x[i].clone()], control_times=[], controls=[], parameters=[])
             for i in range(n_paths)]
    stride = max(1, int(round(trace_dt / dt)))
    steps = int(math.ceil(t_max / dt))
    controller.eval()
    with torch.no_grad():
        for step in range(steps):
            active = torch.where(reasons == 0)[0]
            if not len(active):
                break
            t, next_t = step * dt, min((step + 1) * dt, t_max)
            h = next_t - t
            density_draw = density_rng.uniform(*density_interval, size=n_mc) if uniform_refresh == "step" else fixed_density
            mass_draw = mass_rng.uniform(*mass_interval, size=n_mc) if uniform_refresh == "step" else fixed_mass
            noise = torch.from_numpy(noise_rng.standard_normal((n_mc, 3)))
            current = x[active]
            u = controller(current.float()).double()
            rho, mass = select_parameters(
                aero, current, u, arrays, density_mode=density_mode, mass_mode=mass_mode,
                density_interval=density_interval, mass_interval=mass_interval,
                density_draw=torch.from_numpy(density_draw)[active],
                mass_draw=torch.from_numpy(mass_draw)[active], dt=h, attack=attack)
            effort[active] += h * (u / effort_scale).square().sum(dim=1)
            x[active] = current + h * aero(current, u, density=rho, mass=mass) + math.sqrt(h) * sigma * noise[active]
            reasons[active] = classify(x[active], arrays)
            stop_times[active] = next_t
            for j, i in enumerate(active.tolist()):
                if i >= n_paths:
                    continue
                path = paths[i]
                if step % stride == 0:
                    path["control_times"].append(t)
                    path["controls"].append(u[j].clone())
                    path["parameters"].append(torch.stack([rho[j], mass[j]]))
                if (step + 1) % stride == 0 or reasons[i] != 0 or step == steps - 1:
                    path["times"].append(next_t)
                    path["states"].append(x[i].clone())
    outcomes = ["success" if code == 1 else "timeout" if code == 0 else "fail" for code in reasons.tolist()]
    for i, path in enumerate(paths):
        for key, width in (("states", 3), ("controls", 3), ("parameters", 2)):
            path[key] = torch.stack(path[key]) if path[key] else torch.empty((0, width), dtype=torch.float64)
        for key in ("times", "control_times"):
            path[key] = torch.tensor(path[key], dtype=torch.float64)
        path.update(trajectory=i, outcome=outcomes[i], stop_reason=REASONS[reasons[i]])
    return dict(outcomes=outcomes, stop_reasons=[REASONS[i] for i in reasons.tolist()],
                stop_times=stop_times, normalized_effort=effort, initial_states=initial_states,
                final_states=x, paths=paths)


def compute_stats(result):
    outcomes = result["outcomes"]
    success = torch.tensor([v == "success" for v in outcomes])
    times = result["stop_times"][success]
    effort = result["normalized_effort"]
    finite_effort = effort[torch.isfinite(effort)]
    stats = dict(n_mc=len(outcomes), n_success=int(success.sum()),
                 p_success=float(success.double().mean()),
                 p_fail=outcomes.count("fail") / len(outcomes), p_timeout=outcomes.count("timeout") / len(outcomes),
                 hit_time_s_mean=float(times.mean()) if len(times) else None,
                 hit_time_s_median=float(times.median()) if len(times) else None,
                 normalized_effort_mean=float(finite_effort.mean()) if len(finite_effort) else None,
                 n_nonfinite_effort=int((~torch.isfinite(effort)).sum()))
    stats.update({f"n_{name}": result["stop_reasons"].count(name) for name in REASONS})
    return stats


def aggregate_results(runs):
    """Choose lowest-success attack PER training seed, then average over seeds."""
    groups = {}
    for run in runs:
        key = (run["label"], run["training_seed"], run["density_mode"], run["mass_mode"])
        if key not in groups or (run["stats"]["p_success"], -run["stats"]["p_fail"]) < (
                groups[key]["stats"]["p_success"], -groups[key]["stats"]["p_fail"]):
            groups[key] = run
    summaries = {}
    for key, run in groups.items():
        group = (key[0], key[2], key[3])
        summaries.setdefault(group, []).append(run)
    rows = []
    for (label, density, mass), selected in summaries.items():
        rates = [r["stats"]["p_success"] for r in selected]
        row = dict(controller=label, density_mode=density, mass_mode=mass, n_seeds=len(selected),
                   p_success=float(np.mean(rates)), p_success_seed_std=float(np.std(rates)),
                   selected_attacks={r["training_seed"]: r["attack"] for r in selected})
        for field in ("p_fail", "p_timeout", "hit_time_s_mean", "normalized_effort_mean"):
            values = [r["stats"][field] for r in selected if r["stats"][field] is not None]
            row[field] = float(np.mean(values)) if values else None
        rows.append(row)
    return rows


def write_reports(cache, output_dir):
    """Write tables without rerunning trajectories; the cache remains authoritative."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate_results(cache["runs"])
    (output_dir / "summary.json").write_text(json.dumps(dict(metadata=cache["metadata"], summary=rows), indent=2, allow_nan=False))
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{**row, "selected_attacks": json.dumps(row["selected_attacks"])} for row in rows])
    details = [{**{k: r[k] for k in ("label", "training_seed", "density_mode", "mass_mode", "attack")},
                **r["stats"]} for r in cache["runs"]]
    with (output_dir / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print("\nController | density / mass | success | fail | timeout")
    for row in rows:
        print(f"{row['controller']} | {row['density_mode']} / {row['mass_mode']} | "
              f"{row['p_success']:.3f} | {row['p_fail']:.3f} | {row['p_timeout']:.3f}")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", action="append", metavar="LABEL=PATH", help="Repeat to choose controller runs; otherwise discover default controllers")
    parser.add_argument("--config", type=Path, default=HERE / "config.json", help="Common evaluation physics and regions for every controller")
    parser.add_argument("--density-modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--mass-modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--uniform-refresh", choices=("step", "episode"), default="step")
    parser.add_argument("--adversarial-methods", nargs="+", choices=ATTACKS, default=list(ATTACKS))
    for name in ("density-min", "density-max", "mass-min", "mass-max"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--n-mc", type=int, default=500)
    parser.add_argument("--t-max", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--mc-seed", type=int, default=42)
    parser.add_argument("--n-paths", type=int, default=5)
    parser.add_argument("--trace-dt", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", help="Disable Brownian diffusion; parameter uncertainty remains enabled")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.n_mc < 1 or args.threads < 1 or args.n_paths < 0 or args.mc_seed < 0:
        parser.error("n-mc/threads must be positive; n-paths/mc-seed must be nonnegative")
    if not all(math.isfinite(v) and v > 0 for v in (args.t_max, args.dt, args.trace_dt)):
        parser.error("t-max, dt, and trace-dt must be finite and positive")
    for key in ("density_modes", "mass_modes", "adversarial_methods"):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    config = load_config(args.config)
    try:
        for parameter, field, loader in (("density", "air_density_kg_m3", load_density_interval),
                                          ("mass", "mass_kg", load_mass_interval)):
            interval = list(loader(config))
            for i, endpoint in enumerate(("min", "max")):
                value = getattr(args, f"{parameter}_{endpoint}")
                if value is not None:
                    interval[i] = value
            config["uncertainty"][field] = interval
            loader(config)
        load_region_arrays(config)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    specs = list(DEFAULT_CONTROLLERS)
    if args.controller:
        specs = []
        for spec in args.controller:
            label, separator, directory = spec.partition("=")
            if not separator or not label.strip() or not directory.strip():
                parser.error("--controller must be LABEL=PATH")
            specs.append((label.strip(), Path(directory).expanduser()))
    if len({label for label, _ in specs}) != len(specs):
        parser.error("Controller labels must be unique")
    discovered = []
    for label, directory in specs:
        checkpoints = discover_checkpoints(directory)
        if not checkpoints:
            if args.controller:
                parser.error(f"No controller checkpoint found under {directory}")
            print(f"[skip] {label}: no checkpoint under {directory}", flush=True)
        for training_seed, checkpoint in checkpoints:
            discovered.append((label, training_seed, checkpoint))
    if not discovered:
        parser.error("No controller checkpoints found")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = dict(format="xv15_mc_v1", metadata=dict(
        config=config, n_mc=args.n_mc, t_max=args.t_max, dt=args.dt, mc_seed=args.mc_seed,
        density_modes=args.density_modes, mass_modes=args.mass_modes,
        uniform_refresh=args.uniform_refresh, adversarial_methods=args.adversarial_methods,
        stochastic=not args.deterministic, n_paths=args.n_paths, trace_dt=args.trace_dt,
        method="Euler-Maruyama; events checked at step boundaries; empirical, not a certification proof",
        effort="Integral of (T/(m0*g))^2 + (alpha/alpha_max)^2 + (delta/delta_max)^2 up to stopping; not physical energy",
        adversarial_summary="Lowest success rate among tested attacks per training seed, then mean across seeds",
        controllers=[]), runs=[])
    print(f"Density interval: {load_density_interval(config)} kg/m^3; mass interval: {load_mass_interval(config)} kg", flush=True)
    for label, training_seed, checkpoint in discovered:
        controller, kind = load_controller(checkpoint)
        cache["metadata"]["controllers"].append(dict(label=label, training_seed=training_seed,
                                                       checkpoint=str(checkpoint.resolve()), kind=kind))
        for density_mode, mass_mode in product(args.density_modes, args.mass_modes):
            attacks = args.adversarial_methods if "adversarial" in (density_mode, mass_mode) else ["none"]
            for attack in attacks:
                print(f"{label} [{training_seed}]: density={density_mode}, mass={mass_mode}, attack={attack}", flush=True)
                result = rollout_mc(controller, config, n_mc=args.n_mc, t_max=args.t_max, dt=args.dt,
                    seed=args.mc_seed, density_mode=density_mode, mass_mode=mass_mode,
                    uniform_refresh=args.uniform_refresh, attack=attack if attack != "none" else "lookahead",
                    stochastic=not args.deterministic, n_paths=args.n_paths, trace_dt=args.trace_dt)
                cache["runs"].append(dict(label=label, training_seed=training_seed,
                    density_mode=density_mode, mass_mode=mass_mode, attack=attack,
                    result=result, stats=compute_stats(result)))
    cache_path = args.output_dir / "mc_cache.pth"
    torch.save(cache, cache_path)
    write_reports(cache, args.output_dir)
    if not args.no_plots:
        from examples.xv15_uncertain.postprocess_mc import plot_results
        plot_results(cache, args.output_dir)
    print(f"Saved Monte Carlo results: {cache_path.resolve()}")
    return cache


if __name__ == "__main__":
    main()
