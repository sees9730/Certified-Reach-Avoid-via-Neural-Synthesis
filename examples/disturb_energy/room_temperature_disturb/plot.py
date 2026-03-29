"""
Compare multiple room-temperature controllers on identical MC noise realizations.

Usage examples:
  0) default
    python plot.py \
    --controller Cert.=../room_temperature_disturb_baseline/outputs/resume_checkpoint.pth \
    --controller Cert.(energy)=outputs/beta_latest_resume_checkpoint.pth \
    --controller RL=../room_temperature_disturb_rl/outputs/rl_controller.pth \
    --animate

    python plot.py \
  --controller certified=../room_temperature_disturb_baseline/outputs/resume_checkpoint.pth \
  --controller energy=outputs/beta_latest_resume_checkpoint.pth \
  --controller rl=../room_temperature_disturb_rl/outputs/rl_controller.pth \
  --animate --save-animation-dir outputs/comapre_anim

  1) Hand-designed only (default):
     python plot.py --animate

  2) Hand-designed + trained checkpoints:
     python plot.py \
       --controller nominal=outputs/resume_checkpoint.pth \
       --controller energy=outputs/energy_latest_resume_checkpoint.pth \
       --animate

  3) Checkpoints from different folders and save GIFs:
     python plot.py \
       --controller nominal=../room_temperature_disturb/outputs/resume_checkpoint.pth \
       --controller baseline=../room_temperature_disturb_baseline/outputs/checkpoints.pth \
       --save-animation-dir outputs/compare_anims --no-show
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.control_network import RoomTempControlNN, RoomTempControlWrapper
from test import (
    B_INPUT,
    T_E_MIN,
    T_E_MAX,
    U_MAX,
    X0_INIT,
    X_DOMAIN,
    XS_SAFE,
    XG_GOAL,
    classify_trajectory,
    diffusion,
    drift,
    first_goal_hit_step,
    bang_bang_policy,
)


def _sanitize_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip())
    s = s.strip("_")
    return s or "controller"


def _parse_controller_spec(spec: str) -> tuple[str, Path]:
    """
    Parse a controller spec of form:
      - "label=path/to/file.pth"
    Returns (label, path).
    """
    text = str(spec).strip()
    if "=" not in text:
        raise ValueError(
            f"Controller spec must include a label, got: '{spec}'. "
            "Use format 'label=path/to/checkpoint.pth'."
        )
    label, path_str = text.split("=", 1)
    label = label.strip()
    if label == "":
        raise ValueError(
            f"Controller label is empty in spec: '{spec}'. "
            "Use format 'label=path/to/checkpoint.pth'."
        )
    path = Path(path_str.strip()).expanduser()
    return label, path


def _make_trained_policy_from_checkpoint_only(checkpoint_path: Path):
    """
    Load controller policy from:
      1) trainer checkpoint dict with key 'control_state_dict', or
      2) raw controller state_dict (.pth) saved by pretraining.

    Accepted:
      - checkpoint dict containing key 'control_state_dict'
      - raw state_dict where all values are tensors
    Rejected: eval bundles and non-checkpoint files.
    """
    p = Path(checkpoint_path)
    if "eval_bundle" in p.name.lower():
        raise ValueError(
            f"Refusing eval bundle in plot.py: {p}\n"
            "Please pass a checkpoint file (e.g., resume_checkpoint.pth or checkpoints.pth)."
        )

    loaded = torch.load(str(p), map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(loaded)} from {p}")

    # eval_bundle.pth generally carries these keys; reject explicitly by content too.
    if ("regions" in loaded) or ("final_results" in loaded):
        raise ValueError(
            f"Detected eval-bundle-like content in {p}. "
            "Use checkpoint files only."
        )

    if "control_state_dict" in loaded and isinstance(loaded["control_state_dict"], dict):
        control_state = loaded["control_state_dict"]
    elif loaded and all(isinstance(v, torch.Tensor) for v in loaded.values()):
        # Raw controller state_dict from pretraining save path.
        control_state = loaded
    else:
        raise ValueError(
            f"{p} is not a supported controller checkpoint. "
            "Expected either key 'control_state_dict' or a raw tensor state_dict."
        )
    policy_net = RoomTempControlNN(input_dim=2, hidden_dim=32, output_dim=2, u_max=U_MAX)
    wrapper = RoomTempControlWrapper(policy_net, B_INPUT)

    state_keys = list(control_state.keys())
    if any(k.startswith("policy_net.") for k in state_keys) or ("B" in control_state):
        # Wrapper-format state dict saved from main training.
        wrapper.load_state_dict(control_state, strict=False)
    else:
        # Rare case: policy-only control state.
        policy_net.load_state_dict(control_state, strict=True)

    policy_net.eval()

    @torch.no_grad()
    def trained_policy(x_np):
        x_t = torch.tensor(np.asarray(x_np, dtype=np.float32)).reshape(1, 2)
        u_t = policy_net(x_t).reshape(-1)
        return u_t.cpu().numpy()

    return trained_policy


def _make_policy_bank(controller_specs: list[str], include_hand_designed: bool) -> list[tuple[str, callable]]:
    """Build list of (name, policy(x)->u)."""
    policies = []

    if include_hand_designed:
        policies.append(("hand_designed", bang_bang_policy))

    for spec in controller_specs:
        label, path = _parse_controller_spec(spec)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Controller checkpoint not found: {path}")
        policy = _make_trained_policy_from_checkpoint_only(path)
        policies.append((label, policy))

    if not policies:
        raise ValueError("No controllers provided. Use --controller or keep hand-designed enabled.")

    names = [n for n, _ in policies]
    if len(set(names)) != len(names):
        # Make duplicate names unique while keeping readable labels.
        seen = {}
        unique = []
        for name, policy in policies:
            k = seen.get(name, 0)
            seen[name] = k + 1
            if k == 0:
                unique.append((name, policy))
            else:
                unique.append((f"{name}_{k+1}", policy))
        policies = unique

    return policies


def _build_shared_scenarios(
    n_mc: int,
    n_steps: int,
    dt: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create identical per-trajectory initial states and Brownian increments
    that will be reused for every controller.

    Returns
    -------
    x0s : (n_mc, 2)
    dws : (n_mc, n_steps, 2)
    tes : (n_mc, n_steps), time-varying ambient temperature samples.
    """
    sqrt_dt = float(np.sqrt(dt))
    x0s = np.zeros((n_mc, 2), dtype=float)
    dws = np.zeros((n_mc, n_steps, 2), dtype=float)
    tes = np.zeros((n_mc, n_steps), dtype=float)

    for i in range(n_mc):
        rng_i = np.random.default_rng(np.random.SeedSequence([int(seed), int(i)]))
        x0s[i] = rng_i.uniform(X0_INIT[0], X0_INIT[1], size=2)
        dws[i] = rng_i.standard_normal((n_steps, 2)) * sqrt_dt
        tes[i] = rng_i.uniform(T_E_MIN, T_E_MAX, size=n_steps)

    return x0s, dws, tes


def _simulate_with_shared_noise(
    policy,
    x0: np.ndarray,
    dw_seq: np.ndarray,
    te_seq: np.ndarray,
    dt: float,
):
    """
    Euler-Maruyama rollout with fixed Brownian increments.

    dw_seq shape: (N, 2), where each row is sqrt(dt)*N(0, I).
    """
    n_steps = int(dw_seq.shape[0])
    xs = np.zeros((n_steps + 1, 2), dtype=float)
    us = np.zeros((n_steps, 2), dtype=float)
    xs[0] = np.asarray(x0, dtype=float)

    g = diffusion()
    for k in range(n_steps):
        u = np.clip(np.asarray(policy(xs[k]), dtype=float), -U_MAX, U_MAX)
        us[k] = u
        xs[k + 1] = xs[k] + drift(xs[k], u, T_e=float(te_seq[k])) * dt + g @ dw_seq[k]

    return xs, us


def _evaluate_controller(
    policy,
    x0s: np.ndarray,
    dws: np.ndarray,
    tes: np.ndarray,
    dt: float,
    T_horizon: float,
):
    """Run full MC batch for one controller with shared scenarios."""
    n_mc, n_steps, _ = dws.shape
    ts = np.linspace(0.0, T_horizon, n_steps + 1)

    statuses = []
    hit_steps = []
    reach_times = []
    energies_hit = []
    energies_to_hit = []

    paths = np.zeros((n_mc, n_steps + 1, 2), dtype=float)
    controls = np.zeros((n_mc, n_steps, 2), dtype=float)
    energy_hist = np.zeros((n_mc, n_steps + 1), dtype=float)

    for i in range(n_mc):
        xs, us = _simulate_with_shared_noise(policy, x0s[i], dws[i], tes[i], dt=dt)
        status, step = classify_trajectory(xs)

        statuses.append(status)
        hit_steps.append(int(step))
        paths[i] = xs
        controls[i] = us

        bu = us @ B_INPUT.T
        power = np.sum(bu ** 2, axis=1)
        e = np.zeros(n_steps + 1, dtype=float)
        e[1:] = np.cumsum(power) * dt
        energy_hist[i] = e

        goal_step = first_goal_hit_step(xs)
        if goal_step is not None:
            bu_hit = us[:goal_step] @ B_INPUT.T
            e_hit = float(np.sum(bu_hit ** 2) * dt)
            energies_hit.append(e_hit)
            energies_to_hit.append(e_hit)
            reach_times.append(goal_step * dt)
        else:
            energies_to_hit.append(np.nan)

    return {
        "statuses": np.asarray(statuses, dtype=object),
        "hit_steps": np.asarray(hit_steps, dtype=int),
        "reach_times": np.asarray(reach_times, dtype=float),
        "energies": np.asarray(energies_hit, dtype=float),
        "energies_to_hit": np.asarray(energies_to_hit, dtype=float),
        "paths": paths,
        "controls": controls,
        "energy_hist": energy_hist,
        "ts": ts,
    }


def _print_stats(name: str, mc: dict, n_mc: int):
    statuses = mc["statuses"]
    n_goal = int(np.sum(statuses == "goal"))
    n_unsafe = int(np.sum(statuses == "unsafe"))
    n_timeout = int(np.sum(statuses == "timeout"))

    reach_times = mc["reach_times"]
    energies = mc["energies"]
    energies_to_hit = mc["energies_to_hit"]
    full_energy = mc["energy_hist"][:, -1]

    print(f"[{name}]")
    print(f"  Success rate: {n_goal / n_mc:.3f}")
    print(f"  Unsafe rate:  {n_unsafe / n_mc:.3f}")
    print(f"  Timeout rate: {n_timeout / n_mc:.3f}")

    if np.any(np.isfinite(energies_to_hit)):
        print(f"  Avg energy to t_hit (B@u): {float(np.nanmean(energies_to_hit)):.3f}")
    else:
        print("  Avg energy to t_hit (B@u): n/a")

    print(f"  Avg terminal energy at horizon (B@u): {float(np.mean(full_energy)):.3f}")

    if reach_times.size > 0:
        print(
            f"  T_hit [s]: mean={float(np.mean(reach_times)):.3f}, "
            f"std={float(np.std(reach_times)):.3f}, "
            f"p50={float(np.percentile(reach_times, 50)):.3f}, "
            f"p90={float(np.percentile(reach_times, 90)):.3f}"
        )
    else:
        print("  T_hit [s]: n/a")

    if energies.size > 0:
        print(
            f"  E_hit: mean={float(np.mean(energies)):.3f}, "
            f"std={float(np.std(energies)):.3f}, "
            f"p50={float(np.percentile(energies, 50)):.3f}, "
            f"p90={float(np.percentile(energies, 90)):.3f}"
        )
    else:
        print("  E_hit: n/a")

    print()


def animate_all_controllers(
    all_results: dict,
    controller_names: list[str],
    *,
    skip: int = 3,
    max_traj: int = 30,
    save_path: Path | str | None = None,
    show: bool = True,
):
    """
    Animate all controllers together:
      - phase-plane trajectories (subset) on the left
      - control channels vs time on the top-right
      - energy vs time on the bottom-right
    Each controller has one color.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError("matplotlib is required for animation. Install with `pip install matplotlib`.") from exc

    if not controller_names:
        return None

    first_mc = all_results[controller_names[0]]
    ts = first_mc["ts"]
    n_total = int(first_mc["paths"].shape[0])
    n_show = min(int(max_traj), n_total)
    if n_show <= 0:
        return None

    if n_total > n_show:
        sel = np.linspace(0, n_total - 1, n_show, dtype=int)
    else:
        sel = np.arange(n_total, dtype=int)

    palette = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(controller_names)}
    frame_step = max(1, int(skip))

    fig = plt.figure(figsize=(12, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    ax_u = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1], sharex=ax_u)
    x_min, x_max = X_DOMAIN
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("MC State Trajectories: All Controllers")
    ax.grid(True, alpha=0.25)

    ax.add_patch(Rectangle((x_min, x_min), x_max - x_min, x_max - x_min, fill=False, lw=1.5, color="black"))
    ax.add_patch(Rectangle(
        (XS_SAFE[0], XS_SAFE[0]),
        XS_SAFE[1] - XS_SAFE[0],
        XS_SAFE[1] - XS_SAFE[0],
        facecolor="#d9f2d9",
        edgecolor="#3a7f3a",
        alpha=0.35,
        lw=1.2,
    ))
    ax.add_patch(Rectangle(
        (XG_GOAL[0], XG_GOAL[0]),
        XG_GOAL[1] - XG_GOAL[0],
        XG_GOAL[1] - XG_GOAL[0],
        facecolor="#c9ddff",
        edgecolor="#2f5da8",
        alpha=0.7,
        lw=1.2,
    ))

    line_bank: dict[str, list] = {}
    point_bank: dict[str, list] = {}
    hit_bank: dict[str, np.ndarray] = {}
    path_bank: dict[str, np.ndarray] = {}
    control_bank: dict[str, np.ndarray] = {}
    energy_stat_bank: dict[str, dict[str, np.ndarray]] = {}
    best_hit_bank: dict[str, int] = {}
    best_desc_bank: dict[str, str] = {}
    best_energy_bank: dict[str, float] = {}

    t_u = ts[:-1]

    ctrl_u1_lines = {}
    ctrl_u1_points = {}
    ctrl_u2_lines = {}
    ctrl_u2_points = {}
    ene_mean_lines = {}
    ene_mean_points = {}
    ene_bands = {}

    for name in controller_names:
        mc = all_results[name]
        p = mc["paths"][sel]
        u_full = mc["controls"]
        e_full = mc["energy_hist"]
        h_full = mc["hit_steps"]
        e_hit_full = mc["energies_to_hit"]
        h = mc["hit_steps"][sel]
        path_bank[name] = p
        hit_bank[name] = h

        finite_hit = np.isfinite(e_hit_full)
        if np.any(finite_hit):
            best_idx = int(np.nanargmax(e_hit_full))
            best_desc = f"idx={best_idx}, max E_hit={float(e_hit_full[best_idx]):.3f}"
        else:
            best_idx = int(np.argmax(e_full[:, -1]))
            best_desc = f"idx={best_idx}, max E(T)={float(e_full[best_idx, -1]):.3f} (no goal-hit)"

        control_bank[name] = u_full[best_idx]
        best_hit = int(h_full[best_idx])
        best_hit = max(0, min(best_hit, int(e_full.shape[1] - 1)))
        best_hit_bank[name] = best_hit
        best_energy_bank[name] = float(e_full[best_idx, best_hit])
        best_desc_bank[name] = best_desc

        # Energy summary over all MC trajectories for this controller:
        # mean/min/max at each time step across trajectories.
        e_min = np.min(e_full, axis=0)
        e_max = np.max(e_full, axis=0)
        e_lo = np.minimum(e_min, e_max)
        e_hi = np.maximum(e_min, e_max)
        e_mean = np.mean(e_full, axis=0)
        # Sanity check: mathematically, mean must lie in [min, max].
        # Keep this explicit to detect real computation/plot-indexing bugs.
        tol = 1e-9
        bad = (e_mean < (e_lo - tol)) | (e_mean > (e_hi + tol))
        if np.any(bad):
            max_low = float(np.max(e_lo[bad] - e_mean[bad]))
            max_high = float(np.max(e_mean[bad] - e_hi[bad]))
            print(
                f"[EnergyBandWarning:{name}] mean outside [min,max] at {int(np.sum(bad))} time steps; "
                f"max below={max_low:.3e}, max above={max_high:.3e}"
            )
        energy_stat_bank[name] = {
            "mean": e_mean,
            "min": e_lo,
            "max": e_hi,
        }

        c = color_map[name]
        lines = []
        points = []
        for _ in range(n_show):
            ln, = ax.plot([], [], color=c, lw=1.0, alpha=0.35)
            pt, = ax.plot([], [], marker="o", ms=2.5, color=c, alpha=0.8)
            lines.append(ln)
            points.append(pt)
        line_bank[name] = lines
        point_bank[name] = points
        ln_u1, = ax_u.plot([], [], color=c, lw=1.8, linestyle="-", alpha=0.9)
        pt_u1, = ax_u.plot([], [], marker="o", ms=3, color=c, alpha=0.95)
        ln_u2, = ax_u.plot([], [], color=c, lw=1.8, linestyle="--", alpha=0.9)
        pt_u2, = ax_u.plot([], [], marker="s", ms=3, color=c, alpha=0.95)
        ln_e_mean, = ax_e.plot([], [], color=c, lw=2.0, linestyle="-", alpha=0.95)
        pt_e_mean, = ax_e.plot([], [], marker="o", ms=3, color=c, alpha=0.95)
        band = ax_e.fill_between(ts[:1], np.array([0.0]), np.array([0.0]), color=c, alpha=0.16, linewidth=0.0)
        ctrl_u1_lines[name] = ln_u1
        ctrl_u1_points[name] = pt_u1
        ctrl_u2_lines[name] = ln_u2
        ctrl_u2_points[name] = pt_u2
        ene_mean_lines[name] = ln_e_mean
        ene_mean_points[name] = pt_e_mean
        ene_bands[name] = band

    from matplotlib.lines import Line2D
    ctrl_handles = [Line2D([0], [0], color=color_map[name], lw=2.0) for name in controller_names]
    ctrl_labels = list(controller_names)
    style_handles = [
        Line2D([0], [0], color="black", lw=1.6, linestyle="-"),
        Line2D([0], [0], color="black", lw=1.6, linestyle="--"),
    ]
    style_labels = ["u1", "u2"]
    ax.legend(ctrl_handles, ctrl_labels, loc="upper right", fontsize=8, frameon=True, title="Controller")
    ax_u.legend(style_handles + ctrl_handles, style_labels + ctrl_labels, loc="upper right", fontsize=8, frameon=True)

    max_hit_step = 0
    for name in controller_names:
        h_best = int(best_hit_bank[name])
        h_best = max(0, min(h_best, int(len(ts) - 1)))
        max_hit_step = max(max_hit_step, h_best)
    t_max_hit = float(ts[max_hit_step]) if max_hit_step > 0 else float(ts[-1])

    ax_u.set_title("Control Channels vs Time (max E_hit trajectory per controller)")
    ax_u.set_ylabel("u1, u2")
    ax_u.set_xlim(0.0, t_max_hit)
    ax_u.set_ylim(-float(U_MAX) - 0.05, float(U_MAX) + 0.05)
    ax_u.grid(True, alpha=0.25)

    ax_e.set_title("Energy vs Time (mean with min/max over shown trajectories)")
    ax_e.set_xlabel("time [s]")
    ax_e.set_ylabel(r"$E(t)")
    ax_e.set_xlim(0.0, t_max_hit)
    max_e = 1.0
    for name in controller_names:
        max_e = max(max_e, float(best_energy_bank[name]))
    ax_e.set_ylim(0.0, 1.05 * max_e)
    ax_e.grid(True, alpha=0.25)

    text_time = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    sel_text = " | ".join([f"{name}: {best_desc_bank[name]}" for name in controller_names])
    # text_sel = ax_u.text(0.01, 0.02, sel_text, transform=ax_u.transAxes, va="bottom", fontsize=7)
    text_sel = ""

    def _all_artists():
        out = []
        for name in controller_names:
            out.extend(line_bank[name])
            out.extend(point_bank[name])
            out.extend([ctrl_u1_lines[name], ctrl_u1_points[name], ctrl_u2_lines[name], ctrl_u2_points[name]])
            out.extend([ene_mean_lines[name], ene_mean_points[name]])
        out.extend([text_time, text_sel])
        return out

    def init_anim():
        for name in controller_names:
            for ln, pt in zip(line_bank[name], point_bank[name]):
                ln.set_data([], [])
                pt.set_data([], [])
            ctrl_u1_lines[name].set_data([], [])
            ctrl_u1_points[name].set_data([], [])
            ctrl_u2_lines[name].set_data([], [])
            ctrl_u2_points[name].set_data([], [])
            ene_mean_lines[name].set_data([], [])
            ene_mean_points[name].set_data([], [])
            try:
                ene_bands[name].remove()
            except Exception:
                pass
            ene_bands[name] = ax_e.fill_between(
                ts[:1],
                np.array([0.0]),
                np.array([0.0]),
                color=color_map[name],
                alpha=0.16,
                linewidth=0.0,
            )
        text_time.set_text("")
        return _all_artists()

    def update(frame_idx):
        for name in controller_names:
            p = path_bank[name]
            u = control_bank[name]
            e_stat = energy_stat_bank[name]
            e_mean = e_stat["mean"]
            e_min = e_stat["min"]
            e_max = e_stat["max"]
            h_best = int(best_hit_bank[name])
            h = hit_bank[name]
            lines = line_bank[name]
            points = point_bank[name]
            for i, (ln, pt) in enumerate(zip(lines, points)):
                k = min(int(frame_idx), int(h[i]))
                ln.set_data(p[i, :k + 1, 0], p[i, :k + 1, 1])
                pt.set_data([p[i, k, 0]], [p[i, k, 1]])
            k = min(int(frame_idx), h_best)
            k_u = min(k, u.shape[0])
            if k_u > 0:
                ctrl_u1_lines[name].set_data(t_u[:k_u], u[:k_u, 0])
                ctrl_u1_points[name].set_data([t_u[k_u - 1]], [u[k_u - 1, 0]])
                ctrl_u2_lines[name].set_data(t_u[:k_u], u[:k_u, 1])
                ctrl_u2_points[name].set_data([t_u[k_u - 1]], [u[k_u - 1, 1]])
            else:
                ctrl_u1_lines[name].set_data([], [])
                ctrl_u1_points[name].set_data([], [])
                ctrl_u2_lines[name].set_data([], [])
                ctrl_u2_points[name].set_data([], [])
            ene_mean_lines[name].set_data(ts[:k + 1], e_mean[:k + 1])
            ene_mean_points[name].set_data([ts[k]], [e_mean[k]])
            try:
                ene_bands[name].remove()
            except Exception:
                pass
            ene_bands[name] = ax_e.fill_between(
                ts[:k + 1],
                e_min[:k + 1],
                e_max[:k + 1],
                color=color_map[name],
                alpha=0.16,
                linewidth=0.0,
            )
        text_time.set_text(f"t = {float(ts[frame_idx]):.2f} s")
        return _all_artists()

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, int(max_hit_step) + 1, frame_step),
        init_func=init_anim,
        interval=35,
        blit=False,
        repeat=False,
    )
    plt.tight_layout()

    if save_path:
        save_path_str = str(save_path)
        writer = "pillow" if save_path_str.lower().endswith(".gif") else None
        ani.save(save_path_str, dpi=140, writer=writer)
        print(f"Saved combined animation to: {save_path_str}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple room-temperature controllers under identical MC noise realizations"
    )
    parser.add_argument(
        "--controller",
        action="append",
        default=[],
        help="Controller spec (required label): 'label=path/to/checkpoint.pth'. Repeatable.",
    )
    parser.add_argument(
        "--no-hand-designed",
        action="store_true",
        help="Disable default hand-designed bang-bang controller.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for shared MC scenarios.")
    parser.add_argument("--n-mc", type=int, default=100, help="Number of MC trajectories.")
    parser.add_argument("--mc-horizon", type=float, default=60.0, help="MC horizon [s].")
    parser.add_argument("--mc-dt", type=float, default=0.005, help="Euler-Maruyama step [s].")
    parser.add_argument("--animate", action="store_true", help="Animate all controllers together on one plot.")
    parser.add_argument("--anim-trajs", type=int, default=100, help="Number of trajectories shown in animation.")
    parser.add_argument("--anim-skip", type=int, default=5, help="Animation frame stride.")
    parser.add_argument(
        "--save-animation-dir",
        type=str,
        default="",
        help="If set, save combined animation GIF in this directory.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive animation windows.")

    args = parser.parse_args()

    include_hand_designed = not bool(args.no_hand_designed)
    policies = _make_policy_bank(controller_specs=list(args.controller), include_hand_designed=include_hand_designed)

    n_mc = int(args.n_mc)
    dt = float(args.mc_dt)
    T_horizon = float(args.mc_horizon)
    if n_mc <= 0:
        raise ValueError("--n-mc must be positive")
    if dt <= 0.0:
        raise ValueError("--mc-dt must be positive")
    if T_horizon <= 0.0:
        raise ValueError("--mc-horizon must be positive")

    n_steps = int(np.ceil(T_horizon / dt))
    x0s, dws, tes = _build_shared_scenarios(n_mc=n_mc, n_steps=n_steps, dt=dt, seed=int(args.seed))

    print("=" * 60)
    print("Controller Comparison on Shared MC Noise")
    print("=" * 60)
    print(f"Seed: {int(args.seed)} | n_mc: {n_mc} | horizon: {T_horizon}s | dt: {dt}")
    print(f"Ambient process: T_e[k] ~ Uniform({T_E_MIN}, {T_E_MAX}) (shared across controllers)")
    print(f"Controllers: {', '.join([name for name, _ in policies])}")
    print()

    save_dir = None
    if args.save_animation_dir:
        save_dir = Path(args.save_animation_dir).expanduser()
        if not save_dir.is_absolute():
            save_dir = (Path.cwd() / save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name, policy in policies:
        mc = _evaluate_controller(policy, x0s=x0s, dws=dws, tes=tes, dt=dt, T_horizon=T_horizon)
        all_results[name] = mc
        _print_stats(name=name, mc=mc, n_mc=n_mc)

    if args.animate or save_dir is not None:
        save_path = None
        if save_dir is not None:
            save_path = save_dir / "all_controllers.gif"
        print("Animating all controllers together...")
        animate_all_controllers(
            all_results=all_results,
            controller_names=[name for name, _ in policies],
            skip=int(args.anim_skip),
            max_traj=int(args.anim_trajs),
            save_path=save_path,
            show=(not args.no_show),
        )


if __name__ == "__main__":
    main()
