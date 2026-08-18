"""
Adversarial MC Comparison — Room Temperature (Two-Phase)
=========================================================

Two-phase adversarial evaluation comparing:
  A) examples/disturb_energy/room_temperature_disturb/outputs/beta_runs/eval_bundle.pth
       Cert. (energy)
  B) examples/disturb_energy/room_temperature_disturb_baseline/outputs/eval_bundle.pth
       Cert. (baseline)
  C) examples/disturb_energy/room_temperature_disturb_rl/outputs/rl_controller.pth
       RL

Phase 1 — run RL with select_T_e_adv; record (x0, T_e_adv, dW) per step.
Phase 2 — for each certified controller, evaluate via:
           [A] Replay: same (x0, T_e_adv, dW) recorded from the RL run.
           [B] Adversarial: select_T_e_adv computed from this controller's own trajectory.
           Report the worst-case (lowest p_success); use that for viz.

Usage (from this directory):
    python adversarial_mc_fixed.py
"""

import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.stats import gaussian_kde
import seaborn as sns

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm":      "Times New Roman",
    "mathtext.it":      "Times New Roman:italic",
    "mathtext.bf":      "Times New Roman:bold",
    "font.size":        16,
    "axes.titlesize":   16,
    "axes.labelsize":   16,
    "legend.fontsize":  16,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "axes.linewidth":   1.2,
    "grid.linewidth":   0.7,
    "lines.linewidth":  2.0,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

ROOT       = Path(__file__).resolve().parents[3]
HERE       = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import RoomTempControlNN, RoomTempControlWrapper
from test import (
    B_INPUT,
    T_E_MIN, T_E_MAX,
    U_MAX,
    X0_INIT, X_DOMAIN, XS_SAFE, XG_GOAL,
    drift, diffusion,
    bang_bang_policy,
    make_trained_policy_from_checkpoint,
)

# Palette used inside main() to assign per-controller colours.
_PAL = sns.color_palette("Set2", 8)

# -----------------------------------------------------------------------
# Set helpers
# -----------------------------------------------------------------------
G_DIFF = diffusion()   # sigma * I_2,  shape (2, 2)


def in_goal(x):
    return np.all(x >= XG_GOAL[0]) and np.all(x <= XG_GOAL[1])


def in_safe(x):
    return np.all(x >= XS_SAFE[0]) and np.all(x <= XS_SAFE[1])


def in_domain(x):
    return np.all(x >= X_DOMAIN[0]) and np.all(x <= X_DOMAIN[1])


# -----------------------------------------------------------------------
# Adversarial ambient-temperature selection
# -----------------------------------------------------------------------
def select_T_e_adv(x, u, dt):
    """
    Exact adversarial T_e via one-step lookahead over the two corners
    {T_E_MIN, T_E_MAX}.

    Why corners?
    -----------
    The dynamics are affine in T_e:
        x_next = x + (f_nom(x,u) + E_PARAM * T_e) * dt + noise
    so the predicted next state is linear in T_e over [T_E_MIN, T_E_MAX].
    Any convex harm function (e.g., squared distance from goal) therefore
    achieves its maximum at a corner — evaluating exactly 2 candidates
    gives the global optimum with no approximation.

    Why not just use sign(sum(x - goal_center))?
    --------------------------------------------
    The adversarial direction depends on
        sum(x + f_nom(x,u)*dt - goal_center),
    not on sum(x - goal_center).  When the nominal drift is large (e.g.,
    the heater is strongly pushing temperatures upward), x can be below the
    goal centre while the one-step lookahead already overshoots it.  In that
    case the two approaches choose *opposite* T_e values.

    Score
    -----
    Maximise squared distance from the goal-set centre in x_next.
    This is the minimal, unambiguous harm metric; it pushes the predicted
    state as far as possible from the comfort zone on every step.
    """
    goal_center = np.full(2, (XG_GOAL[0] + XG_GOAL[1]) / 2.0)

    best_T_e    = T_E_MIN
    best_score  = -np.inf

    for T_e in (T_E_MIN, T_E_MAX):
        x_next = x + drift(x, u, T_e) * dt          # deterministic lookahead
        score  = float(np.sum((x_next - goal_center) ** 2))
        if score > best_score:
            best_score = score
            best_T_e   = T_e

    return best_T_e


# -----------------------------------------------------------------------
# MC rollout (standard or adversarial)
# -----------------------------------------------------------------------
def rollout_mc(
    policy,
    n_mc:     int   = 500,
    T_max:    float = 200.0,
    dt:       float = 0.05,
    seed:     int   = 42,
    n_paths:  int   = 20,
    use_adversarial: bool = False,
    uniform_T_e:     bool = False,
):
    """
    Euler-Maruyama MC rollout.

    When use_adversarial=True, T_e is chosen at every step by
    select_T_e_adv (exact 2-corner lookahead) based on current (x, u).
    When uniform_T_e=True, T_e ~ Uniform[T_E_MIN, T_E_MAX] is sampled
    at every step (random ambient temperature drawn each time step).
    Otherwise T_e is fixed at the midpoint (T_E_MIN+T_E_MAX)/2.

    Returns dict:
        outcomes  : list[str]           'success' | 'unsafe' | 'timeout'
        energies  : list[float]         integral ||u||^2 dt up to goal
        hit_times : list[float]         T_hit (T_max for non-success)
        paths     : list[ndarray(K,2)]  first n_paths state trajectories
        upaths    : list[ndarray(K,2)]  first n_paths control histories
    """
    root_rng = np.random.default_rng(seed)
    x0s = root_rng.uniform(X0_INIT[0], X0_INIT[1], size=(n_mc, 2))

    N_steps = int(T_max / dt)
    T_e_nom = (T_E_MIN + T_E_MAX) / 2.0

    outcomes, energies, hit_times = [], [], []
    paths_out, upaths_out = [], []

    for i in range(n_mc):
        x = x0s[i].copy()
        traj_rng = np.random.default_rng([seed, i])

        energy_acc = 0.0
        outcome    = "timeout"
        T_hit      = T_max
        store      = i < n_paths
        if store:
            path  = [x.copy()]
            upath = []

        for k in range(N_steps):
            if in_goal(x):
                T_hit   = k * dt
                outcome = "success"
                break
            if not in_safe(x) or not in_domain(x):
                T_hit   = k * dt
                outcome = "unsafe"
                break

            u = np.clip(policy(x), -U_MAX, U_MAX)
            energy_acc += np.sum(u ** 2) * dt
            if store:
                upath.append(u.copy())

            if use_adversarial:
                T_e = select_T_e_adv(x, u, dt)
            elif uniform_T_e:
                T_e = traj_rng.uniform(T_E_MIN, T_E_MAX)  # sampled at every step
            else:
                T_e = T_e_nom
            dW  = np.sqrt(dt) * traj_rng.standard_normal(2)
            x   = x + drift(x, u, T_e) * dt + G_DIFF @ dW

            if store:
                path.append(x.copy())

        outcomes.append(outcome)
        energies.append(energy_acc)
        hit_times.append(T_hit)
        if store:
            paths_out.append(np.array(path))
            upaths_out.append(np.array(upath) if upath else np.zeros((0, 2)))

    return dict(outcomes=outcomes, energies=energies, hit_times=hit_times,
                paths=paths_out, upaths=upaths_out)


# -----------------------------------------------------------------------
# Two-phase adversarial evaluation
# -----------------------------------------------------------------------
def rollout_mc_record(
    policy,
    n_mc:    int   = 500,
    T_max:   float = 200.0,
    dt:      float = 0.05,
    seed:    int   = 42,
    n_paths: int   = 20,
):
    """
    Run policy with select_T_e_adv and record (x0, T_e_adv, dW) per step.

    Pre-generates all dWs upfront so the replay controller sees identical noise
    even when the two trajectories have different lengths.

    Returns
    -------
    results  : dict  — same layout as rollout_mc
    recorded : dict  — {
        "x0s":      ndarray (n_mc, 2)
        "T_e_advs": ndarray (n_mc, N_steps)   adversarial T_e at each step
        "dWs":      ndarray (n_mc, N_steps, 2) Brownian increments
    }
    """
    root_rng = np.random.default_rng(seed)
    x0s_arr  = root_rng.uniform(X0_INIT[0], X0_INIT[1], size=(n_mc, 2))

    N_steps = int(T_max / dt)

    outcomes, energies, hit_times = [], [], []
    paths_out, upaths_out = [], []

    T_e_advs = np.zeros((n_mc, N_steps),    dtype=float)
    dWs      = np.zeros((n_mc, N_steps, 2), dtype=float)

    # Pre-generate all dWs upfront per trajectory.
    for i in range(n_mc):
        traj_rng  = np.random.default_rng([seed, i])
        dWs[i]    = np.sqrt(dt) * traj_rng.standard_normal((N_steps, 2))

    for i in range(n_mc):
        x = x0s_arr[i].copy()

        energy_acc = 0.0
        outcome    = "timeout"
        T_hit      = T_max
        store      = i < n_paths
        if store:
            path  = [x.copy()]
            upath = []

        for k in range(N_steps):
            if in_goal(x):
                T_hit   = k * dt
                outcome = "success"
                break
            if not in_safe(x) or not in_domain(x):
                T_hit   = k * dt
                outcome = "unsafe"
                break

            u = np.clip(policy(x), -U_MAX, U_MAX)
            energy_acc += np.sum(u ** 2) * dt
            if store:
                upath.append(u.copy())

            T_e = select_T_e_adv(x, u, dt)
            T_e_advs[i, k] = T_e
            dW  = dWs[i, k]
            x   = x + drift(x, u, T_e) * dt + G_DIFF @ dW

            if store:
                path.append(x.copy())

        outcomes.append(outcome)
        energies.append(energy_acc)
        hit_times.append(T_hit)
        if store:
            paths_out.append(np.array(path))
            upaths_out.append(np.array(upath) if upath else np.zeros((0, 2)))

    results  = dict(outcomes=outcomes, energies=energies, hit_times=hit_times,
                    paths=paths_out, upaths=upaths_out)
    recorded = dict(x0s=x0s_arr, T_e_advs=T_e_advs, dWs=dWs)
    return results, recorded


def rollout_mc_replay(
    policy,
    recorded: dict,
    dt:       float = 0.05,
    n_paths:  int   = 20,
):
    """
    Evaluate policy under the pre-recorded (x0, T_e_adv, dW) realizations.

    The policy computes its own control u; it faces the same T_e sequence and
    Brownian noise that was generated during the RL evaluation.

    Returns dict with same layout as rollout_mc.
    """
    x0s      = recorded["x0s"]       # (n_mc, 2)
    T_e_advs = recorded["T_e_advs"]  # (n_mc, N_steps)
    dWs      = recorded["dWs"]       # (n_mc, N_steps, 2)
    n_mc    = len(x0s)
    N_steps = dWs.shape[1]
    T_max   = N_steps * dt

    outcomes, energies, hit_times = [], [], []
    paths_out, upaths_out = [], []

    for i in range(n_mc):
        x = x0s[i].copy()

        energy_acc = 0.0
        outcome    = "timeout"
        T_hit      = T_max
        store      = i < n_paths
        if store:
            path  = [x.copy()]
            upath = []

        for k in range(N_steps):
            if in_goal(x):
                T_hit   = k * dt
                outcome = "success"
                break
            if not in_safe(x) or not in_domain(x):
                T_hit   = k * dt
                outcome = "unsafe"
                break

            u  = np.clip(policy(x), -U_MAX, U_MAX)
            energy_acc += np.sum(u ** 2) * dt
            if store:
                upath.append(u.copy())

            T_e = T_e_advs[i, k]
            dW  = dWs[i, k]
            x   = x + drift(x, u, T_e) * dt + G_DIFF @ dW

            if store:
                path.append(x.copy())

        outcomes.append(outcome)
        energies.append(energy_acc)
        hit_times.append(T_hit)
        if store:
            paths_out.append(np.array(path))
            upaths_out.append(np.array(upath) if upath else np.zeros((0, 2)))

    return dict(outcomes=outcomes, energies=energies, hit_times=hit_times,
                paths=paths_out, upaths=upaths_out)


# -----------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------
def compute_stats(results: dict):
    outcomes  = results["outcomes"]
    energies  = results["energies"]
    hit_times = results["hit_times"]
    n = len(outcomes)

    success_mask = np.array([o == "success" for o in outcomes], dtype=bool)
    unsafe_mask  = np.array([o == "unsafe"  for o in outcomes], dtype=bool)
    timeout_mask = np.array([o == "timeout" for o in outcomes], dtype=bool)

    e_arr = np.array(energies,  dtype=float)
    t_arr = np.array(hit_times, dtype=float)
    e_suc = e_arr[success_mask]
    t_suc = t_arr[success_mask]

    def _s(fn, a): return float(fn(a)) if a.size > 0 else float("nan")

    return {
        "n_mc":              n,
        "p_success":         float(success_mask.sum()) / n,
        "p_unsafe":          float(unsafe_mask.sum())  / n,
        "p_timeout":         float(timeout_mask.sum()) / n,
        "n_success":         int(success_mask.sum()),
        "n_unsafe":          int(unsafe_mask.sum()),
        "n_timeout":         int(timeout_mask.sum()),
        "energy_mean":       _s(np.mean,   e_suc),
        "energy_median":     _s(np.median, e_suc),
        "energy_std":        _s(np.std,    e_suc),
        "t_hit_mean":        _s(np.mean,   t_suc),
        "t_hit_median":      _s(np.median, t_suc),
        "t_hit_std":         _s(np.std,    t_suc),
        "energies_success":  e_suc,
        "t_hits_success":    t_suc,
        "success_mask":      success_mask,
        "hit_times_all":     t_arr,
    }


# -----------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------
def print_summary(label, stats):
    w = 62
    print(f"\n{'='*w}")
    print(f"  {label}")
    print(f"{'='*w}")
    print(f"  n_mc           = {stats['n_mc']}")
    print(f"  p_success      = {stats['p_success']:.4f}  ({stats['n_success']} / {stats['n_mc']})")
    print(f"  p_unsafe       = {stats['p_unsafe']:.4f}  ({stats['n_unsafe']} / {stats['n_mc']})")
    print(f"  p_timeout      = {stats['p_timeout']:.4f}  ({stats['n_timeout']} / {stats['n_mc']})")
    print(f"  -- energy ||u||^2 dt (successful) --")
    print(f"  mean           = {stats['energy_mean']:.4f}")
    print(f"  median         = {stats['energy_median']:.4f}")
    print(f"  std            = {stats['energy_std']:.4f}")
    print(f"  -- first-hitting time (successful) --")
    print(f"  mean T_hit     = {stats['t_hit_mean']:.4f} s")
    print(f"  median T_hit   = {stats['t_hit_median']:.4f} s")
    print(f"  std T_hit      = {stats['t_hit_std']:.4f} s")


# -----------------------------------------------------------------------
# Viz-only rollouts
# -----------------------------------------------------------------------
def rollout_viz_paths(
    policy,
    n_paths: int   = 10,
    T_max:   float = 200.0,
    t_extra: float = 10.0,
    dt:      float = 0.05,
    seed:    int   = 42,
    use_adversarial: bool = False,
):
    """Returns list of (K, 2) state arrays extended t_extra beyond goal hit."""
    root_rng = np.random.default_rng(seed)
    x0s = root_rng.uniform(X0_INIT[0], X0_INIT[1], size=(n_paths, 2))

    N_steps     = int((T_max + t_extra) / dt)
    T_e_nom     = (T_E_MIN + T_E_MAX) / 2.0
    paths_out   = []

    for i in range(n_paths):
        x        = x0s[i].copy()
        traj_rng = np.random.default_rng([seed, i])
        path     = [x.copy()]
        T_hit    = None

        for k in range(N_steps):
            if T_hit is None and in_goal(x):
                T_hit = k * dt
            if T_hit is not None and (k * dt) >= T_hit + t_extra:
                break

            u   = np.clip(policy(x), -U_MAX, U_MAX)
            T_e = select_T_e_adv(x, u, dt) if use_adversarial else T_e_nom
            dW  = np.sqrt(dt) * traj_rng.standard_normal(2)
            x   = x + drift(x, u, T_e) * dt + G_DIFF @ dW
            path.append(x.copy())

        paths_out.append(np.array(path))

    return paths_out


def rollout_viz_paths_replay(
    policy,
    recorded: dict,
    t_extra:  float = 10.0,
    dt:       float = 0.05,
    n_paths:  int   = 10,
):
    """
    Viz-only rollout using pre-recorded (x0, T_e_adv, dW) realizations,
    so phase trajectories are consistent with rollout_mc_replay stats.
    """
    x0s      = recorded["x0s"]
    T_e_advs = recorded["T_e_advs"]
    dWs      = recorded["dWs"]
    N_steps  = dWs.shape[1]
    n_paths  = min(n_paths, len(x0s))
    extra_steps = int(t_extra / dt)
    paths_out   = []

    for i in range(n_paths):
        x     = x0s[i].copy()
        path  = [x.copy()]
        T_hit = None

        for k in range(N_steps + extra_steps):
            if T_hit is None and in_goal(x):
                T_hit = k * dt
            if T_hit is not None and (k * dt) >= T_hit + t_extra:
                break

            u  = np.clip(policy(x), -U_MAX, U_MAX)
            if k < N_steps:
                T_e = T_e_advs[i, k]
                dW  = dWs[i, k]
            else:
                T_e = (T_E_MIN + T_E_MAX) / 2.0
                dW  = np.zeros(2)
            x = x + drift(x, u, T_e) * dt + G_DIFF @ dW
            path.append(x.copy())

        paths_out.append(np.array(path))

    return paths_out


# -----------------------------------------------------------------------
# Region drawing helper
# -----------------------------------------------------------------------
def draw_regions(ax):
    kw_border = dict(fill=False,  lw=2.0, edgecolor="crimson", zorder=2)
    kw_init   = dict(alpha=0.18, facecolor="#1f77b4", edgecolor="#1f77b4",
                     linestyle="--", lw=2.0, zorder=1)
    kw_goal   = dict(alpha=0.20, facecolor="green",   edgecolor="green",   lw=2.0, zorder=1)
    kw_unsafe = dict(alpha=0.15, facecolor="crimson", edgecolor="crimson", lw=2.0, zorder=1)

    lo, hi = X_DOMAIN

    def _rect(lo_x, lo_y, w, h, **kw):
        return Rectangle((lo_x, lo_y), w, h, **kw)

    # Domain border
    ax.add_patch(_rect(lo, lo, hi - lo, hi - lo, **kw_border))
    # Init set
    ax.add_patch(_rect(X0_INIT[0], X0_INIT[0],
                       X0_INIT[1] - X0_INIT[0], X0_INIT[1] - X0_INIT[0], **kw_init))
    # Goal set
    ax.add_patch(_rect(XG_GOAL[0], XG_GOAL[0],
                       XG_GOAL[1] - XG_GOAL[0], XG_GOAL[1] - XG_GOAL[0], **kw_goal))
    # Unsafe corners (below safe or above safe)
    xs_lo, xs_hi = XS_SAFE
    # below-safe band: x < xs_lo
    ax.add_patch(_rect(lo, lo, hi - lo, xs_lo - lo, **kw_unsafe))
    ax.add_patch(_rect(lo, lo, xs_lo - lo, hi - lo, **kw_unsafe))
    # above-safe band: x > xs_hi
    ax.add_patch(_rect(lo, xs_hi, hi - lo, hi - xs_hi, **kw_unsafe))
    ax.add_patch(_rect(xs_hi, lo, hi - xs_hi, hi - lo, **kw_unsafe))

    fs = 13
    gc = (XG_GOAL[0] + XG_GOAL[1]) / 2.0
    ax.text(gc, gc + 0.3, r"$\mathcal{X}_{\rm goal}$",
            color="green", fontsize=fs, ha="center")
    ax.text(X0_INIT[0] + 0.1, X0_INIT[0] + 0.1,
            r"$\mathcal{X}_{\rm init}$", color="#1f77b4", fontsize=fs)

    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.set_ylim(lo - 0.2, hi + 0.2)
    ax.set_xlabel(r"$x_1$ (°C)")
    ax.set_ylabel(r"$x_2$ (°C)")


# -----------------------------------------------------------------------
# Figure 1 — Phase trajectories
# -----------------------------------------------------------------------
def plot_phase_trajectories(entries, viz_paths_per_ctrl, styles, save_dir=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_regions(ax)

    for style, (label, res, stats), viz_paths in zip(styles, entries, viz_paths_per_ctrl):
        first = True
        for path in viz_paths:
            lbl = label if first else None
            first = False
            ax.plot(path[:, 0], path[:, 1],
                    color=style["color"], ls=style["ls"], lw=style["lw"],
                    alpha=1.0, label=lbl, zorder=3)
        for path in viz_paths:
            ax.plot(path[0, 0], path[0, 1], "o",
                    color=style["color"], markersize=4, alpha=0.8, zorder=4)

    ax.legend(loc="upper left", framealpha=0.9, fontsize=16)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "adv_fig1_phase_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Figure 2 — Energy vs Hitting-time scatter
# -----------------------------------------------------------------------
def plot_energy_vs_thit(entries, styles, save_dir=None):
    fig = plt.figure(figsize=(8, 5))
    gs  = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.2], height_ratios=[1.2, 3.5])
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    for style, (label, res, stats) in zip(styles, entries):
        mask  = stats["success_mask"]
        t_hit = np.array(res["hit_times"])[mask]
        e_hit = stats["energies_success"]
        color = style["color"]
        ls    = style["ls"]

        ax_main.scatter(t_hit, e_hit, color=color, alpha=1.0, s=40,
                        label=label, zorder=3)
        if t_hit.size >= 2:
            pad = 0.05 * np.ptp(t_hit)
            xs  = np.linspace(t_hit.min() - pad, t_hit.max() + pad, 400)
            ys  = gaussian_kde(t_hit, bw_method="scott")(xs)
            ax_top.plot(xs, ys, color=color, ls=ls, lw=2.0)
            ax_top.fill_between(xs, ys, color=color, alpha=0.3)
        if e_hit.size >= 2:
            pad = 0.05 * np.ptp(e_hit)
            ys  = np.linspace(e_hit.min() - pad, e_hit.max() + pad, 400)
            xs  = gaussian_kde(e_hit, bw_method="scott")(ys)
            ax_right.plot(xs, ys, color=color, ls=ls, lw=2.0)
            ax_right.fill_betweenx(ys, xs, color=color, alpha=0.3)

    ax_main.set_xlabel("Reach-avoid time (s)")
    ax_main.set_ylabel(r"Control energy $\int ||u||^2\,dt$")
    ax_main.legend(framealpha=0.9, fontsize=14)
    ax_main.grid(True, alpha=0.35)
    ax_top.set_ylabel("Density", labelpad=6)
    ax_top.grid(True, alpha=0.35)
    ax_top.spines["bottom"].set_visible(False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.yaxis.get_major_locator().set_params(nbins=4)
    ax_right.set_xlabel("Density", labelpad=6)
    ax_right.grid(True, alpha=0.35)
    ax_right.spines["left"].set_visible(False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.xaxis.get_major_locator().set_params(nbins=3)
    fig.align_labels()
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "adv_fig2_energy_vs_thit.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Figure 3 — Control input trajectories
# -----------------------------------------------------------------------
def plot_u_trajectories(entries, dt, styles, save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for style, (label, res, __) in zip(styles, entries):
        upaths = res["upaths"]
        if not upaths:
            continue
        max_len = max(u.shape[0] for u in upaths if u.shape[0] > 0)
        if max_len == 0:
            continue
        for dim, ax in enumerate(axes):
            mat = np.full((len(upaths), max_len), np.nan)
            for j, u in enumerate(upaths):
                if u.shape[0] > 0:
                    mat[j, :u.shape[0]] = u[:, dim]
            t_grid = np.arange(max_len) * dt
            mean   = np.nanmean(mat, axis=0)
            lo     = np.nanmin(mat, axis=0)
            hi     = np.nanmax(mat, axis=0)
            ax.fill_between(t_grid, lo, hi, color=style["color"], alpha=0.25)
            ax.plot(t_grid, mean, color=style["color"], ls=style["ls"],
                    lw=2.0, label=label)

    for dim, ax in enumerate(axes):
        ax.axhline(0, color="grey", lw=1.5, ls="--", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(fr"$u_{dim+1}(t)$")
        ax.legend(framealpha=0.9, fontsize=14)
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    if save_dir is not None:
        p = Path(save_dir) / "adv_fig3_u_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Figure 4 — State vs time trajectories
# -----------------------------------------------------------------------
def plot_state_trajectories(entries, dt, styles, save_dir=None):
    """
    Plot x1(t) and x2(t) for each controller (mean ± min/max band over
    stored paths).  Horizontal bands mark the initial set, goal set, and
    unsafe zones so the reader can judge constraint satisfaction at a glance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for style, (label, res, __) in zip(styles, entries):
        spaths = res["paths"]
        if not spaths:
            continue
        max_len = max(p.shape[0] for p in spaths if p.shape[0] > 0)
        if max_len == 0:
            continue
        for dim, ax in enumerate(axes):
            mat = np.full((len(spaths), max_len), np.nan)
            for j, p in enumerate(spaths):
                if p.shape[0] > 0:
                    mat[j, :p.shape[0]] = p[:, dim]
            t_grid = np.arange(max_len) * dt
            mean   = np.nanmean(mat, axis=0)
            lo     = np.nanmin(mat,  axis=0)
            hi     = np.nanmax(mat,  axis=0)
            ax.fill_between(t_grid, lo, hi, color=style["color"], alpha=0.20)
            ax.plot(t_grid, mean, color=style["color"], ls=style["ls"],
                    lw=2.0, label=label)

    # Annotate each axis with set boundaries
    for ax in axes:
        lo_t, hi_t = ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else (0, 1)

        # Unsafe zones (below XS_SAFE[0] and above XS_SAFE[1]) — red fill
        ax.axhspan(X_DOMAIN[0],  XS_SAFE[0], color="crimson", alpha=0.12,
                   label="Unsafe", zorder=0)
        ax.axhspan(XS_SAFE[1],   X_DOMAIN[1], color="crimson", alpha=0.12,
                   zorder=0)

        # # Initial set — blue dashed lines
        # ax.axhline(X0_INIT[0], color="#1f77b4", ls="--", lw=1.4, alpha=0.8,
        #            label="Init range", zorder=1)
        # ax.axhline(X0_INIT[1], color="#1f77b4", ls="--", lw=1.4, alpha=0.8,
        #            zorder=1)

        # Goal set — green fill
        ax.axhspan(XG_GOAL[0], XG_GOAL[1], color="green", alpha=0.15,
                   label="Goal", zorder=0)
        ax.axhline(XG_GOAL[0], color="green", ls="-",  lw=1.2, alpha=0.8, zorder=1)
        ax.axhline(XG_GOAL[1], color="green", ls="-",  lw=1.2, alpha=0.8, zorder=1)

        ax.set_ylim(X_DOMAIN[0] - 0.5, X_DOMAIN[1] + 0.5)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.35)

    axes[0].set_ylabel(r"$x_1$ (°C)")
    axes[1].set_ylabel(r"$x_2$ (°C)")

    # Single combined legend: controllers + region markers (deduplicated)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels)
                  if not (l in seen or seen.add(l))]
        ax.legend(*zip(*unique), framealpha=0.9, fontsize=12, loc="upper left")

    fig.tight_layout()
    if save_dir is not None:
        p = Path(save_dir) / "adv_fig4_state_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------------
    # Controller registry
    # -----------------------------------------------------------------------
    # Each entry is a dict with keys:
    #   label   : display name
    #   path    : Path to checkpoint, or None for analytic policies (bang-bang)
    #   is_raw  : True if file is a raw state_dict; False for eval_bundle
    #   role    : "rl"   → Phase 1: evaluated with select_T_e_adv, realizations recorded
    #             "eval" → Phase 2: worst-case of [replay vs. adversarial] reported
    #   color, ls, lw : matplotlib style
    #
    # To skip a controller, comment out its entry — no other changes needed.
    # Exactly one entry must have role="rl".
    # -----------------------------------------------------------------------
    CONTROLLERS = [
        # dict(
        #     label  = "Cert. (energy)",
        #     path   = HERE / "outputs" / "beta_runs" / "eval_bundle.pth",
        #     is_raw = False,
        #     role   = "eval",
        #     color  = _PAL[0], ls="-",  lw=2.0,
        # ),
        dict(
            label  = "Cert.",
            path   = ROOT / "examples" / "disturb_energy"
                          / "room_temperature_disturb_baseline"
                          / "outputs" / "eval_bundle.pth",
            is_raw = False,
            role   = "eval",
            color  = _PAL[0], ls="--", lw=2.0,
        ),
        dict(
            label  = "RL",
            path   = ROOT / "examples" / "disturb_energy"
                          / "room_temperature_disturb_rl"
                          / "outputs" / "rl_controller.pth",
            is_raw = True,
            role   = "rl",    # Phase 1: record adversarial realizations
            color  = _PAL[3], ls=":",  lw=2.0,
        ),
        dict(
            label  = "Bang-bang",
            path   = None,    # analytic policy — no checkpoint file
            is_raw = None,
            role   = "eval",
            color  = _PAL[1], ls="-.", lw=2.0,
        ),
    ]

    # --- validate paths ---
    for ctrl in CONTROLLERS:
        if ctrl["path"] is not None and not ctrl["path"].exists():
            raise FileNotFoundError(
                f"Bundle not found for '{ctrl['label']}':\n  {ctrl['path']}"
            )

    # --- derived subsets ---
    rl_list   = [c for c in CONTROLLERS if c["role"] == "rl"]
    eval_list = [c for c in CONTROLLERS if c["role"] == "eval"]
    assert len(rl_list) == 1, "Exactly one controller must have role='rl'."
    rl_ctrl = rl_list[0]

    # Style list in CONTROLLERS order (passed to plotting functions)
    styles = [{"color": c["color"], "ls": c["ls"], "lw": c["lw"],
               "label": c["label"]} for c in CONTROLLERS]

    def load_policy(ctrl):
        if ctrl["path"] is None:
            return bang_bang_policy
        return make_trained_policy_from_checkpoint(str(ctrl["path"]))

    # --- MC config ---
    N_MC    = 100
    T_MAX   = 60.0
    DT      = 0.005
    SEED    = 42
    N_PATHS = 4

    # -----------------------------------------------------------------------
    # Phase 1 — run RL with select_T_e_adv; record (x0, T_e_adv, dW)
    # -----------------------------------------------------------------------
    print(f"\n[Phase 1] Running '{rl_ctrl['label']}' with adversarial T_e "
          f"(select_T_e_adv) — recording realizations ...")
    if rl_ctrl["path"]:
        print(f"  {rl_ctrl['path']}")
    rl_policy        = load_policy(rl_ctrl)
    rl_res, recorded = rollout_mc_record(
        rl_policy, n_mc=N_MC, T_max=T_MAX, dt=DT, seed=SEED, n_paths=N_PATHS,
    )
    rl_stats = compute_stats(rl_res)
    print_summary(f"{rl_ctrl['label']} [adversarial]", rl_stats)

    # [C] Uniform T_e — complementary non-adversarial evaluation
    print(f"  [C] Uniform T_e (T_e ~ Uniform[{T_E_MIN},{T_E_MAX}] per step) ...")
    rl_res_uni   = rollout_mc(rl_policy, n_mc=N_MC, T_max=T_MAX, dt=DT,
                              seed=SEED, n_paths=N_PATHS, uniform_T_e=True)
    rl_stats_uni = compute_stats(rl_res_uni)
    print_summary(f"{rl_ctrl['label']} [uniform T_e]", rl_stats_uni)

    rl_ctrl["_entry"]      = (rl_ctrl["label"], rl_res, rl_stats)
    rl_ctrl["_worst_mode"] = "rl"

    # -----------------------------------------------------------------------
    # Phase 2 — evaluate each "eval" controller via replay AND adversarial;
    #           report the worst case (lowest p_success).
    # -----------------------------------------------------------------------
    for ctrl in eval_list:
        label  = ctrl["label"]
        policy = load_policy(ctrl)

        print(f"\n[Phase 2] Evaluating '{label}' ...")
        if ctrl["path"] is not None:
            print(f"  {ctrl['path']}")

        # [A] Replay RL realizations
        print(f"  [A] Replay (RL realizations) ...")
        res_replay   = rollout_mc_replay(policy, recorded=recorded, dt=DT, n_paths=N_PATHS)
        stats_replay = compute_stats(res_replay)
        print_summary(f"{label} [replay]", stats_replay)

        # [B] Per-controller adversarial
        print(f"  [B] Adversarial (per-controller, select_T_e_adv) ...")
        res_adv   = rollout_mc(policy, n_mc=N_MC, T_max=T_MAX, dt=DT,
                               seed=SEED, n_paths=N_PATHS, use_adversarial=True)
        stats_adv = compute_stats(res_adv)
        print_summary(f"{label} [adversarial]", stats_adv)

        # [C] Uniform T_e — complementary non-adversarial evaluation
        print(f"  [C] Uniform T_e (T_e ~ Uniform[{T_E_MIN},{T_E_MAX}] per step) ...")
        res_uni   = rollout_mc(policy, n_mc=N_MC, T_max=T_MAX, dt=DT,
                               seed=SEED, n_paths=N_PATHS, uniform_T_e=True)
        stats_uni = compute_stats(res_uni)
        print_summary(f"{label} [uniform T_e]", stats_uni)

        # Pick worst case (A vs B only — uniform T_e is complementary)
        if stats_replay["p_success"] <= stats_adv["p_success"]:
            res, stats, mode = res_replay, stats_replay, "replay"
        else:
            res, stats, mode = res_adv, stats_adv, "adversarial"
        print(f"  → Worst case: [{mode}]  p_success = {stats['p_success']:.4f}")

        ctrl["_entry"]      = (label, res, stats)
        ctrl["_worst_mode"] = mode

    # Assemble entries in CONTROLLERS order
    entries = [c["_entry"] for c in CONTROLLERS]

    # -----------------------------------------------------------------------
    # Viz paths — method matches worst-case mode so trajectories are
    # consistent with the reported statistics.
    # -----------------------------------------------------------------------
    viz_paths_per_ctrl = []
    for ctrl in CONTROLLERS:
        policy = load_policy(ctrl)
        if ctrl["role"] == "rl":
            viz_paths_per_ctrl.append(rollout_viz_paths(
                policy, n_paths=N_PATHS, T_max=T_MAX, t_extra=10.0,
                dt=DT, seed=SEED, use_adversarial=True,
            ))
        elif ctrl["_worst_mode"] == "replay":
            viz_paths_per_ctrl.append(rollout_viz_paths_replay(
                policy, recorded=recorded, t_extra=10.0, dt=DT, n_paths=N_PATHS,
            ))
        else:
            viz_paths_per_ctrl.append(rollout_viz_paths(
                policy, n_paths=N_PATHS, T_max=T_MAX, t_extra=10.0,
                dt=DT, seed=SEED, use_adversarial=True,
            ))

    SAVE_DIR = str(OUTPUT_DIR)
    plot_phase_trajectories(entries, viz_paths_per_ctrl, styles, save_dir=SAVE_DIR)
    plot_energy_vs_thit(entries,                         styles, save_dir=SAVE_DIR)
    plot_u_trajectories(entries, DT,                     styles, save_dir=SAVE_DIR)
    plot_state_trajectories(entries, DT,                 styles, save_dir=SAVE_DIR)


if __name__ == "__main__":
    main()
