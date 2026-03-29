"""
MC Validation Comparison — Four Controllers
============================================

Compares:
  A) examples/disturb_energy/inv_pend_syn_disturb_baseline/outputs/eval_bundle.pth
       Cert. (plain disturbance synthesis)
  B) examples/disturb_temporal/inv_pend_syn_disturb/outputs/time_runs/eval_bundle.pth
       Cert. (time) (temporal curriculum)
  C) examples/disturb_energy/inv_pend_syn_disturb/outputs/beta_runs/eval_bundle.pth
       Cert. (energy) (beta / energy curriculum)
  D) examples/disturb_energy/inv_pend_syn_disturb_rl/outputs/rl_controller.pth
       RL (same architecture, additive box disturbance training)

Metrics (successful trajectories only unless stated):
  1. Reach-avoid probability  p_success / p_fail / p_timeout
  2. Control energy  E = integral_0^{T_hit}  u_raw(t)^2  dt
  3. First-hitting time  T_hit

Output (PDF, saved to outputs/):
  fig1_phase_trajectories.pdf
  fig2_energy_vs_thit.pdf
  fig5_u_raw_trajectories

Usage (from this directory):
    python plot.py
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

# -----------------------------------------------------------------------
# Publication-quality style
# -----------------------------------------------------------------------
matplotlib.rcParams.update({
    # --- text font ---
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    # --- math font: use the same serif family instead of the default cm ---
    "mathtext.fontset":     "custom",
    "mathtext.rm":          "Times New Roman",
    "mathtext.it":          "Times New Roman:italic",
    "mathtext.bf":          "Times New Roman:bold",
    # --- sizes ---
    "font.size":            16,
    "axes.titlesize":       16,
    "axes.labelsize":       16,
    "legend.fontsize":      16,
    "xtick.labelsize":      16,
    "ytick.labelsize":      16,
    # --- layout ---
    "axes.linewidth":       1.2,
    "grid.linewidth":       0.7,
    "lines.linewidth":      2.0,
    # --- PDF embedding (TrueType, not Type 3) ---
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[3]
HERE       = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN
from src.save_load_utils import load_eval_bundle

# -----------------------------------------------------------------------
# Physical constants  (must match main.py / test_invpend_dynamics.py)
# -----------------------------------------------------------------------
g_grav = 9.81
L      = 0.5
m      = 0.15
b      = 0.1
M      = 6.0
sigma  = 0.2
pi     = np.pi

TORQUE_SCALE = M / (m * L**2) 

# -----------------------------------------------------------------------
# Per-controller style  (color, linestyle, marker)
# -----------------------------------------------------------------------
_pal = sns.color_palette("Set2", 4)
CTRL_STYLES = [
    {"color": _pal[0], "ls": "-",  "lw": 2.0, "label": "Cert."},      # solid
    {"color": _pal[1], "ls": "--", "lw": 2.0, "label": "Cert. (time)"},    # dashed
    {"color": _pal[2], "ls": "-.", "lw": 2.0, "label": "Cert. (energy)"},  # dash-dot
    {"color": _pal[3], "ls": ":",  "lw": 2.0, "label": "RL"},              # dotted
]

# -----------------------------------------------------------------------
# Spec regions
# -----------------------------------------------------------------------
X_init_bounds = {"x1_min": 3*pi/4,  "x1_max": 5*pi/4,  "x2_min": -1.0,  "x2_max":  1.0}
X_goal_bounds = {"x1_min": -0.4*pi, "x1_max": 0.4*pi,  "x2_min": -4.0,  "x2_max":  4.0}
X_unsafe_1    = {"x1_min": -2*pi,   "x1_max": -3*pi/2, "x2_min": -20.0, "x2_max": -10.0}
X_unsafe_2    = {"x1_min":  3*pi/2, "x1_max":  2*pi,   "x2_min":  10.0, "x2_max":  20.0}
X_bounds      = {"x1_min": -2*pi,   "x1_max":  2*pi,   "x2_min": -20.0, "x2_max":  20.0}


def in_box(x1, x2, box):
    return (box["x1_min"] <= x1 <= box["x1_max"]) and (box["x2_min"] <= x2 <= box["x2_max"])


# -----------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------
def f_drift(x, u):
    x1, x2 = x
    u1, u2 = u
    return np.array([
        x2,
        (g_grav / L) * np.sin(x1) + (-b * x2) / (m * L**2) + u2,
    ], dtype=float)


def g_diff(x):
    return np.array([0.0, sigma], dtype=float)


# -----------------------------------------------------------------------
# Controller helpers
# -----------------------------------------------------------------------
def load_control_net(bundle_path, device="cpu", pretrained_state_dict=False):
    """
    Load a WrapperConterlNN from either:
      - an eval_bundle.pth  (pretrained_state_dict=False, default): dict with key "control_state_dict"
      - a controller_pretrained.pth  (pretrained_state_dict=True): raw state_dict saved by pretrain step
    """
    rl_policy_net = InvertControlNN(input_dim=2, hidden_dim=8, output_dim=1)
    control_net   = WrapperConterlNN(rl_policy_net).to(device)
    if pretrained_state_dict:
        state = torch.load(bundle_path, map_location="cpu")
        control_net.load_state_dict(state)
    else:
        bundle = load_eval_bundle(Path(bundle_path), map_location="cpu")
        if bundle.get("control_state_dict") is not None:
            control_net.load_state_dict(bundle["control_state_dict"])
    control_net.eval()
    return control_net


def get_u_and_raw(x1, x2, controller):
    """Return (u_applied [2], u_raw scalar)."""
    with torch.no_grad():
        xt        = torch.tensor([[x1, x2]], dtype=torch.float32)
        u_applied = controller(xt).detach().numpy().reshape(-1)
        if hasattr(controller, "raw_control") and callable(controller.raw_control):
            u_raw = float(controller.raw_control(xt).detach().numpy().reshape(-1)[0])
        else:
            u_raw = float(u_applied[1]) / TORQUE_SCALE
    return u_applied, u_raw


# -----------------------------------------------------------------------
# MC rollout
# -----------------------------------------------------------------------
def rollout_mc(
    controller,
    n_mc:             int   = 500,
    T_max:            float = 8.0,
    dt:               float = 0.005,
    d:                np.ndarray = np.array([1.0, 1.0], dtype=float),
    disturbance_mode: str   = "uniform",
    seed:             int   = 42,
    n_paths:          int   = 20,
    energy_budget:    float = None,   # fail if ∫u_raw² dt exceeds this
    time_budget:      float = None,   # fail if T_hit > this (checked at goal entry)
):
    """
    Euler-Maruyama rollout.  Returns dict:
        outcomes  : list[str]         'success' | 'fail' | 'timeout'
        energies  : list[float]       integral u_raw^2 dt  up to T_hit / budget violation
        hit_times : list[float]       T_hit (T_max for non-success)
        paths     : list[ndarray(K,2)]  first n_paths trajectories

    Budget semantics:
        energy_budget : if accumulated ∫u_raw² dt exceeds the budget at any step
                        the trajectory is classified 'fail' immediately.
        time_budget   : if the trajectory enters X_goal but T_hit > time_budget
                        it is reclassified 'fail' instead of 'success'.

    Initial conditions are pre-generated from a root RNG so that every controller
    sees the same x0_i for trajectory i.  Per-step noise (w and dW) uses a
    per-trajectory RNG seeded by (seed, i) so the noise sequence is also identical
    across controllers for each trajectory, regardless of early termination.
    """
    # Pre-generate all initial conditions so every controller gets the same
    # x0_i regardless of how long other trajectories ran.
    root_rng = np.random.default_rng(seed)
    x1_init  = root_rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"], size=n_mc)
    x2_init  = root_rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"], size=n_mc)

    N_steps = int(T_max / dt)

    outcomes   = []
    energies   = []
    hit_times  = []
    paths_out  = []
    uraw_out   = []   # u_raw(t) history for stored paths

    for i in range(n_mc):
        x1, x2 = float(x1_init[i]), float(x2_init[i])

        # Per-trajectory noise RNG — same seed for every controller on trajectory i
        traj_rng = np.random.default_rng([seed, i])

        energy_acc = 0.0
        outcome    = "timeout"
        T_hit      = T_max
        store_path = i < n_paths
        if store_path:
            path  = [[x1, x2]]
            uraw  = []

        for k in range(N_steps):
            if in_box(x1, x2, X_goal_bounds):
                T_hit   = k * dt
                outcome = "fail" if (time_budget is not None and T_hit > time_budget) else "success"
                break
            if in_box(x1, x2, X_unsafe_1) or in_box(x1, x2, X_unsafe_2):
                outcome = "fail"
                T_hit   = k * dt
                break

            u_applied, u_raw = get_u_and_raw(x1, x2, controller)
            energy_acc += u_raw**2 * dt

            if store_path:
                uraw.append(u_raw)

            if energy_budget is not None and energy_acc > energy_budget:
                outcome = "fail"
                T_hit   = k * dt
                break

            w      = traj_rng.uniform(-d, d) if disturbance_mode == "uniform" else np.zeros(2)
            drift  = f_drift(np.array([x1, x2]), u_applied) + w
            dW     = np.sqrt(dt) * traj_rng.standard_normal()
            x_next = np.array([x1, x2]) + drift * dt + g_diff(np.array([x1, x2])) * dW
            x1, x2 = float(x_next[0]), float(x_next[1])

            if x1 > 2 * pi:   x1 -= 4 * pi
            elif x1 < -2 * pi: x1 += 4 * pi

            if store_path:
                path.append([x1, x2])

        outcomes.append(outcome)
        energies.append(energy_acc)
        hit_times.append(T_hit)
        if store_path:
            paths_out.append(np.array(path,  dtype=float))
            uraw_out.append(np.array(uraw,   dtype=float))

    return dict(outcomes=outcomes, energies=energies, hit_times=hit_times,
                paths=paths_out, uraw_paths=uraw_out)


# -----------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------
def compute_stats(results: dict):
    outcomes  = results["outcomes"]
    energies  = results["energies"]
    hit_times = results["hit_times"]
    n = len(outcomes)

    success_mask = np.array([o == "success" for o in outcomes], dtype=bool)
    fail_mask    = np.array([o == "fail"    for o in outcomes], dtype=bool)
    timeout_mask = np.array([o == "timeout" for o in outcomes], dtype=bool)

    e_arr = np.array(energies,  dtype=float)
    t_arr = np.array(hit_times, dtype=float)
    e_suc = e_arr[success_mask]
    t_suc = t_arr[success_mask]

    def _s(fn, a): return float(fn(a)) if a.size > 0 else float("nan")

    return {
        "n_mc":            n,
        "p_success":       float(success_mask.sum()) / n,
        "p_fail":          float(fail_mask.sum())    / n,
        "p_timeout":       float(timeout_mask.sum()) / n,
        "n_success":       int(success_mask.sum()),
        "n_fail":          int(fail_mask.sum()),
        "n_timeout":       int(timeout_mask.sum()),
        "energy_mean":     _s(np.mean,   e_suc),
        "energy_median":   _s(np.median, e_suc),
        "energy_std":      _s(np.std,    e_suc),
        "t_hit_mean":      _s(np.mean,   t_suc),
        "t_hit_median":    _s(np.median, t_suc),
        "t_hit_std":       _s(np.std,    t_suc),
        "energies_success": e_suc,
        "t_hits_success":   t_suc,
        "success_mask":     success_mask,
        "hit_times_all":    t_arr,
    }


# -----------------------------------------------------------------------
# Region drawing helper
# -----------------------------------------------------------------------
def draw_regions(ax):
    kw_border = dict(fill=False, lw=2.0, edgecolor="crimson", zorder=2)
    kw_init   = dict(alpha=0.18, facecolor="#1f77b4", edgecolor="#1f77b4",
                     linestyle="--", lw=2.0, zorder=1)
    kw_goal   = dict(alpha=0.20, facecolor="green",   edgecolor="green",   lw=2.0, zorder=1)
    kw_unsafe = dict(alpha=0.20, facecolor="crimson", edgecolor="crimson", lw=2.0, zorder=1)

    def _rect(box, **kw):
        return Rectangle(
            (box["x1_min"], box["x2_min"]),
            box["x1_max"] - box["x1_min"],
            box["x2_max"] - box["x2_min"],
            **kw,
        )

    ax.add_patch(_rect(X_bounds, **kw_border))
    ax.add_patch(_rect(X_init_bounds, **kw_init))
    ax.add_patch(_rect(X_goal_bounds, **kw_goal))
    ax.add_patch(_rect(X_unsafe_1,    **kw_unsafe))
    ax.add_patch(_rect(X_unsafe_2,    **kw_unsafe))

    # text labels
    fs = 14
    ax.text(-0.35*pi,  3.5, r"$\mathcal{X}_{\rm goal}$",  color="green",   fontsize=fs)
    ax.text(-5.8,     -14,  r"$\mathcal{X}_{\rm unsafe}$", color="crimson", fontsize=fs)
    ax.text( 4.9,      14,  r"$\mathcal{X}_{\rm unsafe}$", color="crimson", fontsize=fs)
    ax.text( 2.6,      0.3, r"$\mathcal{X}_{\rm init}$",   color="#1f77b4", fontsize=fs)

    ax.set_xlim(X_bounds["x1_min"] - 0.1, X_bounds["x1_max"] + 0.1)
    ax.set_ylim(X_bounds["x2_min"] - 0.5, X_bounds["x2_max"] + 0.5)
    ax.set_xlabel(r"$x_1$ (rad)")
    ax.set_ylabel(r"$x_2$ (rad/s)")


# -----------------------------------------------------------------------
# Viz-only rollout: simulate up to T_hit + t_extra  (no stats produced)
# -----------------------------------------------------------------------
def rollout_viz_paths(
    controller,
    n_paths:          int   = 10,
    T_max:            float = 8.0,
    t_extra:          float = 1.0,
    dt:               float = 0.005,
    d:                np.ndarray = np.array([1.0, 1.0], dtype=float),
    disturbance_mode: str   = "uniform",
    seed:             int   = 42,
):
    """
    Returns a list of (K, 2) phase-state arrays.
    Each trajectory runs until T_hit (first goal entry) + t_extra seconds,
    or until T_max + t_extra, whichever comes first.  Unsafe termination
    is not enforced so the curve continues through for visual context.

    Uses pre-generated initial conditions and per-trajectory noise RNGs so
    all controllers start from identical (x0, w, dW) per path.
    """
    # Pre-generate initial conditions shared across all controllers
    root_rng = np.random.default_rng(seed)
    x1_init  = root_rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"], size=n_paths)
    x2_init  = root_rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"], size=n_paths)

    N_steps   = int((T_max + t_extra) / dt)
    paths_out = []

    for i in range(n_paths):
        x1, x2 = float(x1_init[i]), float(x2_init[i])

        # Per-trajectory noise RNG — same for every controller on path i
        traj_rng = np.random.default_rng([seed, i])

        path    = [[x1, x2]]
        T_hit   = None   # time of first goal entry

        for k in range(N_steps):
            # record goal entry time once
            if T_hit is None and in_box(x1, x2, X_goal_bounds):
                T_hit = k * dt

            # stop t_extra seconds after goal hit
            if T_hit is not None and (k * dt) >= T_hit + t_extra:
                break

            u_applied, _ = get_u_and_raw(x1, x2, controller)
            w      = traj_rng.uniform(-d, d) if disturbance_mode == "uniform" else np.zeros(2)
            drift  = f_drift(np.array([x1, x2]), u_applied) + w
            dW     = np.sqrt(dt) * traj_rng.standard_normal()
            x_next = np.array([x1, x2]) + drift * dt + g_diff(np.array([x1, x2])) * dW
            x1, x2 = float(x_next[0]), float(x_next[1])

            if x1 > 2 * pi:    x1 -= 4 * pi
            elif x1 < -2 * pi: x1 += 4 * pi

            path.append([x1, x2])

        paths_out.append(np.array(path, dtype=float))

    return paths_out


# -----------------------------------------------------------------------
# Figure 1 — Phase trajectories (all controllers, one plot)
# -----------------------------------------------------------------------
def plot_phase_trajectories(entries, viz_paths_per_ctrl, save_dir=None):
    """
    viz_paths_per_ctrl : list of path lists, one per controller,
                         produced by rollout_viz_paths (extended to T_hit+1s).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_regions(ax)

    for style, (label, res, stats), viz_paths in zip(CTRL_STYLES, entries, viz_paths_per_ctrl):
        color = style["color"]
        ls    = style["ls"]
        lw    = style["lw"]
        first = True
        for path in viz_paths:
            lbl = label if first else None
            first = False
            ax.plot(path[:, 0], path[:, 1],
                    color=color, ls=ls, lw=lw, alpha=1.0, label=lbl, zorder=3)
        # mark start points
        for path in viz_paths:
            ax.plot(path[0, 0], path[0, 1], "o",
                    color=color, markersize=4, alpha=0.8, zorder=4)

    ax.legend(loc="upper left", framealpha=0.9, fontsize=18)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(r"$\theta$, rad")
    ax.set_ylabel(r"$\dot\theta$, rad/s")
    # ax.set_title("Phase-Plane Trajectories")
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "fig1_phase_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Figure 2 — Energy vs Hitting-time scatter with marginal KDEs
# -----------------------------------------------------------------------
def plot_energy_vs_thit(entries, budgets, save_dir=None, styles=None):
    """
    Joint scatter of (T_hit, energy) for successful trajectories,
    with marginal KDE of T_hit on the top panel and marginal KDE of
    energy on the right panel.
    """
    if styles is None:
        styles = CTRL_STYLES
    fig = plt.figure(figsize=(8, 5))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[3.5, 1.2], height_ratios=[1.2, 3.5],
        # hspace=0.06, wspace=0.06,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    for style, (label, res, stats) in zip(styles, entries):
        mask  = stats["success_mask"]
        t_hit = np.array(res["hit_times"])[mask]
        e_hit = stats["energies_success"]
        color = style["color"]
        ls    = style["ls"]

        # --- main scatter ---
        ax_main.scatter(t_hit, e_hit,
                        color=color, alpha=1.0, s=40,
                        label=label, zorder=3)

        # --- top marginal: T_hit KDE ---
        if t_hit.size >= 2:
            pad = 0.05 * np.ptp(t_hit)
            xs  = np.linspace(t_hit.min() - pad, t_hit.max() + pad, 400)
            ys  = gaussian_kde(t_hit, bw_method="scott")(xs)
            ax_top.plot(xs, ys, color=color, ls=ls, lw=2.0)
            ax_top.fill_between(xs, ys, color=color, alpha=0.3)

        # --- right marginal: energy KDE ---
        if e_hit.size >= 2:
            pad = 0.05 * np.ptp(e_hit)
            ys  = np.linspace(e_hit.min() - pad, e_hit.max() + pad, 400)
            xs  = gaussian_kde(e_hit, bw_method="scott")(ys)
            ax_right.plot(xs, ys, color=color, ls=ls, lw=2.0)
            ax_right.fill_betweenx(ys, xs, color=color, alpha=0.3)

    # --- main axes ---
    ax_main.set_xlabel("Reach-avoid time (s)")
    ax_main.set_ylabel("Control energy")
    ax_main.legend(framealpha=0.9, fontsize=14)
    ax_main.grid(True, alpha=0.35)

    # --- top marginal ---
    ax_top.set_ylabel("Density", labelpad=6)
    ax_top.yaxis.set_label_position("left")
    ax_top.tick_params(axis="y", labelsize=12)
    ax_top.grid(True, alpha=0.35)
    ax_top.spines["bottom"].set_visible(False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    # remove zero from top y-axis to avoid clutter
    ax_top.yaxis.get_major_locator().set_params(nbins=4)

    # --- right marginal ---
    ax_right.set_xlabel("Density", labelpad=6)
    ax_right.xaxis.set_label_position("bottom")
    ax_right.tick_params(axis="x", labelsize=12, rotation=45)
    ax_right.grid(True, alpha=0.35)
    ax_right.spines["left"].set_visible(False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.xaxis.get_major_locator().set_params(nbins=3)

    fig.align_labels()
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "fig2_energy_vs_thit.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Shared KDE helper
# -----------------------------------------------------------------------
def _kde_plot(ax, data_list, styles, x_label, title, mean_key, stats_list):
    """Overlay KDE curves for each controller on ax."""
    all_vals = np.concatenate([d for d in data_list if d.size > 0])
    if all_vals.size == 0:
        return
    xlo = max(0.0, all_vals.min() - 0.05 * np.ptp(all_vals))
    xhi = all_vals.max() + 0.05 * np.ptp(all_vals)
    xs  = np.linspace(xlo, xhi, 400)

    for style, data, stats in zip(styles, data_list, stats_list):
        if data.size < 2:
            continue
        color = style["color"]
        kde   = gaussian_kde(data, bw_method="scott")
        ys    = kde(xs)
        ax.plot(xs, ys, color=color, ls=style["ls"], lw=2.0,
                label=f"{style['label']}  ($n={data.size}$)")
        ax.fill_between(xs, ys, alpha=0.3, color=color)
        mu = stats[mean_key]
        if not np.isnan(mu):
            ax.axvline(mu, color=color, lw=2.0, ls=":", alpha=1.0,
                       label=fr"$\mu={mu:.3f}$")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.35)


# -----------------------------------------------------------------------
# Figure 3 — Energy distribution
# -----------------------------------------------------------------------
def plot_energy_distribution(entries, save_dir=None):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _kde_plot(
        ax,
        data_list  = [s["energies_success"] for _, _, s in entries],
        styles     = CTRL_STYLES,
        stats_list = [s for _, _, s in entries],
        x_label    = r"$\int_0^{T_{\rm hit}} u_{\rm raw}^2\, dt$",
        title      = "Control Energy Distribution (Successful Trajectories)",
        mean_key   = "energy_mean",
    )
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "fig3_energy_distribution.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Figure 4 — Hitting-time distribution
# -----------------------------------------------------------------------
def plot_thit_distribution(entries, save_dir=None):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _kde_plot(
        ax,
        data_list  = [s["t_hits_success"] for _, _, s in entries],
        styles     = CTRL_STYLES,
        stats_list = [s for _, _, s in entries],
        x_label    = r"$T_{\rm hit}$ (s)",
        title      = "First-Hitting Time Distribution (Successful Trajectories)",
        mean_key   = "t_hit_mean",
    )
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "fig4_thit_distribution.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Figure 5 — u_raw(t) vs time (single plot, all controllers)
# -----------------------------------------------------------------------
def plot_u_raw_trajectories(entries, dt, save_dir=None, styles=None):
    """
    All controllers in one plot.
    Each trajectory is clipped to its first-hitting time T_hit.
    Individual traces are shown in light colour; the per-controller
    mean (over stored paths, aligned on [0, T_hit]) is drawn in bold.
    """
    if styles is None:
        styles = CTRL_STYLES
    fig, ax = plt.subplots(figsize=(8, 5))

    # Pre-compute statistics for all controllers
    ctrl_data = []
    for style, (label, res, __) in zip(styles, entries):
        uraw_all = res["uraw_paths"]
        if not uraw_all:
            ctrl_data.append(None)
            continue
        max_len = max(u.size for u in uraw_all)
        mat     = np.full((len(uraw_all), max_len), np.nan)
        for i, u in enumerate(uraw_all):
            mat[i, :u.size] = u
        ctrl_data.append({
            "t_grid": np.arange(max_len) * dt,
            "mean":   np.nanmean(mat, axis=0),
            "min":    np.nanmin(mat,  axis=0),
            "max":    np.nanmax(mat,  axis=0),
        })

    # Pass 1 — bands (drawn first, underneath everything)
    for zorder, (style, (label, _, __), data) in enumerate(
            zip(styles, entries, ctrl_data), start=1):
        if data is None:
            continue
        ax.fill_between(data["t_grid"], data["min"], data["max"],
                        color=style["color"], alpha=0.3, zorder=zorder)

    # Pass 2 — mean lines (drawn on top of all bands)
    # Cert. (energy) is last in styles → highest zorder automatically
    n = len(styles)
    for zorder, (style, (label, _, __), data) in enumerate(
            zip(styles, entries, ctrl_data), start=n + 1):
        if data is None:
            continue
        ax.plot(data["t_grid"], data["mean"],
                color=style["color"], ls=style["ls"], lw=2.0,
                label=label, zorder=zorder)

    ax.axhline(0, color="grey", lw=2.0, ls="--", alpha=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$u_{\rm raw}(t)$")
    ax.legend(framealpha=0.9, fontsize=18)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "fig5_u_raw_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


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
    print(f"  p_fail         = {stats['p_fail']:.4f}  ({stats['n_fail']} / {stats['n_mc']})")
    print(f"  p_timeout      = {stats['p_timeout']:.4f}  ({stats['n_timeout']} / {stats['n_mc']})")
    print(f"  -- energy ∫u_raw² dt (successful) --")
    print(f"  mean           = {stats['energy_mean']:.6f}")
    print(f"  median         = {stats['energy_median']:.6f}")
    print(f"  std            = {stats['energy_std']:.6f}")
    print(f"  -- first-hitting time (successful) --")
    print(f"  mean T_hit     = {stats['t_hit_mean']:.4f} s")
    print(f"  median T_hit   = {stats['t_hit_median']:.4f} s")
    print(f"  std T_hit      = {stats['t_hit_std']:.4f} s")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    bundle_specs = [
        (
            ROOT / "examples" / "disturb_energy" / "inv_pend_syn_disturb_baseline"
                 / "outputs" / "eval_bundle.pth",
        ),
        (
            ROOT / "examples" / "disturb_temporal" / "inv_pend_syn_disturb"
                 / "outputs" / "time_runs" / "eval_bundle.pth",
        ),
        (
            OUTPUT_DIR / "beta_runs" / "eval_bundle.pth",
        ),
        (
            ROOT / "examples" / "disturb_energy" / "inv_pend_syn_disturb_rl"
                 / "outputs" / "rl_controller.pth",
        ),
    ]

    for (path,), style in zip(bundle_specs, CTRL_STYLES):
        if not path.exists():
            raise FileNotFoundError(
                f"Bundle not found for '{style['label']}':\n  {path}"
            )

    # --- per-controller budgets ---
    # energy_budget : fail if ∫u_raw² dt > threshold  (energy controller)
    # time_budget   : fail if T_hit > threshold         (time controller)
    BUDGETS = [
        {},                                       # Cert.: no budget
        {"time_budget": 4.5},                     # Cert. (time): T_hit must be <= 4.5 s
        {"energy_budget": 0.6},                   # Cert. (energy): energy must be <= 0.6
        {},                                       # RL: no budget
    ]
    IS_PRETRAINED = [False, False, False, True]   # only D uses raw state_dict

    # --- MC config ---
    N_MC    = 100
    T_MAX   = 8.0
    DT      = 0.005
    D       = np.array([1.0, 1.0], dtype=float)
    MODE    = "uniform"
    SEED    = 42
    N_PATHS = 10   # trajectories to store for phase-plane plot

    # entries: list of (label, res, stats)  — same order as CTRL_STYLES / BUDGETS
    entries = []
    for (path,), style, bgt, is_pt in zip(bundle_specs, CTRL_STYLES, BUDGETS, IS_PRETRAINED):
        label = style["label"]
        print(f"\nLoading '{label}'\n  {path}")
        ctrl  = load_control_net(path, pretrained_state_dict=is_pt)
        print(f"  Running MC (n={N_MC}, budgets={bgt}) ...")
        res   = rollout_mc(
            ctrl, n_mc=N_MC, T_max=T_MAX, dt=DT,
            d=D, disturbance_mode=MODE, seed=SEED, n_paths=N_PATHS,
            energy_budget=bgt.get("energy_budget"),
            time_budget=bgt.get("time_budget"),
        )
        stats = compute_stats(res)
        print_summary(label, stats)
        entries.append((label, res, stats))

    # --- viz-only extended paths for Fig 1 (T_hit + 1 s), stats unaffected ---
    viz_paths_per_ctrl = []
    for (path,), style, bgt, is_pt in zip(bundle_specs, CTRL_STYLES, BUDGETS, IS_PRETRAINED):
        ctrl = load_control_net(path, pretrained_state_dict=is_pt)
        viz_paths_per_ctrl.append(rollout_viz_paths(
            ctrl, n_paths=N_PATHS, T_max=T_MAX, t_extra=1.0,
            dt=DT, d=D, disturbance_mode=MODE, seed=SEED,
        ))

    SAVE_DIR = str(OUTPUT_DIR)
    plot_phase_trajectories(entries, viz_paths_per_ctrl, save_dir=SAVE_DIR)
    plot_energy_vs_thit(entries, BUDGETS, save_dir=SAVE_DIR, styles=CTRL_STYLES)
    # plot_energy_distribution(entries,             save_dir=SAVE_DIR)
    # plot_thit_distribution(entries,               save_dir=SAVE_DIR)
    plot_u_raw_trajectories(entries, DT,           save_dir=SAVE_DIR, styles=CTRL_STYLES)


if __name__ == "__main__":
    main()
