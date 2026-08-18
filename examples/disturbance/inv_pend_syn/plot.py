"""
MC Validation Comparison — Three Controllers
=============================================

Compares:
  A) examples/disturbance/inv_pend_syn/outputs/eval_bundle.pth
       Cert. (closed-form) (parametric L-uncertainty synthesis)
  B) examples/disturbance/inv_pend_syn_global/outputs/eval_bundle.pth
       Cert. (partition-based) (constant-but-unknown parameter synthesis)
  C) examples/disturbance/inv_pend_syn_rl/outputs/rl_controller.pth
       RL (finite-time + energy reward, same architecture)

Simulation follows the uniform setting in test_invpend_dynamics.py:
  - Parametric set-valued drift: (g, L, b, m) sampled once per trajectory
  - L ~ Uniform(0.40, 0.60); g, b, m fixed at nominal
  - Euler-Maruyama integration, dt = 0.005

Metrics (successful trajectories only unless stated):
  1. Reach-avoid probability  p_success / p_fail / p_timeout
  2. Control energy  E = integral_0^{T_hit}  u_raw(t)^2  dt
  3. First-hitting time  T_hit

Output (PDF, saved to outputs/):
  fig1_phase_trajectories.pdf
  fig2_energy_vs_thit.pdf
  fig5_u_raw_trajectories.pdf

Usage (from this directory):
    python plot.py
"""

import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.stats import gaussian_kde
import seaborn as sns

# -----------------------------------------------------------------------
# Publication-quality style
# -----------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":     "custom",
    "mathtext.rm":          "Times New Roman",
    "mathtext.it":          "Times New Roman:italic",
    "mathtext.bf":          "Times New Roman:bold",
    "font.size":            16,
    "axes.titlesize":       16,
    "axes.labelsize":       16,
    "legend.fontsize":      16,
    "xtick.labelsize":      14,
    "ytick.labelsize":      14,
    "axes.linewidth":       1.2,
    "grid.linewidth":       0.7,
    "lines.linewidth":      2.0,
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
# Physical constants  (must match test_invpend_dynamics.py)
# -----------------------------------------------------------------------
g_grav = 9.81
L_nom  = 0.5
m_nom  = 0.15
b_nom  = 0.1
sigma  = 0.2
M      = 6.0
pi     = np.pi

TORQUE_SCALE = M / (m_nom * L_nom**2)   # ≈ 160

# -----------------------------------------------------------------------
# Parametric uncertainty ranges  (must match test_invpend_dynamics.py main())
# -----------------------------------------------------------------------
PARAM_RANGES = {
    "g": (9.81, 9.81),
    "L": (0.40, 0.60),
    "b": (0.10, 0.10),
    "m": (0.15, 0.15),
}

# -----------------------------------------------------------------------
# Per-controller style
# -----------------------------------------------------------------------
_pal = sns.color_palette("Set2", 4)
CTRL_STYLES = [
    {"color": _pal[0], "ls": "-",  "lw": 2.0, "label": "Cert. (closed-form)"},    # solid
    {"color": _pal[1], "ls": "--", "lw": 2.0, "label": "Cert. (partition-based)"},# dashed
    {"color": _pal[2], "ls": ":",  "lw": 2.0, "label": "RL"},                     # dotted
    {"color": _pal[3], "ls": "-.", "lw": 2.0, "label": "REINFORCE"},              # dash-dot
]

# -----------------------------------------------------------------------
# Spec regions  (from test_invpend_dynamics.py)
# -----------------------------------------------------------------------
X_init_bounds = {"x1_min":  3*pi/4,  "x1_max":  5*pi/4,  "x2_min": -1.0,  "x2_max":  1.0}
X_goal_bounds = {"x1_min": -0.4*pi,  "x1_max":  0.4*pi,  "x2_min": -4.0,  "x2_max":  4.0}
X_unsafe_1    = {"x1_min": -2*pi,    "x1_max": -3*pi/2,  "x2_min": -20.0, "x2_max": -10.0}
X_unsafe_2    = {"x1_min":  3*pi/2,  "x1_max":  2*pi,    "x2_min":  10.0, "x2_max":  20.0}
X_bounds      = {"x1_min": -2*pi,    "x1_max":  2*pi,    "x2_min": -20.0, "x2_max":  20.0}


def in_box(x1, x2, box):
    return (box["x1_min"] <= x1 <= box["x1_max"]) and (box["x2_min"] <= x2 <= box["x2_max"])


# -----------------------------------------------------------------------
# Dynamics  (parametric, matching test_invpend_dynamics.py)
# -----------------------------------------------------------------------
def f_parametric(x, u, g_val, L_val, b_val, m_val):
    """Parametric drift: physical parameters sampled per trajectory."""
    x1, x2 = x
    u1, u2 = u
    dx1 = x2
    dx2 = (g_val / L_val) * np.sin(x1) + (-b_val * x2) / (m_val * L_val**2) + u2
    return np.array([dx1, dx2], dtype=float)


def g_diff(x):
    """Diffusion vector."""
    return np.array([0.0, sigma], dtype=float)


def sample_param_set(rng, param_ranges):
    """Sample (g, L, b, m) uniformly from param_ranges, once per trajectory."""
    return np.array([
        rng.uniform(*param_ranges["g"]),
        rng.uniform(*param_ranges["L"]),
        rng.uniform(*param_ranges["b"]),
        rng.uniform(*param_ranges["m"]),
    ], dtype=float)


# -----------------------------------------------------------------------
# Parameter vertices  (corners of the hyper-rectangular set Λ)
# -----------------------------------------------------------------------
def get_param_vertices(param_ranges: dict = None) -> list:
    """
    Return all vertices of the parameter hyper-rectangle Λ.

    Each vertex is a dict {param: value} where every parameter is fixed at
    either its minimum or maximum.  Degenerate dimensions (min == max) produce
    only one value, so the total number of vertices is 2^d where d is the
    number of non-degenerate parameters.

    For the current PARAM_RANGES (only L varies), this yields 2 vertices:
        [{"g": 9.81, "L": 0.40, "b": 0.10, "m": 0.15},
         {"g": 9.81, "L": 0.60, "b": 0.10, "m": 0.15}]
    """
    import itertools
    if param_ranges is None:
        param_ranges = PARAM_RANGES
    keys   = list(param_ranges.keys())
    levels = [sorted(set(param_ranges[k])) for k in keys]   # {min} or {min, max}
    return [{k: v for k, v in zip(keys, combo)}
            for combo in itertools.product(*levels)]


# -----------------------------------------------------------------------
# Controller helpers
# -----------------------------------------------------------------------
def load_control_net(bundle_path, device="cpu", pretrained_state_dict=False):
    """
    Load a WrapperConterlNN from either:
      - an eval_bundle.pth  (pretrained_state_dict=False): dict with key "control_state_dict"
      - a controller_pretrained.pth  (pretrained_state_dict=True): raw state_dict
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
# MC rollout  (parametric set-valued drift, matching test_invpend_dynamics.py)
# -----------------------------------------------------------------------
def rollout_mc(
    controller,
    n_mc:          int   = 500,
    T_max:         float = 8.0,
    dt:            float = 0.005,
    param_ranges:  dict  = None,
    seed:          int   = 42,
    n_paths:       int   = 20,
    fixed_params:  dict  = None,
    fixed_x0:      tuple = None,
):
    """
    Euler-Maruyama rollout with parametric set-valued drift.

    Nominal mode (fixed_params=None):
        Parameters sampled uniformly once per trajectory from param_ranges.
        Initial conditions and parameters are pre-generated so every
        controller sees the same (x0_i, params_i) for trajectory i.
        Per-step noise uses a per-trajectory RNG seeded by (seed, i).

    Vertex mode (fixed_params=dict):
        All trajectories use the same fixed parameter values.
        Used for adversarial vertex evaluation: run at each corner of Λ,
        then take the worst-case success rate across vertices.

    Returns dict:
        outcomes  : list[str]            'success' | 'fail' | 'timeout'
        energies  : list[float]          integral u_raw^2 dt  up to T_hit
        hit_times : list[float]          T_hit (T_max for timeout)
        paths     : list[ndarray(K,2)]   first n_paths trajectories
        uraw_paths: list[ndarray(K,)]    u_raw(t) for stored paths
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    root_rng = np.random.default_rng(seed)
    if fixed_x0 is not None:
        x1_init = np.full(n_mc, fixed_x0[0])
        x2_init = np.full(n_mc, fixed_x0[1])
    else:
        x1_init = root_rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"], size=n_mc)
        x2_init = root_rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"], size=n_mc)

    if fixed_params is not None:
        gf, Lf, bf, mf = (fixed_params["g"], fixed_params["L"],
                          fixed_params["b"], fixed_params["m"])
        params_list = [(gf, Lf, bf, mf)] * n_mc
    else:
        params_list = [sample_param_set(root_rng, param_ranges) for _ in range(n_mc)]

    N_steps = int(T_max / dt)

    outcomes   = []
    energies   = []
    hit_times  = []
    paths_out  = []
    uraw_out   = []

    for i in range(n_mc):
        x1, x2 = float(x1_init[i]), float(x2_init[i])
        gk, Lk, bk, mk = params_list[i]   # Lk used only in nominal mode

        # Per-trajectory noise RNG — same seed for every controller on trajectory i
        traj_rng = np.random.default_rng([seed, i])

        energy_acc = 0.0
        outcome    = "timeout"
        T_hit      = T_max
        store_path = i < n_paths
        if store_path:
            path = [[x1, x2]]
            uraw = []

        for k in range(N_steps):
            if in_box(x1, x2, X_goal_bounds):
                T_hit   = k * dt
                outcome = "success"
                break
            if in_box(x1, x2, X_unsafe_1) or in_box(x1, x2, X_unsafe_2):
                outcome = "fail"
                T_hit   = k * dt
                break

            u_applied, u_raw = get_u_and_raw(x1, x2, controller)
            energy_acc += u_raw**2 * dt

            if store_path:
                uraw.append(u_raw)

            drift  = f_parametric(np.array([x1, x2]), u_applied, gk, Lk, bk, mk)
            dW     = np.sqrt(dt) * traj_rng.standard_normal()
            x_next = np.array([x1, x2]) + drift * dt + g_diff(np.array([x1, x2])) * dW
            x1, x2 = float(x_next[0]), float(x_next[1])

            if x1 > 2 * pi:    x1 -= 4 * pi
            elif x1 < -2 * pi: x1 += 4 * pi

            if store_path:
                path.append([x1, x2])

        outcomes.append(outcome)
        energies.append(energy_acc)
        hit_times.append(T_hit)
        if store_path:
            paths_out.append(np.array(path, dtype=float))
            uraw_out.append(np.array(uraw,  dtype=float))

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
        "n_mc":             n,
        "p_success":        float(success_mask.sum()) / n,
        "p_fail":           float(fail_mask.sum())    / n,
        "p_timeout":        float(timeout_mask.sum()) / n,
        "n_success":        int(success_mask.sum()),
        "n_fail":           int(fail_mask.sum()),
        "n_timeout":        int(timeout_mask.sum()),
        "energy_mean":      _s(np.mean,   e_suc),
        "energy_median":    _s(np.median, e_suc),
        "energy_std":       _s(np.std,    e_suc),
        "t_hit_mean":       _s(np.mean,   t_suc),
        "t_hit_median":     _s(np.median, t_suc),
        "t_hit_std":        _s(np.std,    t_suc),
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

    ax.add_patch(_rect(X_bounds,      **kw_border))
    ax.add_patch(_rect(X_init_bounds, **kw_init))
    ax.add_patch(_rect(X_goal_bounds, **kw_goal))
    ax.add_patch(_rect(X_unsafe_1,    **kw_unsafe))
    ax.add_patch(_rect(X_unsafe_2,    **kw_unsafe))

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
# Viz-only rollout: run to T_hit + t_extra  (no stats)
# -----------------------------------------------------------------------
def rollout_viz_paths(
    controller,
    n_paths:       int   = 10,
    T_max:         float = 8.0,
    t_extra:       float = 1.0,
    dt:            float = 0.005,
    param_ranges:  dict  = None,
    seed:          int   = 42,
    fixed_params:  dict  = None,
    fixed_x0:      tuple = None,
):
    """Returns a list of (K, 2) phase-state arrays extended to T_hit + t_extra.
    Uses pre-generated initial conditions and per-trajectory noise RNGs so
    all controllers start from identical (x0, noise) per path.
    If fixed_params is provided, all paths use those fixed parameter values.
    If fixed_x0 is provided, all paths start from that point.
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    root_rng = np.random.default_rng(seed)
    if fixed_x0 is not None:
        x1_init = np.full(n_paths, fixed_x0[0])
        x2_init = np.full(n_paths, fixed_x0[1])
    else:
        x1_init = root_rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"], size=n_paths)
        x2_init = root_rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"], size=n_paths)

    if fixed_params is not None:
        gf, Lf, bf, mf = (fixed_params["g"], fixed_params["L"],
                          fixed_params["b"], fixed_params["m"])
        params_list = [(gf, Lf, bf, mf)] * n_paths
    else:
        params_list = [sample_param_set(root_rng, param_ranges) for _ in range(n_paths)]

    N_steps   = int((T_max + t_extra) / dt)
    paths_out = []

    for i in range(n_paths):
        x1, x2 = float(x1_init[i]), float(x2_init[i])
        gk, Lk, bk, mk = params_list[i]

        # Per-trajectory noise RNG — same for every controller on path i
        traj_rng = np.random.default_rng([seed, i])

        path  = [[x1, x2]]
        T_hit = None

        for k in range(N_steps):
            if T_hit is None and in_box(x1, x2, X_goal_bounds):
                T_hit = k * dt
            if T_hit is None and (in_box(x1, x2, X_unsafe_1) or in_box(x1, x2, X_unsafe_2)):
                T_hit = k * dt
            if T_hit is not None and (k * dt) >= T_hit + t_extra:
                break

            u_applied, _ = get_u_and_raw(x1, x2, controller)
            drift  = f_parametric(np.array([x1, x2]), u_applied, gk, Lk, bk, mk)
            dW     = np.sqrt(dt) * traj_rng.standard_normal()
            x_next = np.array([x1, x2]) + drift * dt + g_diff(np.array([x1, x2])) * dW
            x1, x2 = float(x_next[0]), float(x_next[1])

            if x1 > 2 * pi:    x1 -= 4 * pi
            elif x1 < -2 * pi: x1 += 4 * pi

            path.append([x1, x2])

        paths_out.append(np.array(path, dtype=float))

    return paths_out


# -----------------------------------------------------------------------
# Figure 0 — State time-series  θ(t) and θ̇(t)
# -----------------------------------------------------------------------
def plot_state_trajectories(entries, viz_paths_per_ctrl, dt, save_dir=None):
    """
    Two-panel time-domain plot: angle θ(t) on top, angular velocity θ̇(t) on bottom.
    Each controller is shown with its own colour/linestyle; individual trajectories
    are plotted as semi-transparent lines so their spread is visible.
    Goal and init region bounds are shown as horizontal shaded bands.
    """
    fig, (ax_th, ax_om) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for style, (label, _, __), viz_paths in zip(CTRL_STYLES, entries, viz_paths_per_ctrl):
        color = style["color"]
        ls    = style["ls"]
        lw    = style["lw"]
        first = True
        for path in viz_paths:
            K  = path.shape[0]
            t  = np.arange(K) * dt
            lbl = label if first else None
            first = False
            ax_th.plot(t, path[:, 0], color=color, ls=ls, lw=lw, alpha=1.0, label=lbl)
            ax_om.plot(t, path[:, 1], color=color, ls=ls, lw=lw, alpha=1.0)

    # Goal region: shaded band on each axis
    ax_th.axhspan(X_goal_bounds["x1_min"], X_goal_bounds["x1_max"],
                  color="green", alpha=0.12, label="Goal", zorder=0)
    ax_om.axhspan(X_goal_bounds["x2_min"], X_goal_bounds["x2_max"],
                  color="green", alpha=0.12, zorder=0)

    # Init region: shaded band
    ax_th.axhspan(X_init_bounds["x1_min"], X_init_bounds["x1_max"],
                  color="#4c72b0", alpha=0.10, label="Init", zorder=0)
    ax_om.axhspan(X_init_bounds["x2_min"], X_init_bounds["x2_max"],
                  color="#4c72b0", alpha=0.10, zorder=0)

    ax_th.set_ylabel(r"$\theta$  (rad)", fontsize=17)
    ax_om.set_ylabel(r"$\dot\theta$  (rad/s)", fontsize=17)
    ax_om.set_xlabel("Time  (s)", fontsize=17)

    ax_th.legend(loc="upper right", framealpha=0.9, fontsize=13)
    ax_th.grid(True, alpha=0.35)
    ax_om.grid(True, alpha=0.35)

    fig.align_ylabels([ax_th, ax_om])
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "fig0_state_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Figure 1 — Phase trajectories
# -----------------------------------------------------------------------
def plot_phase_trajectories(entries, viz_paths_per_ctrl, save_dir=None):
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
        for path in viz_paths:
            ax.plot(path[0, 0], path[0, 1], "o",
                    color=color, markersize=4, alpha=1.0, zorder=4)

    ax.legend(loc="upper left", framealpha=0.9, fontsize=18)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(r"$\theta$, rad")
    ax.set_ylabel(r"$\dot\theta$, rad/s")
    fig.tight_layout()

    if save_dir is not None:
        p = Path(save_dir) / "fig1_phase_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        print(f"[saved] {p}")
    plt.show()


# -----------------------------------------------------------------------
# Figure 2 — Energy vs Hitting-time scatter with marginal KDEs
# -----------------------------------------------------------------------
def plot_energy_vs_thit(entries, save_dir=None, styles=None):
    if styles is None:
        styles = CTRL_STYLES
    fig = plt.figure(figsize=(8, 5))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[3.5, 1.2], height_ratios=[1.2, 3.5],
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

        ax_main.scatter(t_hit, e_hit, color=color, alpha=1.0, s=40, label=label, zorder=3)

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
    ax_main.set_ylabel("Control energy")
    ax_main.legend(framealpha=0.9, fontsize=14)
    ax_main.grid(True, alpha=0.35)

    ax_top.set_ylabel("Density", labelpad=6)
    ax_top.yaxis.set_label_position("left")
    ax_top.tick_params(axis="y", labelsize=12)
    ax_top.grid(True, alpha=0.35)
    ax_top.spines["bottom"].set_visible(False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.yaxis.get_major_locator().set_params(nbins=4)

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
# Figure 5 — u_raw(t) vs time
# -----------------------------------------------------------------------
def plot_u_raw_trajectories(entries, dt, save_dir=None, styles=None):
    if styles is None:
        styles = CTRL_STYLES
    fig, ax = plt.subplots(figsize=(8, 5))

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

    for zorder, (style, (label, _, __), data) in enumerate(
            zip(styles, entries, ctrl_data), start=1):
        if data is None:
            continue
        ax.fill_between(data["t_grid"], data["min"], data["max"],
                        color=style["color"], alpha=0.3, zorder=zorder)

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
def _print_vertex_table(vertex_results: dict):
    """
    Print worst-case reach-avoid probability table across all vertices.

    vertex_results : {label: [(vertex_dict, stats), ...]}
    """
    w = 70
    print(f"\n{'='*w}")
    print("  Adversarial Vertex Evaluation  (worst-case over corners of Λ)")
    print(f"{'='*w}")
    print(f"  {'Controller':<28}  {'Worst vertex (L)':<18}  {'p_success':>10}")
    print(f"  {'-'*28}  {'-'*18}  {'-'*10}")
    for label, vlist in vertex_results.items():
        worst = min(vlist, key=lambda x: x[1]["p_success"])
        v, st = worst
        v_str = "  ".join(f"{k}={v:.2f}" for k, v in v.items() if PARAM_RANGES[k][0] != PARAM_RANGES[k][1])
        print(f"  {label:<28}  {v_str:<18}  {st['p_success']:>10.4f}  "
              f"({st['n_success']}/{st['n_mc']})")
    print(f"{'='*w}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MC comparison of inv-pend controllers under parametric uncertainty."
    )
    parser.add_argument(
        "--use-adversarial", type=int, default=0,
        help=(
            "0 (default): parameters sampled uniformly from Λ per trajectory. "
            "1: evaluate at every vertex of Λ (corners of the hyper-rectangle) "
            "and report the worst-case reach-avoid probability across all vertices."
        ),
    )
    parser.add_argument(
        "--naive_rl", type=int, default=0,
        help=(
            "0 (default): compare the three standard controllers. "
            "1: also load naive_rl_controller.pth (REINFORCE) "
            "from inv_pend_syn_rl/outputs/ and add it to the comparison."
        ),
    )
    args = parser.parse_args()
    USE_ADVERSARIAL = bool(args.use_adversarial)
    NAIVE_RL        = bool(args.naive_rl)

    bundle_specs = [
        (OUTPUT_DIR / "eval_bundle.pth",),
        (ROOT / "examples" / "disturbance" / "inv_pend_syn_global"
              / "outputs" / "eval_bundle.pth",),

        (OUTPUT_DIR / "controller_pretrained.pth",),
        # (ROOT / "examples" / "disturbance" / "inv_pend_syn_rl"
        #       / "outputs" / "rl_controller.pth",),
        
    ]
    IS_PRETRAINED = [False, False, True]

    if NAIVE_RL:
        bundle_specs.append((
            ROOT / "examples" / "disturbance" / "inv_pend_syn_rl"
                 / "outputs" / "naive_rl_controller.pth",
        ))
        IS_PRETRAINED.append(True)

    for (path,), style, is_pt in zip(bundle_specs, CTRL_STYLES, IS_PRETRAINED):
        if not path.exists():
            raise FileNotFoundError(
                f"Bundle not found for '{style['label']}':\n  {path}"
            )

    # --- MC config ---
    N_MC    = 100
    T_MAX   = 6.0
    DT      = 0.005
    SEED    = 42
    N_PATHS = 10

    # -----------------------------------------------------------------------
    # Nominal evaluation  (uniform parameter sampling)
    # -----------------------------------------------------------------------
    print(f"\nNominal MC: parameters sampled uniformly from Λ  (n={N_MC})")
    entries = []
    for (path,), style, is_pt in zip(bundle_specs, CTRL_STYLES, IS_PRETRAINED):
        label = style["label"]
        print(f"  [{label}]")
        ctrl  = load_control_net(path, pretrained_state_dict=is_pt)
        res   = rollout_mc(ctrl, n_mc=N_MC, T_max=T_MAX, dt=DT, seed=SEED, n_paths=N_PATHS)
        stats = compute_stats(res)
        print_summary(label, stats)
        entries.append((label, res, stats))

    viz_paths_per_ctrl = []
    for (path,), style, is_pt in zip(bundle_specs, CTRL_STYLES, IS_PRETRAINED):
        ctrl = load_control_net(path, pretrained_state_dict=is_pt)
        viz_paths_per_ctrl.append(
            rollout_viz_paths(ctrl, n_paths=N_PATHS, T_max=T_MAX, t_extra=1.0, dt=DT, seed=SEED)
        )

    SAVE_DIR = str(OUTPUT_DIR)
    plot_state_trajectories(entries, viz_paths_per_ctrl, DT, save_dir=SAVE_DIR)
    plot_phase_trajectories(entries, viz_paths_per_ctrl, save_dir=SAVE_DIR)
    plot_energy_vs_thit(entries,         save_dir=SAVE_DIR)
    plot_u_raw_trajectories(entries, DT, save_dir=SAVE_DIR)

    # -----------------------------------------------------------------------
    # Adversarial vertex evaluation
    # -----------------------------------------------------------------------
    if USE_ADVERSARIAL:
        vertices = get_param_vertices(PARAM_RANGES)
        print(f"\nAdversarial vertex evaluation: {len(vertices)} vertices  (n={N_MC} each)")
        for v in vertices:
            v_str = "  ".join(f"{k}={val:.2f}"
                              for k, val in v.items()
                              if PARAM_RANGES[k][0] != PARAM_RANGES[k][1])
            print(f"  vertex: {v_str}")

        vertex_results = {}   # label → [(vertex_dict, stats), ...]
        for (path,), style, is_pt in zip(bundle_specs, CTRL_STYLES, IS_PRETRAINED):
            label = style["label"]
            print(f"\n  [{label}]")
            ctrl  = load_control_net(path, pretrained_state_dict=is_pt)
            vlist = []
            for v in vertices:
                v_str = "  ".join(f"{k}={val:.2f}"
                                  for k, val in v.items()
                                  if PARAM_RANGES[k][0] != PARAM_RANGES[k][1])
                res   = rollout_mc(ctrl, n_mc=N_MC, T_max=T_MAX, dt=DT,
                                   seed=SEED, n_paths=0, fixed_params=v)
                stats = compute_stats(res)
                print(f"    vertex ({v_str}):  p_success = {stats['p_success']:.4f}")
                vlist.append((v, stats))
            vertex_results[label] = vlist

        _print_vertex_table(vertex_results)


if __name__ == "__main__":
    main()
