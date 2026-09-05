"""
MC Validation Comparison
============================================

Compares every controller listed in `bundle_specs` (inside `main()`) against
six disturbance regimes, applied independently to each controller (no
shared "record on one controller, replay on the others" scheme — every
controller faces its own instance of each regime, using the same seeded RNG
so trajectories are directly comparable):

  1. fast            adversarial disturbance  (select_lambda_adv_fast)
  2. lookahead       adversarial disturbance  (select_lambda_adv_lookahead,
                     one-step worst-case endpoint of the disturbance box)
  3. nearest_unsafe  adversarial disturbance  (select_lambda_adv_nearest_unsafe,
                     one-step endpoint that minimizes distance to the
                     nearest unsafe region)
  4. velocity        adversarial disturbance  (select_lambda_adv_velocity,
                     one-step endpoint that maximizes |x2|, targeting the
                     goal region's angular-rate window directly)
  5. uniform         disturbance, w = [0, w2], w2 ~ Uniform[-drift_mag, drift_mag] i.i.d. each step
  6. zero            no uncertain disturbance, w = [0, 0]

Disturbance is physical: it only perturbs x2 (angular acceleration / torque),
never x1 (theta has no direct force input — dtheta/dt = omega is kinematic).

Metrics (successful trajectories only unless stated):
  1. Reach-avoid probability  p_success / p_fail / p_timeout
  2. Control energy  E = integral_0^{T_hit}  u_raw(t)^2  dt
  3. First-hitting time  T_hit

The final success-rate table reports three columns -- "uniform", "zero", and
"adversarial" -- where "adversarial" is the worst case over fast/lookahead/
nearest_unsafe/velocity (a real adversary picks whichever attack is more
damaging): per seed, take min(p_success under "fast", "lookahead",
"nearest_unsafe", "velocity"), then average that per-seed minimum across
seeds. Figures and --verbose per-seed detail still break out each
adversarial submode separately.

Multi-seed controllers
-----------------------
Each `bundle_specs` entry names a controller directory. If it contains
`seed<N>/` subfolders (as trained by e.g. `neural_certified_nominal_drift/main.py`),
every seed's checkpoint is loaded and evaluated independently; otherwise the
directory's own checkpoint is used as a single implicit run. Per-seed MC
results are pooled per controller (for plotting) and the per-seed success
rates are averaged (for the summary table), so a controller trained with
multiple seeds is reported by its mean success rate across seeds.

Caption: Mean Monte Carlo success probability per controller across seeds

Output:
  run_mc_results/mc_cache.pth
      -- every result/stat/viz-path computed by this run, plus the aggregated
         success-rate table, so `postprocess_mc.py` can re-render all plots
         and the summary table without re-running the (slow) Monte Carlo rollouts
  run_mc_results/<mode>/fig1_phase_trajectories.pdf
  run_mc_results/<mode>/fig2_energy_vs_thit.pdf
  run_mc_results/<mode>/fig5_u_raw_trajectories.pdf
  run_mc_results/fig6_success_rate_summary.pdf
  run_mc_results/fig7_failure_trajectories.pdf
  run_mc_results/fig8_energy_distribution.pdf  (energy comparisons)

With --baseline-controller-dir, only figures 6, 7, and 8 are generated.
Use --figures to select a different set. Figure 8 compares energy boxplots in four panels:
overall, zero disturbance, uniform disturbance, and all adversarial modes.
Each panel includes successes, failures, and timeouts.

Usage (from this directory):
    python run_mc.py                # run the MC, save mc_cache.pth + plots, print summary table
    python run_mc.py --verbose      # also print per-controller/per-seed/per-mode detail
    python run_mc.py --energy-controller-dir /path/to/energy/run  # include Cert. (energy)
    python run_mc.py \
        --baseline-controller-dir neural_certified/seed4 \
        --energy-controller-dir neural_certified/seed4/energy/20260905_091107_645787 \
        --n-mc 100 \
        --output-dir run_mc_energy_seed4 \
        --verbose # compare the cert with/without energy constraints.
    python postprocess_mc.py        # re-render plots/table from mc_cache.pth only
"""

import argparse
import re
import sys
import json
import time
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
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
ROOT          = Path(__file__).resolve().parents[2]
HERE          = Path(__file__).resolve().parent
OUTPUT_DIR    = HERE / "run_mc_results"
MC_CACHE_PATH = OUTPUT_DIR / "mc_cache.pth"
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN
from src.inv_pend_adversarial_config import (
    load_controller_hidden_dim, load_dynamics_params, load_region_boxes,
)
from src.save_load_utils import load_eval_bundle

with (HERE / "config.json").open("r", encoding="utf-8") as f:
    EXAMPLE_CONFIG = json.load(f)

# -----------------------------------------------------------------------
# Physical constants  (single source of truth: config.json "dynamics")
# -----------------------------------------------------------------------
DYNAMICS = load_dynamics_params(EXAMPLE_CONFIG)
g_grav = DYNAMICS["g"]
L      = DYNAMICS["L"]
m      = DYNAMICS["m"]
b      = DYNAMICS["b"]
M      = DYNAMICS["M_torque"]
sigma  = DYNAMICS["sigma"]
pi     = np.pi

TORQUE_SCALE = M / (m * L**2)
ADV_MAG = float(EXAMPLE_CONFIG["drift_mag"])
REGION_BOXES = load_region_boxes(EXAMPLE_CONFIG)
CONTROLLER_HIDDEN_DIM = load_controller_hidden_dim(EXAMPLE_CONFIG)

EVAL_MODES = ("fast", "lookahead", "nearest_unsafe", "velocity", "uniform", "zero")
ADVERSARIAL_MODES = ("fast", "lookahead", "nearest_unsafe", "velocity")

# Success-rate table columns: "fast", "lookahead", "nearest_unsafe", and
# "velocity" (all adversarial disturbance regimes) are pooled into one
# "adversarial" column; figures and --verbose detail still report every mode
# in EVAL_MODES separately.
REPORT_GROUPS = (
    ("zero",        ("zero",)),
    ("uniform",     ("uniform",)),
    ("adversarial", ADVERSARIAL_MODES),
)

# -----------------------------------------------------------------------
# Spec regions
# -----------------------------------------------------------------------
X_init_bounds = REGION_BOXES["init"]
X_goal_bounds = REGION_BOXES["goal"]
X_bounds      = REGION_BOXES["full"]
X_unsafe_bounds = REGION_BOXES["unsafe"]


def in_box(x1, x2, box):
    return (box["x1_min"] <= x1 <= box["x1_max"]) and (box["x2_min"] <= x2 <= box["x2_max"])


def in_unsafe_union(x1, x2):
    return any(in_box(x1, x2, box) for box in X_unsafe_bounds)


def _ensure_save_dir(save_dir):
    """Create figure output directory on demand."""
    if save_dir is None:
        return None
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def build_controller_styles(labels):
    """Assign a distinct (color, linestyle) pair to each controller label."""
    palette = sns.color_palette("Set2", max(len(labels), 3))
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
    return [
        {
            "color": palette[i % len(palette)],
            "ls": linestyles[i % len(linestyles)],
            "lw": 2.0,
            "label": label,
        }
        for i, label in enumerate(labels)
    ]


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
# Verbose-gated printing
# -----------------------------------------------------------------------
def vprint(*args, verbose=True, **kwargs):
    if verbose:
        print(*args, **kwargs)


# -----------------------------------------------------------------------
# Seed discovery
# -----------------------------------------------------------------------
def _seed_sort_key(seed_dir_name):
    m = re.search(r"\d+", seed_dir_name)
    return int(m.group()) if m else seed_dir_name


def discover_seed_checkpoints(base_dir, rel_path):
    """
    Find every `base_dir/seed<N>/<rel_path>` checkpoint, sorted by seed number.

    Falls back to a single `(None, base_dir/rel_path)` entry (no seed
    subfolders) when no `seed*` directories contain that checkpoint, so
    controllers that were trained without seed sweeps keep working.
    """
    base_dir = Path(base_dir)
    seed_dirs = sorted(
        (p for p in base_dir.glob("seed*") if p.is_dir()),
        key=lambda p: _seed_sort_key(p.name),
    )
    found = [(p.name, p / rel_path) for p in seed_dirs if (p / rel_path).exists()]
    if found:
        return found

    fallback = base_dir / rel_path
    return [(None, fallback)] if fallback.exists() else []


def merge_rollout_results(res_list):
    """Concatenate a list of `rollout_mc` result dicts into one pooled dict."""
    merged = dict(outcomes=[], energies=[], hit_times=[], paths=[], uraw_paths=[])
    for res in res_list:
        for key in merged:
            merged[key].extend(res[key])
    return merged


# -----------------------------------------------------------------------
# Controller helpers
# -----------------------------------------------------------------------
def load_control_net(bundle_path, device="cpu", pretrained_state_dict=False):
    """
    Load a WrapperConterlNN from either:
      - an eval_bundle.pth  (pretrained_state_dict=False, default): dict with key "control_state_dict"
      - a controller_pretrained.pth / rl_controller.pth  (pretrained_state_dict=True): raw state_dict
    """
    rl_policy_net = InvertControlNN(input_dim=2, hidden_dim=CONTROLLER_HIDDEN_DIM, output_dim=1)
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
# Adversarial disturbance selection
# -----------------------------------------------------------------------
def select_lambda_adv_fast(x, X_goal_bounds):
    """
    Physical disturbance: only the x2 (angular-acceleration/torque) channel
    is perturbed, so it cannot push x1 (theta) directly -- x1 only responds
    through the kinematic identity dx1/dt = x2. To be genuinely adversarial
    toward the reach-avoid objective (getting x1 into the goal window), this
    reinforces the existing x1 error: if x1 is displaced from the goal center
    in one direction, push x2 (angular velocity) in the *same* direction --
    like a disturbance torque that resists the controller's effort to swing
    x1 back toward goal.
    """
    goal_center_x1 = 0.5 * (X_goal_bounds["x1_min"] + X_goal_bounds["x1_max"])
    direction_x1 = x[0] - goal_center_x1
    return np.array([0.0, np.sign(direction_x1) * ADV_MAG])


def _goal_center(box):
    return np.array([
        0.5 * (box["x1_min"] + box["x1_max"]),
        0.5 * (box["x2_min"] + box["x2_max"]),
    ], dtype=float)


def _adversarial_state_score(x_next):
    """
    Score larger when the next state is less favorable for reach-avoid.

    The score prioritizes immediate safety violations first, then states farther
    from the goal center after one deterministic step.
    """
    x1n, x2n = float(x_next[0]), float(x_next[1])

    out_of_domain = not (
        X_bounds["x1_min"] <= x1n <= X_bounds["x1_max"]
        and X_bounds["x2_min"] <= x2n <= X_bounds["x2_max"]
    )
    if out_of_domain:
        return 1.0e6
    if in_unsafe_union(x1n, x2n):
        return 5.0e5
    if in_box(x1n, x2n, X_goal_bounds):
        return -1.0e5

    goal_center = _goal_center(X_goal_bounds)
    scale = np.array([
        max(1e-6, 0.5 * (X_goal_bounds["x1_max"] - X_goal_bounds["x1_min"])),
        max(1e-6, 0.5 * (X_goal_bounds["x2_max"] - X_goal_bounds["x2_min"])),
    ], dtype=float)
    z = (np.array([x1n, x2n], dtype=float) - goal_center) / scale
    return float(np.dot(z, z))


def select_lambda_adv_lookahead(x, u_applied, dt):
    """
    Stronger online adversary: choose the disturbance-box endpoint (on the
    x2 / torque channel only) that makes the one-step deterministic successor
    as unfavorable as possible.
    """
    x = np.asarray(x, dtype=float)
    base_drift = f_drift(x, u_applied)
    candidates = np.array([
        [0.0, -ADV_MAG],
        [0.0,  ADV_MAG],
    ], dtype=float)

    best_adv = candidates[0]
    best_score = -np.inf
    for lam in candidates:
        x_next = x + (base_drift + lam) * dt
        score = _adversarial_state_score(x_next)
        if score > best_score:
            best_score = score
            best_adv = lam
    return best_adv.copy()


def _dist_to_box(x1, x2, box):
    """Euclidean distance from (x1, x2) to a rectangular box (0 if inside)."""
    dx1 = max(box["x1_min"] - x1, 0.0, x1 - box["x1_max"])
    dx2 = max(box["x2_min"] - x2, 0.0, x2 - box["x2_max"])
    return float(np.hypot(dx1, dx2))


def _dist_to_nearest_unsafe(x1, x2):
    return min(_dist_to_box(x1, x2, box) for box in X_unsafe_bounds)


def select_lambda_adv_nearest_unsafe(x, u_applied, dt):
    """
    One-step online adversary that targets whichever unsafe region is
    currently closest: choose the disturbance-box endpoint (x2 / torque
    channel only) whose one-step deterministic successor minimizes distance
    to the nearest unsafe box. Unlike `select_lambda_adv_lookahead` (which
    also weighs domain-exit and distance-from-goal), this only cares about
    closing the gap to the nearest unsafe region.
    """
    x = np.asarray(x, dtype=float)
    base_drift = f_drift(x, u_applied)
    candidates = np.array([
        [0.0, -ADV_MAG],
        [0.0,  ADV_MAG],
    ], dtype=float)

    best_adv = candidates[0]
    best_dist = np.inf
    for lam in candidates:
        x_next = x + (base_drift + lam) * dt
        dist = _dist_to_nearest_unsafe(float(x_next[0]), float(x_next[1]))
        if dist < best_dist:
            best_dist = dist
            best_adv = lam
    return best_adv.copy()


def select_lambda_adv_velocity(x, u_applied, dt):
    """
    One-step online adversary that directly targets the goal region's
    angular-rate window (x2 in [X_goal_bounds.x2_min, x2_max], typically
    [-1, 1] rad/s): choose the disturbance-box endpoint (x2 / torque channel
    only) whose one-step deterministic successor has the largest |x2|
    magnitude, pushing angular velocity away from the goal band regardless
    of x1 or unsafe-region proximity.

    Disturbance is the system's only lever on x2 (x1 is purely kinematic,
    dx1/dt = x2), and the goal's x2 window is proportionally tighter than
    its x1 window relative to the full state domain -- so this closes a gap
    left by `select_lambda_adv_fast`, which goes inert (returns zero
    disturbance) once x1 is centered in the goal window, even if x2 is
    still far outside its own band.
    """
    x = np.asarray(x, dtype=float)
    base_drift = f_drift(x, u_applied)
    candidates = np.array([
        [0.0, -ADV_MAG],
        [0.0,  ADV_MAG],
    ], dtype=float)

    best_adv = candidates[0]
    best_score = -np.inf
    for lam in candidates:
        x_next = x + (base_drift + lam) * dt
        score = abs(float(x_next[1]))
        if score > best_score:
            best_score = score
            best_adv = lam
    return best_adv.copy()


def select_lambda_adv(x, X_goal_bounds, mode="fast", u_applied=None, dt=0.005):
    """
    Dispatch between the adversarial disturbance strategies.

    Modes
    -----
    fast:
        Heuristic that pushes each state component away from the goal center
        using the disturbance-box corner with matching sign.
    lookahead:
        One-step online adversary that evaluates all disturbance-box corners and
        picks the corner whose deterministic next state is worst for reach-avoid.
    nearest_unsafe:
        One-step online adversary that picks the corner whose deterministic
        next state is closest to the nearest unsafe region.
    velocity:
        One-step online adversary that picks the corner whose deterministic
        next state has the largest |x2|, targeting the goal's angular-rate window.
    """
    if mode == "fast":
        return select_lambda_adv_fast(x=x, X_goal_bounds=X_goal_bounds)
    if mode == "lookahead":
        if u_applied is None:
            raise ValueError("u_applied is required when adversarial mode='lookahead'")
        return select_lambda_adv_lookahead(x=x, u_applied=u_applied, dt=dt)
    if mode == "nearest_unsafe":
        if u_applied is None:
            raise ValueError("u_applied is required when adversarial mode='nearest_unsafe'")
        return select_lambda_adv_nearest_unsafe(x=x, u_applied=u_applied, dt=dt)
    if mode == "velocity":
        if u_applied is None:
            raise ValueError("u_applied is required when adversarial mode='velocity'")
        return select_lambda_adv_velocity(x=x, u_applied=u_applied, dt=dt)
    raise ValueError(f"Unknown adversarial mode: {mode!r}")


def _step_disturbance(x1, x2, u_applied, mode, dt, traj_rng):
    """
    Draw the disturbance drift term for one step under the given eval mode.

    fast / lookahead / nearest_unsafe / velocity : adversarial endpoint of
                       the disturbance box (x2 channel only)
    uniform          : i.i.d. draw w = [0, w2], w2 ~ Uniform[-ADV_MAG, ADV_MAG]
    zero             : no disturbance
    """
    if mode == "zero":
        return np.array([0.0, 0.0], dtype=float)
    if mode == "uniform":
        return np.array([0.0, traj_rng.uniform(-ADV_MAG, ADV_MAG)])
    return select_lambda_adv(
        x=np.array([x1, x2]), X_goal_bounds=X_goal_bounds,
        mode=mode, u_applied=u_applied, dt=dt,
    )


# -----------------------------------------------------------------------
# MC rollout
# -----------------------------------------------------------------------
def rollout_mc(
    controller,
    n_mc:    int   = 500,
    T_max:   float = 8.0,
    dt:      float = 0.005,
    mode:    str   = "fast",   # "fast" | "lookahead" | "uniform" | "zero"
    seed:    int   = 42,
    n_paths: int   = 20,
):
    """
    Euler-Maruyama rollout under one disturbance regime (see `mode`). Returns dict:
        outcomes  : list[str]         'success' | 'fail' | 'timeout'
        energies  : list[float]       integral u_raw^2 dt  up to T_hit
        hit_times : list[float]       T_hit (T_max for non-success)
        paths     : list[ndarray(K,2)]  first n_paths trajectories
        uraw_paths: list[ndarray(K,)]   first n_paths u_raw(t) histories

    Initial conditions are pre-generated from a root RNG so that every controller
    sees the same x0_i for trajectory i.  Per-step noise (disturbance and dW) uses
    a per-trajectory RNG seeded by (seed, i), so results are directly comparable
    across controllers and across eval modes for the same trajectory index.
    """
    if mode not in EVAL_MODES:
        raise ValueError(f"Unknown mode: {mode!r} (expected one of {EVAL_MODES})")

    root_rng = np.random.default_rng(seed)
    x1_init  = root_rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"], size=n_mc)
    x2_init  = root_rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"], size=n_mc)

    N_steps = int(T_max / dt)

    outcomes   = []
    energies   = []
    hit_times  = []
    paths_out  = []
    uraw_out   = []

    for i in range(n_mc):
        x1, x2 = float(x1_init[i]), float(x2_init[i])
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
                T_hit, outcome = k * dt, "success"
                break
            if in_unsafe_union(x1, x2):
                T_hit, outcome = k * dt, "fail"
                break

            u_applied, u_raw = get_u_and_raw(x1, x2, controller)
            energy_acc += u_raw**2 * dt
            if store_path:
                uraw.append(u_raw)

            drift  = f_drift(np.array([x1, x2]), u_applied) + \
                _step_disturbance(x1, x2, u_applied, mode, dt, traj_rng)
            dW     = np.sqrt(dt) * traj_rng.standard_normal()
            x_next = np.array([x1, x2]) + drift * dt + g_diff(np.array([x1, x2])) * dW
            x1, x2 = float(x_next[0]), float(x_next[1])

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
    for unsafe_box in X_unsafe_bounds:
        ax.add_patch(_rect(unsafe_box, **kw_unsafe))

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
    n_paths: int   = 10,
    T_max:   float = 8.0,
    t_extra: float = 1.0,
    dt:      float = 0.005,
    mode:    str   = "fast",   # "fast" | "lookahead" | "uniform" | "zero"
    seed:    int   = 42,
):
    """
    Returns (paths_out, success_out).

    paths_out : list of (K, 2) phase-state arrays.
    success_out : list of bool, one per path -- True iff that trajectory
                  reached the goal box within the simulated window.

    Trajectories stop exactly at the point they first reach the goal
    (success) or first enter an unsafe region (failure) -- no dwell time
    either way. Trajectories that do neither keep running for
    T_max + t_extra so their excursion is still visible (timeout).

    Uses the same pre-generated initial conditions and per-trajectory noise
    RNGs as `rollout_mc` for the given seed, so phase plots are consistent
    with the reported statistics for the same (mode, seed).
    """
    if mode not in EVAL_MODES:
        raise ValueError(f"Unknown mode: {mode!r} (expected one of {EVAL_MODES})")

    root_rng = np.random.default_rng(seed)
    x1_init  = root_rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"], size=n_paths)
    x2_init  = root_rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"], size=n_paths)

    N_steps     = int((T_max + t_extra) / dt)
    paths_out   = []
    success_out = []

    for i in range(n_paths):
        x1, x2 = float(x1_init[i]), float(x2_init[i])
        traj_rng = np.random.default_rng([seed, i])

        path  = [[x1, x2]]

        for k in range(N_steps):
            if in_box(x1, x2, X_goal_bounds):
                # success: stop plotting right at goal entry, no dwell time
                success_out.append(True)
                break
            if in_unsafe_union(x1, x2):
                # failure: stop plotting right at unsafe entry, no dwell time
                success_out.append(False)
                break

            u_applied, _ = get_u_and_raw(x1, x2, controller)
            drift  = f_drift(np.array([x1, x2]), u_applied) + \
                _step_disturbance(x1, x2, u_applied, mode, dt, traj_rng)
            dW     = np.sqrt(dt) * traj_rng.standard_normal()
            x_next = np.array([x1, x2]) + drift * dt + g_diff(np.array([x1, x2])) * dW
            x1, x2 = float(x_next[0]), float(x_next[1])

            path.append([x1, x2])
        else:
            success_out.append(False)

        paths_out.append(np.array(path, dtype=float))

    return paths_out, success_out


# -----------------------------------------------------------------------
# Figure 1 — Phase trajectories (all controllers, one plot)
# -----------------------------------------------------------------------
def plot_phase_trajectories(entries, viz_paths_per_ctrl, styles, save_dir=None, verbose=True):
    """
    viz_paths_per_ctrl : list of path lists, one per controller,
                         produced by rollout_viz_paths (extended to T_hit+1s).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_regions(ax)

    for style, (label, res, stats), viz_paths in zip(styles, entries, viz_paths_per_ctrl):
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

    ax.legend(loc="upper left", framealpha=0.9, fontsize=13)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(r"$\theta$, rad")
    ax.set_ylabel(r"$\dot\theta$, rad/s")
    fig.tight_layout()

    save_path = _ensure_save_dir(save_dir)
    if save_path is not None:
        p = save_path / "fig1_phase_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        vprint(f"[saved] {p}", verbose=verbose)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 7 — Failure phase trajectories (all failures per controller,
# pooled across every eval mode; adversarial successes fill in for
# controllers with none)
# -----------------------------------------------------------------------
def _pick_failure_paths(viz_paths, viz_success):
    """Return every non-successful trajectory (never reached goal within the
    plotted window). Empty list if the controller had no failures."""
    return [path for path, success in zip(viz_paths, viz_success) if not success]


def pool_viz_across_modes(viz_paths, viz_success, modes):
    """
    Pool per-mode `viz_paths`/`viz_success` (each `{mode: [per-controller
    list]}`, as produced by `main()`/`postprocess_mc.py`) across `modes` into
    one list per controller, so trajectories can be picked across every
    tested disturbance regime rather than just one mode.
    """
    n_ctrl = len(viz_paths[modes[0]])
    pooled_paths   = [[] for _ in range(n_ctrl)]
    pooled_success = [[] for _ in range(n_ctrl)]
    for mode in modes:
        for i in range(n_ctrl):
            pooled_paths[i].extend(viz_paths[mode][i])
            pooled_success[i].extend(viz_success[mode][i])
    return pooled_paths, pooled_success


def plot_failure_trajectories(labels, fail_viz_paths, fail_viz_success,
                               adv_viz_paths, adv_viz_success,
                               styles, save_dir=None, verbose=True, max_per_controller=None):
    """
    One phase-plane plot with every failure trajectory (never reached goal)
    found per controller, pooled across every eval mode -- see
    `pool_viz_across_modes`. Pass `max_per_controller` to cap how many
    failure paths are drawn per controller (first N of the pooled set) --
    useful when a controller has dozens of failures and the plot gets
    cluttered; default None draws all of them.

    A controller with zero pooled failures would otherwise be absent from
    the plot/legend entirely; instead it's given `n_target` successful
    trajectories drawn from its own `adv_viz_paths`/`adv_viz_success`
    (pooled across `ADVERSARIAL_MODES` only, i.e. fast/lookahead/
    nearest_unsafe/velocity -- never uniform/zero), so it still shows a
    comparable number of trajectories to the other controllers rather than
    dropping out. `n_target` is the largest (capped) failure count among the
    controllers in this same figure that do have failures (or a small
    per-adversarial-mode default if none do). These substitute lines are
    labeled "(success)" and drawn with reduced opacity so they're never
    mistaken for real failures.
    """
    all_fail_paths = [
        _pick_failure_paths(viz_paths, viz_success)[:max_per_controller]
        for viz_paths, viz_success in zip(fail_viz_paths, fail_viz_success)
    ]
    fail_counts = [len(fp) for fp in all_fail_paths if fp]
    n_target = max(fail_counts) if fail_counts else len(ADVERSARIAL_MODES)

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_regions(ax)

    for style, label, fail_paths, aviz, asucc in zip(
            styles, labels, all_fail_paths, adv_viz_paths, adv_viz_success):
        if fail_paths:
            paths, suffix, alpha = fail_paths, "", 1.0
        else:
            success_paths = [path for path, success in zip(aviz, asucc) if success]
            paths, suffix, alpha = success_paths[:n_target], " (success)", 0.5
        if not paths:
            continue

        first = True
        for path in paths:
            lbl = f"{label}{suffix}" if first else None
            first = False
            ax.plot(path[:, 0], path[:, 1],
                    color=style["color"], ls=style["ls"], lw=style["lw"],
                    alpha=alpha, label=lbl, zorder=3)
        for path in paths:
            ax.plot(path[0, 0], path[0, 1], "o",
                    color=style["color"], markersize=5, alpha=min(alpha + 0.1, 0.9), zorder=4)

    ax.legend(loc="upper left", framealpha=0.9, fontsize=13)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(r"$\theta$, rad")
    ax.set_ylabel(r"$\dot\theta$, rad/s")
    fig.tight_layout()

    save_path = _ensure_save_dir(save_dir)
    if save_path is not None:
        p = save_path / "fig7_failure_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        vprint(f"[saved] {p}", verbose=verbose)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 2 — Energy vs Hitting-time scatter with marginal KDEs
# -----------------------------------------------------------------------
def plot_energy_vs_thit(entries, styles, save_dir=None, verbose=True):
    """
    Joint scatter of (T_hit, energy) for successful trajectories,
    with marginal KDE of T_hit on the top panel and marginal KDE of
    energy on the right panel.
    """
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
    ax_main.legend(framealpha=0.9, fontsize=11)
    ax_main.grid(True, alpha=0.35)

    # --- top marginal ---
    ax_top.set_ylabel("Density", labelpad=6)
    ax_top.yaxis.set_label_position("left")
    ax_top.tick_params(axis="y", labelsize=12)
    ax_top.grid(True, alpha=0.35)
    ax_top.spines["bottom"].set_visible(False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
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

    save_path = _ensure_save_dir(save_dir)
    if save_path is not None:
        p = save_path / "fig2_energy_vs_thit.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        vprint(f"[saved] {p}", verbose=verbose)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 8 — Pooled energy distributions across all MC runs
# -----------------------------------------------------------------------
def plot_energy_distribution(results, styles, save_dir=None, verbose=True):
    """Compare energy across all outcomes, pooled within four disturbance groups.

    Energy is accumulated until success, failure, or timeout. Boxes show
    quartiles and medians, with 1.5-IQR whiskers and individual outliers.
    White diamonds and numeric labels indicate the arithmetic means.
    Adversarial energies pool all four attack modes without taking a minimum.
    """
    groups = (
        ("Overall", EVAL_MODES),
        ("Zero disturbance", ("zero",)),
        ("Uniform disturbance", ("uniform",)),
        ("Adversarial", ADVERSARIAL_MODES),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, modes) in zip(axes, groups):
        pooled = {style["label"]: [] for style in styles}
        for mode in modes:
            for label, res, _stats in results.get(mode, []):
                pooled[label].extend(res["energies"])
        boxes = ax.boxplot(list(pooled.values()), patch_artist=True, widths=0.5,
                           medianprops={"color": "black", "linewidth": 2}, showmeans=True,
                           meanprops={"marker": "D", "markerfacecolor": "white",
                                      "markeredgecolor": "black", "markersize": 5})
        for box, style in zip(boxes["boxes"], styles):
            box.set_facecolor(style["color"])
        for position, values in enumerate(pooled.values(), start=1):
            if values:
                mean = float(np.mean(values))
                ax.annotate(f"Mean: {mean:.4f}", xy=(position, mean), xytext=(0, 10),
                            textcoords="offset points", ha="center", va="bottom", fontsize=10,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
        ax.set_xticks(range(1, len(pooled) + 1), list(pooled))
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.35)
    fig.supylabel(r"Control energy $E = \int u_{\rm raw}(t)^2\,dt$")
    fig.tight_layout()
    save_path = _ensure_save_dir(save_dir)
    if save_path is not None:
        path = save_path / "fig8_energy_distribution.pdf"
        fig.savefig(path, format="pdf", bbox_inches="tight")
        vprint(f"[saved] {path}", verbose=verbose)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 5 — u_raw(t) vs time (single plot, all controllers)
# -----------------------------------------------------------------------
def plot_u_raw_trajectories(entries, dt, styles, save_dir=None, verbose=True):
    """
    All controllers in one plot.
    Each trajectory is clipped to its first-hitting time T_hit.
    Individual traces are shown in light colour; the per-controller
    mean (over stored paths, aligned on [0, T_hit]) is drawn in bold.
    """
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
    ax.legend(framealpha=0.9, fontsize=13)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    save_path = _ensure_save_dir(save_dir)
    if save_path is not None:
        p = save_path / "fig5_u_raw_trajectories.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        vprint(f"[saved] {p}", verbose=verbose)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 6 — Success-rate summary bar chart (controller x report column)
# -----------------------------------------------------------------------
def plot_success_rate_summary(labels, success_table, per_seed_success, styles, save_dir=None, verbose=True):
    """
    Grouped bar chart: one group of bars per report column (from
    `REPORT_GROUPS`), one bar per controller within each group, bar height =
    `success_table[label][col]` (mean p_success across seeds).

    Individual per-seed values (`per_seed_success[label][col]`) are overlaid
    as small jittered dots so seed-to-seed spread stays visible even though
    the bar itself only shows the mean.

    A dedicated, bar-free fourth column holds the legend so it never
    overlaps the rightmost (adversarial) data column.
    """
    col_names = [name for name, _modes in REPORT_GROUPS]
    n_ctrl = len(labels)
    n_col  = len(col_names)
    group_centers = np.arange(n_col)
    bar_w = 0.8 / max(n_ctrl, 1)
    legend_col = n_col   # reserved column index, one past the last data column

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, (style, label) in enumerate(zip(styles, labels)):
        bar_x = group_centers - 0.4 + bar_w * (i + 0.5)
        heights = [success_table[label][col] for col in col_names]
        ax.bar(bar_x, heights, width=bar_w * 0.9,
               color=style["color"], edgecolor="black", linewidth=0.6,
               label=label, zorder=2)

        for j, col in enumerate(col_names):
            seed_vals = per_seed_success[label][col]
            n_seeds = len(seed_vals)
            if n_seeds == 0:
                continue
            jitter = (np.arange(n_seeds) - (n_seeds - 1) / 2) * (bar_w * 0.15)
            ax.scatter(np.full(n_seeds, bar_x[j]) + jitter, seed_vals,
                       color="black", s=14, zorder=3, alpha=0.75)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([col.capitalize() for col in col_names])
    ax.set_xlim(-0.5, legend_col + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel(r"$p_{\rm success}$")

    legend_handles = [
        Patch(facecolor=style["color"], edgecolor="black", label=label)
        for style, label in zip(styles, labels)
    ]
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(legend_col - 0.45, 0.5), bbox_transform=ax.transData,
              framealpha=0.9, fontsize=18, handlelength=2.2, handleheight=1.6, labelspacing=0.9)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()

    save_path = _ensure_save_dir(save_dir)
    if save_path is not None:
        p = save_path / "fig6_success_rate_summary.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        vprint(f"[saved] {p}", verbose=verbose)
    plt.close(fig)


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


def save_mc_cache(cache_path, *, controller_labels, results, viz_paths, viz_success,
                   success_table, per_seed_success, dt, meta=None, verbose=True):
    """
    Persist everything needed to re-render `postprocess_mc.py`'s plots and
    summary table without re-running the (slow) Monte Carlo rollouts.

    `results`/`viz_paths`/`viz_success` are pooled across every controller-seed
    (see `merge_rollout_results`); `viz_success[mode][i]` is index-aligned
    with `viz_paths[mode][i]` (one bool per stored path, see
    `rollout_viz_paths`), and `success_table[label][col]` is the report table
    value for that column, with `per_seed_success[label][col]` the list of
    per-seed values it was averaged from (see `print_success_rate_table` for
    how each `REPORT_GROUPS` column is aggregated across modes and seeds).

    `styles` are deliberately NOT stored -- `build_controller_styles` is a
    pure function of `controller_labels`, so postprocessing recomputes them
    to stay in sync with any later styling changes rather than freezing them.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "controller_labels": controller_labels,
        "results": results,
        "viz_paths": viz_paths,
        "viz_success": viz_success,
        "success_table": success_table,
        "per_seed_success": per_seed_success,
        "dt": dt,
        "meta": meta or {},
    }
    torch.save(payload, cache_path)
    vprint(f"[saved] {cache_path}", verbose=verbose)
    return cache_path


def load_mc_cache(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"MC cache not found: {cache_path}\n"
            "Run `python run_mc.py` first to generate it."
        )
    return torch.load(cache_path, weights_only=False)


def print_success_rate_table(labels, success_table):
    """
    Print a controller x report-column table.

    `success_table[label][col]` (col from `REPORT_GROUPS`) is the mean,
    across seeds, of each seed's worst-case p_success over the modes pooled
    into that column -- e.g. "adversarial" takes, per seed, the min p_success
    over "fast", "lookahead", "nearest_unsafe", and "velocity" before
    averaging over seeds, since a real adversary would pick whichever attack
    is more damaging.
    """
    col_names   = [name for name, _modes in REPORT_GROUPS]
    col_headers = [f"p_success ({name})" for name in col_names]
    label_w = max(len("Controller"), *(len(l) for l in labels))
    col_w   = [len(h) for h in col_headers]

    def _row(cells):
        first, *rest = cells
        return first.ljust(label_w) + "  " + "  ".join(
            c.rjust(w) for c, w in zip(rest, col_w)
        )

    print(f"\n{'=' * 72}")
    print("  Success-rate summary (mean p_success across seeds)")
    print(f"{'=' * 72}")
    print(_row(["Controller", *col_headers]))
    print(_row(["-" * label_w, *("-" * w for w in col_w)]))
    for label in labels:
        row = [f"{success_table[label][name]:.4f}" for name in col_names]
        print(_row([label, *row]))


# -----------------------------------------------------------------------
# Per-controller-seed evaluation
# -----------------------------------------------------------------------
def evaluate_controller_seed(ckpt_path, *, pretrained, n_mc, t_max, dt, mc_seed, n_paths):
    """Load one checkpoint and run every eval mode against it. Returns
    {mode: {"res": ..., "stats": ..., "viz_paths": ..., "viz_success": ...}}."""
    ctrl = load_control_net(ckpt_path, pretrained_state_dict=pretrained)
    per_mode = {}
    for mode in EVAL_MODES:
        res = rollout_mc(
            ctrl, n_mc=n_mc, T_max=t_max, dt=dt, mode=mode, seed=mc_seed, n_paths=n_paths,
        )
        stats = compute_stats(res)
        viz_paths, viz_success = rollout_viz_paths(
            ctrl, n_paths=n_paths, T_max=t_max, t_extra=1.0, dt=dt, mode=mode, seed=mc_seed,
        )
        per_mode[mode] = dict(res=res, stats=stats, viz_paths=viz_paths, viz_success=viz_success)
    return per_mode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-controller/per-seed/per-mode detail (default: only the final summary table).",
    )
    parser.add_argument(
        "--max-fail-trajectories", type=int, default=None, metavar="N",
        help="Cap the number of failure trajectories drawn per controller in "
             "fig7_failure_trajectories.pdf (default: draw every pooled failure).",
    )
    parser.add_argument(
        "--energy-controller-dir", type=Path, default=None,
        help="Include Cert. (energy): a run containing outputs/eval_bundle.pth, "
             "or a directory containing seed<N>/outputs/eval_bundle.pth. "
             "Uses the same raw-control energy metric and MC regimes as Cert.",
    )
    parser.add_argument(
        "--baseline-controller-dir", type=Path, default=None,
        help="Compare only this certified baseline with --energy-controller-dir. "
             "Select a seed<N> directory to evaluate a single training seed.",
    )
    parser.add_argument(
        "--n-mc", type=int, default=int(EXAMPLE_CONFIG["n_mc"]), metavar="N",
        help="Trajectories per controller, training seed, and disturbance mode (default: config.json).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for the MC cache and figures (default: run_mc_results).",
    )
    parser.add_argument(
        "--figures", type=int, nargs="+", choices=(1, 2, 5, 6, 7, 8), default=None,
        help="Figures to generate (default: 6 7 8 for a baseline/energy comparison; "
             "1 2 5 6 7 otherwise).",
    )
    args = parser.parse_args(argv)
    if args.figures is None:
        args.figures = [6, 7, 8] if args.baseline_controller_dir is not None else [1, 2, 5, 6, 7]
    if args.n_mc < 1:
        parser.error("--n-mc must be positive")
    if args.baseline_controller_dir is not None:
        if args.energy_controller_dir is None:
            parser.error("--baseline-controller-dir requires --energy-controller-dir")
        if not discover_seed_checkpoints(args.baseline_controller_dir, "outputs/eval_bundle.pth"):
            parser.error("No baseline controller checkpoint found under --baseline-controller-dir")
    if args.energy_controller_dir is not None and not discover_seed_checkpoints(
        args.energy_controller_dir, "outputs/eval_bundle.pth"
    ):
        parser.error("No energy controller checkpoint found under --energy-controller-dir")
    return args


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    verbose = args.verbose
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Controllers to compare — add/remove entries here.
    # `base_dir`: controller directory. If it holds `seed<N>/<rel_path>`
    #             checkpoints, every seed is evaluated independently and
    #             aggregated; otherwise `base_dir/<rel_path>` is used as a
    #             single implicit run.
    # `pretrained`: False loads an eval_bundle.pth (key "control_state_dict"),
    #               True loads a raw WrapperConterlNN state_dict.
    # Controllers with no checkpoint found on disk are skipped with a warning.
    bundle_specs = [
        dict(label="Cert.",                 base_dir=HERE / "neural_certified", rel_path="outputs/eval_bundle.pth", pretrained=False),
        dict(label="Cert. (nominal drift)", base_dir=HERE / "neural_certified_nominal_drift", rel_path="outputs/eval_bundle.pth", pretrained=False),
        dict(label="RL (SB3 PPO)",          base_dir=HERE / "rl_sb3_ppo", rel_path="outputs/rl_controller.pth", pretrained=True),
        dict(label="RL (SB3 PPO reg.)",     base_dir=HERE / "rl_sb3_ppo_regularize", rel_path="outputs/rl_controller.pth", pretrained=True),
        dict(label="RL (SB3 SAC)",          base_dir=HERE / "rl_sb3_sac", rel_path="outputs/rl_controller.pth", pretrained=True),
        dict(label="RL (SB3 DDPG)",         base_dir=HERE / "rl_sb3_ddpg", rel_path="outputs/rl_controller.pth", pretrained=True),
        dict(label="RL (SB3 RPO)",          base_dir=HERE / "rl_sb3_rpo", rel_path="outputs/rl_controller.pth", pretrained=True),
    ]
    if args.baseline_controller_dir is not None:
        bundle_specs = [dict(label="Cert.", base_dir=args.baseline_controller_dir,
                             rel_path="outputs/eval_bundle.pth", pretrained=False)]
    if args.energy_controller_dir is not None:
        bundle_specs.insert(1, dict(label="Cert. (energy)", base_dir=args.energy_controller_dir,
                                    rel_path="outputs/eval_bundle.pth", pretrained=False))

    controllers = []
    for spec in bundle_specs:
        checkpoints = discover_seed_checkpoints(spec["base_dir"], spec["rel_path"])
        if checkpoints:
            controllers.append({**spec, "checkpoints": checkpoints})
        else:
            print(f"[skip] '{spec['label']}': no checkpoint found under {spec['base_dir']}")
    if not controllers:
        raise FileNotFoundError(
            "No controller checkpoints found among bundle_specs. "
            "Train at least one controller before running run_mc.py."
        )

    styles = build_controller_styles([c["label"] for c in controllers])

    # --- MC config ---
    N_MC    = args.n_mc
    T_MAX   = 30.0
    DT      = 0.005
    MC_SEED = 42
    N_PATHS = 5   # trajectories to store (per seed) for phase-plane / u_raw plots

    # mode -> list[(label, pooled_res, pooled_stats)] / list[pooled_viz_paths]
    results      = {mode: [] for mode in EVAL_MODES}
    viz_paths    = {mode: [] for mode in EVAL_MODES}
    viz_success  = {mode: [] for mode in EVAL_MODES}
    success_table    = {}   # label -> col -> mean p_success across seeds
    per_seed_success = {}   # label -> col -> list[p_success], one per seed
    seed_counts      = {}   # label -> number of seeds evaluated

    for spec in controllers:
        label = spec["label"]
        seed_ids = [seed_id or "default" for seed_id, _ in spec["checkpoints"]]
        vprint(f"\n{'#' * 72}", verbose=verbose)
        vprint(f"Evaluating '{label}'  (seeds: {', '.join(seed_ids)})", verbose=verbose)
        vprint(f"{'#' * 72}", verbose=verbose)

        seed_runs = []
        for seed_id, ckpt_path in spec["checkpoints"]:
            seed_label = f"{label} [{seed_id}]" if seed_id else label
            vprint(f"\n  -- {seed_label}  ({ckpt_path}) --", verbose=verbose)
            per_mode = evaluate_controller_seed(
                ckpt_path, pretrained=spec["pretrained"],
                n_mc=N_MC, t_max=T_MAX, dt=DT, mc_seed=MC_SEED, n_paths=N_PATHS,
            )
            if verbose:
                for mode in EVAL_MODES:
                    print_summary(f"{seed_label} [{mode}]", per_mode[mode]["stats"])
            seed_runs.append(per_mode)

        seed_counts[label] = len(seed_runs)
        per_seed_success[label] = {
            col_name: [
                min(run[mode]["stats"]["p_success"] for mode in modes)
                for run in seed_runs
            ]
            for col_name, modes in REPORT_GROUPS
        }
        success_table[label] = {
            col_name: float(np.mean(vals))
            for col_name, vals in per_seed_success[label].items()
        }

        for mode in EVAL_MODES:
            pooled_res = merge_rollout_results([run[mode]["res"] for run in seed_runs])
            pooled_stats = compute_stats(pooled_res)
            results[mode].append((label, pooled_res, pooled_stats))
            pooled_viz = [p for run in seed_runs for p in run[mode]["viz_paths"]]
            pooled_viz_success = [s for run in seed_runs for s in run[mode]["viz_success"]]
            viz_paths[mode].append(pooled_viz)
            viz_success[mode].append(pooled_viz_success)

    save_mc_cache(
        output_dir / "mc_cache.pth",
        controller_labels=[c["label"] for c in controllers],
        results=results,
        viz_paths=viz_paths,
        viz_success=viz_success,
        success_table=success_table,
        per_seed_success=per_seed_success,
        dt=DT,
        meta=dict(
            n_mc=N_MC,
            t_max=T_MAX,
            mc_seed=MC_SEED,
            n_paths=N_PATHS,
            eval_modes=list(EVAL_MODES),
            figures=args.figures,
            seed_counts=seed_counts,
            bundle_specs=[
                {
                    "label": c["label"],
                    "base_dir": str(c["base_dir"]),
                    "rel_path": c["rel_path"],
                    "pretrained": c["pretrained"],
                    "checkpoints": [str(p) for _, p in c["checkpoints"]],
                }
                for c in controllers
            ],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
        verbose=verbose,
    )

    for mode in EVAL_MODES:
        mode_dir = output_dir / mode
        if 1 in args.figures:
            plot_phase_trajectories(results[mode], viz_paths[mode], styles, save_dir=mode_dir, verbose=verbose)
        if 2 in args.figures:
            plot_energy_vs_thit(results[mode], styles, save_dir=mode_dir, verbose=verbose)
        if 5 in args.figures:
            plot_u_raw_trajectories(results[mode], DT, styles, save_dir=mode_dir, verbose=verbose)

    labels = [c["label"] for c in controllers]
    if 6 in args.figures:
        plot_success_rate_summary(labels, success_table, per_seed_success, styles,
                                  save_dir=output_dir, verbose=verbose)
    if 7 in args.figures:
        pooled_viz_paths, pooled_viz_success = pool_viz_across_modes(viz_paths, viz_success, list(EVAL_MODES))
        adv_viz_paths, adv_viz_success = pool_viz_across_modes(viz_paths, viz_success, list(ADVERSARIAL_MODES))
        plot_failure_trajectories(labels, pooled_viz_paths, pooled_viz_success,
                                 adv_viz_paths, adv_viz_success, styles,
                                 save_dir=output_dir, verbose=verbose,
                                 max_per_controller=args.max_fail_trajectories)
    if 8 in args.figures:
        plot_energy_distribution(results, styles, save_dir=output_dir, verbose=verbose)

    print_success_rate_table(labels, success_table)


if __name__ == "__main__":
    main()
