"""
Inverted pendulum SDE simulation + animation.

State:
  x1 = angle (rad)
  x2 = angular velocity (rad/s)

SDE (u(x)=0):
  dx1 = x2 dt
  dx2 = (g/L * sin(x1) - b*x2/(m*L^2)) dt + sigma dW

Specs:
  X          = [-2pi, 2pi] x [-20, 20]
  X_init     = [3pi/4, 5pi/4] x [-1, 1]
  X_goal     = [-pi/2, pi/2] x [-4, 4]
  X_unsafe   = ([-2pi, -3pi/2] x [-20, -10]) U ([3pi/2, 2pi] x [10, 20])
"""
import numpy as np
import torch
from math import pi
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]   # repo_root
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN
from src.save_load_utils import load_eval_bundle

# ----------------------------
# Parameters
# ----------------------------
g_grav = 9.81
L = 0.5
m = 0.15
b = 0.1
M = 6.0           # not used when u=0
sigma = 0.2

# ----------------------------
# Spec sets
# ----------------------------
pi = np.pi

X_bounds = {
    "x1_min": -2*pi, "x1_max":  2*pi,
    "x2_min": -20.0, "x2_max": 20.0
}

X_init_bounds = {
    "x1_min": 3*pi/4, "x1_max": 5*pi/4,
    "x2_min": -1.0,   "x2_max": 1.0
}

X_goal_bounds = {
    "x1_min": -0.4*pi,  "x1_max": 0.4*pi,
    "x2_min": -4.0,   "x2_max": 4.0
}

# Unsafe = union of two rectangles
X_unsafe_1 = {
    "x1_min": -2*pi,   "x1_max": -3*pi/2,
    "x2_min": -20.0,   "x2_max": -10.0
}
X_unsafe_2 = {
    "x1_min":  3*pi/2, "x1_max":  2*pi,
    "x2_min":  10.0,   "x2_max":  20.0
}

def in_box(x1, x2, box):
    return (box["x1_min"] <= x1 <= box["x1_max"]) and (box["x2_min"] <= x2 <= box["x2_max"])


# Control Network
def load_control_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    rl_policy_net = InvertControlNN(input_dim=2, hidden_dim=8, output_dim=1)
    control_net = WrapperConterlNN(rl_policy_net).to(device)
    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    # 4) eval mode for rollout
    control_net.eval()
    return control_net

from src.set_values import AdditiveBoxSetDrift, InvertedPendulumSetDrift

# ----------------------------
# Uncertainty helpers (same style as GBM)
# ----------------------------
def sample_uniform_box(rng, low, high):
    return rng.uniform(low, high).astype(float)

def sample_corner_box(rng, rad):
    s = rng.choice([-1.0, 1.0], size=rad.shape)
    return (s * rad).astype(float)

def sample_uniform_interval(rng, lo, hi):
    return float(rng.uniform(lo, hi))

def sample_corner_interval(rng, lo, hi):
    return float(rng.choice([lo, hi]))

def get_default_param_ranges():
    return {
        "g": (g_grav, g_grav),
        "L": (L, L),
        "b": (b, b),
        "m": (m, m),
    }

def validate_param_ranges(param_ranges):
    required = ("g", "L", "b", "m")
    for key in required:
        if key not in param_ranges:
            raise ValueError(f"Missing param range '{key}'")
        r = param_ranges[key]
        if len(r) != 2:
            raise ValueError(f"Range for '{key}' must have length 2, got {r}")
        lo = float(r[0])
        hi = float(r[1])
        if lo > hi:
            raise ValueError(f"Invalid range for '{key}': {r}")

    if float(param_ranges["L"][0]) <= 0.0:
        raise ValueError("L range must be strictly positive")
    if float(param_ranges["m"][0]) <= 0.0:
        raise ValueError("m range must be strictly positive")

def sample_param_set(rng, mode, param_ranges):
    if mode == "none":
        return np.array([
            float(param_ranges["g"][0]),
            float(param_ranges["L"][0]),
            float(param_ranges["b"][0]),
            float(param_ranges["m"][0]),
        ], dtype=float)
    if mode == "uniform":
        return np.array([
            sample_uniform_interval(rng, *param_ranges["g"]),
            sample_uniform_interval(rng, *param_ranges["L"]),
            sample_uniform_interval(rng, *param_ranges["b"]),
            sample_uniform_interval(rng, *param_ranges["m"]),
        ], dtype=float)
    if mode == "corners":
        return np.array([
            sample_corner_interval(rng, *param_ranges["g"]),
            sample_corner_interval(rng, *param_ranges["L"]),
            sample_corner_interval(rng, *param_ranges["b"]),
            sample_corner_interval(rng, *param_ranges["m"]),
        ], dtype=float)
    raise ValueError(f"Unknown parametric mode: {mode}")

def worstcase_wrt_V_box(V_net, x_np, rad, device="cpu"):
    """
    argmax_{|w_i|<=rad_i} <∇V(x), w> = rad ⊙ sign(∇V(x))
    """
    xt = torch.tensor(x_np.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)
    V = V_net(xt)
    grad = torch.autograd.grad(V.sum(), xt, create_graph=False)[0].detach().cpu().numpy().reshape(-1)
    return (rad * np.sign(grad)).astype(float)

# Optional: torch nominal drift for AdditiveBoxSetDrift instantiation (not required by numpy sim)
def f_nominal_torch(x: torch.Tensor) -> torch.Tensor:
    # x: (N,2) -> (N,2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    dx1 = x2
    dx2 = (g_grav / L) * torch.sin(x1) + (-b * x2) / (m * L**2)
    return torch.stack([dx1, dx2], dim=1)

# ----------------------------
# small helper to get u
# ----------------------------
def get_u(x1, x2, controller=None):
    if controller is None:
        return np.zeros(2)
    if "torch" in globals() and hasattr(controller, "forward"):
        with torch.no_grad():
            xt = torch.tensor([[x1, x2]], dtype=torch.float32)
            u_val = controller(xt).detach().numpy().reshape(-2,)
    else:
        u_val = controller(np.array([x1, x2], dtype=float))
    return u_val


"""Stochastic Inverted Pendulum Dynamics"""
def f(x, u):
    """Drift dynamics f(x,u). x = [x1, x2]. u in [-1,1]."""
    u1, u2 = u
    x1, x2 = x
    dx1_dt = x2
    dx2_dt = (g_grav / L) * np.sin(x1) + (- b * x2) / (m * L**2) + u2
    return np.array([dx1_dt, dx2_dt], dtype=float)

def f_parametric(x, u, g_val, L_val, b_val, m_val):
    u1, u2 = u
    x1, x2 = x
    dx1_dt = x2
    dx2_dt = (g_val / L_val) * np.sin(x1) + (-b_val * x2) / (m_val * L_val**2) + u2
    return np.array([dx1_dt, dx2_dt], dtype=float)


def g(x):
    """Diffusion vector g(x) for scalar Wiener dW."""
    return np.array([0.0, sigma], dtype=float)


def test_single_traj_run(
    controller=None,
    V_net=None,
    unc_type="none",          # "none" | "additive" | "parametric"
    mode="uniform",           # additive: "none" | "uniform" | "corners" | "worstcase_V"
                              # parametric: "none" | "uniform" | "corners"
    d=np.array([0.0, 0.0], dtype=float),  # bounds on w(t)
    param_ranges=None,        # dict: {"g":(lo,hi),"L":(lo,hi),"b":(lo,hi),"m":(lo,hi)}
    param_time_varying=False, # False: sample once per trajectory (default, physical); True: resample each step
    device="cpu",
    plot_uncertainty=True,
    T=100.0,
    seed=None,
    # NEW:
    n_traj=10,
    pend_idx=0,               # which traj to draw in pendulum + u(t)
    shared_uncertainty=False, # if True: all traj share same w(t); not allowed with worstcase_V
    unc_plot_max=5,
):
    """
    Multi-trajectory SDE rollout + animation, with optional set-valued disturbance:
        dx = (f(x,u) + w(t)) dt + g(x) dW,   w(t) in [-d, d] (box)
    or:
        dx2 = (g/L sin(x1) - b*x2/(m*L^2) + u2) dt + sigma dW,
        (g,L,b,m) sampled within specified intervals.

    - pendulum + u(t): shows trajectory pend_idx only
    - phase plane: shows all trajectories
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed={seed}, n_traj={n_traj}, unc_type={unc_type}, mode={mode}, shared={shared_uncertainty}")

    if shared_uncertainty and (mode == "worstcase_V"):
        raise ValueError("shared_uncertainty=True is incompatible with mode='worstcase_V' (state dependent).")

    if param_ranges is None:
        param_ranges = get_default_param_ranges()
    validate_param_ranges(param_ranges)

    # (optional) instantiate set-valued drift module for consistency with main.py style
    if unc_type == "additive":
        _ = AdditiveBoxSetDrift(f_nominal_torch, torch.tensor(d, dtype=torch.float32)).to(device)
    elif unc_type == "parametric":
        _ = InvertedPendulumSetDrift(
            g_range=param_ranges["g"],
            L_range=param_ranges["L"],
            b_range=param_ranges["b"],
            m_range=param_ranges["m"],
        ).to(device)

    dt = 0.01
    N  = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, N)

    # ----------------------------
    # init (sample each traj from X_init)
    # ----------------------------
    x = np.zeros((n_traj, N, 2), dtype=float)
    for i in range(n_traj):
        x1_0 = rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"])
        x2_0 = rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"])
        x[i, 0] = [x1_0, x2_0]
    print("x0[pend_idx] =", x[pend_idx, 0])

    u_hist = np.zeros((n_traj, N, 2), dtype=float)

    # disturbance history
    w_hist = None
    p_hist = None
    if unc_type == "additive":
        w_hist = np.zeros((n_traj, N, 2), dtype=float)
    elif unc_type == "parametric":
        p_hist = np.zeros((n_traj, N, 4), dtype=float)  # [g, L, b, m]

    # shared uncertainty sequences
    w_shared = None
    p_shared = None
    p_traj = None
    if shared_uncertainty and (unc_type == "additive"):
        w_shared = np.zeros((N, 2), dtype=float)
        for k in range(N):
            if mode == "none":
                w_shared[k] = np.zeros(2)
            elif mode == "uniform":
                w_shared[k] = sample_uniform_box(rng, -d, d)
            elif mode == "corners":
                w_shared[k] = sample_corner_box(rng, d)
            else:
                raise ValueError(f"Unknown additive mode: {mode}")
    elif shared_uncertainty and (unc_type == "parametric"):
        if param_time_varying:
            p_shared = np.zeros((N, 4), dtype=float)
            for k in range(N):
                p_shared[k] = sample_param_set(rng, mode, param_ranges)
        else:
            p_shared = sample_param_set(rng, mode, param_ranges)
    elif (unc_type == "parametric") and (not shared_uncertainty) and (not param_time_varying):
        p_traj = np.zeros((n_traj, 4), dtype=float)
        for i in range(n_traj):
            p_traj[i] = sample_param_set(rng, mode, param_ranges)

    # ----------------------------
    # simulate (Euler–Maruyama)
    # ----------------------------
    for k in range(N - 1):
        # independent Brownian per trajectory
        dW_all = np.sqrt(dt) * rng.standard_normal(size=(n_traj,))

        for i in range(n_traj):
            x_curr = x[i, k].copy()
            x1, x2 = x_curr

            u = get_u(x1, x2, controller=controller)
            u_hist[i, k] = u

            drift = f(x_curr, u)

            if unc_type == "additive":
                if shared_uncertainty:
                    w = w_shared[k].copy()
                else:
                    if mode == "none":
                        w = np.zeros(2)
                    elif mode == "uniform":
                        w = sample_uniform_box(rng, -d, d)
                    elif mode == "corners":
                        w = sample_corner_box(rng, d)
                    elif mode == "worstcase_V":
                        if V_net is None:
                            raise ValueError("mode='worstcase_V' requires V_net.")
                        w = worstcase_wrt_V_box(V_net, x_curr, d, device=device)
                    else:
                        raise ValueError(f"Unknown additive mode: {mode}")

                w_hist[i, k] = w
                drift = drift + w
            elif unc_type == "parametric":
                if mode == "worstcase_V":
                    raise ValueError("mode='worstcase_V' is only supported for additive uncertainty.")
                if shared_uncertainty:
                    if param_time_varying:
                        gk, Lk, bk, mk = p_shared[k]
                    else:
                        gk, Lk, bk, mk = p_shared
                else:
                    if param_time_varying:
                        gk, Lk, bk, mk = sample_param_set(rng, mode, param_ranges)
                    else:
                        gk, Lk, bk, mk = p_traj[i]
                p_hist[i, k] = [gk, Lk, bk, mk]
                drift = f_parametric(x_curr, u, gk, Lk, bk, mk)

            diff = g(x_curr)          # (2,)
            dW = dW_all[i]            # scalar
            x_next = x_curr + drift * dt + diff * dW

            # wrap angle to [-2pi, 2pi]
            if x_next[0] > 2*pi:
                x_next[0] -= 4*pi
            elif x_next[0] < -2*pi:
                x_next[0] += 4*pi

            x[i, k + 1] = x_next

    u_hist[:, -1] = u_hist[:, -2]
    if w_hist is not None:
        if shared_uncertainty:
            w_hist[:, :, :] = w_shared[None, :, :]
        else:
            w_hist[:, -1] = w_hist[:, -2]
    if p_hist is not None:
        p_hist[:, -1] = p_hist[:, -2]

    # torque history for ALL trajectories
    if controller is not None:
        torque_scale = (M / (m * L**2))
        u_torque_hist_all = u_hist[:, :, 1] / torque_scale   # (n_traj, N)
    else:
        u_torque_hist_all = np.zeros((n_traj, N), dtype=float)


    # ----------------------------
    # Animation
    # ----------------------------
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])

    ax_pend  = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_u     = fig.add_subplot(gs[1, :])

    ax_pend.set_aspect("equal", adjustable="box")
    L_plot = L
    if unc_type == "parametric":
        L_plot = max(L_plot, float(param_ranges["L"][1]))
    ax_pend.set_xlim(-L_plot*1.4, L_plot*1.4)
    ax_pend.set_ylim(-L_plot*1.4, L_plot*1.4)
    title = "Inverted pendulum SDE (u=0)" if controller is None else "Inverted pendulum SDE (controlled)"
    if unc_type == "additive":
        title += f" + additive({mode})"
        if shared_uncertainty:
            title += " [shared]"
    elif unc_type == "parametric":
        title += f" + parametric({mode})"
        if shared_uncertainty:
            title += " [shared]"
    title += f"  (n_traj={n_traj})"
    ax_pend.set_title(title)
    ax_pend.set_xticks([])
    ax_pend.set_yticks([])

    pivot, = ax_pend.plot([0], [0], marker="o")
    rods = [ax_pend.plot([], [], lw=2)[0] for _ in range(n_traj)]
    bobs = [ax_pend.plot([], [], marker="o", markersize=6, linestyle="None")[0] for _ in range(n_traj)]

    time_text = ax_pend.text(0.02, 0.95, "", transform=ax_pend.transAxes)
    u_text = ax_pend.text(0.02, 0.88, "", transform=ax_pend.transAxes) if controller is not None else None

    # Phase axis setup (same as yours)
    ax_phase.set_xlim(X_bounds["x1_min"], X_bounds["x1_max"])
    ax_phase.set_ylim(X_bounds["x2_min"], X_bounds["x2_max"])
    ax_phase.set_xlabel(r"$x_1$ (angle)")
    ax_phase.set_ylabel(r"$x_2$ (angular velocity)")
    ax_phase.set_title("Phase plane with specifications")

    ax_phase.add_patch(Rectangle(
        (X_bounds["x1_min"], X_bounds["x2_min"]),
        X_bounds["x1_max"] - X_bounds["x1_min"],
        X_bounds["x2_max"] - X_bounds["x2_min"],
        fill=False, lw=1.5
    ))
    ax_phase.text(X_bounds["x1_min"]+0.1, X_bounds["x2_max"]-1.5, r"$X$")

    ax_phase.add_patch(Rectangle(
        (X_init_bounds["x1_min"], X_init_bounds["x2_min"]),
        X_init_bounds["x1_max"] - X_init_bounds["x1_min"],
        X_init_bounds["x2_max"] - X_init_bounds["x2_min"],
        alpha=0.18, linestyle="--", lw=1.5
    ))
    ax_phase.text(X_init_bounds["x1_min"]+0.1, X_init_bounds["x2_max"]-0.5, r"$X_{\mathrm{init}}$")

    ax_phase.add_patch(Rectangle(
        (X_goal_bounds["x1_min"], X_goal_bounds["x2_min"]),
        X_goal_bounds["x1_max"] - X_goal_bounds["x1_min"],
        X_goal_bounds["x2_max"] - X_goal_bounds["x2_min"],
        alpha=0.20, color="green"
    ))
    ax_phase.text(X_goal_bounds["x1_min"]+0.1, X_goal_bounds["x2_max"]-0.7, r"$X_{\mathrm{goal}}$")

    ax_phase.add_patch(Rectangle(
        (X_unsafe_1["x1_min"], X_unsafe_1["x2_min"]),
        X_unsafe_1["x1_max"] - X_unsafe_1["x1_min"],
        X_unsafe_1["x2_max"] - X_unsafe_1["x2_min"],
        alpha=0.25, color="red"
    ))
    ax_phase.add_patch(Rectangle(
        (X_unsafe_2["x1_min"], X_unsafe_2["x2_min"]),
        X_unsafe_2["x1_max"] - X_unsafe_2["x1_min"],
        X_unsafe_2["x2_max"] - X_unsafe_2["x2_min"],
        alpha=0.25, color="red"
    ))
    ax_phase.text(X_unsafe_1["x1_min"]+0.1, X_unsafe_1["x1_max"]*0 + X_unsafe_1["x2_max"]-0.7, r"$X_{\mathrm{unsafe}}$")
    ax_phase.text(X_unsafe_2["x1_min"]+0.1, X_unsafe_2["x2_max"]-0.7, r"$X_{\mathrm{unsafe}}$")

    status_text = ax_phase.text(0.02, 0.95, "", transform=ax_phase.transAxes)

    # phase trajectories (multi)
    traj_lines = [ax_phase.plot([], [], lw=1.0)[0] for _ in range(n_traj)]
    points = [ax_phase.plot([], [], marker="o", markersize=3, linestyle="None")[0] for _ in range(n_traj)]

    # u(t) subplot (rep traj only)
    ax_u.set_xlim(0.0, T)
    ax_u.set_xlabel("time [s]")
    ax_u.set_ylabel("u_torque")
    if controller is not None:
        u_min = float(u_torque_hist_all.min())
        u_max = float(u_torque_hist_all.max())
        if np.isclose(u_min, u_max):
            margin = max(1.0, abs(u_min) * 0.2)
            u_min -= margin
            u_max += margin
        else:
            margin = 0.1 * (u_max - u_min)
            u_min -= margin
            u_max += margin
    else:
        u_min, u_max = -1.0, 1.0
    ax_u.set_ylim(u_min, u_max)
    ax_u.set_title("Control torque vs time (all trajs)")

    u_lines   = [ax_u.plot([], [], lw=1.0)[0] for _ in range(n_traj)]
    u_markers = [ax_u.plot([], [], marker="o", markersize=3, linestyle="None")[0] for _ in range(n_traj)]

    def init_anim():
        for r, b_ in zip(rods, bobs):
            r.set_data([], [])
            b_.set_data([], [])
        for ln, pt in zip(traj_lines, points):
            ln.set_data([], [])
            pt.set_data([], [])
        time_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        status_text.set_text("")
        for ln, mk in zip(u_lines, u_markers):
            ln.set_data([], [])
            mk.set_data([], [])

        artists = (
            rods + bobs +
            [time_text, status_text] +
            u_lines + u_markers +
            traj_lines + points
        )
        if u_text is not None:
            artists.append(u_text)
        return artists

    def update(frame):
        # draw ALL pendulums
        for i in range(n_traj):
            x1i = x[i, frame, 0]
            Li = float(p_hist[i, frame, 1]) if p_hist is not None else L
            px = Li * np.sin(x1i)
            py = Li * np.cos(x1i)
            rods[i].set_data([0, px], [0, py])
            bobs[i].set_data([px], [py])

        # update all phase trajectories
        n_goal = 0
        n_unsafe = 0
        for i in range(n_traj):
            x1, x2 = x[i, frame]
            traj_lines[i].set_data(x[i, :frame+1, 0], x[i, :frame+1, 1])
            points[i].set_data([x1], [x2])

            if in_box(x1, x2, X_goal_bounds):
                n_goal += 1
            if in_box(x1, x2, X_unsafe_1) or in_box(x1, x2, X_unsafe_2):
                n_unsafe += 1

        time_text.set_text(f"t = {t_grid[frame]:.2f}s")

        if u_text is not None:
            u_torque = u_torque_hist_all[pend_idx, frame]
            u_text.set_text(f"traj {pend_idx} u_torque = {u_torque:+.3f}")

        status_text.set_text(f"in goal: {n_goal}/{n_traj}   in unsafe: {n_unsafe}/{n_traj}")

        # u(t) rep traj
        # u(t) for ALL trajectories
        t_now = t_grid[frame]
        for i in range(n_traj):
            ui = u_torque_hist_all[i]
            u_lines[i].set_data(t_grid[:frame+1], ui[:frame+1])
            u_markers[i].set_data([t_now], [ui[frame]])

        artists = rods + bobs + [time_text, status_text] + u_lines + u_markers + traj_lines + points
        if u_text is not None:
            artists.append(u_text)
        return artists

    skip = 5
    frames = range(0, N, skip)
    _ = FuncAnimation(fig, update, frames=frames, init_func=init_anim, blit=True, interval=30)
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Uncertainty time history (plot a few trajectories)
    # ----------------------------
    if plot_uncertainty and (w_hist is not None):
        mshow = min(n_traj, unc_plot_max)
        figw, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        for i in range(mshow):
            ax[0].plot(t_grid, w_hist[i, :, 0], alpha=0.7)
            ax[1].plot(t_grid, w_hist[i, :, 1], alpha=0.7)
        ax[0].axhline(+d[0], linestyle="--")
        ax[0].axhline(-d[0], linestyle="--")
        ax[1].axhline(+d[1], linestyle="--")
        ax[1].axhline(-d[1], linestyle="--")
        ax[0].set_ylabel("w1(t)")
        ax[1].set_ylabel("w2(t)")
        ax[1].set_xlabel("time (s)")
        ax[0].grid(True)
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()
    if plot_uncertainty and (p_hist is not None):
        mshow = min(n_traj, unc_plot_max)
        figp, axp = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        labels = ("g(t)", "L(t)", "b(t)", "m(t)")
        keys = ("g", "L", "b", "m")
        for i in range(mshow):
            for j in range(4):
                axp[j].plot(t_grid, p_hist[i, :, j], alpha=0.7)
        for j, key in enumerate(keys):
            lo = float(param_ranges[key][0])
            hi = float(param_ranges[key][1])
            axp[j].axhline(lo, linestyle="--")
            axp[j].axhline(hi, linestyle="--")
            axp[j].set_ylabel(labels[j])
            axp[j].grid(True)
        axp[-1].set_xlabel("time (s)")
        plt.tight_layout()
        plt.show()


def estimate_reach_avoid_mc(
    controller=None,
    V_net=None,
    # NEW:
    unc_type="none",          # "none" | "additive" | "parametric"
    mode="uniform",           # additive: "none" | "uniform" | "corners" | "worstcase_V"
                              # parametric: "none" | "uniform" | "corners"
    d=np.array([0.0, 0.0], dtype=float),
    param_ranges=None,
    param_time_varying=False, # False: sample once per trajectory/path; True: resample each step
    device="cpu",
    # existing:
    n_mc=2000,
    T_mc=4.0,
    dt_mc=0.005,
    seed_mc=123,
    return_example_paths=False,
    n_example_paths=5,
):
    if seed_mc is None:
        seed_mc = int(np.random.SeedSequence().entropy)
    rng_mc = np.random.default_rng(seed_mc)
    print(f"[estimate_reach_avoid_mc] seed={seed_mc}, unc_type={unc_type}, mode={mode}")

    if param_ranges is None:
        param_ranges = get_default_param_ranges()
    validate_param_ranges(param_ranges)

    if unc_type == "additive":
        _ = AdditiveBoxSetDrift(f_nominal_torch, torch.tensor(d, dtype=torch.float32)).to(device)
    elif unc_type == "parametric":
        _ = InvertedPendulumSetDrift(
            g_range=param_ranges["g"],
            L_range=param_ranges["L"],
            b_range=param_ranges["b"],
            m_range=param_ranges["m"],
        ).to(device)

    N_mc = int(T_mc / dt_mc) + 1
    success = 0
    fail = 0
    timeout = 0
    example_paths = []

    for _ in range(n_mc):
        x1 = rng_mc.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"])
        x2 = rng_mc.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"])
        p_path = None
        if unc_type == "parametric" and (not param_time_varying):
            p_path = sample_param_set(rng_mc, mode, param_ranges)

        if return_example_paths and len(example_paths) < n_example_paths:
            path = np.zeros((N_mc, 2), dtype=float)
            path[0] = [x1, x2]

        outcome_recorded = False

        for k in range(N_mc - 1):
            in_goal = in_box(x1, x2, X_goal_bounds)
            in_unsafe = in_box(x1, x2, X_unsafe_1) or in_box(x1, x2, X_unsafe_2)

            if in_unsafe:
                fail += 1
                outcome_recorded = True
                if return_example_paths and len(example_paths) < n_example_paths:
                    example_paths.append(path[:k+1].copy())
                break

            if in_goal:
                success += 1
                outcome_recorded = True
                if return_example_paths and len(example_paths) < n_example_paths:
                    example_paths.append(path[:k+1].copy())
                break

            u = get_u(x1, x2, controller)
            x_curr = np.array([x1, x2], dtype=float)

            drift = f(x_curr, u)

            # NEW: additive bounded disturbance
            if unc_type == "additive":
                if mode == "none":
                    w = np.zeros(2)
                elif mode == "uniform":
                    w = sample_uniform_box(rng_mc, -d, d)
                elif mode == "corners":
                    w = sample_corner_box(rng_mc, d)
                elif mode == "worstcase_V":
                    if V_net is None:
                        raise ValueError("mode='worstcase_V' requires V_net.")
                    w = worstcase_wrt_V_box(V_net, x_curr, d, device=device)
                else:
                    raise ValueError(f"Unknown additive mode: {mode}")
                drift = drift + w
            elif unc_type == "parametric":
                if mode == "worstcase_V":
                    raise ValueError("mode='worstcase_V' is only supported for additive uncertainty.")
                if param_time_varying:
                    gk, Lk, bk, mk = sample_param_set(rng_mc, mode, param_ranges)
                else:
                    gk, Lk, bk, mk = p_path
                drift = f_parametric(x_curr, u, gk, Lk, bk, mk)

            diff = g(x_curr)
            dW = np.sqrt(dt_mc) * rng_mc.standard_normal()

            x_next = x_curr + drift * dt_mc + diff * dW
            x1, x2 = x_next

            if x1 > 2*pi:
                x1 -= 4*pi
            elif x1 < -2*pi:
                x1 += 4*pi

            if return_example_paths and len(example_paths) < n_example_paths:
                path[k+1] = [x1, x2]

        if not outcome_recorded:
            timeout += 1
            if return_example_paths and len(example_paths) < n_example_paths:
                example_paths.append(path.copy())

    p_hat = success / n_mc
    stats = {
        "n_mc": n_mc,
        "success": success,
        "fail": fail,
        "timeout": timeout,
        "p_hat": p_hat,
        "success_rate": success / n_mc,
        "fail_rate": fail / n_mc,
        "timeout_rate": timeout / n_mc,
    }
    return (p_hat, stats, example_paths) if return_example_paths else (p_hat, stats)


def test_mc(controller=None):
    p_hat, stats = estimate_reach_avoid_mc(
        controller=controller,
        n_mc=100,
        T_mc=8.0,
        dt_mc=0.005,
        seed_mc=0
    )
    print("Reach-avoid MC estimate:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def run_rollout_case(title, controller, param_ranges, param_time_varying, T=30.0, seed=1):
    print(f"\n=== {title} ===")
    test_single_traj_run(
        controller=controller,
        unc_type="parametric",
        mode="uniform",
        param_ranges=param_ranges,
        param_time_varying=param_time_varying,
        T=T,
        seed=seed,
        plot_uncertainty=True,
    )


def run_mc_case(title, controller, param_ranges, param_time_varying, n_mc=1000, T_mc=8.0, dt_mc=0.005, seed_mc=1):
    print(f"\n=== {title} ===")
    _, stats = estimate_reach_avoid_mc(
        controller=controller,
        unc_type="parametric",
        mode="uniform",
        param_ranges=param_ranges,
        param_time_varying=param_time_varying,
        n_mc=n_mc,
        T_mc=T_mc,
        dt_mc=dt_mc,
        seed_mc=seed_mc,
    )
    print(stats)


def main():
    control_net = load_control_net(OUTPUT_DIR / "eval_bundle.pth")

    # Keep g, b, m fixed; vary L in [0.4, 0.6].
    param_ranges = {
        "g": (9.81, 9.81),
        "L": (0.40, 0.60),
        "b": (0.10, 0.10),
        "m": (0.15, 0.15),
    }
    run_rollout_case("1) Uncontrolled rollout (parametric, time-constant)", None, param_ranges, False)
    run_rollout_case("2) Controlled rollout (parametric, time-constant)", control_net, param_ranges, False)
    run_rollout_case("3) Controlled rollout (parametric, time-varying)", control_net, param_ranges, True)
    run_mc_case("4) MC estimate (parametric, time-constant)", control_net, param_ranges, False)
    run_mc_case("5) MC estimate (parametric, time-varying)", control_net, param_ranges, True)


if __name__ == "__main__":
    main()    
