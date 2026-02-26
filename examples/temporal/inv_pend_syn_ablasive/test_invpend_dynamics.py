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
sigma = 2.0

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
    "x1_min": -pi/2,  "x1_max": pi/2,
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


# Control Network (from RL pretrained)
# def load_control_net(control_save_path, device="cpu"):
#     # 1) create a fresh network with SAME architecture as when saved
#     rl_policy_net = InvertControlNN().to(device)
#     # 2) load checkpoint
#     ckpt = torch.load(control_save_path, map_location=device)
#     # 3) restore weights
#     rl_policy_net.load_state_dict(ckpt["model_state_dict"])
#     # 4) eval mode for rollout
#     rl_policy_net.eval()
#     return rl_policy_net
# control_net = load_control_net(OUTPUT_DIR / "control_net.pth")


# Control Network
def load_control_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    rl_policy_net = InvertControlNN(hidden_dim=64)
    control_net = WrapperConterlNN(rl_policy_net).to(device)
    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    # 4) eval mode for rollout
    control_net.eval()
    return control_net


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


def g(x):
    """Diffusion vector g(x) for scalar Wiener dW."""
    return np.array([0.0, sigma], dtype=float)


def test_single_traj_run(controller=None, T=10.0, seed=None):
    """
    Single SDE rollout + animation.
    controller:
      - None  -> u(x)=0
      - callable(x_np)->u scalar
      - torch nn.Module mapping R^2->R
    """
    # ----------------------------
    # RNG (fresh seed each call unless user provides one)
    # ----------------------------
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed = {seed}")

    # Simulation horizon
    dt = 0.01
    N  = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, N)

    # ----------------------------
    # Initial condition (sample from X_init)
    # ----------------------------
    x1_0 = rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"])
    x2_0 = rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"])
    x = np.zeros((N, 2), dtype=float)
    x[0] = [x1_0, x2_0]
    print("x0 =", x[0])

    u_hist = np.zeros((N, 2), dtype=float)

    # ----------------------------
    # Euler–Maruyama simulation
    # ----------------------------
    for k in range(N - 1):
        x1, x2 = x[k]
        u = get_u(x1, x2, controller=controller)
        u_hist[k] = u

        drift = f(x[k], u)
        diff  = g(x[k])

        dW = np.sqrt(dt) * rng.standard_normal()

        x_next = x[k] + drift * dt + diff * dW

        # wrap angle to [-2pi, 2pi] for plotting clarity
        if x_next[0] > 2*pi:
            x_next[0] -= 4*pi
        elif x_next[0] < -2*pi:
            x_next[0] += 4*pi

        x[k + 1] = x_next

    u_hist[-1] = u_hist[-2]

    # --- NEW: precompute torque history in physical units ---
    if controller is not None:
        torque_scale = (M / (m * L**2))  # same scale used in u_text
        u_torque_hist = u_hist[:, 1] / torque_scale
    else:
        u_torque_hist = np.zeros(N)

    # ----------------------------
    # Build animation figure
    # ----------------------------
    # --- NEW: use a gridspec: top row has 2 subplots, bottom spans both ---
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])

    ax_pend  = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_u     = fig.add_subplot(gs[1, :])  # bottom, across both columns

    # --- Pendulum axis setup ---
    ax_pend.set_aspect("equal", adjustable="box")
    ax_pend.set_xlim(-L*1.4, L*1.4)
    ax_pend.set_ylim(-L*1.4, L*1.4)
    title = "Inverted pendulum SDE (u=0)" if controller is None else "Inverted pendulum SDE (controlled)"
    ax_pend.set_title(title)
    ax_pend.set_xticks([])
    ax_pend.set_yticks([])

    pivot, = ax_pend.plot([0], [0], marker="o")
    rod, = ax_pend.plot([], [], lw=2)
    bob, = ax_pend.plot([], [], marker="o", markersize=10)
    time_text = ax_pend.text(0.02, 0.95, "", transform=ax_pend.transAxes)
    u_text = ax_pend.text(0.02, 0.88, "", transform=ax_pend.transAxes) if controller is not None else None

    # --- Phase axis setup ---
    ax_phase.set_xlim(X_bounds["x1_min"], X_bounds["x1_max"])
    ax_phase.set_ylim(X_bounds["x2_min"], X_bounds["x2_max"])
    ax_phase.set_xlabel(r"$x_1$ (angle)")
    ax_phase.set_ylabel(r"$x_2$ (angular velocity)")
    ax_phase.set_title("Phase plane with specifications")

    # Draw X (domain)
    ax_phase.add_patch(Rectangle(
        (X_bounds["x1_min"], X_bounds["x2_min"]),
        X_bounds["x1_max"] - X_bounds["x1_min"],
        X_bounds["x2_max"] - X_bounds["x2_min"],
        fill=False, lw=1.5
    ))
    ax_phase.text(X_bounds["x1_min"]+0.1, X_bounds["x2_max"]-1.5, r"$X$")

    # Draw X_init
    ax_phase.add_patch(Rectangle(
        (X_init_bounds["x1_min"], X_init_bounds["x2_min"]),
        X_init_bounds["x1_max"] - X_init_bounds["x1_min"],
        X_init_bounds["x2_max"] - X_init_bounds["x2_min"],
        alpha=0.18, linestyle="--", lw=1.5
    ))
    ax_phase.text(X_init_bounds["x1_min"]+0.1, X_init_bounds["x2_max"]-0.5, r"$X_{\mathrm{init}}$")

    # Draw X_goal
    ax_phase.add_patch(Rectangle(
        (X_goal_bounds["x1_min"], X_goal_bounds["x2_min"]),
        X_goal_bounds["x1_max"] - X_goal_bounds["x1_min"],
        X_goal_bounds["x2_max"] - X_goal_bounds["x2_min"],
        alpha=0.20, color="green"
    ))
    ax_phase.text(X_goal_bounds["x1_min"]+0.1, X_goal_bounds["x2_max"]-0.7, r"$X_{\mathrm{goal}}$")

    # Draw unsafe regions
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
    ax_phase.text(X_unsafe_1["x1_min"]+0.1, X_unsafe_1["x2_max"]-0.7, r"$X_{\mathrm{unsafe}}$")
    ax_phase.text(X_unsafe_2["x1_min"]+0.1, X_unsafe_2["x2_max"]-0.7, r"$X_{\mathrm{unsafe}}$")

    traj_line, = ax_phase.plot([], [], lw=1.5)
    point, = ax_phase.plot([], [], marker="o")
    status_text = ax_phase.text(0.02, 0.95, "", transform=ax_phase.transAxes)

    # --- NEW: u(t) axis setup ---
    ax_u.set_xlim(0.0, T)
    ax_u.set_xlabel("time [s]")
    ax_u.set_ylabel("u_torque")

    if controller is not None:
        u_min = float(u_torque_hist.min())
        u_max = float(u_torque_hist.max())
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
    ax_u.set_title("Control torque vs time")

    u_line,   = ax_u.plot([], [], lw=1.5)
    u_marker, = ax_u.plot([], [], marker="o")

    def init_anim():
        rod.set_data([], [])
        bob.set_data([], [])
        traj_line.set_data([], [])
        point.set_data([], [])
        time_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        status_text.set_text("")
        u_line.set_data([], [])
        u_marker.set_data([], [])

        artists = [rod, bob, traj_line, point, time_text, status_text, u_line, u_marker]
        if u_text is not None:
            artists.append(u_text)
        return artists

    def update(frame):
        x1, x2 = x[frame]

        # geometry (x1=0 upright, positive CCW)
        px = L * np.sin(x1)
        py = L * np.cos(x1)
        rod.set_data([0, px], [0, py])
        bob.set_data([px], [py])

        traj_line.set_data(x[:frame+1, 0], x[:frame+1, 1])
        point.set_data([x1], [x2])

        in_goal = in_box(x1, x2, X_goal_bounds)
        in_unsafe = in_box(x1, x2, X_unsafe_1) or in_box(x1, x2, X_unsafe_2)
        status = "UNSAFE" if in_unsafe else ("GOAL" if in_goal else "OK")

        time_text.set_text(f"t = {t_grid[frame]:.2f}s")
        if u_text is not None:
            u_torque = u_torque_hist[frame]
            u_text.set_text(f"u_torque = {u_torque:+.3f}")

        status_text.set_text(f"status: {status}")

        # --- NEW: update u(t) plot ---
        t_now = t_grid[frame]
        u_line.set_data(t_grid[:frame+1], u_torque_hist[:frame+1])
        u_marker.set_data([t_now], [u_torque_hist[frame]])

        artists = [rod, bob, traj_line, point, time_text, status_text, u_line, u_marker]
        if u_text is not None:
            artists.append(u_text)
        return artists

    skip = 5
    frames = range(0, N, skip)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init_anim, blit=True, interval=30)
    plt.tight_layout()
    plt.show()



def estimate_reach_avoid_mc(
    controller=None,
    n_mc=2000,
    T_mc=4.0,
    dt_mc=0.005,
    seed_mc=123,
    return_example_paths=False,
    n_example_paths=5,
):
    """
    Monte Carlo estimate of reach-avoid probability:
      P( reach X_goal before X_unsafe within horizon T_mc )

    controller:
      - None  -> u(x)=0
      - callable(x_np)->u scalar
      - torch nn.Module mapping R^2->R
    """
    # RNG
    if seed_mc is None:
        seed_mc = int(np.random.SeedSequence().entropy)
    rng_mc = np.random.default_rng(seed_mc)
    print(f"[estimate_reach_avoid_mc] seed = {seed_mc}")

    N_mc = int(T_mc / dt_mc) + 1

    success = 0
    fail = 0
    timeout = 0
    example_paths = []

    for r in range(n_mc):
        # sample initial condition from X_init
        x1 = rng_mc.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"])
        x2 = rng_mc.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"])

        if return_example_paths and len(example_paths) < n_example_paths:
            path = np.zeros((N_mc, 2), dtype=float)
            path[0] = [x1, x2]

        outcome_recorded = False

        for k in range(N_mc - 1):
            # check stopping sets
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

            # Euler–Maruyama step (controlled)
            u = get_u(x1, x2, controller)
            x_curr = np.array([x1, x2], dtype=float)

            drift = f(x_curr, u)
            diff  = g(x_curr)

            dW = np.sqrt(dt_mc) * rng_mc.standard_normal()

            x_next = x_curr + drift * dt_mc + diff * dW
            x1, x2 = x_next

            # wrap angle
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

    if return_example_paths:
        return p_hat, stats, example_paths
    return p_hat, stats


def test_mc(controller=None):
    p_hat, stats = estimate_reach_avoid_mc(
        controller=controller,
        n_mc=100,
        T_mc=10.0,
        dt_mc=0.005,
        seed_mc=0
    )
    print("Reach-avoid MC estimate:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    test_single_traj_run()
    control_net = load_control_net(OUTPUT_DIR / "eval_bundle.pth")
    test_single_traj_run(controller=control_net)
    test_mc(controller=None)
    test_mc(controller=control_net)


if __name__ == "__main__":
    main()    