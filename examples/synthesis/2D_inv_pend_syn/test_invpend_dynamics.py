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
from src.network import create_V
from src.hyperparameters import Hyperparameters

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
    "x1_min": (3/4)*pi, "x1_max": (5/4)*pi,
    "x2_min": -1.0,     "x2_max": 1.0
}

X_goal_bounds = {
    "x1_min": -0.4*pi,  "x1_max": 0.4*pi,
    "x2_min": -4.0,     "x2_max": 4.0
}

# Unsafe regions - matching main.py
X_unsafe_down1 = {
    "x1_min": -2*pi,        "x1_max": -2*pi+0.5*pi,
    "x2_min": -20.0,        "x2_max": -10.0
}
X_unsafe_down2 = {
    "x1_min":  2*pi-0.5*pi, "x1_max":  2*pi,
    "x2_min":  10.0,        "x2_max":  20.0
}
X_unsafe_lb = {
    "x1_min": -2*pi,        "x1_max": -2*pi+0.5,
    "x2_min": -20.0,        "x2_max": 20.0
}
X_unsafe_rb = {
    "x1_min":  2*pi-0.5,    "x1_max":  2*pi,
    "x2_min": -20.0,        "x2_max": 20.0
}
X_unsafe_tb = {
    "x1_min": -2*pi,        "x1_max":  2*pi,
    "x2_min":  20.0-0.5,    "x2_max":  20.0
}
X_unsafe_bb = {
    "x1_min": -2*pi,        "x1_max":  2*pi,
    "x2_min": -20.0,        "x2_max": -20.0+0.5
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


# Control Network and Value Function
def load_control_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    rl_policy_net = InvertControlNN(input_dim=2, hidden_dim=8, output_dim=1)
    control_net = WrapperConterlNN(rl_policy_net).to(device)
    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    # eval mode for rollout
    control_net.eval()
    return control_net, bundle

def load_V_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    params = Hyperparameters.from_dict(bundle["hyperparameters"])
    V_net = create_V(params.network).to(device)
    V_net.load_state_dict(bundle["V_state_dict"])
    V_net.eval()
    return V_net


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


def test_single_traj_run(controller=None, T=100.0, seed=None, V_net=None, n_trajectories=5):
    """
    Multiple SDE rollouts + animation.
    controller:
      - None  -> u(x)=0
      - callable(x_np)->u scalar
      - torch nn.Module mapping R^2->R
    V_net: optional value function network for visualization
    n_trajectories: number of trajectories to simulate
    """
    # ----------------------------
    # RNG (fresh seed each call unless user provides one)
    # ----------------------------
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed = {seed}, simulating {n_trajectories} trajectories")

    # Simulation horizon
    dt = 0.01
    N_max = int(T / dt) + 1

    # Time to continue after reaching goal (in seconds)
    post_goal_time = 3.0
    post_goal_steps = int(post_goal_time / dt)

    # Simulate multiple trajectories
    all_trajectories = []
    all_controls = []
    all_times = []

    print(f"Simulating {n_trajectories} trajectories...")
    for traj_idx in range(n_trajectories):
        # Sample initial condition
        x1_0 = rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"])
        x2_0 = rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"])

        x = np.zeros((N_max, 2), dtype=float)
        x[0] = [x1_0, x2_0]
        u_hist = np.zeros((N_max, 2), dtype=float)

        N = N_max
        goal_reach_step = None

        for k in range(N_max - 1):
            x1, x2 = x[k]

            # Check if goal reached
            if in_box(x1, x2, X_goal_bounds) and goal_reach_step is None:
                goal_reach_step = k
                print(f"Trajectory {traj_idx+1}: Goal reached at t={k*dt:.2f}s")

            # Stop shortly after reaching goal
            if goal_reach_step is not None and k >= goal_reach_step + post_goal_steps:
                N = k + 1
                x = x[:N]
                u_hist = u_hist[:N]
                print(f"Trajectory {traj_idx+1}: Stopped at t={(N-1)*dt:.2f}s (0.5s after goal)")
                break

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

        u_hist[-1] = u_hist[-2] if len(u_hist) > 1 else u_hist[0]
        t_traj = np.linspace(0.0, (N-1)*dt, N)

        all_trajectories.append(x)
        all_controls.append(u_hist)
        all_times.append(t_traj)

    # Use longest trajectory for animation
    max_len = max(len(traj) for traj in all_trajectories)
    t_grid = np.linspace(0.0, (max_len-1)*dt, max_len)
    N = max_len
    T = t_grid[-1]

    # Precompute torque history for all trajectories
    all_torques = []
    if controller is not None:
        torque_scale = (M / (m * L**2))
        for u_hist in all_controls:
            u_torque_hist = u_hist[:, 1] / torque_scale
            all_torques.append(u_torque_hist)
    else:
        for u_hist in all_controls:
            all_torques.append(np.zeros(len(u_hist)))

    # Generate colors for trajectories
    traj_colors = plt.cm.tab10(np.linspace(0, 1, n_trajectories))

    # --- Precompute V(x) contour data if V_net provided ---
    V_contour_data = None
    if V_net is not None:
        # Match resolution from src/visualization.py
        resolution = 100
        x1_grid = np.linspace(X_bounds["x1_min"], X_bounds["x1_max"], resolution)
        x2_grid = np.linspace(X_bounds["x2_min"], X_bounds["x2_max"], resolution)
        X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
        grid_pts = np.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=1)

        with torch.no_grad():
            V_net.eval()
            grid_tensor = torch.tensor(grid_pts, dtype=torch.float32)
            V_vals = V_net(grid_tensor).detach().numpy().reshape(X1_mesh.shape)

        V_contour_data = (X1_mesh, X2_mesh, V_vals)

    # ----------------------------
    # Build animation figure
    # ----------------------------
    # --- NEW: use a gridspec: top row has 2 subplots, bottom spans both ---
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], width_ratios=[1.0, 1.5], hspace=0.3, wspace=0.3)

    ax_pend  = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_u     = fig.add_subplot(gs[1, :])  # bottom, across both columns

    # --- Pendulum axis setup ---
    ax_pend.set_aspect("equal", adjustable="box")
    ax_pend.set_xlim(-L*1.4, L*1.4)
    ax_pend.set_ylim(-L*1.4, L*1.4)
    title = "Physical System" if controller is None else "Physical System (controlled)"
    ax_pend.set_title(title, fontsize=11)
    ax_pend.set_xticks([])
    ax_pend.set_yticks([])

    # Add radial grid background
    # Concentric circles at different radii extending to plot bounds
    max_radius = L * 1.4
    for radius_frac in [0.33, 0.67, 1.0, 1.4]:
        circle = plt.Circle((0, 0), L * radius_frac, fill=False, color='lightgray',
                           linestyle='--', linewidth=0.5, alpha=0.3, zorder=1)
        ax_pend.add_patch(circle)

    # Radial lines at cardinal and diagonal directions extending to plot bounds
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        x_end = max_radius * np.sin(angle_rad)
        y_end = max_radius * np.cos(angle_rad)
        ax_pend.plot([0, x_end], [0, y_end], color='lightgray', linestyle='--',
                    linewidth=0.5, alpha=0.3, zorder=1)

    pivot, = ax_pend.plot([0], [0], marker="o", color='k', markersize=8, zorder=10)

    # Pendulum rods and bob snapshots with time gradient
    pend_rods = []
    pend_snapshots = []
    n_snapshots = 15  # Number of snapshots to show in gradient
    for i in range(n_trajectories):
        # Current rod for this trajectory
        rod, = ax_pend.plot([], [], lw=2, color=traj_colors[i], alpha=0.6, zorder=5)
        pend_rods.append(rod)

        # Snapshots for gradient effect
        snapshot_list = []
        for j in range(n_snapshots):
            alpha_val = 0.1 + 0.5 * (j / n_snapshots)  # Gradient from 0.1 to 0.6
            size_val = 3 + 8 * (j / n_snapshots)  # Size from 3 to 11
            snap, = ax_pend.plot([], [], marker="o", markersize=size_val, color=traj_colors[i],
                               alpha=alpha_val, zorder=4+j, markeredgecolor='none')
            snapshot_list.append(snap)
        pend_snapshots.append(snapshot_list)

    time_text = ax_pend.text(0.02, 0.95, "", transform=ax_pend.transAxes, fontsize=9)

    # --- Phase axis setup ---
    ax_phase.set_xlim(X_bounds["x1_min"], X_bounds["x1_max"])
    ax_phase.set_ylim(X_bounds["x2_min"], X_bounds["x2_max"])
    ax_phase.set_xlabel(r"$x_1$ (angle)", fontsize=10)
    ax_phase.set_ylabel(r"$x_2$ (angular velocity)", fontsize=10)
    phase_title = "Phase Plane" if V_net is None else "Phase Plane with V(x)"
    ax_phase.set_title(phase_title, fontsize=11)

    # Draw V contour if available - match style from src/visualization.py but dimmer
    if V_contour_data is not None:
        X1_mesh, X2_mesh, V_vals = V_contour_data
        # Use same style as visualize_value_function: 20 levels, viridis, but with alpha for dimness
        contour = ax_phase.contourf(X1_mesh, X2_mesh, V_vals, levels=40, cmap='viridis', alpha=1)

    # Draw regions with colors matching src/visualization.py
    # Draw X_init - seagreen
    ax_phase.add_patch(Rectangle(
        (X_init_bounds["x1_min"], X_init_bounds["x2_min"]),
        X_init_bounds["x1_max"] - X_init_bounds["x1_min"],
        X_init_bounds["x2_max"] - X_init_bounds["x2_min"],
        fill=False, edgecolor='seagreen', lw=3
    ))

    # Draw X_goal - darkgoldenrod
    ax_phase.add_patch(Rectangle(
        (X_goal_bounds["x1_min"], X_goal_bounds["x2_min"]),
        X_goal_bounds["x1_max"] - X_goal_bounds["x1_min"],
        X_goal_bounds["x2_max"] - X_goal_bounds["x2_min"],
        fill=False, edgecolor='darkgoldenrod', lw=3
    ))

    # Draw all unsafe regions - firebrick
    for unsafe_bounds in [X_unsafe_down1, X_unsafe_down2, X_unsafe_lb, X_unsafe_rb, X_unsafe_tb, X_unsafe_bb]:
        ax_phase.add_patch(Rectangle(
            (unsafe_bounds["x1_min"], unsafe_bounds["x2_min"]),
            unsafe_bounds["x1_max"] - unsafe_bounds["x1_min"],
            unsafe_bounds["x2_max"] - unsafe_bounds["x2_min"],
            fill=False, edgecolor='firebrick', lw=3
        ))

    # Trajectory lines and points for each trajectory
    traj_lines = []
    points = []
    for i in range(n_trajectories):
        line, = ax_phase.plot([], [], lw=1.5, color=traj_colors[i], alpha=0.8, zorder=5)
        pt, = ax_phase.plot([], [], marker="o", markersize=6, color=traj_colors[i], zorder=10,
                           markeredgecolor='white', markeredgewidth=1.0)
        traj_lines.append(line)
        points.append(pt)
    status_text = ax_phase.text(0.02, 0.95, "", transform=ax_phase.transAxes, fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- NEW: u(t) axis setup ---
    ax_u.set_xlim(0.0, T)
    ax_u.set_xlabel("Time [s]", fontsize=10)
    ax_u.set_ylabel("Control Torque [N·m]", fontsize=10)
    ax_u.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

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
    ax_u.set_title("Control Input History", fontsize=11)

    # Multiple control lines
    u_lines = []
    for i in range(n_trajectories):
        line, = ax_u.plot([], [], lw=1.5, color=traj_colors[i], alpha=0.7)
        u_lines.append(line)

    def init_anim():
        artists = [pivot, time_text, status_text]
        for rod in pend_rods:
            rod.set_data([], [])
            artists.append(rod)
        for snapshot_list in pend_snapshots:
            for snap in snapshot_list:
                snap.set_data([], [])
                artists.append(snap)
        for line, pt in zip(traj_lines, points):
            line.set_data([], [])
            pt.set_data([], [])
            artists.extend([line, pt])
        for line in u_lines:
            line.set_data([], [])
            artists.append(line)
        return artists

    def update(frame):
        current_time = t_grid[frame]
        time_text.set_text(f"t = {current_time:.2f}s")

        artists = [pivot, time_text, status_text]

        # Update all trajectories
        for traj_idx in range(n_trajectories):
            x_traj = all_trajectories[traj_idx]
            u_torque = all_torques[traj_idx]
            t_traj = all_times[traj_idx]

            # Find index for this trajectory at current time
            if current_time <= t_traj[-1]:
                traj_i = int(current_time / dt)
                traj_i = min(traj_i, len(x_traj) - 1)

                # Current pendulum rod
                x1_curr = x_traj[traj_i, 0]
                pend_x = L * np.sin(x1_curr)
                pend_y = L * np.cos(x1_curr)
                pend_rods[traj_idx].set_data([0, pend_x], [0, pend_y])
                artists.append(pend_rods[traj_idx])

                # Pendulum snapshots with gradient
                n_snapshots = len(pend_snapshots[traj_idx])
                for snap_idx, snap in enumerate(pend_snapshots[traj_idx]):
                    # Calculate which trajectory index this snapshot should show
                    history_i = max(0, traj_i - (n_snapshots - snap_idx - 1) * max(1, traj_i // n_snapshots))
                    x1_snap = x_traj[history_i, 0]
                    snap_x = L * np.sin(x1_snap)
                    snap_y = L * np.cos(x1_snap)
                    snap.set_data([snap_x], [snap_y])
                    artists.append(snap)

                # Phase plane trajectory
                traj_lines[traj_idx].set_data(x_traj[:traj_i+1, 0], x_traj[:traj_i+1, 1])
                points[traj_idx].set_data([x_traj[traj_i, 0]], [x_traj[traj_i, 1]])

                # Control plot
                u_lines[traj_idx].set_data(t_traj[:traj_i+1], u_torque[:traj_i+1])
            else:
                # Trajectory has ended
                # Final pendulum rod
                x1_final = x_traj[-1, 0]
                pend_x = L * np.sin(x1_final)
                pend_y = L * np.cos(x1_final)
                pend_rods[traj_idx].set_data([0, pend_x], [0, pend_y])
                artists.append(pend_rods[traj_idx])

                # Show evenly spaced snapshots
                n_snapshots = len(pend_snapshots[traj_idx])
                traj_len = len(x_traj)
                for snap_idx, snap in enumerate(pend_snapshots[traj_idx]):
                    history_i = int((snap_idx / (n_snapshots - 1)) * (traj_len - 1)) if n_snapshots > 1 else traj_len - 1
                    x1_snap = x_traj[history_i, 0]
                    snap_x = L * np.sin(x1_snap)
                    snap_y = L * np.cos(x1_snap)
                    snap.set_data([snap_x], [snap_y])
                    artists.append(snap)

                traj_lines[traj_idx].set_data(x_traj[:, 0], x_traj[:, 1])
                points[traj_idx].set_data([x_traj[-1, 0]], [x_traj[-1, 1]])

                u_lines[traj_idx].set_data(t_traj, u_torque)

            artists.extend([traj_lines[traj_idx], points[traj_idx], u_lines[traj_idx]])

        return artists

    skip = 5
    frames = range(0, N, skip)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init_anim, blit=True, interval=30)
    plt.tight_layout()

    # Save the final frame as a PDF
    update(N - 1)  # Draw the final frame
    fig.savefig(HERE / "results" / "animation_final_frame.pdf", dpi=300, format='pdf', bbox_inches='tight')
    print(f"Saved final frame to {HERE / 'results' / 'animation_final_frame.pdf'}")

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
            in_unsafe = (in_box(x1, x2, X_unsafe_down1) or in_box(x1, x2, X_unsafe_down2) or
                         in_box(x1, x2, X_unsafe_lb) or in_box(x1, x2, X_unsafe_rb) or
                         in_box(x1, x2, X_unsafe_tb) or in_box(x1, x2, X_unsafe_bb))

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
        T_mc=8.0,
        dt_mc=0.005,
        seed_mc=0
    )
    print("Reach-avoid MC estimate:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def test_comparison_animation(n_trajectories=5, T=10.0, seed=42):
    """
    Side-by-side comparison animation: No controller vs Trained controller.
    Shows n trajectories for each case in a single plot.
    """
    # Load controller and value function
    control_net, _ = load_control_net(OUTPUT_DIR / "eval_bundle.pth")
    V_net = load_V_net(OUTPUT_DIR / "eval_bundle.pth")

    # Set up RNG
    rng = np.random.default_rng(seed)
    dt = 0.01

    print(f"Simulating {n_trajectories} trajectories for each case...")
    print(f"Seed: {seed}")

    # Helper function to simulate trajectories
    def simulate_trajectories(controller, rng_seed):
        rng_local = np.random.default_rng(rng_seed)
        N_max = int(T / dt) + 1
        all_trajectories = []
        all_times = []
        all_reached_goal = []

        # Time to continue after reaching goal (in seconds)
        post_goal_time = 3.0
        post_goal_steps = int(post_goal_time / dt)

        for traj_idx in range(n_trajectories):
            # Sample initial condition
            x1_0 = rng_local.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"])
            x2_0 = rng_local.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"])

            x = np.zeros((N_max, 2), dtype=float)
            x[0] = [x1_0, x2_0]

            N = N_max
            reached_goal = False
            goal_reach_step = None

            for k in range(N_max - 1):
                x1, x2 = x[k]

                # Check if goal reached
                if in_box(x1, x2, X_goal_bounds) and goal_reach_step is None:
                    goal_reach_step = k
                    reached_goal = True

                # Stop shortly after reaching goal
                if goal_reach_step is not None and k >= goal_reach_step + post_goal_steps:
                    N = k + 1
                    x = x[:N]
                    break

                u = get_u(x1, x2, controller=controller)
                drift = f(x[k], u)
                diff = g(x[k])
                dW = np.sqrt(dt) * rng_local.standard_normal()
                x_next = x[k] + drift * dt + diff * dW

                # Wrap angle
                if x_next[0] > 2*pi:
                    x_next[0] -= 4*pi
                elif x_next[0] < -2*pi:
                    x_next[0] += 4*pi

                x[k + 1] = x_next

            t_traj = np.linspace(0.0, (N-1)*dt, N)
            all_trajectories.append(x)
            all_times.append(t_traj)
            all_reached_goal.append(reached_goal)

        return all_trajectories, all_times, all_reached_goal

    # Simulate both cases with same initial conditions (same seed)
    trajs_open, times_open, goals_open = simulate_trajectories(None, seed)
    trajs_closed, times_closed, goals_closed = simulate_trajectories(control_net, seed)

    print(f"Open-loop: {sum(goals_open)}/{n_trajectories} reached goal")
    print(f"Closed-loop: {sum(goals_closed)}/{n_trajectories} reached goal")

    # Use longest trajectory for animation duration
    max_len = max(
        max(len(traj) for traj in trajs_open),
        max(len(traj) for traj in trajs_closed)
    )
    t_grid = np.linspace(0.0, (max_len-1)*dt, max_len)
    N = max_len
    T_final = t_grid[-1]

    # Colors: use distinct colormaps that are visible against dark background
    # cmap_open = plt.cm.binary  # Red shades for uncontrolled
    # cmap_closed = plt.cm.inferno  # Green shades for controlled

    # Generate colors from colormaps (use brighter end of spectrum)
    # colors_open = [cmap_open(0.3 + 0.1 * i) for i in range(n_trajectories)]
    # colors_closed = [cmap_closed(0.3 + 0.1 * i) for i in range(n_trajectories)]
    colors_open = ["black"] * n_trajectories
    colors_closed = ["deeppink"] * n_trajectories

    # Base colors for legend
    color_open_base = "black"
    color_closed_base = "deeppink"

    # Precompute V(x) contour data
    resolution = 100
    x1_grid = np.linspace(X_bounds["x1_min"], X_bounds["x1_max"], resolution)
    x2_grid = np.linspace(X_bounds["x2_min"], X_bounds["x2_max"], resolution)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
    grid_pts = np.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=1)

    with torch.no_grad():
        V_net.eval()
        grid_tensor = torch.tensor(grid_pts, dtype=torch.float32)
        V_vals = V_net(grid_tensor).detach().numpy().reshape(X1_mesh.shape)

    # ----------------------------
    # Build figure: single row layout
    # Left: pendulum, Middle: phase plane, Right: control inputs
    # ----------------------------
    fig = plt.figure(figsize=(26, 5.5))
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'Times New Roman',
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'mathtext.fontset': 'stix'})
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.6, 1.4], hspace=0.3, wspace=0.15)

    ax_pend = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[0, 2])

    # ----------------------------
    # Setup physical pendulum axis
    # ----------------------------
    ax_pend.set_aspect("equal", adjustable="box")
    ax_pend.set_xlim(-L*1.5, L*1.5)
    ax_pend.set_ylim(-L*1.5, L*1.5)
    ax_pend.set_title("Physical System", fontsize=22)
    ax_pend.set_xticks([])
    ax_pend.set_yticks([])

    # Add radial grid background
    # Concentric circles at different radii extending to plot bounds
    max_radius = L * 1.5
    for radius_frac in [0.33, 0.67, 1.0, 1.5]:
        circle = plt.Circle((0, 0), L * radius_frac, fill=False, color='lightgray',
                           linestyle='--', linewidth=1, alpha=1, zorder=1)
        ax_pend.add_patch(circle)

    # Radial lines at cardinal and diagonal directions extending to plot bounds
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        x_end = max_radius * np.sin(angle_rad)
        y_end = max_radius * np.cos(angle_rad)
        ax_pend.plot([0, x_end], [0, y_end], color='lightgray', linestyle='--',
                    linewidth=1, alpha=1, zorder=1)
    pivot, = ax_pend.plot([0], [0], marker="o", color='k', markersize=8, zorder=10)

    # Pendulum rods and snapshots with gradient
    n_snapshots = 20
    pend_rods_open = []
    pend_rods_closed = []
    pend_snapshots_open = []
    pend_snapshots_closed = []

    for i in range(n_trajectories):
        # Open-loop rod
        rod, = ax_pend.plot([], [], lw=2, color=colors_open[i], alpha=0.5, zorder=20)
        pend_rods_open.append(rod)

        # Open-loop snapshots
        snapshot_list = []
        for j in range(n_snapshots):
            alpha_val = 0.1 + 0.4 * (j / n_snapshots)
            size_val = 4 + 6 * (j / n_snapshots)
            snap, = ax_pend.plot([], [], marker="o", markersize=size_val, color=colors_open[i],
                               alpha=alpha_val, zorder=4+j, markeredgecolor='none')
            snapshot_list.append(snap)
        pend_snapshots_open.append(snapshot_list)

        # Closed-loop rod
        rod, = ax_pend.plot([], [], lw=2, color=colors_closed[i], alpha=0.6, zorder=21)
        pend_rods_closed.append(rod)

        # Closed-loop snapshots
        snapshot_list = []
        for j in range(n_snapshots):
            alpha_val = 0.1 + 0.5 * (j / n_snapshots)
            size_val = 4 + 6 * (j / n_snapshots)
            snap, = ax_pend.plot([], [], marker="o", markersize=size_val, color=colors_closed[i],
                               alpha=alpha_val, zorder=4+j, markeredgecolor='none')
            snapshot_list.append(snap)
        pend_snapshots_closed.append(snapshot_list)

    # ----------------------------
    # Setup axes - Phase plane (with both controlled and uncontrolled)
    # ----------------------------
    ax_phase.set_xlim(X_bounds["x1_min"], X_bounds["x1_max"])
    ax_phase.set_ylim(X_bounds["x2_min"], X_bounds["x2_max"])
    ax_phase.set_xlabel(r"$x_1$ (angle)", fontsize=22)
    ax_phase.set_ylabel(r"$x_2$ (angular velocity)", fontsize=22)
    ax_phase.set_title("Phase Plane", fontsize=22)

    # Draw V contour
    contour = ax_phase.contourf(X1_mesh, X2_mesh, V_vals, levels=40, cmap='viridis', alpha=1)
    plt.colorbar(contour, ax=ax_phase, label='V(x)')

    # Draw regions
    ax_phase.add_patch(Rectangle(
        (X_init_bounds["x1_min"], X_init_bounds["x2_min"]),
        X_init_bounds["x1_max"] - X_init_bounds["x1_min"],
        X_init_bounds["x2_max"] - X_init_bounds["x2_min"],
        fill=False, edgecolor='seagreen', lw=3, label = "Initial Region"
    ))
    ax_phase.add_patch(Rectangle(
        (X_goal_bounds["x1_min"], X_goal_bounds["x2_min"]),
        X_goal_bounds["x1_max"] - X_goal_bounds["x1_min"],
        X_goal_bounds["x2_max"] - X_goal_bounds["x2_min"],
        fill=False, edgecolor='darkgoldenrod', lw=3, label = "Goal Region"
    ))
    for unsafe_bounds in [X_unsafe_down1, X_unsafe_down2, X_unsafe_lb, X_unsafe_rb, X_unsafe_tb, X_unsafe_bb]:
        ax_phase.add_patch(Rectangle(
            (unsafe_bounds["x1_min"], unsafe_bounds["x2_min"]),
            unsafe_bounds["x1_max"] - unsafe_bounds["x1_min"],
            unsafe_bounds["x2_max"] - unsafe_bounds["x2_min"],
            fill=False, edgecolor='firebrick', lw=3, label = "Unsafe Region"
        ))

    # Trajectory lines - both open and closed on same plot
    traj_lines_open = []
    points_open = []
    traj_lines_closed = []
    points_closed = []

    # Add open-loop trajectories (thinner, dashed, dimmer)
    for i in range(n_trajectories):
        line, = ax_phase.plot([], [], lw=1.3, color=colors_open[i], alpha=0.6, zorder=4,
                             linestyle='--')
        pt, = ax_phase.plot([], [], marker="o", markersize=4, color=colors_open[i], zorder=9,
                           markeredgecolor='white', markeredgewidth=0.5, alpha=0.7)
        traj_lines_open.append(line)
        points_open.append(pt)

    # Add closed-loop trajectories (thicker, solid, brighter)
    for i in range(n_trajectories):
        line, = ax_phase.plot([], [], lw=1.8, color=colors_closed[i], alpha=0.85, zorder=5)
        pt, = ax_phase.plot([], [], marker="o", markersize=6, color=colors_closed[i], zorder=10,
                           markeredgecolor='white', markeredgewidth=0.5, alpha=0.95)
        traj_lines_closed.append(line)
        points_closed.append(pt)

    # Create legend with base colors
    legend_open = plt.Line2D([0], [0], color=color_open_base, lw=2, linestyle='--', label='Uncontrolled Trajectories')
    legend_closed = plt.Line2D([0], [0], color=color_closed_base, lw=2, label='Controlled Trajectories')
    legend_unsafe = plt.Line2D([0], [0], color='firebrick', lw=2, label='Unsafe Region')
    legend_goal = plt.Line2D([0], [0], color='darkgoldenrod', lw=2, label='Goal Region')
    legend_init = plt.Line2D([0], [0], color='seagreen', lw=2, label='Initial Region')
    ax_phase.legend(handles=[legend_open, legend_closed, legend_unsafe, legend_goal, legend_init], loc='upper left', fontsize=12, framealpha=0.95)

    # ----------------------------
    # Setup control input axis
    # ----------------------------
    # Find longest closed-loop trajectory time
    max_time_closed = max(t_traj[-1] for t_traj in times_closed)

    ax_u.set_xlim(0.0, max_time_closed)
    ax_u.set_xlabel("Time [s]", fontsize=22)
    ax_u.set_ylabel("Norm. Torque", fontsize=22)
    ax_u.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_u.set_title("Control Input History", fontsize=22)

    # Precompute torque history for closed-loop trajectories
    all_torques_closed = []
    torque_scale = (M / (m * L**2))
    for traj_idx in range(n_trajectories):
        # Get control history for this closed-loop trajectory
        # We need to recompute u at each step
        x_traj = trajs_closed[traj_idx]
        u_torque_hist = np.zeros(len(x_traj))
        for k in range(len(x_traj)):
            u = get_u(x_traj[k, 0], x_traj[k, 1], controller=control_net)
            u_torque_hist[k] = u[1] / torque_scale
        all_torques_closed.append(u_torque_hist)

    # Set y-limits based on control range
    if len(all_torques_closed) > 0:
        all_u = np.concatenate(all_torques_closed)
        u_min = float(all_u.min())
        u_max = float(all_u.max())
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

    # Control lines for closed-loop
    u_lines_closed = []
    for i in range(n_trajectories):
        line, = ax_u.plot([], [], lw=1.5, color=colors_closed[i], alpha=0.8, label='Controlled' if i == 0 else None)
        u_lines_closed.append(line)

    # Control lines for open-loop (u=0)
    u_lines_open = []
    for i in range(n_trajectories):
        line, = ax_u.plot([], [], lw=1.5, color=colors_open[i], alpha=0.7, linestyle='--', label='Uncontrolled' if i == 0 else None)
        u_lines_open.append(line)

    # Add legend to control plot
    ax_u.legend(loc='best', framealpha=0.9, fontsize=14)

    # Add zero line for reference
    ax_u.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # Statistics text
    stats_text_phase = ax_phase.text(0.02, 0.02, "", transform=ax_phase.transAxes,
                                     fontsize=10, ha='left', va='bottom',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def init_anim():
        artists = [pivot, stats_text_phase]
        for rod in pend_rods_open + pend_rods_closed:
            rod.set_data([], [])
            artists.append(rod)
        for snapshot_list in pend_snapshots_open + pend_snapshots_closed:
            for snap in snapshot_list:
                snap.set_data([], [])
                artists.append(snap)
        for line, pt in zip(traj_lines_open + traj_lines_closed,
                           points_open + points_closed):
            line.set_data([], [])
            pt.set_data([], [])
            artists.extend([line, pt])
        for line in u_lines_closed + u_lines_open:
            line.set_data([], [])
            artists.append(line)
        return artists

    def update(frame):
        current_time = t_grid[frame]

        artists = [pivot, stats_text_phase]

        # Update pendulum rods and snapshots for open-loop
        for traj_idx in range(n_trajectories):
            x_traj = trajs_open[traj_idx]
            t_traj = times_open[traj_idx]

            if current_time <= t_traj[-1]:
                traj_i = int(current_time / dt)
                traj_i = min(traj_i, len(x_traj) - 1)

                # Current pendulum rod
                x1_curr = x_traj[traj_i, 0]
                pend_x = L * np.sin(x1_curr)
                pend_y = L * np.cos(x1_curr)
                pend_rods_open[traj_idx].set_data([0, pend_x], [0, pend_y])
                artists.append(pend_rods_open[traj_idx])

                # Snapshots
                n_snaps = len(pend_snapshots_open[traj_idx])
                for snap_idx, snap in enumerate(pend_snapshots_open[traj_idx]):
                    history_i = max(0, traj_i - (n_snaps - snap_idx - 1) * max(1, traj_i // n_snaps))
                    x1_snap = x_traj[history_i, 0]
                    snap_x = L * np.sin(x1_snap)
                    snap_y = L * np.cos(x1_snap)
                    snap.set_data([snap_x], [snap_y])
                    artists.append(snap)
            else:
                # Final rod position
                x1_final = x_traj[-1, 0]
                pend_x = L * np.sin(x1_final)
                pend_y = L * np.cos(x1_final)
                pend_rods_open[traj_idx].set_data([0, pend_x], [0, pend_y])
                artists.append(pend_rods_open[traj_idx])

                # Snapshots
                n_snaps = len(pend_snapshots_open[traj_idx])
                traj_len = len(x_traj)
                for snap_idx, snap in enumerate(pend_snapshots_open[traj_idx]):
                    history_i = int((snap_idx / (n_snaps - 1)) * (traj_len - 1)) if n_snaps > 1 else traj_len - 1
                    x1_snap = x_traj[history_i, 0]
                    snap_x = L * np.sin(x1_snap)
                    snap_y = L * np.cos(x1_snap)
                    snap.set_data([snap_x], [snap_y])
                    artists.append(snap)

        # Update pendulum rods and snapshots for closed-loop
        for traj_idx in range(n_trajectories):
            x_traj = trajs_closed[traj_idx]
            t_traj = times_closed[traj_idx]

            if current_time <= t_traj[-1]:
                traj_i = int(current_time / dt)
                traj_i = min(traj_i, len(x_traj) - 1)

                # Current pendulum rod
                x1_curr = x_traj[traj_i, 0]
                pend_x = L * np.sin(x1_curr)
                pend_y = L * np.cos(x1_curr)
                pend_rods_closed[traj_idx].set_data([0, pend_x], [0, pend_y])
                artists.append(pend_rods_closed[traj_idx])

                # Snapshots
                n_snaps = len(pend_snapshots_closed[traj_idx])
                for snap_idx, snap in enumerate(pend_snapshots_closed[traj_idx]):
                    history_i = max(0, traj_i - (n_snaps - snap_idx - 1) * max(1, traj_i // n_snaps))
                    x1_snap = x_traj[history_i, 0]
                    snap_x = L * np.sin(x1_snap)
                    snap_y = L * np.cos(x1_snap)
                    snap.set_data([snap_x], [snap_y])
                    artists.append(snap)
            else:
                # Final rod position
                x1_final = x_traj[-1, 0]
                pend_x = L * np.sin(x1_final)
                pend_y = L * np.cos(x1_final)
                pend_rods_closed[traj_idx].set_data([0, pend_x], [0, pend_y])
                artists.append(pend_rods_closed[traj_idx])

                # Snapshots
                n_snaps = len(pend_snapshots_closed[traj_idx])
                traj_len = len(x_traj)
                for snap_idx, snap in enumerate(pend_snapshots_closed[traj_idx]):
                    history_i = int((snap_idx / (n_snaps - 1)) * (traj_len - 1)) if n_snaps > 1 else traj_len - 1
                    x1_snap = x_traj[history_i, 0]
                    snap_x = L * np.sin(x1_snap)
                    snap_y = L * np.cos(x1_snap)
                    snap.set_data([snap_x], [snap_y])
                    artists.append(snap)

        # Update open-loop trajectories on phase plane
        goals_reached_open = 0
        for traj_idx in range(n_trajectories):
            x_traj = trajs_open[traj_idx]
            t_traj = times_open[traj_idx]

            if current_time <= t_traj[-1]:
                traj_i = int(current_time / dt)
                traj_i = min(traj_i, len(x_traj) - 1)

                # Phase plane
                traj_lines_open[traj_idx].set_data(x_traj[:traj_i+1, 0], x_traj[:traj_i+1, 1])
                points_open[traj_idx].set_data([x_traj[traj_i, 0]], [x_traj[traj_i, 1]])

                # Control plot (zero control for open-loop)
                u_lines_open[traj_idx].set_data(t_traj[:traj_i+1], np.zeros(traj_i+1))

                # Check if in goal
                if in_box(x_traj[traj_i, 0], x_traj[traj_i, 1], X_goal_bounds):
                    goals_reached_open += 1
            else:
                # Trajectory ended
                traj_lines_open[traj_idx].set_data(x_traj[:, 0], x_traj[:, 1])
                points_open[traj_idx].set_data([x_traj[-1, 0]], [x_traj[-1, 1]])

                # Control plot (zero control for open-loop)
                u_lines_open[traj_idx].set_data(t_traj, np.zeros(len(t_traj)))

                if goals_open[traj_idx]:
                    goals_reached_open += 1

            artists.extend([traj_lines_open[traj_idx], points_open[traj_idx], u_lines_open[traj_idx]])

        # Update closed-loop trajectories
        goals_reached_closed = 0
        for traj_idx in range(n_trajectories):
            x_traj = trajs_closed[traj_idx]
            t_traj = times_closed[traj_idx]
            u_torque = all_torques_closed[traj_idx]

            if current_time <= t_traj[-1]:
                traj_i = int(current_time / dt)
                traj_i = min(traj_i, len(x_traj) - 1)

                # Phase plane
                traj_lines_closed[traj_idx].set_data(x_traj[:traj_i+1, 0], x_traj[:traj_i+1, 1])
                points_closed[traj_idx].set_data([x_traj[traj_i, 0]], [x_traj[traj_i, 1]])

                # Control plot
                u_lines_closed[traj_idx].set_data(t_traj[:traj_i+1], u_torque[:traj_i+1])

                # Check if in goal
                if in_box(x_traj[traj_i, 0], x_traj[traj_i, 1], X_goal_bounds):
                    goals_reached_closed += 1
            else:
                # Trajectory ended
                traj_lines_closed[traj_idx].set_data(x_traj[:, 0], x_traj[:, 1])
                points_closed[traj_idx].set_data([x_traj[-1, 0]], [x_traj[-1, 1]])

                # Control plot
                u_lines_closed[traj_idx].set_data(t_traj, u_torque)

                if goals_closed[traj_idx]:
                    goals_reached_closed += 1

            artists.extend([traj_lines_closed[traj_idx], points_closed[traj_idx], u_lines_closed[traj_idx]])

        return artists

    skip = 5
    frames = range(0, N, skip)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init_anim, blit=False, interval=30)

    plt.tight_layout()
    fig.tight_layout()

    # Save final frame
    update(N - 1)
    fig.savefig(HERE / "results" / "comparison_animation_final.pdf", dpi=300, format='pdf', bbox_inches='tight')
    print(f"\nSaved final frame to {HERE / 'results' / 'comparison_animation_final.pdf'}")

    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='comparison',
                       choices=['comparison', 'single', 'mc'],
                       help='Animation mode: comparison (side-by-side), single (detailed with control), mc (Monte Carlo)')
    parser.add_argument('--n_traj', type=int, default=5, help='Number of trajectories')
    parser.add_argument('--T', type=float, default=10.0, help='Time horizon (seconds)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load control network and value function
    control_net, _ = load_control_net(OUTPUT_DIR / "eval_bundle.pth")
    V_net = load_V_net(OUTPUT_DIR / "eval_bundle.pth")

    if args.mode == 'comparison':
        # Run comparison animation (5 trajectories each, side-by-side)
        test_comparison_animation(n_trajectories=args.n_traj, T=args.T, seed=args.seed)
    elif args.mode == 'single':
        # Run single detailed animation with physical system and control inputs
        test_single_traj_run(controller=control_net, V_net=V_net, T=args.T,
                            n_trajectories=args.n_traj, seed=args.seed)
    elif args.mode == 'mc':
        # Monte Carlo estimates
        test_mc(controller=None)
        test_mc(controller=control_net)


if __name__ == "__main__":
    main()    