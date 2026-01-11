"""
Test and visualize 5D acrobot-inspired control synthesis
Double pendulum with movable base

Goal: Swing up and balance at upright position (θ1≈π) - unstable equilibrium!
Init: Start near hanging down (θ1≈0) - stable equilibrium
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from pathlib import Path

# Set up directories
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.save_load_utils import load_eval_bundle

# Import the control network class
import sys
sys.path.insert(0, str(HERE))
from src.control_network import InvertControlNN


# ========================================================================
# REGION DEFINITIONS (matching main.py)
# ========================================================================
# Init: start near hanging down (stable equilibrium)
init_range = np.array([
    [-0.5,  0.5],   # θ1 near 0 (hanging down)
    [-0.5,  0.5],   # θ2 near 0
    [-0.3,  0.3],   # ω1 small
    [-0.3,  0.3],   # ω2 small
    [-0.5,  0.5],   # x_base near origin
], dtype=np.float32)

# Goal: upright position (UNSTABLE equilibrium) - this is the challenge!
# θ1 = π means first link pointing up, θ2 = 0 means second link aligned
goal_range = np.array([
    [ 2.8,  3.14159],   # θ1 near π (upright, unstable!)
    [-0.3,  0.3],       # θ2 near 0 (aligned)
    [-0.2,  0.2],       # ω1 near 0 (balanced)
    [-0.2,  0.2],       # ω2 near 0 (balanced)
    [-0.4,  0.4],       # x_base near origin
], dtype=np.float32)

# Unsafe: horizontal position or base far from origin
unsafe_range = np.array([
    [ 1.2,  1.9],   # θ1 near horizontal (dangerous)
    [-2.0, -1.3],   # θ2 large negative
    [ 1.2,  1.5],   # ω1 too large
    [ 1.2,  1.5],   # ω2 too large
    [ 2.0,  2.5],   # x_base too far
], dtype=np.float32)

# Full range
full_range = np.array([
    [-3.0,  3.0],   # θ1 (±π approximately)
    [-3.0,  3.0],   # θ2
    [-1.5,  1.5],   # ω1
    [-1.5,  1.5],   # ω2
    [-3.0,  3.0],   # x_base
], dtype=np.float32)


# ========================================================================
# HELPERS
# ========================================================================
def in_box_5d(x: np.ndarray, bounds_5d: np.ndarray) -> bool:
    """Check if x (5,) is inside bounds (5,2)"""
    return bool(np.all((x >= bounds_5d[:, 0]) & (x <= bounds_5d[:, 1])))


# ========================================================================
# DYNAMICS (numpy versions)
# ========================================================================
NOISE_DIAG = np.array([0.0, 0.0, 0.1, 0.1, 0.05], dtype=float)

# Physical parameters (matching main.py)
g = 9.8
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
lc1 = 0.5
lc2 = 0.5
I1 = m1 * l1**2 / 3.0
I2 = m2 * l2**2 / 3.0
d1 = 0.5
d2 = 0.5
d_base = 0.3

# For visualization
L1 = l1
L2 = l2


def f_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Closed-loop drift for proper acrobot dynamics.
    x: (5,) = [θ1, θ2, ω1, ω2, x_base]
    u: (2,) = [τ_elbow, F_base]
    returns: (5,) drift
    """
    theta1, theta2, omega1, omega2, x_base = x
    tau_elbow, F_base = u

    # Trigonometric shortcuts
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    s12 = np.sin(theta1 + theta2)
    c2 = np.cos(theta2)

    # Inertia matrix M(q)
    M11 = I1 + I2 + m2 * l1**2 + 2 * m2 * l1 * lc2 * c2
    M12 = I2 + m2 * l1 * lc2 * c2
    M21 = M12
    M22 = I2
    det_M = M11 * M22 - M12 * M21

    # Coriolis matrix C(q,q̇)
    h = -m2 * l1 * lc2 * s2
    C11 = -2 * h * omega2
    C12 = -h * omega2
    C21 = h * omega1
    C22 = 0.0

    # Gravity torques
    tau_g1 = -m1 * g * lc1 * s1 - m2 * g * (l1 * s1 + lc2 * s12)
    tau_g2 = -m2 * g * lc2 * s12

    # Coriolis terms
    C_term1 = C11 * omega1 + C12 * omega2
    C_term2 = C21 * omega1 + C22 * omega2

    # Right-hand side with control
    rhs1 = tau_g1 - C_term1 - d1 * omega1  # no control on shoulder
    rhs2 = tau_g2 - C_term2 - d2 * omega2 + tau_elbow  # control on elbow

    # Solve for accelerations
    alpha1 = (M22 * rhs1 - M12 * rhs2) / det_M
    alpha2 = (-M21 * rhs1 + M11 * rhs2) / det_M

    # Base dynamics
    dx_base = -d_base * x_base + F_base

    return np.array([
        omega1,    # dθ1/dt
        omega2,    # dθ2/dt
        alpha1,    # dω1/dt
        alpha2,    # dω2/dt
        dx_base    # dx_base/dt
    ], dtype=float)


def g_diag_np(_x: np.ndarray) -> np.ndarray:
    """Constant diagonal diffusion (5,)"""
    return NOISE_DIAG


# ========================================================================
# CONTROL NETWORK LOADING
# ========================================================================
def load_control_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location=device)

    control_net = InvertControlNN(input_dim=5, hidden_dim=512, output_dim=2)
    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    control_net.eval()
    return control_net


def get_u(x_vec: np.ndarray, controller) -> np.ndarray:
    """
    Get control from controller.
    x_vec: (5,) state
    returns: (2,) control [τ, F]
    """
    if controller is None:
        return np.zeros(2, dtype=float)

    if hasattr(controller, "forward"):  # torch module
        with torch.no_grad():
            u = controller(torch.tensor(x_vec[None, :], dtype=torch.float32))
            u = u.detach().cpu().numpy()
    else:  # callable
        u = controller(np.asarray(x_vec, dtype=float))

    u = np.asarray(u, dtype=float).reshape(-1)
    if u.size != 2:
        raise ValueError(f"controller must return shape (2,), got {u.shape}")
    return u


# ========================================================================
# PENDULUM KINEMATICS
# ========================================================================
def compute_pendulum_positions(x_base, theta1, theta2):
    """
    Compute positions of pendulum links for visualization.
    θ1 = 0 means hanging down (vertical)

    Returns: (x0, y0), (x1, y1), (x2, y2)
    - (x0, y0): base position
    - (x1, y1): end of link 1
    - (x2, y2): end of link 2
    """
    x0, y0 = x_base, 0.0

    # Link 1: angle measured from downward vertical (positive = clockwise)
    x1 = x0 + L1 * np.sin(theta1)
    y1 = y0 - L1 * np.cos(theta1)

    # Link 2: relative angle to link 1
    total_angle = theta1 + theta2
    x2 = x1 + L2 * np.sin(total_angle)
    y2 = y1 - L2 * np.cos(total_angle)

    return (x0, y0), (x1, y1), (x2, y2)


# ========================================================================
# SINGLE TRAJECTORY SIMULATION + ANIMATION
# ========================================================================
def test_single_traj_run(controller=None, T=20.0, seed=None):
    """
    Run single SDE trajectory and animate.
    Shows pendulum animation + state plots.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed = {seed}")

    dt = 0.01
    N = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, N)

    # Initial condition
    x0 = np.array([
        rng.uniform(init_range[0, 0], init_range[0, 1]),
        rng.uniform(init_range[1, 0], init_range[1, 1]),
        rng.uniform(init_range[2, 0], init_range[2, 1]),
        rng.uniform(init_range[3, 0], init_range[3, 1]),
        rng.uniform(init_range[4, 0], init_range[4, 1]),
    ], dtype=float)

    x = np.zeros((N, 5), dtype=float)
    x[0] = x0
    print("x0 =", x0)

    u_hist = np.zeros((N, 2), dtype=float)

    # Euler-Maruyama
    for k in range(N - 1):
        x_curr = x[k]
        u = get_u(x_curr, controller)
        u_hist[k] = u

        drift = f_np(x_curr, u)
        g_vec = g_diag_np(x_curr)
        dW = np.sqrt(dt) * rng.standard_normal(size=5)
        x[k + 1] = x_curr + drift * dt + g_vec * dW

    u_hist[-1] = u_hist[-2]

    # ========================================================================
    # CHECK FINAL OUTCOME
    # ========================================================================
    reached_goal = False
    reached_unsafe = False
    goal_time = None
    unsafe_time = None

    for k in range(N):
        if in_box_5d(x[k], goal_range):
            reached_goal = True
            goal_time = t_grid[k]
            break
        if in_box_5d(x[k], unsafe_range):
            reached_unsafe = True
            unsafe_time = t_grid[k]
            break

    print("\n" + "="*80)
    print("TRAJECTORY OUTCOME")
    print("="*80)
    if reached_goal:
        print(f"✓ SUCCESS: Reached goal at t={goal_time:.2f}s")
    elif reached_unsafe:
        print(f"✗ FAILURE: Reached unsafe at t={unsafe_time:.2f}s")
    else:
        print(f"⊗ TIMEOUT: Did not reach goal or unsafe within T={T}s")

    print(f"\nFinal state: θ1={x[-1,0]:.3f}, θ2={x[-1,1]:.3f}, ω1={x[-1,2]:.3f}, ω2={x[-1,3]:.3f}, x_base={x[-1,4]:.3f}")
    print("="*80 + "\n")

    # ========================================================================
    # ANIMATION
    # ========================================================================
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    ax_pend = fig.add_subplot(gs[:, 0])  # Pendulum visualization (left, spans both rows)
    ax_angles = fig.add_subplot(gs[0, 1])  # θ1, θ2 vs time
    ax_vels = fig.add_subplot(gs[1, 1])    # ω1, ω2 vs time
    ax_base = fig.add_subplot(gs[0, 2])    # x_base vs time
    ax_control = fig.add_subplot(gs[1, 2]) # control inputs vs time

    title = "5D Acrobot (u=0)" if controller is None else "5D Acrobot (controlled)"
    fig.suptitle(title, fontsize=14)

    # Pendulum visualization
    ax_pend.set_xlim(-4, 4)
    ax_pend.set_ylim(-3, 1)
    ax_pend.set_xlabel("x")
    ax_pend.set_ylabel("y")
    ax_pend.set_title("Double Pendulum (Goal: Upright ↑)")
    ax_pend.set_aspect('equal')
    ax_pend.grid(True, alpha=0.3)
    ax_pend.axhline(0, color='black', linestyle='-', linewidth=0.8)

    # Show goal region (upright zone) with light green shading
    goal_y_top = 0.5
    goal_y_bottom = L1 + L2 - 0.5  # Height when upright
    ax_pend.axhspan(goal_y_bottom, 1, color='green', alpha=0.1, label='Goal zone (upright)')

    # Show init region (hanging down zone) with light blue shading
    init_y_top = -L1 - L2 + 0.5
    init_y_bottom = -L1 - L2 - 0.5
    ax_pend.axhspan(init_y_bottom, init_y_top, color='blue', alpha=0.1, label='Init zone (hanging)')

    # Pendulum artists
    pendulum_line, = ax_pend.plot([], [], 'o-', lw=3, markersize=10, color='blue', label='Pendulum')
    base_pt, = ax_pend.plot([], [], 'ks', markersize=12, label='Base')
    trail, = ax_pend.plot([], [], 'r-', lw=0.5, alpha=0.3, label='Trail')
    time_text_pend = ax_pend.text(0.02, 0.05, "", transform=ax_pend.transAxes, fontsize=10, verticalalignment='bottom')
    status_text = ax_pend.text(0.02, 0.12, "", transform=ax_pend.transAxes, fontsize=10, verticalalignment='bottom')
    ax_pend.legend(loc='upper right', fontsize=8)

    # Angles vs time
    ax_angles.set_xlim(0, T)
    ax_angles.set_ylim(full_range[0, 0], full_range[0, 1])
    ax_angles.set_xlabel("Time (s)")
    ax_angles.set_ylabel("Angle (rad)")
    ax_angles.set_title("Pendulum Angles")
    ax_angles.grid(True, alpha=0.3)
    # Goal: θ1 ≈ π (upright), θ2 ≈ 0 (aligned)
    ax_angles.axhspan(goal_range[0,0], goal_range[0,1], color='green', alpha=0.15, label='θ1 goal (upright)')
    ax_angles.axhspan(goal_range[1,0], goal_range[1,1], color='lightgreen', alpha=0.15, label='θ2 goal (aligned)')
    # Init: θ1 ≈ 0 (hanging), θ2 ≈ 0
    ax_angles.axhspan(init_range[0,0], init_range[0,1], color='blue', alpha=0.1)
    ax_angles.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5, label='θ=0 (hanging)')
    ax_angles.axhline(np.pi, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='θ=π (upright)')

    traj_theta1, = ax_angles.plot([], [], 'b-', lw=1.5, label='θ1')
    traj_theta2, = ax_angles.plot([], [], 'g-', lw=1.5, label='θ2')
    ax_angles.legend(loc='upper right', fontsize=8)

    # Angular velocities vs time
    ax_vels.set_xlim(0, T)
    ax_vels.set_ylim(full_range[2, 0], full_range[2, 1])
    ax_vels.set_xlabel("Time (s)")
    ax_vels.set_ylabel("Angular velocity (rad/s)")
    ax_vels.set_title("Angular Velocities")
    ax_vels.grid(True, alpha=0.3)
    ax_vels.axhspan(goal_range[2,0], goal_range[2,1], color='green', alpha=0.15)
    ax_vels.axhspan(goal_range[3,0], goal_range[3,1], color='lightgreen', alpha=0.15)
    ax_vels.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    traj_omega1, = ax_vels.plot([], [], 'b-', lw=1.5, label='ω1')
    traj_omega2, = ax_vels.plot([], [], 'g-', lw=1.5, label='ω2')
    ax_vels.legend(loc='upper right', fontsize=8)

    # Base position vs time
    ax_base.set_xlim(0, T)
    ax_base.set_ylim(full_range[4, 0], full_range[4, 1])
    ax_base.set_xlabel("Time (s)")
    ax_base.set_ylabel("x_base (m)")
    ax_base.set_title("Base Position")
    ax_base.grid(True, alpha=0.3)
    ax_base.axhspan(goal_range[4,0], goal_range[4,1], color='green', alpha=0.15, label='Goal')
    ax_base.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    traj_base, = ax_base.plot([], [], 'r-', lw=1.5)
    ax_base.legend(loc='upper right', fontsize=8)

    # Control inputs vs time
    if controller is not None:
        ax_control.set_xlim(0, T)
        ax_control.set_xlabel("Time (s)")
        ax_control.set_ylabel("Control input")
        ax_control.set_title("Control Inputs")
        ax_control.grid(True, alpha=0.3)
        ax_control.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

        traj_u_tau, = ax_control.plot([], [], 'b-', lw=1, label='τ (torque)')
        traj_u_F, = ax_control.plot([], [], 'r-', lw=1, label='F (force)')
        ax_control.legend(loc='upper right', fontsize=8)

    # Trail storage for end effector
    trail_x = []
    trail_y = []

    def init_anim():
        pendulum_line.set_data([], [])
        base_pt.set_data([], [])
        trail.set_data([], [])
        traj_theta1.set_data([], [])
        traj_theta2.set_data([], [])
        traj_omega1.set_data([], [])
        traj_omega2.set_data([], [])
        traj_base.set_data([], [])
        time_text_pend.set_text("")
        status_text.set_text("")
        artists = [pendulum_line, base_pt, trail, traj_theta1, traj_theta2,
                   traj_omega1, traj_omega2, traj_base, time_text_pend, status_text]
        if controller is not None:
            traj_u_tau.set_data([], [])
            traj_u_F.set_data([], [])
            artists.extend([traj_u_tau, traj_u_F])
        return artists

    def update(frame):
        theta1, theta2, omega1, omega2, x_base = x[frame]

        # Compute pendulum positions
        (x0, y0), (x1, y1), (x2, y2) = compute_pendulum_positions(x_base, theta1, theta2)

        # Update pendulum visualization
        pendulum_line.set_data([x0, x1, x2], [y0, y1, y2])
        base_pt.set_data([x0], [y0])

        # Update trail
        trail_x.append(x2)
        trail_y.append(y2)
        trail.set_data(trail_x, trail_y)

        # Update time series plots
        traj_theta1.set_data(t_grid[:frame+1], x[:frame+1, 0])
        traj_theta2.set_data(t_grid[:frame+1], x[:frame+1, 1])
        traj_omega1.set_data(t_grid[:frame+1], x[:frame+1, 2])
        traj_omega2.set_data(t_grid[:frame+1], x[:frame+1, 3])
        traj_base.set_data(t_grid[:frame+1], x[:frame+1, 4])

        # Check status
        x_vec = np.array([theta1, theta2, omega1, omega2, x_base], dtype=float)
        in_goal = in_box_5d(x_vec, goal_range)
        in_unsafe = in_box_5d(x_vec, unsafe_range)
        status = "UNSAFE" if in_unsafe else ("GOAL" if in_goal else "OK")

        time_text_pend.set_text(f"t = {t_grid[frame]:.2f}s")
        status_text.set_text(f"status: {status}")

        artists = [pendulum_line, base_pt, trail, traj_theta1, traj_theta2,
                   traj_omega1, traj_omega2, traj_base, time_text_pend, status_text]

        # Update control plot
        if controller is not None:
            traj_u_tau.set_data(t_grid[:frame+1], u_hist[:frame+1, 0])
            traj_u_F.set_data(t_grid[:frame+1], u_hist[:frame+1, 1])
            # Auto-scale control axes
            if frame > 0:
                u_max = max(np.abs(u_hist[:frame+1, 0]).max(), np.abs(u_hist[:frame+1, 1]).max())
                ax_control.set_ylim(-u_max*1.1, u_max*1.1)
            artists.extend([traj_u_tau, traj_u_F])

        return artists

    skip = 5
    frames = range(0, N, skip)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init_anim, blit=True, interval=30)

    plt.tight_layout()
    plt.show()


# ========================================================================
# MONTE CARLO REACH-AVOID ESTIMATION
# ========================================================================
def estimate_reach_avoid_mc(
    controller=None,
    n_mc=1000,
    T_mc=30.0,
    dt_mc=0.01,
    seed_mc=123,
):
    """
    Monte Carlo estimate of reach-avoid probability.
    P(reach goal before unsafe within T_mc)
    """
    if seed_mc is None:
        seed_mc = int(np.random.SeedSequence().entropy)
    rng_mc = np.random.default_rng(seed_mc)
    print(f"[estimate_reach_avoid_mc] seed = {seed_mc}")

    N_mc = int(T_mc / dt_mc) + 1

    success = 0
    fail = 0
    timeout = 0

    for r in range(n_mc):
        x_curr = np.array([
            rng_mc.uniform(init_range[0, 0], init_range[0, 1]),
            rng_mc.uniform(init_range[1, 0], init_range[1, 1]),
            rng_mc.uniform(init_range[2, 0], init_range[2, 1]),
            rng_mc.uniform(init_range[3, 0], init_range[3, 1]),
            rng_mc.uniform(init_range[4, 0], init_range[4, 1]),
        ], dtype=float)

        outcome_recorded = False

        # Check t=0
        if in_box_5d(x_curr, unsafe_range):
            fail += 1
            outcome_recorded = True
        elif in_box_5d(x_curr, goal_range):
            success += 1
            outcome_recorded = True

        for k in range(N_mc - 1):
            if outcome_recorded:
                break

            u = get_u(x_curr, controller)
            drift = f_np(x_curr, u)
            g_vec = g_diag_np(x_curr)
            dW = np.sqrt(dt_mc) * rng_mc.standard_normal(size=5)
            x_next = x_curr + drift * dt_mc + g_vec * dW

            # Check goal/unsafe
            in_goal = in_box_5d(x_next, goal_range)
            in_unsafe = in_box_5d(x_next, unsafe_range)

            if in_unsafe:
                fail += 1
                outcome_recorded = True
            elif in_goal:
                success += 1
                outcome_recorded = True

            x_curr = x_next

        if not outcome_recorded:
            timeout += 1

    p_reach_avoid = success / n_mc
    stats = {
        "n_mc": n_mc,
        "success": success,
        "fail": fail,
        "timeout": timeout,
        "p_reach_avoid": p_reach_avoid,
        "success_rate": success / n_mc,
        "fail_rate": fail / n_mc,
        "timeout_rate": timeout / n_mc,
    }

    return p_reach_avoid, stats


def test_mc(controller=None):
    p_reach_avoid, stats = estimate_reach_avoid_mc(
        controller=controller,
        n_mc=100,
        T_mc=20.0,
        dt_mc=0.01,
        seed_mc=0
    )
    print("Reach-avoid MC estimate (5D Acrobot):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    # Test open-loop (no control)
    print("="*80)
    print("TESTING OPEN-LOOP (u=0)")
    print("="*80)
    test_single_traj_run(controller=None, T=20.0, seed=42)

    # Try to load trained controller
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"
    if bundle_path.exists():
        print("\n" + "="*80)
        print("TESTING CLOSED-LOOP (with trained controller)")
        print("="*80)
        control_net = load_control_net(bundle_path)
        test_single_traj_run(controller=control_net, T=20.0, seed=42)

        # Monte Carlo comparisons
        print("\n" + "="*80)
        print("MONTE CARLO EVALUATION")
        print("="*80)
        print("\nOpen-loop:")
        test_mc(controller=None)

        print("\nClosed-loop:")
        test_mc(controller=control_net)
    else:
        print(f"\nNo trained controller found at {bundle_path}")
        print("Run 'python main.py --train=1' first")


if __name__ == "__main__":
    main()
