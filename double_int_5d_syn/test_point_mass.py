"""
Test and visualize 5D point mass with orientation control synthesis
2D position + 2D velocity + heading angle
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
# Init: start with some initial positions, velocities, and heading
init_range = np.array([
    [ 0.3,  0.8],   # x (right side)
    [ 0.3,  0.8],   # y (top side)
    [ 0.2,  0.5],   # vx (small velocity)
    [ 0.2,  0.5],   # vy (small velocity)
    [ 0.2,  0.5],   # θ (small angle)
], dtype=np.float32)

# Goal: equilibrium at origin with zero velocities and heading
goal_range = np.array([
    [-0.5,   0.5],   # x near origin
    [-0.5,   0.5],   # y near origin
    [-1.2,   0.2],   # vx near zero
    [-1.2,   0.2],   # vy near zero
    [-1.2,   0.2],   # θ near zero
], dtype=np.float32)

# Unsafe: far from origin with large velocities/heading
unsafe_range = np.array([
    [ 2.0,  3.0],   # x far right
    [ 2.0,  3.0],   # y far top
    [ 1.2,  1.5],   # any vx
    [ 1.2,  1.5],   # any vy
    [ 1.2,  1.5],   # any θ
], dtype=np.float32)

# Full range: symmetric around origin
full_range = np.array([
    [-3.0,  3.0],   # x
    [-3.0,  3.0],   # y
    [-1.5,  1.5],   # vx
    [-1.5,  1.5],   # vy
    [-1.5,  1.5],   # θ
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
NOISE_DIAG = np.array([0.1, 0.0, 0.1, 0.0, 0.0], dtype=float)


def f_ol_np(x: np.ndarray) -> np.ndarray:
    """
    Open-loop drift for 5D point mass with orientation.
    x: (5,) = [x, y, vx, vy, θ]
    returns: (5,) drift

    Dynamics:
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -0.3*vx (damping)
    dvy/dt = -0.3*vy (damping)
    dθ/dt = -0.5*θ (damping)
    """
    x_pos, y_pos, vx, vy, theta = x
    return np.array([
        vx,           # dx/dt = vx
        vy,           # dy/dt = vy
        -0.3 * vx,    # dvx/dt = -0.3*vx (damping)
        -0.3 * vy,    # dvy/dt = -0.3*vy (damping)
        -0.5 * theta  # dθ/dt = -0.5*θ (damping)
    ], dtype=float)


def f_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Closed-loop drift = open-loop + control.
    u: (3,) = [u_x, u_y, u_θ] accelerations + angular control
    """
    f_open = f_ol_np(x)
    # Control affects velocity and heading derivatives
    f_open[2] += u[0]  # dvx/dt += u_x
    f_open[3] += u[1]  # dvy/dt += u_y
    f_open[4] += u[2]  # dθ/dt += u_θ
    return f_open


def g_diag_np(_x: np.ndarray) -> np.ndarray:
    """Constant diagonal diffusion (5,)"""
    return NOISE_DIAG


# ========================================================================
# CONTROL NETWORK LOADING
# ========================================================================
def load_control_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location=device)

    control_net = InvertControlNN(input_dim=5, hidden_dim=8, output_dim=3)
    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    control_net.eval()
    return control_net


def get_u(x_vec: np.ndarray, controller) -> np.ndarray:
    """
    Get control from controller.
    x_vec: (5,) state
    returns: (3,) control [u_x, u_y, u_θ]
    """
    if controller is None:
        return np.zeros(3, dtype=float)

    if hasattr(controller, "forward"):  # torch module
        with torch.no_grad():
            u = controller(torch.tensor(x_vec[None, :], dtype=torch.float32))
            u = u.detach().cpu().numpy()
    else:  # callable
        u = controller(np.asarray(x_vec, dtype=float))

    u = np.asarray(u, dtype=float).reshape(-1)
    if u.size != 3:
        raise ValueError(f"controller must return shape (3,), got {u.shape}")
    return u


# ========================================================================
# SINGLE TRAJECTORY SIMULATION + ANIMATION
# ========================================================================
def test_single_traj_run(controller=None, T=20.0, seed=None):
    """
    Run single SDE trajectory and animate.
    Shows three subplots:
      - Position space (x vs y)
      - Velocity space (vx vs vy)
      - Heading angle θ vs time
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

    u_hist = np.zeros((N, 3), dtype=float)

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
    # CHECK FINAL OUTCOME (with detailed diagnostics)
    # ========================================================================
    reached_goal = False
    reached_unsafe = False
    goal_time = None
    unsafe_time = None

    # Track when each dimension enters goal bounds
    x_in_goal = []
    y_in_goal = []
    vx_in_goal = []
    vy_in_goal = []
    theta_in_goal = []

    for k in range(N):
        # Check full 5D goal
        if in_box_5d(x[k], goal_range):
            reached_goal = True
            goal_time = t_grid[k]
            break
        if in_box_5d(x[k], unsafe_range):
            reached_unsafe = True
            unsafe_time = t_grid[k]
            break

        # Track individual dimensions
        if goal_range[0,0] <= x[k,0] <= goal_range[0,1]:
            x_in_goal.append((t_grid[k], x[k,0]))
        if goal_range[1,0] <= x[k,1] <= goal_range[1,1]:
            y_in_goal.append((t_grid[k], x[k,1]))
        if goal_range[2,0] <= x[k,2] <= goal_range[2,1]:
            vx_in_goal.append((t_grid[k], x[k,2]))
        if goal_range[3,0] <= x[k,3] <= goal_range[3,1]:
            vy_in_goal.append((t_grid[k], x[k,3]))
        if goal_range[4,0] <= x[k,4] <= goal_range[4,1]:
            theta_in_goal.append((t_grid[k], x[k,4]))

    print("\n" + "="*80)
    print("TRAJECTORY OUTCOME")
    print("="*80)
    if reached_goal:
        print(f"✓ SUCCESS: Reached goal at t={goal_time:.2f}s")
    elif reached_unsafe:
        print(f"✗ FAILURE: Reached unsafe at t={unsafe_time:.2f}s")
    else:
        print(f"⊗ TIMEOUT: Did not reach goal or unsafe within T={T}s")

    print(f"\nFinal state: x={x[-1,0]:.3f}, y={x[-1,1]:.3f}, vx={x[-1,2]:.3f}, vy={x[-1,3]:.3f}, θ={x[-1,4]:.3f}")
    print(f"Goal region: x∈[{goal_range[0,0]:.1f},{goal_range[0,1]:.1f}], y∈[{goal_range[1,0]:.1f},{goal_range[1,1]:.1f}]")
    print(f"             vx∈[{goal_range[2,0]:.1f},{goal_range[2,1]:.1f}], vy∈[{goal_range[3,0]:.1f},{goal_range[3,1]:.1f}], θ∈[{goal_range[4,0]:.1f},{goal_range[4,1]:.1f}]")

    # Show per-dimension diagnostics
    print(f"\nPer-dimension goal satisfaction:")
    print(f"  x in goal bounds: {len(x_in_goal)}/{N} steps ({100*len(x_in_goal)/N:.1f}%)")
    if len(x_in_goal) > 0:
        print(f"     First: t={x_in_goal[0][0]:.2f}s, Last: t={x_in_goal[-1][0]:.2f}s")
    print(f"  y in goal bounds: {len(y_in_goal)}/{N} steps ({100*len(y_in_goal)/N:.1f}%)")
    if len(y_in_goal) > 0:
        print(f"     First: t={y_in_goal[0][0]:.2f}s, Last: t={y_in_goal[-1][0]:.2f}s")
    print(f"  vx in goal bounds: {len(vx_in_goal)}/{N} steps ({100*len(vx_in_goal)/N:.1f}%)")
    if len(vx_in_goal) > 0:
        print(f"     First: t={vx_in_goal[0][0]:.2f}s, Last: t={vx_in_goal[-1][0]:.2f}s")
    print(f"  vy in goal bounds: {len(vy_in_goal)}/{N} steps ({100*len(vy_in_goal)/N:.1f}%)")
    if len(vy_in_goal) > 0:
        print(f"     First: t={vy_in_goal[0][0]:.2f}s, Last: t={vy_in_goal[-1][0]:.2f}s")
    print(f"  θ in goal bounds: {len(theta_in_goal)}/{N} steps ({100*len(theta_in_goal)/N:.1f}%)")
    if len(theta_in_goal) > 0:
        print(f"     First: t={theta_in_goal[0][0]:.2f}s, Last: t={theta_in_goal[-1][0]:.2f}s")

    print("="*80 + "\n")

    # ========================================================================
    # ANIMATION
    # ========================================================================
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_pos = fig.add_subplot(gs[:, 0])  # x vs y (spans both rows, first column)
    ax_vel = fig.add_subplot(gs[0, 1])   # vx vs vy (top right)
    ax_theta = fig.add_subplot(gs[1, 1])  # θ vs time (bottom right)
    ax_states = fig.add_subplot(gs[:, 2])  # All states vs time (spans both rows, last column)

    title = "5D Point Mass (u=0)" if controller is None else "5D Point Mass (controlled)"
    fig.suptitle(title)

    # Text overlays
    time_text = ax_pos.text(0.02, 0.95, "", transform=ax_pos.transAxes, fontsize=10)
    u_text = ax_pos.text(0.02, 0.88, "", transform=ax_pos.transAxes, fontsize=10) if controller is not None else None
    status_text = ax_pos.text(0.02, 0.81, "", transform=ax_pos.transAxes, fontsize=10)

    # Position space (x vs y)
    ax_pos.set_xlim(full_range[0, 0], full_range[0, 1])
    ax_pos.set_ylim(full_range[1, 0], full_range[1, 1])
    ax_pos.set_xlabel(r"$x$ position")
    ax_pos.set_ylabel(r"$y$ position")
    ax_pos.set_title("Position Space (x, y)")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_aspect('equal')

    # Draw regions in position space (projections)
    ax_pos.add_patch(Rectangle((init_range[0,0], init_range[1,0]),
                                init_range[0,1]-init_range[0,0],
                                init_range[1,1]-init_range[1,0],
                                fill=True, alpha=0.2, color='blue', label='Init'))
    ax_pos.add_patch(Rectangle((goal_range[0,0], goal_range[1,0]),
                                goal_range[0,1]-goal_range[0,0],
                                goal_range[1,1]-goal_range[1,0],
                                fill=True, alpha=0.2, color='green', label='Goal'))
    ax_pos.add_patch(Rectangle((unsafe_range[0,0], unsafe_range[1,0]),
                                unsafe_range[0,1]-unsafe_range[0,0],
                                unsafe_range[1,1]-unsafe_range[1,0],
                                fill=True, alpha=0.25, color='red', label='Unsafe'))

    # Velocity space (vx vs vy)
    ax_vel.set_xlim(full_range[2, 0], full_range[2, 1])
    ax_vel.set_ylim(full_range[3, 0], full_range[3, 1])
    ax_vel.set_xlabel(r"$v_x$")
    ax_vel.set_ylabel(r"$v_y$")
    ax_vel.set_title("Velocity Space")
    ax_vel.grid(True, alpha=0.3)
    ax_vel.set_aspect('equal')

    # Draw velocity constraints
    ax_vel.add_patch(Rectangle((init_range[2,0], init_range[3,0]),
                                init_range[2,1]-init_range[2,0],
                                init_range[3,1]-init_range[3,0],
                                fill=True, alpha=0.2, color='blue'))
    ax_vel.add_patch(Rectangle((goal_range[2,0], goal_range[3,0]),
                                goal_range[2,1]-goal_range[2,0],
                                goal_range[3,1]-goal_range[3,0],
                                fill=True, alpha=0.2, color='green'))
    ax_vel.add_patch(Rectangle((unsafe_range[2,0], unsafe_range[3,0]),
                                unsafe_range[2,1]-unsafe_range[2,0],
                                unsafe_range[3,1]-unsafe_range[3,0],
                                fill=True, alpha=0.25, color='red'))

    # Heading angle θ vs time
    ax_theta.set_xlim(0, T)
    ax_theta.set_ylim(full_range[4, 0], full_range[4, 1])
    ax_theta.set_xlabel("Time (s)")
    ax_theta.set_ylabel(r"$\theta$ (heading)")
    ax_theta.set_title("Heading Angle")
    ax_theta.grid(True, alpha=0.3)

    # Draw region bands for heading
    ax_theta.axhspan(init_range[4,0], init_range[4,1], color='blue', alpha=0.2, label='Init')
    ax_theta.axhspan(goal_range[4,0], goal_range[4,1], color='green', alpha=0.2, label='Goal')
    ax_theta.axhspan(unsafe_range[4,0], unsafe_range[4,1], color='red', alpha=0.2, label='Unsafe')
    ax_theta.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax_theta.legend(loc='upper right', fontsize=8)

    # All states vs time
    ax_states.set_xlim(0, T)
    ax_states.set_xlabel("Time (s)")
    ax_states.set_ylabel("State values")
    ax_states.set_title("All States vs Time")
    ax_states.grid(True, alpha=0.3)

    # Trajectory artists
    traj_pos, = ax_pos.plot([], [], 'b-', lw=1.5, alpha=0.6)
    pt_pos, = ax_pos.plot([], [], 'ro', markersize=8)
    # Add heading indicator (arrow)
    heading_arrow = ax_pos.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='red', ec='red')

    traj_vel, = ax_vel.plot([], [], 'b-', lw=1.5, alpha=0.6)
    pt_vel, = ax_vel.plot([], [], 'ro', markersize=8)

    traj_theta, = ax_theta.plot([], [], 'b-', lw=1.5, alpha=0.8)
    pt_theta, = ax_theta.plot([], [], 'ro', markersize=6)

    # State trajectories
    traj_x, = ax_states.plot([], [], 'r-', lw=1, alpha=0.7, label='x')
    traj_y, = ax_states.plot([], [], 'g-', lw=1, alpha=0.7, label='y')
    traj_vx, = ax_states.plot([], [], 'b-', lw=1, alpha=0.7, label='vx')
    traj_vy, = ax_states.plot([], [], 'c-', lw=1, alpha=0.7, label='vy')
    traj_theta_states, = ax_states.plot([], [], 'm-', lw=1, alpha=0.7, label='θ')
    ax_states.legend(loc='upper right', fontsize=8)

    ax_pos.legend(loc='upper right')

    def init_anim():
        traj_pos.set_data([], [])
        pt_pos.set_data([], [])
        traj_vel.set_data([], [])
        pt_vel.set_data([], [])
        traj_theta.set_data([], [])
        pt_theta.set_data([], [])
        traj_x.set_data([], [])
        traj_y.set_data([], [])
        traj_vx.set_data([], [])
        traj_vy.set_data([], [])
        traj_theta_states.set_data([], [])
        time_text.set_text("")
        status_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        artists = [traj_pos, pt_pos, traj_vel, pt_vel, traj_theta, pt_theta,
                   traj_x, traj_y, traj_vx, traj_vy, traj_theta_states,
                   time_text, status_text]
        if u_text is not None:
            artists.append(u_text)
        return artists

    def update(frame):
        x_pos, y_pos, vx, vy, theta = x[frame]

        traj_pos.set_data(x[:frame+1, 0], x[:frame+1, 1])
        pt_pos.set_data([x_pos], [y_pos])

        traj_vel.set_data(x[:frame+1, 2], x[:frame+1, 3])
        pt_vel.set_data([vx], [vy])

        traj_theta.set_data(t_grid[:frame+1], x[:frame+1, 4])
        pt_theta.set_data([t_grid[frame]], [theta])

        # State trajectories
        traj_x.set_data(t_grid[:frame+1], x[:frame+1, 0])
        traj_y.set_data(t_grid[:frame+1], x[:frame+1, 1])
        traj_vx.set_data(t_grid[:frame+1], x[:frame+1, 2])
        traj_vy.set_data(t_grid[:frame+1], x[:frame+1, 3])
        traj_theta_states.set_data(t_grid[:frame+1], x[:frame+1, 4])

        x_vec = np.array([x_pos, y_pos, vx, vy, theta], dtype=float)
        in_goal = in_box_5d(x_vec, goal_range)
        in_unsafe = in_box_5d(x_vec, unsafe_range)

        status = "UNSAFE" if in_unsafe else ("GOAL" if in_goal else "OK")

        time_text.set_text(f"t = {t_grid[frame]:.2f}s")
        status_text.set_text(f"status: {status}")

        if u_text is not None:
            u_vec = u_hist[frame]
            u_text.set_text(f"u = [{u_vec[0]:.2f}, {u_vec[1]:.2f}, {u_vec[2]:.2f}]")

        artists = [traj_pos, pt_pos, traj_vel, pt_vel, traj_theta, pt_theta,
                   traj_x, traj_y, traj_vx, traj_vy, traj_theta_states,
                   time_text, status_text]
        if u_text is not None:
            artists.append(u_text)
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
    print("Reach-avoid MC estimate (5D):")
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
