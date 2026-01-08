"""
Test and visualize 4D double integrator control synthesis
Two particles swapping positions while avoiding collision
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
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
# from main import DoubleIntegratorControlNN
from src.control_network import LinearControlNN, NonlinearControlNN, InvertControlNN


# ========================================================================
# REGION DEFINITIONS (matching main.py)
# ========================================================================
# Init: particles start with some initial positions and velocities
init_range = np.array([
    [ 0.2,  0.7],   # p1 (right side)
    [ 0.2,  0.5],   # v1 (small velocity)
    [ 0.2,  0.7],   # p2 (right side)
    [ 0.2,  0.5],   # v2 (small velocity)
], dtype=np.float32)

# Goal: equilibrium at origin with zero velocities
# Dynamics naturally drive velocities to zero (v2 has damping, v1 maintained by control)
goal_range = np.array([
    [-0.5,   0.5],   # p1 near origin
    [-1.2,   0.2],   # v1 near zero
    [-0.5,   0.5],   # p2 near origin
    [-1.2,   0.2],   # v2 near zero
], dtype=np.float32)

# Unsafe: far from origin (representing collision or out-of-bounds)
unsafe_range = np.array([
    [ 2.0,  3.0],   # p1 far right
    [ 1.2,  1.5],   # any v1
    [ 2.0,  3.0],   # p2 also far right
    [ 1.2,  1.5],   # any v2
], dtype=np.float32)

# Full range: symmetric around origin
full_range = np.array([
    [-3.0,  3.0],   # p1
    [-1.5,  1.5],   # v1
    [-3.0,  3.0],   # p2
    [-1.5,  1.5],   # v2
], dtype=np.float32)


# ========================================================================
# HELPERS
# ========================================================================
def in_box_4d(x: np.ndarray, bounds_4d: np.ndarray) -> bool:
    """Check if x (4,) is inside bounds (4,2)"""
    return bool(np.all((x >= bounds_4d[:, 0]) & (x <= bounds_4d[:, 1])))


def collision_distance(x: np.ndarray) -> float:
    """Distance between two particles in position space"""
    p1 = x[0]
    p2 = x[2]
    return abs(p1 - p2)


# ========================================================================
# DYNAMICS (numpy versions)
# ========================================================================
NOISE_DIAG = np.array([0.1, 0.0, 0.1, 0.0], dtype=float)


def f_ol_np(x: np.ndarray) -> np.ndarray:
    """
    Open-loop drift for 4D double integrator.
    x: (4,) = [p1, v1, p2, v2]
    returns: (4,) drift
    """
    p1, v1, p2, v2 = x
    return np.array([
        v1,           # dp1/dt = v1
        -0.3 * v1,          # dv1/dt = 0 (coasting)
        v2,           # dp2/dt = v2
        -0.5 * v2     # dv2/dt = -0.5*v2 (damping)
    ], dtype=float)


def f_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Closed-loop drift = open-loop + control.
    u: (2,) = [a1, a2] accelerations
    """
    f_open = f_ol_np(x)
    # Control affects velocity derivatives
    f_open[1] += u[0]  # dv1/dt += a1
    f_open[3] += u[1]  # dv2/dt += a2
    return f_open


def g_diag_np(_x: np.ndarray) -> np.ndarray:
    """Constant diagonal diffusion (4,)"""
    return NOISE_DIAG


# ========================================================================
# CONTROL NETWORK LOADING
# ========================================================================
def load_control_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location=device)

    # control_net = DoubleIntegratorControlNN(hidden_dim=64)
    control_net = InvertControlNN(input_dim=4, hidden_dim=8, output_dim=2)
    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    control_net.eval()
    return control_net


def get_u(x_vec: np.ndarray, controller) -> np.ndarray:
    """
    Get control from controller.
    x_vec: (4,) state
    returns: (2,) control [a1, a2]
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
# SINGLE TRAJECTORY SIMULATION + ANIMATION
# ========================================================================
def test_single_traj_run(controller=None, T=20.0, seed=None):
    """
    Run single SDE trajectory and animate.
    Shows two subplots:
      - Position space (p1 vs p2)
      - Velocity space (v1 vs v2)
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
    ], dtype=float)

    x = np.zeros((N, 4), dtype=float)
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
        dW = np.sqrt(dt) * rng.standard_normal(size=4)
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
    p1_in_goal = []
    p2_in_goal = []
    v1_in_goal = []
    v2_in_goal = []

    for k in range(N):
        # Check full 4D goal
        if in_box_4d(x[k], goal_range):
            reached_goal = True
            goal_time = t_grid[k]
            break
        if in_box_4d(x[k], unsafe_range):
            reached_unsafe = True
            unsafe_time = t_grid[k]
            break

        # Track individual dimensions
        if goal_range[0,0] <= x[k,0] <= goal_range[0,1]:
            p1_in_goal.append((t_grid[k], x[k,0]))
        if goal_range[2,0] <= x[k,2] <= goal_range[2,1]:
            p2_in_goal.append((t_grid[k], x[k,2]))
        if goal_range[1,0] <= x[k,1] <= goal_range[1,1]:
            v1_in_goal.append((t_grid[k], x[k,1]))
        if goal_range[3,0] <= x[k,3] <= goal_range[3,1]:
            v2_in_goal.append((t_grid[k], x[k,3]))

    print("\n" + "="*80)
    print("TRAJECTORY OUTCOME")
    print("="*80)
    if reached_goal:
        print(f"✓ SUCCESS: Reached goal at t={goal_time:.2f}s")
    elif reached_unsafe:
        print(f"✗ FAILURE: Reached unsafe at t={unsafe_time:.2f}s")
    else:
        print(f"⊗ TIMEOUT: Did not reach goal or unsafe within T={T}s")

    print(f"\nFinal state: p1={x[-1,0]:.3f}, v1={x[-1,1]:.3f}, p2={x[-1,2]:.3f}, v2={x[-1,3]:.3f}")
    print(f"Goal region: p1∈[{goal_range[0,0]:.1f},{goal_range[0,1]:.1f}], v1∈[{goal_range[1,0]:.1f},{goal_range[1,1]:.1f}]")
    print(f"             p2∈[{goal_range[2,0]:.1f},{goal_range[2,1]:.1f}], v2∈[{goal_range[3,0]:.1f},{goal_range[3,1]:.1f}]")

    # Show per-dimension diagnostics
    print(f"\nPer-dimension goal satisfaction:")
    print(f"  p1 in goal bounds: {len(p1_in_goal)}/{N} steps ({100*len(p1_in_goal)/N:.1f}%)")
    if len(p1_in_goal) > 0:
        print(f"     First: t={p1_in_goal[0][0]:.2f}s, Last: t={p1_in_goal[-1][0]:.2f}s")
    print(f"  p2 in goal bounds: {len(p2_in_goal)}/{N} steps ({100*len(p2_in_goal)/N:.1f}%)")
    if len(p2_in_goal) > 0:
        print(f"     First: t={p2_in_goal[0][0]:.2f}s, Last: t={p2_in_goal[-1][0]:.2f}s")
    print(f"  v1 in goal bounds: {len(v1_in_goal)}/{N} steps ({100*len(v1_in_goal)/N:.1f}%)")
    if len(v1_in_goal) > 0:
        print(f"     First: t={v1_in_goal[0][0]:.2f}s, Last: t={v1_in_goal[-1][0]:.2f}s")
    print(f"  v2 in goal bounds: {len(v2_in_goal)}/{N} steps ({100*len(v2_in_goal)/N:.1f}%)")
    if len(v2_in_goal) > 0:
        print(f"     First: t={v2_in_goal[0][0]:.2f}s, Last: t={v2_in_goal[-1][0]:.2f}s")

    print("="*80 + "\n")

    # ========================================================================
    # ANIMATION
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_pos = axes[0]  # p1 vs p2
    ax_vel = axes[1]  # v1 vs v2

    title = "4D Double Integrator (u=0)" if controller is None else "4D Double Integrator (controlled)"
    fig.suptitle(title)

    # Text overlays
    time_text = ax_pos.text(0.02, 0.95, "", transform=ax_pos.transAxes)
    u_text = ax_pos.text(0.02, 0.88, "", transform=ax_pos.transAxes) if controller is not None else None
    status_text = ax_pos.text(0.02, 0.81, "", transform=ax_pos.transAxes)
    dist_text = ax_pos.text(0.02, 0.74, "", transform=ax_pos.transAxes)

    # Position space (p1 vs p2)
    ax_pos.set_xlim(full_range[0, 0], full_range[0, 1])
    ax_pos.set_ylim(full_range[2, 0], full_range[2, 1])
    ax_pos.set_xlabel(r"$p_1$ (particle 1 position)")
    ax_pos.set_ylabel(r"$p_2$ (particle 2 position)")
    ax_pos.set_title("Position Space")
    ax_pos.grid(True, alpha=0.3)

    # Draw regions in position space (projections)
    # Init: p1 ∈ [-2,-1], p2 ∈ [1,2]
    ax_pos.add_patch(Rectangle((init_range[0,0], init_range[2,0]),
                                init_range[0,1]-init_range[0,0],
                                init_range[2,1]-init_range[2,0],
                                fill=True, alpha=0.2, color='blue', label='Init'))
    # Goal: p1 ∈ [1,2], p2 ∈ [-2,-1]
    ax_pos.add_patch(Rectangle((goal_range[0,0], goal_range[2,0]),
                                goal_range[0,1]-goal_range[0,0],
                                goal_range[2,1]-goal_range[2,0],
                                fill=True, alpha=0.2, color='green', label='Goal'))
    # Unsafe: draw actual rectangle from unsafe_range
    ax_pos.add_patch(Rectangle((unsafe_range[0,0], unsafe_range[2,0]),
                                unsafe_range[0,1]-unsafe_range[0,0],
                                unsafe_range[2,1]-unsafe_range[2,0],
                                fill=True, alpha=0.25, color='red', label='Unsafe'))

    # Velocity space (v1 vs v2)
    ax_vel.set_xlim(full_range[1, 0], full_range[1, 1])
    ax_vel.set_ylim(full_range[3, 0], full_range[3, 1])
    ax_vel.set_xlabel(r"$v_1$ (particle 1 velocity)")
    ax_vel.set_ylabel(r"$v_2$ (particle 2 velocity)")
    ax_vel.set_title("Velocity Space")
    ax_vel.grid(True, alpha=0.3)

    # Draw velocity constraints
    # Init/Goal: low velocity
    ax_vel.add_patch(Rectangle((init_range[1,0], init_range[3,0]),
                                init_range[1,1]-init_range[1,0],
                                init_range[3,1]-init_range[3,0],
                                fill=True, alpha=0.2, color='blue'))
    ax_vel.add_patch(Rectangle((goal_range[1,0], goal_range[3,0]),
                                goal_range[1,1]-goal_range[1,0],
                                goal_range[3,1]-goal_range[3,0],
                                fill=True, alpha=0.2, color='green'))
    # Unsafe: velocity constraints
    ax_vel.add_patch(Rectangle((unsafe_range[1,0], unsafe_range[3,0]),
                                unsafe_range[1,1]-unsafe_range[1,0],
                                unsafe_range[3,1]-unsafe_range[3,0],
                                fill=True, alpha=0.25, color='red'))

    # Trajectory artists
    traj_pos, = ax_pos.plot([], [], 'b-', lw=1.5, alpha=0.6)
    pt_pos, = ax_pos.plot([], [], 'ro', markersize=8)

    traj_vel, = ax_vel.plot([], [], 'b-', lw=1.5, alpha=0.6)
    pt_vel, = ax_vel.plot([], [], 'ro', markersize=8)

    ax_pos.legend(loc='upper right')

    def init_anim():
        traj_pos.set_data([], [])
        pt_pos.set_data([], [])
        traj_vel.set_data([], [])
        pt_vel.set_data([], [])
        time_text.set_text("")
        status_text.set_text("")
        dist_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        artists = [traj_pos, pt_pos, traj_vel, pt_vel, time_text, status_text, dist_text]
        if u_text is not None:
            artists.append(u_text)
        return artists

    def update(frame):
        p1, v1, p2, v2 = x[frame]

        traj_pos.set_data(x[:frame+1, 0], x[:frame+1, 2])
        pt_pos.set_data([p1], [p2])

        traj_vel.set_data(x[:frame+1, 1], x[:frame+1, 3])
        pt_vel.set_data([v1], [v2])

        x_vec = np.array([p1, v1, p2, v2], dtype=float)
        in_goal = in_box_4d(x_vec, goal_range)
        in_unsafe = in_box_4d(x_vec, unsafe_range)

        dist = collision_distance(x_vec)

        status = "UNSAFE" if in_unsafe else ("GOAL" if in_goal else "OK")

        time_text.set_text(f"t = {t_grid[frame]:.2f}s")
        status_text.set_text(f"status: {status}")
        dist_text.set_text(f"dist: {dist:.3f}")

        if u_text is not None:
            u_vec = u_hist[frame]
            u_text.set_text(f"u = [{u_vec[0]:.2f}, {u_vec[1]:.2f}]")

        artists = [traj_pos, pt_pos, traj_vel, pt_vel, time_text, status_text, dist_text]
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
        ], dtype=float)

        outcome_recorded = False

        # Check t=0
        if in_box_4d(x_curr, unsafe_range):
            fail += 1
            outcome_recorded = True
        elif in_box_4d(x_curr, goal_range):
            success += 1
            outcome_recorded = True

        for k in range(N_mc - 1):
            if outcome_recorded:
                break

            u = get_u(x_curr, controller)
            drift = f_np(x_curr, u)
            g_vec = g_diag_np(x_curr)
            dW = np.sqrt(dt_mc) * rng_mc.standard_normal(size=4)
            x_next = x_curr + drift * dt_mc + g_vec * dW

            # Check goal/unsafe
            in_goal = in_box_4d(x_next, goal_range)
            in_unsafe = in_box_4d(x_next, unsafe_range)

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
    print("Reach-avoid MC estimate (4D):")
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
