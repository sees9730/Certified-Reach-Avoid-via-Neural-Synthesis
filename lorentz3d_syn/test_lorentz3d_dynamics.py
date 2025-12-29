"""
Empirical Validation of 3D Geometric Brownian Motion (GBM)
Phase animation with region specifications:
  - subplot 1: x1 vs x2
  - subplot 2: x3 vs x2
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from pathlib import Path

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.control_network import NonlinearControlNN
from src.save_load_utils import load_eval_bundle


# ----------------------------
# Region specs from main code
# ----------------------------
# Start near the + lobe equilibrium (~ +8.5, +8.5, 27)
init_range = np.array([
    [ 7.0, 10.0],    # x
    [ 7.0, 10.0],    # y
    [24.0, 30.0],    # z
], dtype=np.float32)

# Define "goal" as reaching near the - lobe equilibrium (~ -8.5, -8.5, 27)
goal_range = np.array([
    [-12.0, -2.0],   # x
    [-17.0, -2.0],   # y
    [ 15.0,  30.0],  # z
], dtype=np.float32)

# Unsafe = "escaped to very high z" (top slice of the usual plotting box)
unsafe_range = np.array([
    [25.0, 30.0],   # x (any)
    [-30.0, 0.0],   # y (any)
    [ 55.0, 60.0],   # z (too high)
], dtype=np.float32)

full_range = np.array([
    [-30.0, 30.0],
    [-30.0, 30.0],
    [0.0, 60.0]
], dtype=np.float32)


# ----------------------------
# Helpers
# ----------------------------
def in_box_3d(x: np.ndarray, bounds_3d: np.ndarray) -> bool:
    """x: (3,), bounds_3d: (3,2)"""
    return bool(np.all((x >= bounds_3d[:, 0]) & (x <= bounds_3d[:, 1])))


def _draw_proj_rect(ax, x_bounds, y_bounds, *, label=None, alpha=0.18, lw=1.5, fill=True, linestyle="--"):
    """
    Draw a rectangle on ax given projected bounds.
      x_bounds: (min,max) for x-axis
      y_bounds: (min,max) for y-axis
    """
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=fill,
        alpha=alpha if fill else 1.0,
        lw=lw,
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if label is not None:
        ax.text(x0 + 0.5, y1 - 2.0, label)
    return rect


# ----------------------------
# Dynamics (numpy) — Lorenz-63 + constant diagonal diffusion
# ----------------------------
# Classic Lorenz-63 parameters
L63_SIGMA = 10.0
L63_RHO   = 28.0
L63_BETA  = 8.0 / 3.0

# Constant diagonal diffusion (per-state noise std dev)
NOISE_DIAG = np.array([0.1, 0.1, 0.1], dtype=float)   # shape (3,)

def f_ol_np(x: np.ndarray) -> np.ndarray:
    """Open-loop Lorenz-63 drift, x: (3,) -> (3,).  (x1,x2,x3) = (X,Y,Z)."""
    x1, x2, x3 = x
    f1 = L63_SIGMA * (x2 - x1)
    f2 = x1 * (L63_RHO - x3) - x2
    f3 = x1 * x2 - L63_BETA * x3
    return np.array([f1, f2, f3], dtype=float)

def f_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Closed-loop drift = Lorenz-63 drift + control u (u can be zero)."""
    return f_ol_np(x) + u

def g_diag_np(_x: np.ndarray) -> np.ndarray:
    """Constant diagonal diffusion vector (3,), used with 3D dW via elementwise multiply."""
    return NOISE_DIAG


# Control Network
def load_control_net(bundle_path, device="cpu"):
    
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    
    control_net = NonlinearControlNN()

    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    # 4) eval mode for rollout
    control_net.eval()
    return control_net


def get_u(x_vec: np.ndarray, controller) -> np.ndarray:
    if controller is None:
        return np.zeros(3, dtype=float)

    if hasattr(controller, "forward"):  # torch module
        with torch.no_grad():
            u = controller(torch.tensor(x_vec[None, :], dtype=torch.float32))
            u = u.detach().cpu().numpy()
    else:  # plain callable
        u = controller(np.asarray(x_vec, dtype=float))

    u = np.asarray(u, dtype=float).reshape(-1)  # handles (1,3), (3,), (3,1), etc.
    if u.size != 3:
        raise ValueError(f"controller must return shape (3,), got {u.shape}")
    return u


def test_single_traj_run(controller=None, T=100.0, seed=None):
    """
    Single SDE rollout + animation (3D).
    controller:
      - None -> u(x)=0 in R^3
      - callable(x_np)->u_np shape (3,)
      - torch nn.Module mapping R^3 -> R^3
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed = {seed}")

    # Simulation horizon
    dt = 0.01
    N = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, N)

    # Initial condition
    x0 = np.array([
        rng.uniform(init_range[0, 0], init_range[0, 1]),
        rng.uniform(init_range[1, 0], init_range[1, 1]),
        rng.uniform(init_range[2, 0], init_range[2, 1]),
    ], dtype=float)

    x = np.zeros((N, 3), dtype=float)
    x[0] = x0
    print("x0 =", x0)

    u_hist = np.zeros((N, 3), dtype=float)

    # Euler–Maruyama
    for k in range(N - 1):
        x_curr = x[k]
        u = get_u(x_curr, controller)
        u_hist[k] = u

        drift = f_np(x_curr, u)          # (3,)
        g_vec = g_diag_np(x_curr)        # (3,)

        dW = np.sqrt(dt) * rng.standard_normal(size=3)  # (3,)
        x[k + 1] = x_curr + drift * dt + g_vec * dW

    u_hist[-1] = u_hist[-2]

    # ----------------------------
    # Build animation figure
    # ----------------------------
    fig = plt.figure(figsize=(11, 5))
    ax12 = fig.add_subplot(1, 2, 1)  # x1 vs x2
    ax32 = fig.add_subplot(1, 2, 2)  # x3 vs x2

    title = "3D GBM (u=0)" if controller is None else "3D GBM (controlled)"
    fig.suptitle(title + " — phase projections with specifications")

    # Common texts (put on left axis)
    time_text = ax12.text(0.02, 0.95, "", transform=ax12.transAxes)
    u_text = ax12.text(0.02, 0.88, "", transform=ax12.transAxes) if controller is not None else None
    status_text = ax12.text(0.02, 0.81, "", transform=ax12.transAxes)

    # Axis setup
    ax12.set_xlim(full_range[0, 0], full_range[0, 1])
    ax12.set_ylim(full_range[1, 0], full_range[1, 1])
    ax12.set_xlabel(r"$x_1$")
    ax12.set_ylabel(r"$x_2$")
    ax12.set_title(r"$x_1$ vs $x_2$")

    ax32.set_xlim(full_range[2, 0], full_range[2, 1])
    ax32.set_ylim(full_range[1, 0], full_range[1, 1])
    ax32.set_xlabel(r"$x_3$")
    ax32.set_ylabel(r"$x_2$")
    ax32.set_title(r"$x_3$ vs $x_2$")

    # Draw projected regions (FULL / INIT / GOAL / UNSAFE)
    # Full region projection:
    _draw_proj_rect(ax12, full_range[0], full_range[1], label=r"$X$", fill=False, alpha=1.0, linestyle="-", lw=1.5)
    _draw_proj_rect(ax32, full_range[2], full_range[1], label=r"$X$", fill=False, alpha=1.0, linestyle="-", lw=1.5)

    # Init projection:
    _draw_proj_rect(ax12, init_range[0], init_range[1], label=r"$X_{\mathrm{init}}$", fill=True, alpha=0.18, linestyle="--", lw=1.5)
    _draw_proj_rect(ax32, init_range[2], init_range[1], label=r"$X_{\mathrm{init}}$", fill=True, alpha=0.18, linestyle="--", lw=1.5)

    # Goal projection:
    _draw_proj_rect(ax12, goal_range[0], goal_range[1], label=r"$X_{\mathrm{goal}}$", fill=True, alpha=0.20, linestyle="-", lw=1.5)
    _draw_proj_rect(ax32, goal_range[2], goal_range[1], label=r"$X_{\mathrm{goal}}$", fill=True, alpha=0.20, linestyle="-", lw=1.5)

    # Unsafe projection:
    _draw_proj_rect(ax12, unsafe_range[0], unsafe_range[1], label=r"$X_{\mathrm{unsafe}}$", fill=True, alpha=0.25, linestyle="-", lw=1.5)
    _draw_proj_rect(ax32, unsafe_range[2], unsafe_range[1], label=r"$X_{\mathrm{unsafe}}$", fill=True, alpha=0.25, linestyle="-", lw=1.5)

    # Trajectory artists (both subplots)
    traj12, = ax12.plot([], [], lw=1.5)
    pt12, = ax12.plot([], [], marker="o")

    traj32, = ax32.plot([], [], lw=1.5)
    pt32, = ax32.plot([], [], marker="o")

    def init_anim():
        traj12.set_data([], [])
        pt12.set_data([], [])
        traj32.set_data([], [])
        pt32.set_data([], [])
        time_text.set_text("")
        status_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        artists = [traj12, pt12, traj32, pt32, time_text, status_text]
        if u_text is not None:
            artists.append(u_text)
        return artists

    def update(frame):
        x1, x2, x3 = x[frame]

        traj12.set_data(x[:frame+1, 0], x[:frame+1, 1])
        pt12.set_data([x1], [x2])

        traj32.set_data(x[:frame+1, 2], x[:frame+1, 1])
        pt32.set_data([x3], [x2])

        x_vec = np.array([x1, x2, x3], dtype=float)
        in_goal = in_box_3d(x_vec, goal_range)
        in_unsafe = in_box_3d(x_vec, unsafe_range)
        status = "UNSAFE" if in_unsafe else ("GOAL" if in_goal else "OK")

        time_text.set_text(f"t = {t_grid[frame]:.2f}s")
        if u_text is not None:
            u_vec = u_hist[frame]
            u_text.set_text("u = [" + ", ".join(f"{v:.2f}" for v in u_vec) + "]")
        status_text.set_text(f"status: {status}")

        artists = [traj12, pt12, traj32, pt32, time_text, status_text]
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
    dt_mc=0.001,
    seed_mc=123,
    return_example_paths=False,
    n_example_paths=5,
):
    """
    Monte Carlo estimate of reach-avoid probability (3D):
      P( reach X_goal before X_unsafe within horizon T_mc )
    """
    if seed_mc is None:
        seed_mc = int(np.random.SeedSequence().entropy)
    rng_mc = np.random.default_rng(seed_mc)
    print(f"[estimate_reach_avoid_mc] seed = {seed_mc}")

    N_mc = int(T_mc / dt_mc) + 1

    def segment_first_entry_time_into_box(x0: np.ndarray, x1: np.ndarray, box: np.ndarray):
        """
        Earliest t in [0,1] such that x(t)=x0 + t*(x1-x0) is inside axis-aligned box.
        box: shape (3,2) with [lo, hi] per dim.
        Returns: t_enter in [0,1] or None if no intersection.
        """
        x0 = np.asarray(x0, dtype=float).reshape(3,)
        x1 = np.asarray(x1, dtype=float).reshape(3,)
        d = x1 - x0

        t_enter = -np.inf
        t_exit  =  np.inf

        for i in range(3):
            lo, hi = float(box[i, 0]), float(box[i, 1])
            if abs(d[i]) < 1e-15:
                # Segment parallel to slabs in this dim
                if x0[i] < lo or x0[i] > hi:
                    return None
                # else no constraint from this dim
                continue

            t0 = (lo - x0[i]) / d[i]
            t1 = (hi - x0[i]) / d[i]
            if t0 > t1:
                t0, t1 = t1, t0

            t_enter = max(t_enter, t0)
            t_exit  = min(t_exit,  t1)

            if t_enter > t_exit:
                return None

        # Intersects infinite line segment; restrict to [0,1]
        if t_exit < 0.0 or t_enter > 1.0:
            return None

        return max(t_enter, 0.0)

    success = 0
    fail = 0
    timeout = 0
    example_paths = []

    for r in range(n_mc):
        x_curr = np.array([
            rng_mc.uniform(init_range[0, 0], init_range[0, 1]),
            rng_mc.uniform(init_range[1, 0], init_range[1, 1]),
            rng_mc.uniform(init_range[2, 0], init_range[2, 1]),
        ], dtype=float)

        want_path = return_example_paths and (len(example_paths) < n_example_paths)
        if want_path:
            path = np.zeros((N_mc, 3), dtype=float)
            path[0] = x_curr

        outcome_recorded = False

        # Handle t=0
        if in_box_3d(x_curr, unsafe_range):
            fail += 1
            outcome_recorded = True
            if want_path:
                example_paths.append(path[:1].copy())
        elif in_box_3d(x_curr, goal_range):
            success += 1
            outcome_recorded = True
            if want_path:
                example_paths.append(path[:1].copy())

        for k in range(N_mc - 1):
            if outcome_recorded:
                break

            # propagate one step
            u = get_u(x_curr, controller)        # (3,)
            drift = f_np(x_curr, u)  # (3,)
            g_vec = g_diag_np(x_curr)  # (3,)

            dW = np.sqrt(dt_mc) * rng_mc.standard_normal(size=3)  # (3,)
            x_next = x_curr + drift * dt_mc + g_vec * dW

            if want_path:
                path[k + 1] = x_next

            # Decide event order within this step using segment entry times
            t_goal = segment_first_entry_time_into_box(x_curr, x_next, goal_range)
            t_unsafe = segment_first_entry_time_into_box(x_curr, x_next, unsafe_range)

            if (t_goal is not None) or (t_unsafe is not None):
                # both hit: whichever enters first wins
                if (t_goal is not None) and (t_unsafe is not None):
                    if t_goal < t_unsafe:
                        success += 1
                    else:
                        # tie or unsafe first -> conservative fail
                        fail += 1
                elif t_goal is not None:
                    success += 1
                else:
                    fail += 1

                outcome_recorded = True
                if want_path:
                    example_paths.append(path[:k + 2].copy())
                break

            # no event: advance
            x_curr = x_next

        if not outcome_recorded:
            timeout += 1
            if want_path:
                example_paths.append(path.copy())

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

    if return_example_paths:
        return p_reach_avoid, stats, example_paths
    return p_reach_avoid, stats


def test_mc(controller=None):
    p_reach_avoid, stats = estimate_reach_avoid_mc(
        controller=controller,
        n_mc=100,
        T_mc=10.0,
        dt_mc=0.005,
        seed_mc=0
    )
    print("Reach-avoid MC estimate (3D):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    test_single_traj_run(controller=None)

    control_net = load_control_net(OUTPUT_DIR / "eval_bundle.pth")
    test_single_traj_run(controller=control_net)

    test_mc(controller=None)
    test_mc(controller=control_net)


if __name__ == "__main__":
    main()
