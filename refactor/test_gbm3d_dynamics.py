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
from control_network import LinearControlNN

ROOT = Path(__file__).resolve().parents[1]   # repo_root
OUTPUT_DIR = ROOT / "refactor" / "outputs"


# ----------------------------
# Region specs from main code
# ----------------------------
init_range = np.array([
    [45.0, 55.0],
    [-55.0, -45.0],
    [50.0, 60.0]
], dtype=np.float32)

goal_range = np.array([
    [-25.0, 25.0],
    [-25.0, 25.0],
    [-25.0, 25.0]
], dtype=np.float32)

unsafe_range = np.array([
    [-100.0, -80.0],
    [-100.0, 100.0],
    [-100.0, -80.0]
], dtype=np.float32)

full_range = np.array([
    [-100.0, 100.0],
    [-100.0, 100.0],
    [-100.0, 100.0]
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
# Dynamics (numpy)
# ----------------------------
def f_ol_np(x: np.ndarray) -> np.ndarray:
    """Open-loop drift, x: (3,) -> (3,)"""
    x1, x2, x3 = x
    f1 = -0.5 * x1 + 1.0 * x2 + 0.0 * x3
    f2 = -1.0 * x1 - 0.5 * x2 + 1.0 * x3
    f3 =  0.0 * x1 - 1.0 * x2 - 0.5 * x3
    return np.array([f1, f2, f3], dtype=float)


def f_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Closed-loop drift = f_ol(x) + u (u can be zero)."""
    return f_ol_np(x) + u


def g_diag_np(x: np.ndarray) -> np.ndarray:
    """Diagonal diffusion vector (3,), used with 3D dW via elementwise multiply."""
    return 0.2 * x


# ----------------------------
# Optional: load a 3D control net
# (Your net must be R^3 -> R^3)
# ----------------------------
def load_control_net(control_save_path, device="cpu"):
    control_net = LinearControlNN(input_dim=3)
    ckpt = torch.load(control_save_path, map_location=device)
    control_net.load_state_dict(ckpt["model_state_dict"])
    control_net.eval()
    return control_net


def test_single_traj_run(controller=None, T=10.0, seed=None):
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

    def get_u(x_vec: np.ndarray) -> np.ndarray:
        if controller is None:
            return np.zeros(3, dtype=float)

        if "torch" in globals() and hasattr(controller, "forward"):
            with torch.no_grad():
                xt = torch.tensor(x_vec[None, :], dtype=torch.float32)  # (1,3)
                u_val = controller(xt)                                  # (1,3) expected
                u_val = u_val.squeeze(0).detach().cpu().numpy()
        else:
            u_val = controller(x_vec.astype(float))

        u_val = np.asarray(u_val, dtype=float).reshape(-1)
        if u_val.shape[0] != 3:
            raise ValueError(f"controller must return shape (3,), got {u_val.shape}")
        return u_val

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
        u = get_u(x_curr)
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
    dt_mc=0.005,
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

    def get_u(x_vec: np.ndarray) -> np.ndarray:
        if controller is None:
            return np.zeros(3, dtype=float)

        if "torch" in globals() and hasattr(controller, "forward"):
            with torch.no_grad():
                xt = torch.tensor(x_vec[None, :], dtype=torch.float32)
                u_val = controller(xt)
                u_val = u_val.squeeze(0).detach().cpu().numpy()
        else:
            u_val = controller(x_vec.astype(float))

        u_val = np.asarray(u_val, dtype=float).reshape(-1)
        if u_val.shape[0] != 3:
            raise ValueError(f"controller must return shape (3,), got {u_val.shape}")
        return u_val

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

        if return_example_paths and len(example_paths) < n_example_paths:
            path = np.zeros((N_mc, 3), dtype=float)
            path[0] = x_curr

        outcome_recorded = False

        for k in range(N_mc - 1):
            if in_box_3d(x_curr, unsafe_range):
                fail += 1
                outcome_recorded = True
                if return_example_paths and len(example_paths) < n_example_paths:
                    example_paths.append(path[:k+1].copy())
                break

            if in_box_3d(x_curr, goal_range):
                success += 1
                outcome_recorded = True
                if return_example_paths and len(example_paths) < n_example_paths:
                    example_paths.append(path[:k+1].copy())
                break

            u = get_u(x_curr)        # (3,)
            drift = f_np(x_curr, u)  # (3,)
            g_vec = g_diag_np(x_curr)  # (3,)

            dW = np.sqrt(dt_mc) * rng_mc.standard_normal(size=3)  # (3,)
            x_curr = x_curr + drift * dt_mc + g_vec * dW

            if return_example_paths and len(example_paths) < n_example_paths:
                path[k+1] = x_curr

        if not outcome_recorded:
            timeout += 1
            if return_example_paths and len(example_paths) < n_example_paths:
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
        n_mc=1000,
        T_mc=8.0,
        dt_mc=0.005,
        seed_mc=0
    )
    print("Reach-avoid MC estimate (3D):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    # test_single_traj_run(controller=None)
    # test_mc(controller=None)

    # If you have a trained 3D controller:
    # from control_network import LinearControlNN3D
    control_net = load_control_net(OUTPUT_DIR / "control_net.pth")
    test_single_traj_run(controller=control_net)
    test_mc(controller=control_net)


if __name__ == "__main__":
    main()
