"""
Empirical Validation of 4D Geometric Brownian Motion (GBM)
Phase animation with region specifications:
  - subplot 1: x1 vs x2
  - subplot 2: x3 vs x4
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

from src.control_network import LinearControlNN
from src.save_load_utils import load_eval_bundle

X_DIM = 4

# ----------------------------
# Region specs from main code (4D)
# ----------------------------
init_range = np.array([
    [45.0, 55.0],
    [-55.0, -45.0],
    [50.0, 60.0],
    [45.0, 55.0],
], dtype=np.float32)

goal_range = np.array([
    [-25.0, 25.0],
    [-25.0, 25.0],
    [-25.0, 25.0],
    [-25.0, 25.0],
], dtype=np.float32)

unsafe_range = np.array([
    [-100.0, -80.0],
    [-100.0, 100.0],
    [-100.0, -80.0],
    [-100.0, -80.0],
], dtype=np.float32)

full_range = np.array([
    [-100.0, 100.0],
    [-100.0, 100.0],
    [-100.0, 100.0],
    [-100.0, 100.0],
], dtype=np.float32)


# ----------------------------
# Helpers
# ----------------------------
def in_box_4d(x: np.ndarray, bounds_4d: np.ndarray) -> bool:
    """x: (4,), bounds_4d: (4,2)"""
    return bool(np.all((x >= bounds_4d[:, 0]) & (x <= bounds_4d[:, 1])))


def _draw_proj_rect(ax, x_bounds, y_bounds, *, label=None, alpha=0.18, lw=1.5, fill=True, linestyle="--",
                    facecolor=None, edgecolor=None):
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
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(rect)
    if label is not None:
        ax.text(x0 + 0.5, y1 - 2.0, label)
    return rect


# ----------------------------
# Dynamics (numpy) - 4D chain drift + diagonal diffusion
# ----------------------------
def f_ol_np(x: np.ndarray) -> np.ndarray:
    """Open-loop drift, x: (4,) -> (4,)"""
    x1, x2, x3, x4 = x
    f1 = -1.5 * x1 + 1.0 * x2 + 0.0 * x3 + 0.0 * x4
    f2 = -1.0 * x1 - 1.5 * x2 + 1.0 * x3 + 0.0 * x4
    f3 =  0.0 * x1 - 1.0 * x2 - 1.5 * x3 + 1.0 * x4
    f4 =  0.0 * x1 + 0.0 * x2 - 1.0 * x3 - 1.5 * x4
    return np.array([f1, f2, f3, f4], dtype=float)


def f_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Closed-loop drift = f_ol(x) + u (u can be zero)."""
    return f_ol_np(x) + u


def g_diag_np(x: np.ndarray) -> np.ndarray:
    """Diagonal diffusion vector (4,), used with 4D dW via elementwise multiply."""
    return 0.2 * x


# Control Network
def load_control_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location="cpu")

    # create a fresh network with SAME architecture as when saved
    control_net = LinearControlNN(input_dim=X_DIM).to(device)

    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    control_net.eval()
    return control_net


def get_u(x_vec: np.ndarray, controller=None) -> np.ndarray:
    if controller is None:
        return np.zeros(X_DIM, dtype=float)

    if "torch" in globals() and hasattr(controller, "forward"):
        with torch.no_grad():
            xt = torch.tensor(x_vec[None, :], dtype=torch.float32)  # (1,4)
            u_val = controller(xt)                                  # (1,4) expected
            u_val = u_val.squeeze(0).detach().cpu().numpy()
    else:
        u_val = controller(x_vec.astype(float))

    u_val = np.asarray(u_val, dtype=float).reshape(-1)
    if u_val.shape[0] != X_DIM:
        raise ValueError(f"controller must return shape (X_DIM,), got {u_val.shape}")
    return u_val


def test_single_traj_run(controller=None, T=10.0, seed=None, n_traj: int = 10):
    """
    Multi-traj SDE rollout + animation (4D).
    controller:
      - None -> u(x)=0 in R^4
      - callable(x_np)->u_np shape (4,)
      - torch nn.Module mapping R^4 -> R^4
    n_traj:
      - number of trajectories to simulate/animate (each starts from random x0 in init_range)
    """
    if n_traj <= 0:
        raise ValueError(f"n_traj must be >= 1, got {n_traj}")

    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed = {seed}, n_traj = {n_traj}")

    # Simulation horizon
    dt = 0.01
    N = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, N)

    # Initial conditions (n_traj, 4)
    x0 = np.column_stack([
        rng.uniform(init_range[0, 0], init_range[0, 1], size=n_traj),
        rng.uniform(init_range[1, 0], init_range[1, 1], size=n_traj),
        rng.uniform(init_range[2, 0], init_range[2, 1], size=n_traj),
        rng.uniform(init_range[3, 0], init_range[3, 1], size=n_traj),
    ]).astype(float)

    # State history: (N, n_traj, 4)
    x = np.zeros((N, n_traj, X_DIM), dtype=float)
    x[0] = x0
    print("x0[0] =", x0[0])

    # Control history: (N, n_traj, 4)
    u_hist = np.zeros((N, n_traj, X_DIM), dtype=float)

    # Euler–Maruyama
    for k in range(N - 1):
        x_curr = x[k]  # (n_traj,4)

        # per-traj control + step
        for i in range(n_traj):
            u = get_u(x_curr[i], controller=controller)  # (4,)
            u_hist[k, i] = u

            drift = f_np(x_curr[i], u)     # (4,)
            g_vec = g_diag_np(x_curr[i])   # (4,)

            dW = np.sqrt(dt) * rng.standard_normal(size=X_DIM)  # (4,)
            x[k + 1, i] = x_curr[i] + drift * dt + g_vec * dW

    u_hist[-1] = u_hist[-2]

    # ----------------------------
    # Build animation figure
    # ----------------------------
    fig = plt.figure(figsize=(11, 5))
    ax12 = fig.add_subplot(1, 2, 1)  # x1 vs x2
    ax34 = fig.add_subplot(1, 2, 2)  # x3 vs x4

    title = "4D GBM (closed loop with u_i = -x_i)" if controller is None else "4D GBM (controlled)"
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

    ax34.set_xlim(full_range[2, 0], full_range[2, 1])
    ax34.set_ylim(full_range[3, 0], full_range[3, 1])
    ax34.set_xlabel(r"$x_3$")
    ax34.set_ylabel(r"$x_4$")
    ax34.set_title(r"$x_3$ vs $x_4$")

    # Draw projected regions (FULL / INIT / GOAL / UNSAFE)
    _draw_proj_rect(ax12, full_range[0], full_range[1], label=r"$X$", fill=False, alpha=1.0, linestyle="-", lw=1.5)
    _draw_proj_rect(ax34, full_range[2], full_range[3], label=r"$X$", fill=False, alpha=1.0, linestyle="-", lw=1.5)

    # INIT = green
    _draw_proj_rect(
        ax12, init_range[0], init_range[1],
        label=r"$X_{\mathrm{init}}$", fill=True, alpha=0.18, linestyle="--", lw=1.5,
        facecolor="green", edgecolor="green"
    )
    _draw_proj_rect(
        ax34, init_range[2], init_range[3],
        label=r"$X_{\mathrm{init}}$", fill=True, alpha=0.18, linestyle="--", lw=1.5,
        facecolor="green", edgecolor="green"
    )

    # GOAL = blue
    _draw_proj_rect(
        ax12, goal_range[0], goal_range[1],
        label=r"$X_{\mathrm{goal}}$", fill=True, alpha=0.20, linestyle="-", lw=1.5,
        facecolor="blue", edgecolor="blue"
    )
    _draw_proj_rect(
        ax34, goal_range[2], goal_range[3],
        label=r"$X_{\mathrm{goal}}$", fill=True, alpha=0.20, linestyle="-", lw=1.5,
        facecolor="blue", edgecolor="blue"
    )

    # UNSAFE = red
    _draw_proj_rect(
        ax12, unsafe_range[0], unsafe_range[1],
        label=r"$X_{\mathrm{unsafe}}$", fill=True, alpha=0.25, linestyle="-", lw=1.5,
        facecolor="red", edgecolor="red"
    )
    _draw_proj_rect(
        ax34, unsafe_range[2], unsafe_range[3],
        label=r"$X_{\mathrm{unsafe}}$", fill=True, alpha=0.25, linestyle="-", lw=1.5,
        facecolor="red", edgecolor="red"
    )

    # Trajectory artists (one per traj, per subplot)
    traj12_list, pt12_list = [], []
    traj34_list, pt34_list = [], []
    for _i in range(n_traj):
        t12, = ax12.plot([], [], lw=1.5)
        p12, = ax12.plot([], [], marker="o")
        t34, = ax34.plot([], [], lw=1.5)
        p34, = ax34.plot([], [], marker="o")
        traj12_list.append(t12); pt12_list.append(p12)
        traj34_list.append(t34); pt34_list.append(p34)

    def init_anim():
        for i in range(n_traj):
            traj12_list[i].set_data([], [])
            pt12_list[i].set_data([], [])
            traj34_list[i].set_data([], [])
            pt34_list[i].set_data([], [])
        time_text.set_text("")
        status_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        artists = []
        for i in range(n_traj):
            artists += [traj12_list[i], pt12_list[i], traj34_list[i], pt34_list[i]]
        artists += [time_text, status_text]
        if u_text is not None:
            artists.append(u_text)
        return artists

    def update(frame):
        # Update all trajectories
        for i in range(n_traj):
            xi = x[:frame+1, i, :]  # (frame+1, 4)
            x1, x2, x3, x4 = xi[-1]

            traj12_list[i].set_data(xi[:, 0], xi[:, 1])
            pt12_list[i].set_data([x1], [x2])

            traj34_list[i].set_data(xi[:, 2], xi[:, 3])
            pt34_list[i].set_data([x3], [x4])

        # Status text based on traj 0 (keeps UI simple)
        x_vec0 = x[frame, 0, :].astype(float)
        in_goal0 = in_box_4d(x_vec0, goal_range)
        in_unsafe0 = in_box_4d(x_vec0, unsafe_range)
        status = "UNSAFE" if in_unsafe0 else ("GOAL" if in_goal0 else "OK")

        time_text.set_text(f"t = {t_grid[frame]:.2f}s")
        if u_text is not None:
            u0 = u_hist[frame, 0, :]
            u_text.set_text("u(traj0) = [" + ", ".join(f"{v:.2f}" for v in u0) + "]")
        status_text.set_text(f"status(traj0): {status}")

        artists = []
        for i in range(n_traj):
            artists += [traj12_list[i], pt12_list[i], traj34_list[i], pt34_list[i]]
        artists += [time_text, status_text]
        if u_text is not None:
            artists.append(u_text)
        return artists

    skip = 5
    frames = range(0, N, skip)
    _ = FuncAnimation(fig, update, frames=frames, init_func=init_anim, blit=True, interval=30)

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
    Monte Carlo estimate of reach-avoid probability (4D):
      P( reach X_goal before X_unsafe within horizon T_mc )
    """
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
        x_curr = np.array([
            rng_mc.uniform(init_range[0, 0], init_range[0, 1]),
            rng_mc.uniform(init_range[1, 0], init_range[1, 1]),
            rng_mc.uniform(init_range[2, 0], init_range[2, 1]),
            rng_mc.uniform(init_range[3, 0], init_range[3, 1]),
        ], dtype=float)

        if return_example_paths and len(example_paths) < n_example_paths:
            path = np.zeros((N_mc, 4), dtype=float)
            path[0] = x_curr

        outcome_recorded = False

        for k in range(N_mc - 1):
            if in_box_4d(x_curr, unsafe_range):
                fail += 1
                outcome_recorded = True
                if return_example_paths and len(example_paths) < n_example_paths:
                    example_paths.append(path[:k+1].copy())
                break

            if in_box_4d(x_curr, goal_range):
                success += 1
                outcome_recorded = True
                if return_example_paths and len(example_paths) < n_example_paths:
                    example_paths.append(path[:k+1].copy())
                break

            u = get_u(x_curr, controller=controller)        # (4,)
            drift = f_np(x_curr, u)  # (4,)
            g_vec = g_diag_np(x_curr)  # (4,)

            dW = np.sqrt(dt_mc) * rng_mc.standard_normal(size=X_DIM)  # (4,)
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
    print("Reach-avoid MC estimate (4D):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    test_single_traj_run(controller=None)
    test_mc(controller=None)


if __name__ == "__main__":
    main()
