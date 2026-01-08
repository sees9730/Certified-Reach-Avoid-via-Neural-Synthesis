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

from src.control_network import LinearControlNN, LorentzLinearControlNN, NonlinearControlNN
from src.save_load_utils import load_eval_bundle


# ----------------------------
# Region specs from main code
# ----------------------------
init_range = np.array([
    [-1.2, 1.2],
    [-1.2, 1.2],
    [-1.2, 1.2],
], dtype=np.float32)

goal_range = np.array([
    [-0.3, 0.3],
    [-0.3, 0.3],
    [-0.3, 0.3],
], dtype=np.float32)

unsafe_range = np.array([
    [-6.0, -5.0],
    [-6.0, -5.0],
    [-6.0, -5.0],
], dtype=np.float32)

full_range = np.array([
    # [-6.0, 6.0],
    # [-6.0, 6.0],
    # [-6.0, 6.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 20.0],
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
    f1 = -10.0*x1 + 10.0*x2
    # f1 = -33.71*x1 - 8.49*x2
    f2 = -x1*x3 + 28.0*x1 - x2
    f3 =  x1*x2 - 8/3 *x3
    return np.array([f1, f2, f3], dtype=float)


def f_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Closed-loop drift = f_ol(x) + u (u can be zero)."""
    return f_ol_np(x) + u


NOISE_DIAG = np.array([0.1, 0.1, 0.1], dtype=float)  # per-state noise std dev
def g_diag_np(x: np.ndarray) -> np.ndarray:
    """Diagonal diffusion vector (3,), used with 3D dW via elementwise multiply."""
    return NOISE_DIAG


def print_net_params(net: torch.nn.Module, *, full_tensor: bool = True, precision: int = 6):
    """
    Print all weights/biases (i.e., all parameters) in `net`.
    - full_tensor=True prints the entire tensor
    - full_tensor=False prints shape + min/max/mean
    """
    torch.set_printoptions(precision=precision, sci_mode=False)

    print(f"\n[print_net_params] {net.__class__.__name__}")
    print("--------------------------------------------------")

    # 1) Named parameters (this is the authoritative list of trainable weights/biases)
    for name, p in net.named_parameters():
        if p is None:
            continue
        p_det = p.detach().cpu()
        print(f"\nPARAM: {name}")
        print(f"  requires_grad: {p.requires_grad}")
        print(f"  dtype/device : {p_det.dtype} / cpu")
        print(f"  shape        : {tuple(p_det.shape)}")

        if full_tensor:
            print(p_det)
        else:
            print(
                f"  stats: min={p_det.min().item():.6g}, "
                f"max={p_det.max().item():.6g}, "
                f"mean={p_det.mean().item():.6g}"
            )

    # 2) Named buffers (not weights/biases, but sometimes useful)
    buffers = list(net.named_buffers())
    if buffers:
        print("\n\nBUFFERS (non-parameter tensors):")
        for name, b in buffers:
            b_det = b.detach().cpu()
            print(f"  {name}: shape={tuple(b_det.shape)} dtype={b_det.dtype}")

    print("\n--------------------------------------------------\n")


# Control Network
def load_control_net(bundle_path, device="cpu"):
    
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    
    # 1) create a fresh network with SAME architecture as when saved
    # control_net = LinearControlNN(input_dim=3).to(device)
    # control_net = NonlinearControlNN()
    control_net = LorentzLinearControlNN()

    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    print_net_params(control_net, full_tensor=True)

    # 4) eval mode for rollout
    control_net.eval()
    return control_net


def get_u(x_vec: np.ndarray, controller=None) -> np.ndarray:
    if controller is None:
        return np.zeros(3, dtype=float)
    else:
        x_vec_tensor = torch.tensor(x_vec, dtype=torch.float32).reshape(1,-1)
        u_val = controller(x_vec_tensor).detach().cpu().numpy().reshape(-1,)

    u_val = np.asarray(u_val, dtype=float).reshape(-1)
    if u_val.shape[0] != 3:
        raise ValueError(f"controller must return shape (3,), got {u_val.shape}")
    return u_val


def test_single_traj_run(controller=None, T=10.0, seed=None, n_traj: int = 5):
    """
    Multi-trajectory SDE rollout + 3D animation (x1,x2,x3) with region boxes.

    n_traj:
      - integer >= 1
      - each trajectory starts from a random x0 sampled uniformly in init_range

    controller:
      - None -> u(x)=0 in R^3
      - callable(x_np)->u_np shape (3,)
      - torch nn.Module mapping R^3 -> R^3
    """
    if n_traj < 1:
        raise ValueError(f"n_traj must be >= 1, got {n_traj}")

    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed = {seed}, n_traj = {n_traj}")

    # ----------------------------
    # Sim horizon
    # ----------------------------
    dt = 0.01
    N = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, N)

    # ----------------------------
    # Initial conditions (n_traj of them)
    # ----------------------------
    x0_all = np.column_stack([
        rng.uniform(init_range[0, 0], init_range[0, 1], size=n_traj),
        rng.uniform(init_range[1, 0], init_range[1, 1], size=n_traj),
        rng.uniform(init_range[2, 0], init_range[2, 1], size=n_traj),
    ]).astype(float)  # (n_traj, 3)

    x = np.zeros((n_traj, N, 3), dtype=float)
    x[:, 0, :] = x0_all
    print("x0[0] =", x0_all[0])

    u_hist = np.zeros((n_traj, N, 3), dtype=float)

    # ----------------------------
    # Euler–Maruyama (loop over time, then traj)
    # ----------------------------
    for k in range(N - 1):
        dW_all = np.sqrt(dt) * rng.standard_normal(size=(n_traj, 3))  # (n_traj,3)

        for i in range(n_traj):
            x_curr = x[i, k]
            u = get_u(x_curr, controller=controller)
            u_hist[i, k] = u

            drift = f_np(x_curr, u)          # (3,)
            g_vec = g_diag_np(x_curr)        # (3,)
            x[i, k + 1] = x_curr + drift * dt + g_vec * dW_all[i]

    u_hist[:, -1, :] = u_hist[:, -2, :]

    # ----------------------------
    # 3D region-box helper
    # ----------------------------
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def _draw_box_3d(ax, bounds_3d, *, color=None, alpha=0.12, lw=1.2, linestyle="--", label=None):
        """
        Draw an axis-aligned 3D box given bounds_3d shape (3,2).
        Returns Poly3DCollection.
        """
        (x0b, x1b), (y0b, y1b), (z0b, z1b) = bounds_3d
        x0b, x1b = float(x0b), float(x1b)
        y0b, y1b = float(y0b), float(y1b)
        z0b, z1b = float(z0b), float(z1b)

        # 8 corners
        p000 = (x0b, y0b, z0b)
        p001 = (x0b, y0b, z1b)
        p010 = (x0b, y1b, z0b)
        p011 = (x0b, y1b, z1b)
        p100 = (x1b, y0b, z0b)
        p101 = (x1b, y0b, z1b)
        p110 = (x1b, y1b, z0b)
        p111 = (x1b, y1b, z1b)

        faces = [
            [p000, p001, p011, p010],  # x=x0
            [p100, p101, p111, p110],  # x=x1
            [p000, p001, p101, p100],  # y=y0
            [p010, p011, p111, p110],  # y=y1
            [p000, p010, p110, p100],  # z=z0
            [p001, p011, p111, p101],  # z=z1
        ]

        poly = Poly3DCollection(faces, alpha=alpha, linewidths=lw, linestyles=linestyle)

        if color is not None:
            poly.set_facecolor(color)
            poly.set_edgecolor(color)
        ax.add_collection3d(poly)

        if label is not None:
            ax.text(x0b, y1b, z1b, label)

        return poly

    # ----------------------------
    # Build 3D animation figure
    # ----------------------------
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    title = "3D Lorentz (u=0)" if controller is None else "3D Lorentz (controlled)"
    ax.set_title(title + f" — 3D phase with specs (n_traj={n_traj})")

    # Axes limits
    ax.set_xlim(full_range[0, 0], full_range[0, 1])
    ax.set_ylim(full_range[1, 0], full_range[1, 1])
    ax.set_zlim(full_range[2, 0], full_range[2, 1])

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")

    # Draw region boxes
    _draw_box_3d(ax, full_range,   alpha=0.0,  lw=1.5, linestyle="-",  label=r"$X$")
    _draw_box_3d(ax, init_range,   color="green", alpha=0.10, lw=1.2, linestyle="--", label=r"$X_{\mathrm{init}}$")
    _draw_box_3d(ax, goal_range,   color="blue",  alpha=0.10, lw=1.2, linestyle="-",  label=r"$X_{\mathrm{goal}}$")
    _draw_box_3d(ax, unsafe_range, color="red",   alpha=0.12, lw=1.2, linestyle="-",  label=r"$X_{\mathrm{unsafe}}$")

    # Text overlays in 2D axes coords
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    status_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes)
    u_text = ax.text2D(0.02, 0.85, "", transform=ax.transAxes) if controller is not None else None

    # Trajectory artists (one per trajectory)
    traj3d_list = []
    pt3d_list = []
    for _ in range(n_traj):
        line, = ax.plot([], [], [], lw=1.5)
        pt,   = ax.plot([], [], [], marker="o")
        traj3d_list.append(line)
        pt3d_list.append(pt)

    def init_anim():
        for line, pt in zip(traj3d_list, pt3d_list):
            line.set_data([], [])
            line.set_3d_properties([])
            pt.set_data([], [])
            pt.set_3d_properties([])
        time_text.set_text("")
        status_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        artists = traj3d_list + pt3d_list + [time_text, status_text]
        if u_text is not None:
            artists.append(u_text)
        return artists

    def update(frame):
        # Update all trajectories
        for i, (line, pt) in enumerate(zip(traj3d_list, pt3d_list)):
            xs = x[i, :frame + 1, 0]
            ys = x[i, :frame + 1, 1]
            zs = x[i, :frame + 1, 2]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

            pt.set_data([x[i, frame, 0]], [x[i, frame, 1]])
            pt.set_3d_properties([x[i, frame, 2]])

        # Status summary across trajectories at this frame
        in_goal_ct = 0
        in_unsafe_ct = 0
        for i in range(n_traj):
            x_vec = x[i, frame]
            if in_box_3d(x_vec, goal_range):
                in_goal_ct += 1
            if in_box_3d(x_vec, unsafe_range):
                in_unsafe_ct += 1

        time_text.set_text(f"t = {t_grid[frame]:.2f}s")
        status_text.set_text(f"goal: {in_goal_ct}/{n_traj}   unsafe: {in_unsafe_ct}/{n_traj}")

        # Keep u display simple: show u for traj 0 at current frame
        if u_text is not None:
            u_vec = u_hist[0, frame]
            u_text.set_text("u(traj0) = [" + ", ".join(f"{v:.2f}" for v in u_vec) + "]")

        artists = traj3d_list + pt3d_list + [time_text, status_text]
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

    Optimization:
      - EARLY TERMINATION: as soon as a traj hits goal or unsafe, stop simulating it.
      - Slightly less overhead in the inner loop (precompute sqrt_dt, fewer branches).
    """
    if seed_mc is None:
        seed_mc = int(np.random.SeedSequence().entropy)
    rng_mc = np.random.default_rng(seed_mc)
    print(f"[estimate_reach_avoid_mc] seed = {seed_mc}")

    N_mc = int(T_mc / dt_mc) + 1
    sqrt_dt = float(np.sqrt(dt_mc))

    success = 0
    fail = 0
    timeout = 0
    example_paths = []

    keep_examples = bool(return_example_paths) and (n_example_paths is not None) and (n_example_paths > 0)

    for r in range(n_mc):
        x_curr = np.array([
            rng_mc.uniform(init_range[0, 0], init_range[0, 1]),
            rng_mc.uniform(init_range[1, 0], init_range[1, 1]),
            rng_mc.uniform(init_range[2, 0], init_range[2, 1]),
        ], dtype=float)

        save_this = keep_examples and (len(example_paths) < n_example_paths)
        path = None
        if save_this:
            path = np.empty((N_mc, 3), dtype=float)
            path[0] = x_curr

        # Early termination loop
        for k in range(N_mc - 1):
            # Check termination first (hit unsafe or goal)
            if in_box_3d(x_curr, unsafe_range):
                fail += 1
                if save_this:
                    example_paths.append(path[:k + 1].copy())
                break

            if in_box_3d(x_curr, goal_range):
                success += 1
                if save_this:
                    example_paths.append(path[:k + 1].copy())
                break

            # One EM step
            u = get_u(x_curr, controller=controller)   # (3,)
            drift = f_np(x_curr, u)                    # (3,)
            g_vec = g_diag_np(x_curr)                  # (3,)

            x_curr = x_curr + drift * dt_mc + g_vec * (sqrt_dt * rng_mc.standard_normal(size=3))

            if save_this:
                path[k + 1] = x_curr
        else:
            # Only executes if the for-loop did NOT break -> timed out
            timeout += 1
            if save_this:
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
        T_mc=20.0,
        dt_mc=0.005,
        seed_mc=0
    )
    print("Reach-avoid MC estimate (3D):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    test_single_traj_run(controller=None, n_traj=20)

    # If you have a trained 3D controller:
    # from control_network import LinearControlNN3D
    control_net = load_control_net(OUTPUT_DIR / "eval_bundle.pth")
    test_single_traj_run(controller=control_net, n_traj=20)

    test_mc(controller=None)
    test_mc(controller=control_net)


if __name__ == "__main__":
    main()
