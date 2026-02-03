"""
Empirical Validation of 2D Geometric Brownian Motion (GBM)
where the goal set does not contain the equilibrium
"""

import numpy as np
import torch
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

from src.control_network import GBMControlNN
from src.save_load_utils import load_eval_bundle

X_bounds = {
    "x1_min": -100.0, "x1_max": 100.0,
    "x2_min": -100.0, "x2_max": 100.0
}

X_init_bounds = {
    "x1_min": 45.0, "x1_max": 55.0,
    "x2_min": -55.0,   "x2_max": -45.0
}

X_goal_bounds = {
    "x1_min":  20.0,  "x1_max": 40.0,
    "x2_min": -25.0,  "x2_max": 25.0
}

# we only define one unsafe because out of X_bounds is also defined as unsafe
X_unsafe_1 = {
    "x1_min": -100.0,   "x1_max": -80.0,
    "x2_min": -100.0,   "x2_max": 100.0
}


def in_box(x1, x2, box):
    return (box["x1_min"] <= x1 <= box["x1_max"]) and (box["x2_min"] <= x2 <= box["x2_max"])


# Control Network
def load_control_net(bundle_path, device="cpu"):
    
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    
    # 1) create a fresh network with SAME architecture as when saved
    control_net = GBMControlNN(input_dim=2).to(device)

    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    # 4) eval mode for rollout
    control_net.eval()
    return control_net


"""Stochastic Inverted Pendulum Dynamics"""
def f(x, u):
    """Drift dynamics f(x,u) """
    x1, x2 = x
    u1, u2 = u
    dx1_dt = -0.5*x1 + x2 + u1
    dx2_dt = -1.0*x1 -0.5*x2 + u2
    return np.array([dx1_dt, dx2_dt], dtype=float)


def g(x):
    """Diffusion vector g(x) for scalar Wiener dW."""
    x1, x2 = x
    return np.array([[0.2*x1, 0.0],
                     [0.0,    0.2*x2]], dtype=float)


def test_single_traj_run(controller=None, T=10.0, seed=None, N_traj=20, save_path=None, show=True):
    """
    Multi-trajectory SDE rollout + animation (2D):
      LEFT:  phase plane (x1,x2) with specs + trajectories
      RIGHT: controls vs time (u1 and u2, stacked)

    controller:
      - None  -> u(x)=0
      - callable(x_np)->u in R^2
      - torch nn.Module mapping R^2->R^2

    N_traj:
      - number of trajectories, each starting from X_init_bounds
    """
    # ----------------------------
    # RNG
    # ----------------------------
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed = {seed} | N_traj = {N_traj}")

    # ----------------------------
    # helper to get u(x) in R^2
    # ----------------------------
    def get_u(x1, x2):
        if controller is None:
            return np.zeros(2, dtype=float)
        if "torch" in globals() and hasattr(controller, "forward"):
            with torch.no_grad():
                xt = torch.tensor([[x1, x2]], dtype=torch.float32)
                u_val = controller(xt)
                u_val = u_val.squeeze(0).detach().cpu().numpy()   # (2,)
        else:
            u_val = controller(np.array([x1, x2], dtype=float))
        return u_val

    # Simulation horizon
    dt = 0.01
    N  = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, N)

    # ----------------------------
    # Initial conditions (sample N_traj from X_init)
    # ----------------------------
    x0 = np.zeros((N_traj, 2), dtype=float)
    x0[:, 0] = rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"], size=N_traj)
    x0[:, 1] = rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"], size=N_traj)
    print("x0[0] =", x0[0])

    # Paths and controls
    X = np.zeros((N, N_traj, 2), dtype=float)
    U = np.zeros((N, N_traj, 2), dtype=float)
    X[0] = x0

    # Per-trajectory terminal bookkeeping
    done = np.zeros((N_traj,), dtype=bool)
    terminal = np.array(["OK"] * N_traj, dtype=object)

    # ----------------------------
    # Euler–Maruyama simulation
    # ----------------------------
    for k in range(N - 1):
        xk = X[k]  # (N_traj,2)

        # controls
        uk = np.zeros((N_traj, 2), dtype=float)
        for i in range(N_traj):
            if done[i]:
                uk[i] = 0.0
                continue
            uk[i] = get_u(xk[i, 0], xk[i, 1])
        U[k] = uk

        # step each traj
        for i in range(N_traj):
            if done[i]:
                X[k + 1, i] = X[k, i]
                U[k + 1, i] = U[k, i]
                continue

            x_curr = X[k, i]
            drift = f(x_curr, U[k, i])          # (2,)
            diff  = g(x_curr)                   # (2,2)

            dW = np.sqrt(dt) * rng.standard_normal(size=2)
            x_next = x_curr + drift * dt + diff @ dW

            X[k + 1, i] = x_next

            # terminal checks (priority: OOB -> unsafe -> goal)
            x1n, x2n = float(x_next[0]), float(x_next[1])
            in_bounds = in_box(x1n, x2n, X_bounds)
            in_goal   = in_box(x1n, x2n, X_goal_bounds)
            in_unsafe = in_box(x1n, x2n, X_unsafe_1)

            if (not in_bounds) and (not in_goal):
                done[i] = True
                terminal[i] = "OOB"
            elif in_unsafe:
                done[i] = True
                terminal[i] = "UNSAFE"
            elif in_goal:
                done[i] = True
                terminal[i] = "GOAL"

    U[-1] = U[-2]

    # ----------------------------
    # Build animation figure (side-by-side: phase | controls)
    #   (simplified: draw full traj once; animate only moving dots + cursor)
    # ----------------------------
    fig = plt.figure(figsize=(14, 6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.25)

    ax_phase = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, hspace=0.15)
    ax_u1 = fig.add_subplot(right[0, 0])
    ax_u2 = fig.add_subplot(right[1, 0], sharex=ax_u1)

    title = "GBM (u=0)" if controller is None else "GBM (controlled)"
    ax_phase.set_title(title + f" | Phase plane ({N_traj} trajectories)")
    ax_phase.set_xlim(X_bounds["x1_min"], X_bounds["x1_max"])
    ax_phase.set_ylim(X_bounds["x2_min"], X_bounds["x2_max"])
    ax_phase.set_xlabel(r"$x_1$")
    ax_phase.set_ylabel(r"$x_2$")
    ax_phase.grid(True, alpha=0.25)

    time_text = ax_phase.text(0.60, 0.96, "", transform=ax_phase.transAxes)
    status_text = ax_phase.text(0.60, 0.90, "", transform=ax_phase.transAxes)
    u_text = ax_phase.text(0.60, 0.84, "", transform=ax_phase.transAxes) if controller is not None else None

    # Specs rectangles
    ax_phase.add_patch(Rectangle(
        (X_bounds["x1_min"], X_bounds["x2_min"]),
        X_bounds["x1_max"] - X_bounds["x1_min"],
        X_bounds["x2_max"] - X_bounds["x2_min"],
        fill=False, lw=1.5
    ))
    ax_phase.add_patch(Rectangle(
        (X_init_bounds["x1_min"], X_init_bounds["x2_min"]),
        X_init_bounds["x1_max"] - X_init_bounds["x1_min"],
        X_init_bounds["x2_max"] - X_init_bounds["x2_min"],
        alpha=0.18, linestyle="--", lw=1.5
    ))
    ax_phase.add_patch(Rectangle(
        (X_goal_bounds["x1_min"], X_goal_bounds["x2_min"]),
        X_goal_bounds["x1_max"] - X_goal_bounds["x1_min"],
        X_goal_bounds["x2_max"] - X_goal_bounds["x2_min"],
        alpha=0.20, color="green"
    ))
    ax_phase.add_patch(Rectangle(
        (X_unsafe_1["x1_min"], X_unsafe_1["x2_min"]),
        X_unsafe_1["x1_max"] - X_unsafe_1["x1_min"],
        X_unsafe_1["x2_max"] - X_unsafe_1["x2_min"],
        alpha=0.25, color="red"
    ))

    # Pre-draw phase trajectories (faint)
    for i in range(N_traj):
        ax_phase.plot(X[:, i, 0], X[:, i, 1], lw=1.0, alpha=0.35)

    # Moving dots in phase (all traj at once)
    phase_sc = ax_phase.scatter(X[0, :, 0], X[0, :, 1], s=18)

    # Controls axes
    ax_u1.set_title("Controls vs time (all trajectories)")
    ax_u1.set_ylabel("u1")
    ax_u2.set_ylabel("u2")
    ax_u2.set_xlabel("t [s]")
    ax_u1.grid(True, alpha=0.3)
    ax_u2.grid(True, alpha=0.3)
    ax_u1.set_xlim(0.0, float(T))

    def _set_ylim(ax, y):
        y = np.asarray(y, dtype=float)
        y0, y1 = float(np.min(y)), float(np.max(y))
        if np.isclose(y0, y1):
            y0 -= 1.0
            y1 += 1.0
        pad = 0.10 * (y1 - y0)
        ax.set_ylim(y0 - pad, y1 + pad)

    _set_ylim(ax_u1, U[:, :, 0])
    _set_ylim(ax_u2, U[:, :, 1])

    # Pre-draw controls (faint)
    for i in range(N_traj):
        ax_u1.plot(t_grid, U[:, i, 0], lw=1.0, alpha=0.35)
        ax_u2.plot(t_grid, U[:, i, 1], lw=1.0, alpha=0.35)

    # Moving dots in controls (all traj at once)
    u1_sc = ax_u1.scatter(np.zeros(N_traj), U[0, :, 0], s=18)
    u2_sc = ax_u2.scatter(np.zeros(N_traj), U[0, :, 1], s=18)

    # Cursor lines
    vline1 = ax_u1.axvline(0.0, lw=1.0, alpha=0.6)
    vline2 = ax_u2.axvline(0.0, lw=1.0, alpha=0.6)

    def init_anim():
        phase_sc.set_offsets(np.c_[X[0, :, 0], X[0, :, 1]])
        u1_sc.set_offsets(np.c_[np.zeros(N_traj), U[0, :, 0]])
        u2_sc.set_offsets(np.c_[np.zeros(N_traj), U[0, :, 1]])
        vline1.set_xdata([0.0, 0.0])
        vline2.set_xdata([0.0, 0.0])
        time_text.set_text("")
        status_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        return (phase_sc, u1_sc, u2_sc, vline1, vline2, time_text, status_text) + (() if u_text is None else (u_text,))

    def update(frame):
        frame = int(frame)
        tt = float(t_grid[frame])

        phase_sc.set_offsets(np.c_[X[frame, :, 0], X[frame, :, 1]])
        u1_sc.set_offsets(np.c_[np.full(N_traj, tt), U[frame, :, 0]])
        u2_sc.set_offsets(np.c_[np.full(N_traj, tt), U[frame, :, 1]])
        vline1.set_xdata([tt, tt])
        vline2.set_xdata([tt, tt])

        # aggregate status counts
        oob = unsafe = goal = 0
        for i in range(N_traj):
            x1, x2 = X[frame, i]
            in_bounds = in_box(x1, x2, X_bounds)
            in_goal   = in_box(x1, x2, X_goal_bounds)
            in_unsafe = in_box(x1, x2, X_unsafe_1)
            if (not in_bounds) and (not in_goal):
                oob += 1
            elif in_unsafe:
                unsafe += 1
            elif in_goal:
                goal += 1

        time_text.set_text(f"t = {tt:.2f}s")
        status_text.set_text(f"GOAL={goal}  UNSAFE={unsafe}  OOB={oob}")

        if u_text is not None:
            u_vec = U[frame, 0]
            u_text.set_text(f"u[traj0] = [{u_vec[0]:.2f}, {u_vec[1]:.2f}]")

        return (phase_sc, u1_sc, u2_sc, vline1, vline2, time_text, status_text) + (() if u_text is None else (u_text,))

    frames = range(0, N, 5)
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init_anim,
        interval=30,
        blit=False,   # IMPORTANT: macOS-safe
    )

    if save_path is not None:
        ani.save(save_path, dpi=150)

    if show:
        plt.show()

    return {"t": t_grid, "X": X, "U": U, "x0": x0, "terminal": terminal, "fig": fig, "ani": ani}


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
      - None  -> u(x) = 0 in R^2
      - callable(x_np: shape (2,)) -> u_np: shape (2,)
      - torch nn.Module mapping R^2 -> R^2
    """
    # RNG
    if seed_mc is None:
        seed_mc = int(np.random.SeedSequence().entropy)
    rng_mc = np.random.default_rng(seed_mc)
    print(f"[estimate_reach_avoid_mc] seed = {seed_mc}")

    N_mc = int(T_mc / dt_mc) + 1

    # helper to get u(x) in R^2
    def get_u(x1, x2):
        if controller is None:
            return np.zeros(2, dtype=float)

        if "torch" in globals() and hasattr(controller, "forward"):
            # torch net case
            with torch.no_grad():
                xt = torch.tensor([[x1, x2]], dtype=torch.float32)  # (1, 2)
                u_val = controller(xt)                              # (1, 2)
                u_val = u_val.squeeze(0).detach().cpu().numpy()     # (2,)
        else:
            # python callable: controller(x_np) -> length-2
            u_val = controller(np.array([x1, x2], dtype=float))

        return u_val  # shape (2,)

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
            in_bounds = in_box(x1, x2, X_bounds)
            in_goal = in_box(x1, x2, X_goal_bounds)
            in_unsafe = in_box(x1, x2, X_unsafe_1)

            if not in_bounds and not in_goal:
                fail += 1
                outcome_recorded = True
                if return_example_paths and len(example_paths) < n_example_paths:
                    example_paths.append(path[:k+1].copy())
                break

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

            # Euler–Maruyama step (controlled, 2D Brownian, 2x2 diffusion)
            u = get_u(x1, x2)                          # (2,)
            x_curr = np.array([x1, x2], dtype=float)   # (2,)

            drift = f(x_curr, u)                       # (2,)
            diff  = g(x_curr)                          # (2, 2)

            dW = np.sqrt(dt_mc) * rng_mc.standard_normal(size=2)  # (2,)
            x_next = x_curr + drift * dt_mc + diff @ dW           # (2,)

            x1, x2 = x_next

            if return_example_paths and len(example_paths) < n_example_paths:
                path[k+1] = [x1, x2]

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