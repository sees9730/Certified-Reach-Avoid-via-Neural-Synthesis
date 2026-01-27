"""
Empirical Validation of 2D Geometric Brownian Motion (GBM)
"""

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.control_network import LinearControlNN
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
    "x1_min": -25.0,  "x1_max": 25.0,
    "x2_min": -25.0,  "x2_max": 25.0
}

# Unsafe = union of two rectangles
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
    control_net = LinearControlNN().to(device)

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

    # ----------------------------
    # small helper to get u
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
        u = get_u(x1, x2)
        u_hist[k] = u

        drift = f(x[k], u)
        diff  = g(x[k])

        # 2D Brownian increment
        dW = np.sqrt(dt) * rng.standard_normal(size=2)   # (2,)

        x_next = x[k] + drift * dt + diff @ dW

        x[k + 1] = x_next

    u_hist[-1] = u_hist[-2]

    # ----------------------------
    # Build animation figure
    # ----------------------------
    fig = plt.figure(figsize=(8, 6))
    ax_phase = fig.add_subplot(1, 1, 1)

    # --- Pendulum axis setup ---
    title = "GBM (u=0)" if controller is None else "GBM (controlled)"
    time_text = ax_phase.text(0.85, 0.9, "", transform=ax_phase.transAxes)
    u_text = ax_phase.text(0.8, 0.8, "", transform=ax_phase.transAxes) if controller is not None else None

    # --- Phase axis setup ---
    ax_phase.set_xlim(X_bounds["x1_min"], X_bounds["x1_max"])
    ax_phase.set_ylim(X_bounds["x2_min"], X_bounds["x2_max"])
    ax_phase.set_xlabel(r"$x_1$ (angle)")
    ax_phase.set_ylabel(r"$x_2$ (angular velocity)")
    ax_phase.set_title(title + " Phase plane with specifications")

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
    ax_phase.text(X_unsafe_1["x1_min"]+0.1, X_unsafe_1["x2_max"]-0.7, r"$X_{\mathrm{unsafe}}$")

    traj_line, = ax_phase.plot([], [], lw=1.5)
    point, = ax_phase.plot([], [], marker="o")
    status_text = ax_phase.text(0.85, 0.95, "", transform=ax_phase.transAxes)

    def init_anim():
        traj_line.set_data([], [])
        point.set_data([], [])
        time_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        status_text.set_text("")
        artists = [traj_line, point, time_text, status_text]
        if u_text is not None:
            artists.append(u_text)
        return artists

    def update(frame):
        x1, x2 = x[frame]


        traj_line.set_data(x[:frame+1, 0], x[:frame+1, 1])
        point.set_data([x1], [x2])

        in_goal = in_box(x1, x2, X_goal_bounds)
        in_unsafe = in_box(x1, x2, X_unsafe_1)
        status = "UNSAFE" if in_unsafe else ("GOAL" if in_goal else "OK")

        time_text.set_text(f"t = {t_grid[frame]:.2f}s")
        if u_text is not None:
            u_vec = u_hist[frame]
            u_str = ", ".join(f"{v:.2f}" for v in u_vec)
            u_text.set_text(f"u = [{u_str}]")
        status_text.set_text(f"status: {status}")

        artists = [traj_line, point, time_text, status_text]
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
            in_goal = in_box(x1, x2, X_goal_bounds)
            in_unsafe = in_box(x1, x2, X_unsafe_1)

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
    test_mc(controller=None)


if __name__ == "__main__":
    main()    