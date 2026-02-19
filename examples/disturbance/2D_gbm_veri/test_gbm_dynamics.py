"""
Empirical Validation of 2D Geometric Brownian Motion (GBM)
- Supports:
    (i) additive bounded disturbance w(t) in drift
    (ii) set-valued linear parameters A(t) in drift:  dx = A(t)x dt + g(x)dW,
         where A_ij(t) ∈ [A_L_ij, A_U_ij]
- Can simulate random or worst-case (w.r.t. ∇V) uncertainty
- Plots uncertainty time history
- Keeps the structure as close as possible to your existing code
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.control_network import LinearControlNN
from src.save_load_utils import load_eval_bundle
from src.network import create_V  # load V_net for worstcase modes

# NEW: import set-valued drift classes from src/
from src.set_values import AdditiveBoxSetDrift, LinearIntervalSetDrift


# ----------------------------
# Sets / boxes
# ----------------------------
X_bounds = {"x1_min": -100.0, "x1_max": 100.0, "x2_min": -100.0, "x2_max": 100.0}
X_init_bounds = {"x1_min": 45.0, "x1_max": 55.0, "x2_min": -55.0, "x2_max": -45.0}
X_goal_bounds = {"x1_min": -25.0, "x1_max": 25.0, "x2_min": -25.0, "x2_max": 25.0}
X_unsafe_1 = {"x1_min": -100.0, "x1_max": -80.0, "x2_min": -100.0, "x2_max": 100.0}


def in_box(x1, x2, box):
    return (box["x1_min"] <= x1 <= box["x1_max"]) and (box["x2_min"] <= x2 <= box["x2_max"])


# ----------------------------
# Bundle loaders
# ----------------------------
def load_control_net(bundle_path, device="cpu"):
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    control_net = LinearControlNN().to(device)
    if bundle.get("control_state_dict", None) is not None:
        control_net.load_state_dict(bundle["control_state_dict"])
    control_net.eval()
    return control_net


def load_value_net(bundle_path, device="cpu"):
    """
    Loads V_net using the saved hyperparameters (architecture) and weights.
    Needed only if you want mode='worstcase_V'.
    """
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    params = bundle["hyperparameters"]
    net_cfg = params["network"] if isinstance(params, dict) else params.network
    V_net = create_V(net_cfg).to(device)
    V_net.load_state_dict(bundle["V_state_dict"])
    V_net.eval()
    return V_net


# ----------------------------
# Nominal GBM dynamics (matches main.py structure)
# ----------------------------
def f_nominal(x, u):
    """
    Nominal drift:
      f1 = -1.5 x1 + 1.0 x2 + u1
      f2 = -1.0 x1 - 1.5 x2 + u2
    """
    x1, x2 = x
    u1, u2 = u
    f1 = -1.5 * x1 + 1.0 * x2 + u1
    f2 = -1.0 * x1 - 1.5 * x2 + u2
    return np.array([f1, f2], dtype=float)


def g(x):
    """
    Diagonal diffusion for 2D Brownian:
      dx = ... + diag(0.2*x1, 0.2*x2) dW
    returns (2,2)
    """
    x1, x2 = x
    return np.array([[0.2 * x1, 0.0],
                     [0.0,      0.2 * x2]], dtype=float)


# ----------------------------
# Torch nominal drift (for set-valued drift modules, if you want consistency)
# ----------------------------
def f_nominal_torch(x: torch.Tensor) -> torch.Tensor:
    # x: (N,2) -> (N,2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    f1 = -1.5 * x1 + 1.0 * x2
    f2 = -1.0 * x1 - 1.5 * x2
    return torch.stack([f1, f2], dim=1)


# ----------------------------
# Uncertainty policies
# ----------------------------
def sample_uniform_box(rng, low, high):
    return rng.uniform(low, high).astype(float)


def sample_corner_box(rng, rad):
    s = rng.choice([-1.0, 1.0], size=rad.shape)
    return (s * rad).astype(float)


def worstcase_wrt_V_box(V_net, x_np, rad, device="cpu"):
    """
    argmax_{|w_i|<=rad_i} <∇V(x), w> = rad ⊙ sign(∇V(x))
    """
    xt = torch.tensor(x_np.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)
    V = V_net(xt)
    grad = torch.autograd.grad(V.sum(), xt, create_graph=False)[0].detach().cpu().numpy().reshape(-1)
    return (rad * np.sign(grad)).astype(float)


def worstcase_A_wrt_V(V_net, x_np, A_L, A_U, device="cpu"):
    """
    Per-entry worst-case for <∇V(x), A x>:
      A_ij = A_U_ij if (p_i * x_j) >= 0 else A_L_ij,
    where p = ∇V(x).
    """
    xt = torch.tensor(x_np.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)
    V = V_net(xt)
    p = torch.autograd.grad(V.sum(), xt, create_graph=False)[0].detach().cpu().numpy().reshape(-1)  # (2,)
    x = x_np.reshape(-1)  # (2,)

    s = np.sign(np.outer(p, x))  # (2,2)
    return np.where(s >= 0.0, A_U, A_L).astype(float)


# ----------------------------
# Single run + animation + uncertainty plot
# ----------------------------
def test_single_traj_run(
    controller=None,
    V_net=None,
    # NEW: choose uncertainty type
    unc_type="additive",      # "none" | "additive" | "linear_interval"
    mode="uniform",           # for additive: "none"|"uniform"|"corners"|"worstcase_V"
                              # for A:       "midpoint"|"uniform"|"worstcase_V"
    # additive params
    d=np.array([0.1, 0.1], dtype=float),
    # linear interval params
    A_L=None,                 # (2,2)
    A_U=None,                 # (2,2)
    # sim params
    T=10.0,
    seed=None,
    device="cpu",
    plot_uncertainty=True,
    # multi-traj
    n_traj=10,
    shared_uncertainty=False,   # shared w(t) or shared A(t) (not allowed with worstcase_V)
    unc_plot_max=5,
):
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    print(f"[test_single_traj_run] seed={seed}, n_traj={n_traj}, unc_type={unc_type}, mode={mode}, shared={shared_uncertainty}")

    def get_u(x1, x2):
        if controller is None:
            return np.zeros(2, dtype=float)
        if hasattr(controller, "forward"):
            with torch.no_grad():
                xt = torch.tensor([[x1, x2]], dtype=torch.float32, device=device)
                u_val = controller(xt).squeeze(0).detach().cpu().numpy()
        else:
            u_val = controller(np.array([x1, x2], dtype=float))
        return u_val.astype(float)

    dt = 0.01
    N = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, N)

    x = np.zeros((n_traj, N, 2), dtype=float)
    u_hist = np.zeros((n_traj, N, 2), dtype=float)

    # histories (only one of them will be used)
    w_hist = None
    A_hist = None

    if unc_type == "additive":
        w_hist = np.zeros((n_traj, N, 2), dtype=float)
        # also instantiate the set-valued module (so the config matches main.py, even if sim uses numpy)
        _ = AdditiveBoxSetDrift(f_nominal_torch, torch.tensor(d, dtype=torch.float32)).to(device)

    if unc_type == "linear_interval":
        if A_L is None or A_U is None:
            raise ValueError("unc_type='linear_interval' requires A_L and A_U.")
        A_L = np.asarray(A_L, dtype=float)
        A_U = np.asarray(A_U, dtype=float)
        A_hist = np.zeros((n_traj, N, 2, 2), dtype=float)
        _ = LinearIntervalSetDrift(torch.tensor(A_L, dtype=torch.float32),
                                  torch.tensor(A_U, dtype=torch.float32)).to(device)

    # init
    for i in range(n_traj):
        x[i, 0, 0] = rng.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"])
        x[i, 0, 1] = rng.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"])

    # shared uncertainty sequence
    w_shared = None
    A_shared = None
    if shared_uncertainty:
        if mode == "worstcase_V":
            raise ValueError("shared_uncertainty=True is incompatible with mode='worstcase_V' (state dependent).")

        if unc_type == "additive":
            w_shared = np.zeros((N, 2), dtype=float)
            for k in range(N):
                if mode == "none":
                    w_shared[k] = np.zeros(2)
                elif mode == "uniform":
                    w_shared[k] = sample_uniform_box(rng, -d, d)
                elif mode == "corners":
                    w_shared[k] = sample_corner_box(rng, d)
                else:
                    raise ValueError(f"Unknown additive mode: {mode}")

        if unc_type == "linear_interval":
            A_shared = np.zeros((N, 2, 2), dtype=float)
            if mode == "midpoint":
                A_shared[:, :, :] = 0.5 * (A_L + A_U)
            elif mode == "uniform":
                for k in range(N):
                    A_shared[k] = sample_uniform_box(rng, A_L, A_U)
            else:
                raise ValueError(f"Unknown linear_interval mode: {mode}")

    # simulate
    for k in range(N - 1):
        dW_all = np.sqrt(dt) * rng.standard_normal(size=(n_traj, 2))
        for i in range(n_traj):
            x_curr = x[i, k].copy()
            u = get_u(x_curr[0], x_curr[1])
            u_hist[i, k] = u

            # drift
            if unc_type == "none":
                drift = f_nominal(x_curr, u)

            elif unc_type == "additive":
                if shared_uncertainty:
                    w = w_shared[k].copy()
                else:
                    if mode == "none":
                        w = np.zeros(2)
                    elif mode == "uniform":
                        w = sample_uniform_box(rng, -d, d)
                    elif mode == "corners":
                        w = sample_corner_box(rng, d)
                    elif mode == "worstcase_V":
                        if V_net is None:
                            raise ValueError("mode='worstcase_V' requires V_net.")
                        w = worstcase_wrt_V_box(V_net, x_curr, d, device=device)
                    else:
                        raise ValueError(f"Unknown additive mode: {mode}")
                w_hist[i, k] = w
                drift = f_nominal(x_curr, u) + w

            elif unc_type == "linear_interval":
                if shared_uncertainty:
                    A = A_shared[k].copy()
                else:
                    if mode == "midpoint":
                        A = 0.5 * (A_L + A_U)
                    elif mode == "uniform":
                        A = sample_uniform_box(rng, A_L, A_U)
                    elif mode == "worstcase_V":
                        if V_net is None:
                            raise ValueError("mode='worstcase_V' requires V_net.")
                        A = worstcase_A_wrt_V(V_net, x_curr, A_L, A_U, device=device)
                    else:
                        raise ValueError(f"Unknown linear_interval mode: {mode}")
                A_hist[i, k] = A
                drift = A @ x_curr  # (2,)   (if you want +u, change to: A@x_curr + u)

            else:
                raise ValueError(f"Unknown unc_type: {unc_type}")

            diff = g(x_curr)
            x_next = x_curr + drift * dt + diff @ dW_all[i]
            x[i, k + 1] = x_next

    u_hist[:, -1] = u_hist[:, -2]
    if w_hist is not None:
        if shared_uncertainty:
            w_hist[:, :, :] = w_shared[None, :, :]
        else:
            w_hist[:, -1] = w_hist[:, -2]
    if A_hist is not None:
        if shared_uncertainty:
            A_hist[:, :, :, :] = A_shared[None, :, :, :]
        else:
            A_hist[:, -1] = A_hist[:, -2]

    # -----------------------------------
    # Animate in phase plane
    # -----------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax_phase = fig.add_subplot(1, 1, 1)

    title = "GBM (u=0)" if controller is None else "GBM (controlled)"
    title += f" + {unc_type}({mode})"
    if shared_uncertainty:
        title += " [shared]"

    ax_phase.set_xlim(X_bounds["x1_min"], X_bounds["x1_max"])
    ax_phase.set_ylim(X_bounds["x2_min"], X_bounds["x2_max"])
    ax_phase.set_xlabel(r"$x_1$")
    ax_phase.set_ylabel(r"$x_2$")
    ax_phase.set_title(title + " phase plane")

    # Domain
    ax_phase.add_patch(Rectangle((X_bounds["x1_min"], X_bounds["x2_min"]),
                                 X_bounds["x1_max"] - X_bounds["x1_min"],
                                 X_bounds["x2_max"] - X_bounds["x2_min"],
                                 fill=False, lw=1.5))
    ax_phase.text(X_bounds["x1_min"] + 0.1, X_bounds["x2_max"] - 1.5, r"$X$")

    # Init
    ax_phase.add_patch(Rectangle((X_init_bounds["x1_min"], X_init_bounds["x2_min"]),
                                 X_init_bounds["x1_max"] - X_init_bounds["x1_min"],
                                 X_init_bounds["x2_max"] - X_init_bounds["x2_min"],
                                 alpha=0.18, linestyle="--", lw=1.5))
    ax_phase.text(X_init_bounds["x1_min"] + 0.1, X_init_bounds["x2_max"] - 0.5, r"$X_{\mathrm{init}}$")

    # Goal
    ax_phase.add_patch(Rectangle((X_goal_bounds["x1_min"], X_goal_bounds["x2_min"]),
                                 X_goal_bounds["x1_max"] - X_goal_bounds["x1_min"],
                                 X_goal_bounds["x2_max"] - X_goal_bounds["x2_min"],
                                 alpha=0.20, color="green"))
    ax_phase.text(X_goal_bounds["x1_min"] + 0.1, X_goal_bounds["x2_max"] - 0.7, r"$X_{\mathrm{goal}}$")

    # Unsafe
    ax_phase.add_patch(Rectangle((X_unsafe_1["x1_min"], X_unsafe_1["x2_min"]),
                                 X_unsafe_1["x1_max"] - X_unsafe_1["x1_min"],
                                 X_unsafe_1["x2_max"] - X_unsafe_1["x2_min"],
                                 alpha=0.25, color="red"))
    ax_phase.text(X_unsafe_1["x1_min"] + 0.1, X_unsafe_1["x2_max"] - 0.7, r"$X_{\mathrm{unsafe}}$")

    time_text = ax_phase.text(0.80, 0.92, "", transform=ax_phase.transAxes)
    summary_text = ax_phase.text(0.02, 0.95, "", transform=ax_phase.transAxes)

    traj_lines = [ax_phase.plot([], [], lw=1.0)[0] for _ in range(n_traj)]
    points = [ax_phase.plot([], [], marker="o", markersize=3, linestyle="None")[0] for _ in range(n_traj)]

    def init_anim():
        for ln, pt in zip(traj_lines, points):
            ln.set_data([], [])
            pt.set_data([], [])
        time_text.set_text("")
        summary_text.set_text("")
        return traj_lines + points + [time_text, summary_text]

    def update(frame):
        n_goal = 0
        n_unsafe = 0
        for i in range(n_traj):
            x1, x2 = x[i, frame]
            traj_lines[i].set_data(x[i, :frame + 1, 0], x[i, :frame + 1, 1])
            points[i].set_data([x1], [x2])
            if in_box(x1, x2, X_goal_bounds):
                n_goal += 1
            if in_box(x1, x2, X_unsafe_1):
                n_unsafe += 1
        time_text.set_text(f"t = {t_grid[frame]:.2f}s")
        summary_text.set_text(f"in goal: {n_goal}/{n_traj}   in unsafe: {n_unsafe}/{n_traj}")
        return traj_lines + points + [time_text, summary_text]

    skip = 5
    frames = range(0, N, skip)
    _ = FuncAnimation(fig, update, frames=frames, init_func=init_anim, blit=True, interval=30)
    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # Uncertainty time history
    # -----------------------------------
    if plot_uncertainty:
        if unc_type == "additive" and w_hist is not None:
            m = min(n_traj, unc_plot_max)
            fig2, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            for i in range(m):
                ax[0].plot(t_grid, w_hist[i, :, 0], alpha=0.7)
                ax[1].plot(t_grid, w_hist[i, :, 1], alpha=0.7)
            ax[0].axhline(+d[0], linestyle="--")
            ax[0].axhline(-d[0], linestyle="--")
            ax[1].axhline(+d[1], linestyle="--")
            ax[1].axhline(-d[1], linestyle="--")
            ax[0].set_ylabel("w1(t)")
            ax[1].set_ylabel("w2(t)")
            ax[1].set_xlabel("time (s)")
            ax[0].grid(True)
            ax[1].grid(True)
            plt.tight_layout()
            plt.show()

        if unc_type == "linear_interval" and A_hist is not None:
            m = min(n_traj, unc_plot_max)
            fig2, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            for i in range(m):
                ax[0].plot(t_grid, A_hist[i, :, 0, 0], alpha=0.7)
                ax[1].plot(t_grid, A_hist[i, :, 0, 1], alpha=0.7)
            ax[0].set_ylabel("A11(t)")
            ax[1].set_ylabel("A12(t)")
            ax[1].set_xlabel("time (s)")
            ax[0].grid(True)
            ax[1].grid(True)
            plt.tight_layout()
            plt.show()


# ----------------------------
# MC reach-avoid (simple)
# ----------------------------
def estimate_reach_avoid_mc(
    controller=None,
    V_net=None,
    unc_type="additive",    # "none" | "additive" | "linear_interval"
    mode="uniform",
    d=np.array([0.1, 0.1], dtype=float),
    A_L=None,
    A_U=None,
    n_mc=2000,
    T_mc=4.0,
    dt_mc=0.005,
    seed_mc=123,
    device="cpu",
):
    if seed_mc is None:
        seed_mc = int(np.random.SeedSequence().entropy)
    rng_mc = np.random.default_rng(seed_mc)
    print(f"[estimate_reach_avoid_mc] seed={seed_mc}, unc_type={unc_type}, mode={mode}")

    N_mc = int(T_mc / dt_mc) + 1

    def get_u(x1, x2):
        if controller is None:
            return np.zeros(2, dtype=float)
        if hasattr(controller, "forward"):
            with torch.no_grad():
                xt = torch.tensor([[x1, x2]], dtype=torch.float32, device=device)
                u_val = controller(xt).squeeze(0).detach().cpu().numpy()
        else:
            u_val = controller(np.array([x1, x2], dtype=float))
        return u_val.astype(float)

    if unc_type == "linear_interval":
        if A_L is None or A_U is None:
            raise ValueError("unc_type='linear_interval' requires A_L and A_U.")
        A_L = np.asarray(A_L, dtype=float)
        A_U = np.asarray(A_U, dtype=float)
        # instantiate for consistency (not used directly)
        _ = LinearIntervalSetDrift(torch.tensor(A_L, dtype=torch.float32),
                                  torch.tensor(A_U, dtype=torch.float32)).to(device)

    if unc_type == "additive":
        _ = AdditiveBoxSetDrift(f_nominal_torch, torch.tensor(d, dtype=torch.float32)).to(device)

    success = 0
    fail = 0
    timeout = 0

    for _ in range(n_mc):
        x1 = rng_mc.uniform(X_init_bounds["x1_min"], X_init_bounds["x1_max"])
        x2 = rng_mc.uniform(X_init_bounds["x2_min"], X_init_bounds["x2_max"])
        outcome = False

        for _k in range(N_mc - 1):
            if in_box(x1, x2, X_unsafe_1):
                fail += 1
                outcome = True
                break
            if in_box(x1, x2, X_goal_bounds):
                success += 1
                outcome = True
                break

            u = get_u(x1, x2)
            x_curr = np.array([x1, x2], dtype=float)

            if unc_type == "none":
                drift = f_nominal(x_curr, u)

            elif unc_type == "additive":
                if mode == "none":
                    w = np.zeros(2)
                elif mode == "uniform":
                    w = sample_uniform_box(rng_mc, -d, d)
                elif mode == "corners":
                    w = sample_corner_box(rng_mc, d)
                elif mode == "worstcase_V":
                    if V_net is None:
                        raise ValueError("mode='worstcase_V' requires V_net.")
                    w = worstcase_wrt_V_box(V_net, x_curr, d, device=device)
                else:
                    raise ValueError(f"Unknown additive mode: {mode}")
                drift = f_nominal(x_curr, u) + w

            elif unc_type == "linear_interval":
                if mode == "midpoint":
                    A = 0.5 * (A_L + A_U)
                elif mode == "uniform":
                    A = sample_uniform_box(rng_mc, A_L, A_U)
                elif mode == "worstcase_V":
                    if V_net is None:
                        raise ValueError("mode='worstcase_V' requires V_net.")
                    A = worstcase_A_wrt_V(V_net, x_curr, A_L, A_U, device=device)
                else:
                    raise ValueError(f"Unknown linear_interval mode: {mode}")
                drift = A @ x_curr

            else:
                raise ValueError(f"Unknown unc_type: {unc_type}")

            diff = g(x_curr)
            dW = np.sqrt(dt_mc) * rng_mc.standard_normal(size=2)
            x_next = x_curr + drift * dt_mc + diff @ dW
            x1, x2 = x_next

        if not outcome:
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


def test_mc(**kwargs):
    p, stats = estimate_reach_avoid_mc(**kwargs)
    print("Reach-avoid MC estimate:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    V_net = None
    controller = None

    # -------------------------------------------------------------------------
    # CASE 2) set-valued linear parameters A(t)
    # -------------------------------------------------------------------------
    d = 0.2  # interval radius
    A_nom = torch.tensor([[-1.5,  1.0],
                        [-1.0, -1.5]], dtype=torch.float32)
    A_L = A_nom.clone()
    A_U = A_nom.clone()
    A_L = A_nom - d
    A_U = A_nom + d

    test_single_traj_run(
        controller=controller,
        V_net=V_net,
        unc_type="linear_interval",
        mode="uniform",      # "midpoint" | "uniform" | "worstcase_V"
        A_L=A_L,
        A_U=A_U,
        T=10.0,
        seed=0,
        device="cpu",
        plot_uncertainty=True,
        n_traj=10,
        shared_uncertainty=False,
    )

    # MC examples (pick matching unc_type/mode)
    for mode in ["midpoint", "uniform"]:
        test_mc(
            controller=controller,
            V_net=V_net,
            unc_type="linear_interval",
            mode=mode,
            A_L=A_L,
            A_U=A_U,
            n_mc=1000,
            T_mc=8.0,
            dt_mc=0.005,
            seed_mc=0,
            device="cpu",
        )


if __name__ == "__main__":
    main()
