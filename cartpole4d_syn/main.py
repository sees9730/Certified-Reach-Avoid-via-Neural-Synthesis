from __future__ import annotations

import math
import argparse
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

# -----------------------------------------------------------------------------
# Repo paths (match your Lorentz script)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # repo_root
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.hyperparameters import Hyperparameters
from src.dynamics import Dynamics
from src.regions import Regions, Region
from src.network import create_V
from src.phi_module import create_GV
from src.discretization import discretize_regions
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.training_utils import (
    compute_total_loss_bounds,
    compute_loss_goal_bounds,
    evaluate_constraints,
    print_loss_summary,
    print_constraint_summary,
    refine_failing_cells,
    merge_passing_neighbor_cells,
)
from src.save_load_utils import save_eval_bundle, load_eval_bundle, log_loaded_training_epochs, enable_terminal_logging
from src.utils import cleanup_and_setup_directories, print_training_config
from src.visualization import (
    visualize_training_progress,
    create_summary_plots
)


torch.manual_seed(0)
np.random.seed(0)

DEG = np.pi / 180.0
pi = np.pi
G = 9.81


class CartpoleControlNN(nn.Module):
    """
    Simple MLP controller u(x) in [-Fmax, Fmax].
    Returns shape (N,1) by default (recommended).
    """
    def __init__(self, input_dim=4, hidden_dim=64, Fmax=50.0,
                 input_scale=(2*np.pi, 20.0, 10.0, 20.0)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        self.Fmax = float(Fmax)

        # store scale as float32 tensor (broadcasts over batch)
        s = torch.tensor(input_scale, dtype=torch.float32).view(1, input_dim)
        self.register_buffer("input_scale", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x/self.input_scale
        h = torch.tanh(self.fc1(x))
        u = torch.tanh(self.fc2(h)) * self.Fmax   # (N,1)
        return u


class ClosedLoopCartPole(nn.Module):
    """
    Closed-loop cart-pole continuous dynamics:
      x = [theta, theta_dot, z, z_dot]
      u = force on cart (along +z)

    Convention:
      theta = 0 is upright.
      theta > 0 tips to the right (matches your rendering: tipx = z + l*sin(theta)).
    """
    def __init__(self, controller: nn.Module, m_c=1.0, m_p=0.1, l=0.5):
        super().__init__()
        self.controller = controller
        self.m_c = float(m_c)
        self.m_p = float(m_p)
        self.l = float(l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,4) -> xdot: (N,4)
        """
        th  = x[:, 0]  # (N,)
        dth = x[:, 1]
        z   = x[:, 2]
        dz  = x[:, 3]

        # u: ensure shape (N,)
        u = self.controller(x)[:, 0]

        m_c = self.m_c
        m_p = self.m_p
        l   = self.l
        total_m = m_c + m_p

        sin_th = torch.sin(th)
        cos_th = torch.cos(th)

        # Standard cartpole equations (continuous-time)
        # temp is the horizontal acceleration contribution from force and centripetal term
        temp = m_c + m_p - m_p * cos_th**2

        # angular acceleration
        ddth = (-m_p*l*cos_th*sin_th*dth**2 + u*cos_th + total_m*G*sin_th)/ (l * temp)

        # cart acceleration
        ddz = (-m_p*l*sin_th*dth**2 + u + m_p*G*cos_th*sin_th)/temp

        return torch.stack([dth, ddth, dz, ddz], dim=1)


def animate_cartpole_state_control(
    *,
    f_cl_module,                 # ClosedLoopCartPole(controller): xdot = f(x)
    g_fn=None,                   # optional diffusion g(x) -> (N,4) diag
    init_range: np.ndarray,      # (4,2) in [theta, theta_dot, z, z_dot]
    goal_range: np.ndarray,      # (4,2)
    full_range: np.ndarray,      # (4,2)
    unsafe_boxes: np.ndarray,    # (4,2) or (K*4,2) or (K,4,2)
    device: str = "cpu",
    dt: float = 0.01,
    T: float = 8.0,
    seed: int = 0,
    # visualization
    cart_w: float = 0.35,
    cart_h: float = 0.18,
    pole_l: float | None = None,   # if None, use getattr(f_cl_module, "l", 0.5)
    pole_l_vis: float | None = None,  # optional visual-only pole length
    pole_w: float = 2.5,
    mass_r: float = 0.06,
    window_pad: float = 1.0,       # half-width padding for moving window around cart
    window_follow: bool = True,    # keep cart centered
    theta_wrap: bool = False,       # wrap theta for plotting/animation
    save_path: str | None = None,
    show: bool = True,
    controller_label: str | None = None,
    frame_skip: int = 5,
):
    """
    Layout (3 columns):
      LEFT:   cart-pole animation (z, theta) with a moving x-window so it's always visible
      MIDDLE: 4 rows state vs time (theta, theta_dot, z, z_dot) with init/unsafe/goal bands
      RIGHT:  control vs time (Force)

    State ordering:
      x = [theta, theta_dot, z, z_dot]
      u(x) = [Force]
    """
    rng = np.random.default_rng(seed)

    init_range = np.asarray(init_range, dtype=np.float32)
    goal_range = np.asarray(goal_range, dtype=np.float32)
    full_range = np.asarray(full_range, dtype=np.float32)

    # -----------------------------
    # Unsafe: accept (4,2), (K*4,2), or (K,4,2)
    # -----------------------------
    def _as_union_boxes(x_unsafe: np.ndarray) -> np.ndarray:
        x = np.asarray(x_unsafe, dtype=np.float32)
        if x.ndim == 2:
            if x.shape == (4, 2):
                return x[None, ...]
            if x.shape[1] == 2 and (x.shape[0] % 4 == 0):
                K = x.shape[0] // 4
                return x.reshape(K, 4, 2)
            raise ValueError(f"unsafe_boxes 2D must be (4,2) or (K*4,2), got {x.shape}")
        if x.ndim == 3:
            if x.shape[1:] != (4, 2):
                raise ValueError(f"unsafe_boxes 3D must be (K,4,2), got {x.shape}")
            return x
        raise ValueError(f"unsafe_boxes must be (4,2), (K*4,2), or (K,4,2), got {x.shape}")

    unsafeK = _as_union_boxes(unsafe_boxes)  # (K,4,2)

    # -----------------------------
    # Rollout
    # -----------------------------
    x0 = np.array([rng.uniform(init_range[d, 0], init_range[d, 1]) for d in range(4)], dtype=np.float32)

    N = int(T / dt) + 1
    t = np.linspace(0.0, T, N, dtype=np.float32)

    X = np.zeros((N, 4), dtype=np.float32)   # [th, dth, z, dz]
    U = np.zeros((N, 1), dtype=np.float32)   # [Force]
    X[0] = x0

    if pole_l is None:
        pole_l = float(getattr(f_cl_module, "l", 0.5))
    if pole_l_vis is None:
        pole_l_vis = pole_l

    def _force_scalar(u_t: torch.Tensor) -> float:
        # accept (1,), (1,1), scalar-ish
        u_np = u_t.detach().cpu().numpy().reshape(-1)
        if u_np.size != 1:
            raise ValueError(f"controller(x) must return one force per sample; got {tuple(u_t.shape)}")
        return float(u_np[0])

    def _gdiag_np(g_out) -> np.ndarray:
        # want diag coeffs in (4,)
        if isinstance(g_out, (float, int)):
            return np.full((4,), float(g_out), dtype=np.float32)
        if torch.is_tensor(g_out):
            g_np = g_out.detach().cpu().numpy()
        else:
            g_np = np.asarray(g_out)
        g_np = np.asarray(g_np).reshape(-1)
        if g_np.size != 4:
            raise ValueError(f"g_fn(x) must return 4 diagonal coeffs; got shape {np.shape(g_out)}")
        return g_np.astype(np.float32)

    f_cl_module.eval()
    with torch.no_grad():
        for k in range(N - 1):
            xk_t = torch.tensor(X[k:k + 1], dtype=torch.float32, device=device)  # (1,4)

            uk_t = f_cl_module.controller(xk_t)
            u_scalar = _force_scalar(uk_t)

            xdot = f_cl_module(xk_t).detach().cpu().numpy().reshape(4).astype(np.float32)

            if g_fn is None:
                xnext = X[k] + dt * xdot
            else:
                gk = _gdiag_np(g_fn(xk_t))  # (4,)
                dW = (np.sqrt(dt) * rng.standard_normal(4)).astype(np.float32)
                xnext = X[k] + dt * xdot + gk * dW

            if not np.isfinite(xnext).all():
                # stop gracefully — still animate what we have
                X = X[:k + 1]
                U = U[:k + 1]
                t = t[:k + 1]
                N = X.shape[0]
                print(f"[warn] rollout became non-finite at step {k}; truncating animation.")
                break

            X[k + 1] = xnext
            U[k, 0] = u_scalar

        if N >= 2:
            U[-1, 0] = U[-2, 0]

    # Optional: wrap theta to [-pi, pi] for nicer plots/animation
    if theta_wrap:
        th = X[:, 0]
        X[:, 0] = (th + np.pi) % (2 * np.pi) - np.pi

    # -----------------------------
    # Bands / limits helpers
    # -----------------------------
    def _is_tighter(rng_1d: np.ndarray, full_1d: np.ndarray, eps: float = 1e-9) -> bool:
        return (rng_1d[0] > full_1d[0] + eps) or (rng_1d[1] < full_1d[1] - eps)

    def _add_band(ax, lo, hi, color, label=None, alpha=0.12):
        ax.axhspan(lo, hi, color=color, alpha=alpha, label=label, zorder=0)

    def _add_state_bands(ax, dim: int, ax_label: str):
        if _is_tighter(init_range[dim], full_range[dim]):
            _add_band(ax, init_range[dim, 0], init_range[dim, 1], color="green", label=f"init ({ax_label})")
        if _is_tighter(goal_range[dim], full_range[dim]):
            _add_band(ax, goal_range[dim, 0], goal_range[dim, 1], color="blue", label=f"goal ({ax_label})")

        any_unsafe = False
        for k in range(unsafeK.shape[0]):
            if _is_tighter(unsafeK[k, dim], full_range[dim]):
                _add_band(
                    ax,
                    unsafeK[k, dim, 0],
                    unsafeK[k, dim, 1],
                    color="red",
                    label=("unsafe" if not any_unsafe else None),
                    alpha=0.10,
                )
                any_unsafe = True

    def _collect_band_extents_for_dim(d: int):
        lows, highs = [], []
        if _is_tighter(init_range[d], full_range[d]):
            lows.append(float(init_range[d, 0])); highs.append(float(init_range[d, 1]))
        if _is_tighter(goal_range[d], full_range[d]):
            lows.append(float(goal_range[d, 0])); highs.append(float(goal_range[d, 1]))
        for k in range(unsafeK.shape[0]):
            if _is_tighter(unsafeK[k, d], full_range[d]):
                lows.append(float(unsafeK[k, d, 0])); highs.append(float(unsafeK[k, d, 1]))
        if not lows:
            return None
        return (min(lows), max(highs))

    def _set_ylim_with_bands(ax, y_data, band_extents):
        y = np.asarray(y_data, dtype=np.float32)
        y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
        if band_extents is not None:
            b0, b1 = band_extents
            y0 = min(y0, b0)
            y1 = max(y1, b1)
        if not np.isfinite([y0, y1]).all() or np.isclose(y0, y1):
            y0 -= 1.0
            y1 += 1.0
        pad = 0.10 * (y1 - y0)
        ax.set_ylim(y0 - pad, y1 + pad)

    def _set_ylim(ax, y_data):
        y = np.asarray(y_data, dtype=np.float32)
        y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
        if not np.isfinite([y0, y1]).all() or np.isclose(y0, y1):
            y0 -= 1.0
            y1 += 1.0
        pad = 0.10 * (y1 - y0)
        ax.set_ylim(y0 - pad, y1 + pad)

    # -----------------------------
    # Figure layout: 4 rows x 3 cols
    # -----------------------------
    fig = plt.figure(figsize=(18, 8))
    title = "Cart-Pole rollout: animation + state + control"
    if controller_label:
        title += f"  |  Controller: {controller_label}"
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92)

    gs = fig.add_gridspec(
        4, 3,
        width_ratios=[1.35, 1.0, 1.0],
        height_ratios=[1, 1, 1, 1],
        wspace=0.30,
        hspace=0.55,
    )

    ax_cp  = fig.add_subplot(gs[:, 0])
    ax_th  = fig.add_subplot(gs[0, 1])
    ax_dth = fig.add_subplot(gs[1, 1], sharex=ax_th)
    ax_z   = fig.add_subplot(gs[2, 1], sharex=ax_th)
    ax_dz  = fig.add_subplot(gs[3, 1], sharex=ax_th)
    ax_u   = fig.add_subplot(gs[:, 2])

    # -----------------------------
    # LEFT: cart-pole animation (always visible)
    # -----------------------------
    ax_cp.set_title("Cart-Pole")
    ax_cp.set_xlabel("z")
    ax_cp.set_ylabel("height")
    ax_cp.grid(True, alpha=0.25)
    ax_cp.set_aspect("auto")  # important: don't shrink the axis

    y_ground = 0.0
    y_top = y_ground + (cart_h + 1.55 * pole_l_vis)
    ax_cp.set_ylim(-0.25 * pole_l_vis, y_top)

    # Initial window around initial z
    z0 = float(X[0, 2])
    halfw = max(window_pad, 1.2 * pole_l_vis)
    ax_cp.set_xlim(z0 - halfw, z0 + halfw)

    ground_line, = ax_cp.plot([z0 - halfw, z0 + halfw], [y_ground, y_ground], lw=2, zorder=1)

    cart = Rectangle(
        (z0 - 0.5 * cart_w, y_ground),
        cart_w, cart_h,
        facecolor="0.6", edgecolor="k", linewidth=1.5, alpha=0.9, zorder=3,
    )
    ax_cp.add_patch(cart)

    pole_line, = ax_cp.plot([], [], lw=pole_w, zorder=4)

    mass = Circle((0.0, 0.0), radius=max(mass_r, 0.06 * pole_l_vis), facecolor="k", alpha=0.9, zorder=5)
    ax_cp.add_patch(mass)

    txt_cp = ax_cp.text(0.02, 0.98, "", transform=ax_cp.transAxes, va="top")

    def _cartpole_geom(th, z):
        # pivot at top-center of cart
        px = float(z)
        py = float(y_ground + cart_h)
        tipx = px + pole_l_vis * np.sin(th)
        tipy = py + pole_l_vis * np.cos(th)
        return (px, py), (tipx, tipy)

    # -----------------------------
    # MIDDLE: states vs time
    # -----------------------------
    ax_th.set_title("States vs time")
    ax_th.set_ylabel("theta")
    ax_dth.set_ylabel("theta_dot")
    ax_z.set_ylabel("z")
    ax_dz.set_ylabel("z_dot")
    ax_dz.set_xlabel("t [s]")
    for ax in (ax_th, ax_dth, ax_z, ax_dz):
        ax.grid(True, alpha=0.3)

    _add_state_bands(ax_th,  0, "theta")
    _add_state_bands(ax_dth, 1, "theta_dot")
    _add_state_bands(ax_z,   2, "z")
    _add_state_bands(ax_dz,  3, "z_dot")

    _set_ylim_with_bands(ax_th,  X[:, 0], _collect_band_extents_for_dim(0))
    ax_th.set_ylim([-2*pi, 2*pi])
    _set_ylim_with_bands(ax_dth, X[:, 1], _collect_band_extents_for_dim(1))
    _set_ylim_with_bands(ax_z,   X[:, 2], _collect_band_extents_for_dim(2))
    _set_ylim_with_bands(ax_dz,  X[:, 3], _collect_band_extents_for_dim(3))
    ax_th.set_xlim(0.0, float(t[-1] if len(t) else T))

    handles, labels = ax_th.get_legend_handles_labels()
    if handles:
        ax_th.legend(loc="upper right", framealpha=0.85)

    l_th,  = ax_th.plot([], [], lw=2)
    l_dth, = ax_dth.plot([], [], lw=2)
    l_z,   = ax_z.plot([], [], lw=2)
    l_dz,  = ax_dz.plot([], [], lw=2)

    # -----------------------------
    # RIGHT: control vs time
    # -----------------------------
    ax_u.set_title("Control vs time")
    ax_u.set_xlabel("t [s]")
    ax_u.set_ylabel("Force")
    ax_u.grid(True, alpha=0.3)
    ax_u.set_xlim(0.0, float(t[-1] if len(t) else T))
    _set_ylim(ax_u, U[:, 0] if len(U) else np.array([0.0], dtype=np.float32))
    l_u, = ax_u.plot([], [], lw=2)

    # -----------------------------
    # Animation init/update
    # -----------------------------
    def init_anim():
        pole_line.set_data([], [])
        mass.center = (0.0, 0.0)
        cart.set_xy((float(X[0, 2]) - 0.5 * cart_w, y_ground))
        txt_cp.set_text("")

        l_th.set_data([], [])
        l_dth.set_data([], [])
        l_z.set_data([], [])
        l_dz.set_data([], [])
        l_u.set_data([], [])

        return (ground_line, cart, pole_line, mass, txt_cp, l_th, l_dth, l_z, l_dz, l_u)

    def update(i: int):
        i = int(i)
        xs = X[:i + 1]
        us = U[:i + 1]

        th = float(xs[-1, 0])
        z  = float(xs[-1, 2])
        u  = float(us[-1, 0]) if len(us) else 0.0

        # Move the view window so the cart never leaves the plot
        if window_follow:
            halfw = max(window_pad, 1.2 * pole_l_vis)
            ax_cp.set_xlim(z - halfw, z + halfw)
            ground_line.set_data([z - halfw, z + halfw], [y_ground, y_ground])

        (px, py), (tx, ty) = _cartpole_geom(th, z)

        cart.set_xy((z - 0.5 * cart_w, y_ground))
        pole_line.set_data([px, tx], [py, ty])
        mass.center = (tx, ty)

        txt_cp.set_text(
            f"t={t[i]:.2f}s\n"
            f"theta={xs[-1,0]:.3f}, theta_dot={xs[-1,1]:.3f}\n"
            f"z={xs[-1,2]:.3f}, z_dot={xs[-1,3]:.3f}\n"
            f"Force={u:.2f}"
        )

        # states
        l_th.set_data(t[:i + 1], xs[:, 0])
        l_dth.set_data(t[:i + 1], xs[:, 1])
        l_z.set_data(t[:i + 1], xs[:, 2])
        l_dz.set_data(t[:i + 1], xs[:, 3])

        # control
        l_u.set_data(t[:i + 1], us[:, 0] if len(us) else np.zeros(i + 1))

        return (ground_line, cart, pole_line, mass, txt_cp, l_th, l_dth, l_z, l_dz, l_u)

    if frame_skip < 1:
        frame_skip = 1
    frames = range(0, N, frame_skip)

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init_anim,
        interval=30,
        blit=False,
    )

    if save_path is not None:
        ani.save(save_path, dpi=150)

    if show:
        plt.show()

    return {"t": t, "X": X, "U": U, "x0": x0}


# def mc_reach_avoid(
#     *,
#     f_cl_module,              # torch module: xdot = f(x), x shape (1,3)
#     g_fn=None,                # torch callable: g(x) -> (3,) or (1,3); if None => deterministic
#     init_range: np.ndarray,   # (3,2)
#     goal_range: np.ndarray,   # (3,2)
#     full_range: np.ndarray,   # (3,2)
#     unsafe_boxes: np.ndarray, # (K*3,2) or (K,3,2) or (3,2)
#     N_trials: int = 200,
#     dt: float = 0.02,
#     T: float = 12.0,
#     seed: int = 0,
#     device: str = "cpu",
# ):
#     """Monte-Carlo reach-avoid probability over [0,T]. Uses ONLY numpy RNG (seeded)."""

#     rng = np.random.default_rng(seed)

#     init_range = np.asarray(init_range, np.float32)
#     goal_range = np.asarray(goal_range, np.float32)
#     full_range = np.asarray(full_range, np.float32)
#     ub = np.asarray(unsafe_boxes, np.float32)

#     # unsafe -> (K,3,2)
#     if ub.ndim == 2:
#         ub = ub[None, ...] if ub.shape == (3, 2) else ub.reshape(-1, 3, 2)

#     def in_box(x, box3x2):
#         return bool(np.all(x >= box3x2[:, 0]) and np.all(x <= box3x2[:, 1]))

#     def in_unsafe(x):
#         for k in range(ub.shape[0]):
#             if in_box(x, ub[k]):
#                 return True
#         return False

#     N_steps = int(T / dt) + 1
#     sqrt_dt = float(np.sqrt(dt))

#     f_cl_module.eval()

#     successes = 0
#     with torch.no_grad():
#         for _ in range(N_trials):
#             # sample x0 in init
#             x = np.array([rng.uniform(*init_range[d]) for d in range(3)], dtype=np.float32)

#             failed = False
#             for _k in range(N_steps - 1):
#                 # fail if outside full or unsafe
#                 if (not in_box(x, full_range)) or in_unsafe(x):
#                     failed = True
#                     break
#                 # succeed if in goal
#                 if in_box(x, goal_range):
#                     break

#                 xt = torch.tensor(x[None, :], dtype=torch.float32, device=device)
#                 xdot = f_cl_module(xt).detach().cpu().numpy().reshape(3).astype(np.float32)

#                 if g_fn is None:
#                     x = x + dt * xdot
#                 else:
#                     gk = g_fn(xt).detach().cpu().numpy().reshape(3).astype(np.float32)
#                     dW = (sqrt_dt * rng.standard_normal(3)).astype(np.float32)
#                     x = x + dt * xdot + gk * dW

#             if (not failed) and in_box(x, goal_range):
#                 successes += 1

#     return {
#         "p_reach_avoid": successes / float(N_trials),
#         "successes": int(successes),
#         "N_trials": int(N_trials),
#     }


def _diag_ggt_from_g(g_out: torch.Tensor, D: int) -> torch.Tensor:
    """
    Convert g(x) output to diag(GG^T) in shape (N,D), (D,) or scalar.

    Supports:
      - (N,D)      : diagonal diffusion vector
      - (N,D,m)    : general diffusion factors
      - (N,D,D)    : full matrix per sample
      - (D,D)      : constant matrix
      - (D,)       : constant diagonal vector
      - scalar     : isotropic
    """
    if isinstance(g_out, (float, int)):
        return float(g_out) ** 2

    if g_out.dim() == 0:
        return float(g_out.item()) ** 2

    if g_out.dim() == 1:
        if g_out.numel() != D:
            raise ValueError(f"g_out is (D,) but numel={g_out.numel()} != D={D}")
        return g_out.square()  # (D,)

    if g_out.dim() == 2:
        # (D,D) constant matrix or (N,D) diagonal vector
        if g_out.shape == (D, D):
            return g_out.square().sum(dim=1)  # (D,)
        if g_out.shape[1] != D:
            raise ValueError(f"g_out is (N,D) but second dim={g_out.shape[1]} != D={D}")
        return g_out.square()  # (N,D)

    if g_out.dim() == 3:
        # (N,D,m) or (N,D,D): diag(GG^T) = sum_k G_{i,k}^2
        if g_out.shape[1] != D:
            raise ValueError(f"g_out is (N,D,*) but dim1={g_out.shape[1]} != D={D}")
        return g_out.square().sum(dim=2)  # (N,D)

    raise ValueError(f"Unsupported g_out shape: {tuple(g_out.shape)}")


def check_gv_matches_autograd_full_range(
    V_net,
    GV_net,
    dynamics,
    full_range,                       # np.ndarray or torch.Tensor with shape (D,2)
    *,
    num_points: int = 512,            # how many samples in full_range
    seed: int = 0,
    batch_size: int = 128,            # autograd Hessian diag is expensive; batch it
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    report_topk: int = 10,
):
    """
    Compare GV_net(x) against an autograd-computed generator on MANY samples drawn uniformly
    from full_range.

    Prints aggregate error stats and worst offenders (top-k).
    """
    torch.manual_seed(seed)

    # --- prepare bounds ---
    if not isinstance(full_range, torch.Tensor):
        full_range_t = torch.tensor(full_range, device=device, dtype=dtype)
    else:
        full_range_t = full_range.to(device=device, dtype=dtype)

    assert full_range_t.dim() == 2 and full_range_t.shape[1] == 2, "full_range must be (D,2)"
    D = int(full_range_t.shape[0])
    lo = full_range_t[:, 0]
    hi = full_range_t[:, 1]

    # --- sample uniformly in full_range ---
    x_all = lo.unsqueeze(0) + (hi - lo).unsqueeze(0) * torch.rand(num_points, D, device=device, dtype=dtype)

    # --- get f,g once ---
    f = dynamics.get_f()
    g = dynamics.get_g()

    # --- storage for errors ---
    errs = []
    gv_fast_list = []
    gv_auto_list = []

    # --- loop in batches (avoid huge graphs) ---
    num_batches = (num_points + batch_size - 1) // batch_size

    for bi in range(num_batches):
        s = bi * batch_size
        e = min(num_points, (bi + 1) * batch_size)
        x = x_all[s:e]  # (B,D)
        B = x.shape[0]

        # ----- fast GV (no grad) -----
        with torch.no_grad():
            gv_fast = GV_net(x)
            if gv_fast.dim() == 2 and gv_fast.shape[1] == 1:
                gv_fast = gv_fast[:, 0]
            else:
                gv_fast = gv_fast.view(-1)
            gv_fast = gv_fast.detach()

        # ----- autograd GV -----
        x_req = x.clone().detach().requires_grad_(True)

        V_out = V_net(x_req)
        if V_out.dim() == 2 and V_out.shape[1] == 1:
            Vs = V_out[:, 0]          # (B,)
        elif V_out.dim() == 1:
            Vs = V_out                # (B,)
        else:
            raise ValueError(f"Expected scalar V output, got shape {tuple(V_out.shape)}")

        # grad: (B,D)
        gradV = torch.autograd.grad(Vs.sum(), x_req, create_graph=True)[0]

        # Hessian diagonal: (B,D)
        Hdiag_cols = []
        for i in range(D):
            gi = gradV[:, i]
            dgi = torch.autograd.grad(gi.sum(), x_req, create_graph=True)[0][:, i]
            Hdiag_cols.append(dgi)
        Hdiag = torch.stack(Hdiag_cols, dim=1)  # (B,D)

        # f(x): expected (B,D)
        if callable(f):
            fx = f(x_req)
            if isinstance(fx, np.ndarray):
                fx = torch.from_numpy(fx).to(device=x_req.device, dtype=x_req.dtype)
        else:
            fx = x_req @ f.to(x_req).T

        if fx.shape != x_req.shape:
            raise ValueError(f"f(x) must be (B,D)={tuple(x_req.shape)}; got {tuple(fx.shape)}")

        # diag(GG^T): (B,D)
        if g is None:
            gdiag = torch.zeros_like(x_req)
        elif callable(g):
            gout = g(x_req)
            if isinstance(gout, np.ndarray):
                gout = torch.from_numpy(gout).to(device=x_req.device, dtype=x_req.dtype)

            gdiag_raw = _diag_ggt_from_g(gout, D)

            if isinstance(gdiag_raw, (float, int)):
                gdiag = torch.full_like(x_req, float(gdiag_raw))
            elif isinstance(gdiag_raw, torch.Tensor) and gdiag_raw.dim() == 1:
                # (D,) -> broadcast to (B,D)
                gdiag = gdiag_raw.to(device=x_req.device, dtype=x_req.dtype).view(1, D).expand_as(x_req)
            else:
                # (B,D)
                gdiag = gdiag_raw.to(device=x_req.device, dtype=x_req.dtype)
        else:
            gout = g.to(x_req)
            gdiag_raw = _diag_ggt_from_g(gout, D)
            if isinstance(gdiag_raw, (float, int)):
                gdiag = torch.full_like(x_req, float(gdiag_raw))
            elif isinstance(gdiag_raw, torch.Tensor) and gdiag_raw.dim() == 1:
                gdiag = gdiag_raw.view(1, D).expand_as(x_req)
            else:
                gdiag = gdiag_raw

        gv_auto = ((fx * gradV) + 0.5 * (gdiag * Hdiag)).sum(dim=1)  # (B,)
        gv_auto = gv_auto.detach()

        # ----- compare -----
        err = (gv_fast - gv_auto).abs()  # (B,)

        errs.append(err.cpu())
        gv_fast_list.append(gv_fast.cpu())
        gv_auto_list.append(gv_auto.cpu())

        # free graph ASAP
        del x_req, Vs, V_out, gradV, Hdiag, fx, gdiag, gv_auto

    errs = torch.cat(errs, dim=0)                 # (N,)
    gv_fast_all = torch.cat(gv_fast_list, dim=0)  # (N,)
    gv_auto_all = torch.cat(gv_auto_list, dim=0)  # (N,)

    # --- stats ---
    max_err = float(errs.max())
    mean_err = float(errs.mean())
    med_err = float(errs.median())
    p95 = float(errs.kthvalue(int(0.95 * (errs.numel() - 1)) + 1).values)

    print("=== GV vs Autograd check over full_range ===")
    print(f"samples: {num_points}, batch_size: {batch_size}, D: {D}")
    print(f"max |diff|  = {max_err}")
    print(f"mean|diff|  = {mean_err}")
    print(f"median|diff|= {med_err}")
    print(f"p95 |diff|  = {p95}")

    # --- worst offenders ---
    k = min(report_topk, errs.numel())
    top_err, top_idx = torch.topk(errs, k=k, largest=True)
    print(f"\nTop-{k} worst errors:")
    for rank in range(k):
        i = int(top_idx[rank].item())
        print(
            f"[{rank:02d}] idx={i:05d}  |diff|={float(top_err[rank]):.6e}  "
            f"gv_fast={float(gv_fast_all[i]):.6e}  gv_auto={float(gv_auto_all[i]):.6e}  "
            f"x={x_all[i].detach().cpu().numpy()}"
        )

    return {
        "errs": errs,
        "gv_fast": gv_fast_all,
        "gv_auto": gv_auto_all,
        "x": x_all.detach().cpu(),
        "stats": {"max": max_err, "mean": mean_err, "median": med_err, "p95": p95},
    }


def pretrain_network_samples(
    model,
    x_goal_range,
    x_unsafe_range,
    x_init_range,
    x_range,
    params,
    GV_net=None,
    num_epochs=1000,
    lr=0.01,
    device='cpu',
    control_net=None,
    n_each: int = 400,   # samples per region per epoch
    lambda_w = 1.0,
    save_v_path=None,            # NEW
    save_control_path=None,      # NEW
):
    """
    Pre-train V network using sampled points to match constraint structure.

    Supports x_unsafe_range as:
      1) (D,2) single box
      2) (K,D,2) union of K boxes
      3) (2*D,2) produced by np.vstack((box1, box2, ...))  <-- your case

    Losses:
      - full-range: enforce v(x) >= 0
      - init-range: enforce v(x) <= 1
      - unsafe-range: enforce v(x) >= pretrain_unsafe_target
      - optional phi loss on x_others: enforce phi(x) <= 0
    """
    print("\n" + "="*80)
    print("PRE-TRAINING: Constraint-Structured Initialization (Sample-Based)")
    print("="*80)

    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    if GV_net is not None:
        print("  GV (Φ) pre-training ENABLED (using provided GV_net)")
    else:
        print("  GV (Φ) pre-training DISABLED (no GV_net provided)")

    best_loss = float('inf')
    best_model_state = None
    best_control_state = None

    # Convert ranges to torch once: each is (D,2)
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t    = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    init_t    = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
    low  = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low
    v_eq = model.output_offset.detach()

    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        """box: (D,2) -> samples: (N,D) uniform in box."""
        b_low = box[:, 0]
        b_high = box[:, 1]
        return torch.rand(N, D, device=device) * (b_high - b_low) + b_low

    def _in_box(x_batch: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """x_batch: (N,D), box: (D,2) -> mask: (N,)"""
        return ((x_batch >= box[:, 0]) & (x_batch <= box[:, 1])).all(dim=1)

    # -----------------------------
    # Unsafe region: allow union of boxes
    # -----------------------------
    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
        """
        Returns unsafe_boxes as torch.Tensor of shape (K,D,2).
        Accepts:
          - (D,2)
          - (K,D,2)
          - (K*D,2) from np.vstack((box1, box2, ...))
        """
        t = torch.as_tensor(x_unsafe, dtype=torch.float32, device=device)

        if t.dim() == 2:
            if t.shape == (D, 2):
                return t.unsqueeze(0)  # (1,D,2)
            if t.shape[1] == 2 and (t.shape[0] % D == 0):
                K = int(t.shape[0] // D)
                return t.view(K, D, 2)  # (K,D,2)  <-- handles vstack case
            raise ValueError(f"x_unsafe_range 2D must be (D,2) or (K*D,2); got {tuple(t.shape)}")

        if t.dim() == 3:
            if t.shape[1:] != (D, 2):
                raise ValueError(f"x_unsafe_range 3D must be (K,D,2) with D={D}; got {tuple(t.shape)}")
            return t

        raise ValueError(f"x_unsafe_range must be (D,2), (K,D,2), or (K*D,2); got {tuple(t.shape)}")

    unsafe_boxes = _unsafe_to_boxes(x_unsafe_range)  # (K,D,2)
    K_unsafe = int(unsafe_boxes.shape[0])

    def _in_unsafe_union(x_batch: torch.Tensor) -> torch.Tensor:
        """mask True if x is inside ANY unsafe box."""
        mask = torch.zeros(x_batch.shape[0], dtype=torch.bool, device=device)
        for k in range(K_unsafe):
            mask |= _in_box(x_batch, unsafe_boxes[k])
        return mask

    def _sample_in_unsafe_union(N: int) -> torch.Tensor:
        """
        Sample N points from EACH of the K unsafe boxes (torch only).

        Returns:
        x: (K_unsafe * N, D)

        Notes:
        - If K_unsafe == 1, this is just N samples from that box.
        - Shuffles so the batch is not grouped by box.
        """
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")

        if K_unsafe == 1:
            return _sample_in_box(unsafe_boxes[0], N)

        xs = []
        for k in range(K_unsafe):
            xs.append(_sample_in_box(unsafe_boxes[k], N))  # (N, D) per box

        x = torch.cat(xs, dim=0)  # (K_unsafe * N, D)
        x = x[torch.randperm(x.shape[0], device=device)]  # shuffle
        return x
    
    def _l2_weight_penalty(model: torch.nn.Module, exclude_bias: bool = True) -> torch.Tensor:
        reg = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_bias and (p.dim() == 1 or name.endswith("bias")):
                continue
            reg = reg + (p ** 2).sum()
        return reg

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # 1) full-range samples -> enforce v(x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # 2) init-range samples -> enforce v(x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # 3) unsafe-range samples -> enforce v(x) >= pretrain_unsafe_target
        x_unsafe = _sample_in_unsafe_union(int(n_each/6))
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.pretrain_unsafe_target - v_unsafe).sum()

        # 4) goal-range samples -> enforce v(x) <= 1.0
        x_goal = _sample_in_box(goal_t, n_each)
        v_goal = model(x_goal).squeeze(-1)
        v_loss_inside_goal = F.relu(0.0 - v_goal).sum()
        # miv_v_goal = torch.min(v_goal)

        # 5) samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= 0.0
        x_others_list = []
        need = n_each
        max_tries = 20
        tries = 0
        while need > 0 and tries < max_tries:
            tries += 1
            x_cand = torch.rand(max(need * 4, 32), D, device=device) * span + low
            cand_in_goal = _in_box(x_cand, goal_t)
            cand_in_unsafe = _in_unsafe_union(x_cand)
            keep = ~(cand_in_goal | cand_in_unsafe)
            x_keep = x_cand[keep]
            if x_keep.shape[0] > 0:
                take = min(need, x_keep.shape[0])
                x_others_list.append(x_keep[:take])
                need -= take

        if len(x_others_list) == 0:
            x_others = x_full.detach()
        else:
            x_others = torch.cat(x_others_list, dim=0)

        v_others = model(x_others).squeeze(-1)
        v_loss_others = F.relu(v_eq - v_others).sum()

        # Total V loss (v_loss_inside_goal and v_loss_others are not used anymore)
        loss_v = (v_loss_full
            + v_loss_init
            + v_loss_unsafe
            + v_loss_inside_goal
            + v_loss_others
        )

        # Phi loss on SAME x_others -> enforce phi(x) <= 0
        loss_phi = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_phi = x_others.detach().clone().requires_grad_(True)
            phi_output = GV_net(x_phi).squeeze(-1)
            loss_phi = F.relu(phi_output).sum()

        total_loss = loss_v + loss_phi

        # Add regularization for V network
        reg_w = _l2_weight_penalty(model, exclude_bias=True)
        reg_w_control = _l2_weight_penalty(control_net, exclude_bias=True)
        # print(reg_w)
        total_loss = total_loss + lambda_w * (reg_w)

        # Track best
        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        # Logging
        if epoch % 100 == 0:
            if GV_net is not None:
                print(
                    f"  Epoch [{epoch}/{num_epochs}]: "
                    f"V={loss_v.item():.3f} "
                    f"(full={v_loss_full.item():.3f}, init={v_loss_init.item():.3f}, "
                    f"unsafe={v_loss_unsafe.item():.3f}, "
                    f"goal={v_loss_inside_goal.item():.3f}), {v_loss_others.item():.3f}"
                    f"Φ={loss_phi.item():.3e}, Reg={reg_w.item():.3f}, {reg_w_control.item():.3f}, Total={total_loss.item():.3e}"
                )
                print(model(model.input_offset))
            else:
                print(
                    f"  Epoch [{epoch}/{num_epochs}]: "
                    f"V={loss_v.item():.6f} "
                    f"(full={v_loss_full.item():.3f}, init={v_loss_init.item():.3f}, "
                    f"unsafe={v_loss_unsafe.item():.3f}, "
                    f"goal={v_loss_inside_goal.item():.3f})"
                )

        # Optimize/update networks
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\n  Best loss: {best_loss:.6f}")

        # NEW: save best pretrained weights (state_dict)
        if save_v_path is not None:
            torch.save(best_model_state, save_v_path)
            print(f"  Saved pretrained V_net to: {save_v_path}")
        if (control_net is not None) and (best_control_state is not None) and (save_control_path is not None):
            torch.save(best_control_state, save_control_path)
            print(f"  Saved pretrained Controller_net to: {save_control_path}")

    print("="*80)
    networks_trained = ["V"]
    if GV_net is not None:
        networks_trained.append("GV (Φ)")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")
    print("="*80 + "\n")


def train_network_bounds(
    V_net,
    GV_net,
    region_cells: dict,
    regions: Regions,
    params: Hyperparameters,
    device: str = 'cpu',
    visualize_interval: int = 5000,
    control_net: nn.Module = None
):
    """
    Train the value network using CROWN bounds.

    Args:
        V_net: Value network
        GV_net: Generator network
        region_cells: Dictionary of discretized cells
        regions: Regions object
        params: Hyperparameters
        device: Device for training
        visualize_interval: Interval for visualization (0 to disable)
        control_net: if this is provided, then we do [control synthesis]
    """
    print("\n" + "="*80)
    print("BOUND-BASED TRAINING (using CROWN)")
    print("="*80)

    # Move models to device
    V_net = V_net.to(device)
    v_eq = V_net.output_offset.detach()

    def _log_V_zero_at_offset(V_net, epoch: int, device: str, every: int = 100, atol: float = 1e-6):
        if epoch % every != 0:
            return
        V_net.eval()
        with torch.no_grad():
            x0 = V_net.input_offset.to(device=device, dtype=next(V_net.parameters()).dtype)
            y0 = V_net(x0).item()  # (out,) or scalar-ish
            print(f"[Check] epoch={epoch:06d}  {y0:.3e} ")
        V_net.train()

    # Collect ALL cells in the same order as original (init, goal, unsafe, outside, generator)
    print("\nCollecting all cells for V network...")
    all_cells_V = []
    cell_counts_V = {}
    region_order_V = ['init', 'goal', 'unsafe', 'outside', 'boundary']  # Added boundary

    for name in region_order_V:
        cells = region_cells[name]
        all_cells_V.extend(cells)
        cell_counts_V[name] = len(cells)
        print(f"  {name}: {len(cells)} cells")

    total_cells_V = len(all_cells_V)
    print(f"  Total V cells: {total_cells_V}")

    # Prepare ALL input bounds at once (matching original)
    print("\nPreparing concatenated input bounds for V network...")
    if total_cells_V > 0:
        input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)
    else:
        input_lowers_all = torch.empty(0, params.network.n_inputs, device=device)
        input_uppers_all = torch.empty(0, params.network.n_inputs, device=device)

    # Create ONE big CROWN cache for ALL V cells (matching original!)
    print(f"\nInitializing CROWN cache for ALL {total_cells_V} V cells...")
    crown_cache_all = SymbolicCROWNCache(
        model=V_net,
        num_cells=total_cells_V,
        input_dim=params.network.n_inputs,
        device=device
    )

    # Create CROWN cache for generator (Phi) - separate cache
    crown_cache_phi = None
    input_lowers_gen = None
    input_uppers_gen = None
    if len(region_cells['generator']) > 0 and params.training.generator_weight > 0:
        print(f"\nCreating CROWN cache for 'generator' (Phi)...")
        print(f"  generator: {len(region_cells['generator'])} cells")
        crown_cache_phi = SymbolicCROWNCache_Phi(
            phi_module=GV_net,
            num_cells=len(region_cells['generator']),
            input_dim=params.network.n_inputs,
            device=device
        )
        # Prepare generator input bounds
        input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

    # Check if beta_s should be learnable
    learnable_beta_s = None
    beta_s_value = None

    if params.training.learnable_beta_s or params.constraints.beta_s is None:
        # Create learnable beta_s
        initial_beta_s = params.constraints.beta_s if params.constraints.beta_s is not None else 0.6
        learnable_beta_s = LearnableBetaS(initial_value=initial_beta_s).to(device)
        print(f"\nUsing LEARNABLE beta_s (initialized to {initial_beta_s})")

        # # Optimizer includes both V_net and learnable beta_s
        # optimizer = torch.optim.Adam(
        #     list(V_net.parameters()) + list(learnable_beta_s.parameters()),
        #     lr=params.training.learning_rate
        # )
        opt_params = list(V_net.parameters()) + list(learnable_beta_s.parameters())
    else:
        # Use constant beta_s
        beta_s_value = params.constraints.beta_s
        print(f"\nUsing CONSTANT beta_s = {beta_s_value}")

        # # Optimizer only for V_net
        # optimizer = torch.optim.Adam(
        #     V_net.parameters(),
        #     lr=params.training.learning_rate
        # )
        opt_params = list(V_net.parameters())

    # [control synthesis]
    if control_net is not None:
        opt_params += list(control_net.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)

    # Scheduler (matching testing_simple3.py)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,   # every 2000 epochs
        gamma=0.95         # multiply lr by 0.5
    )

    # Training loop
    loss_history = []
    refinement_epochs = {'outside': [], 'generator': []}
    start_time = time.time()
    final_beta_s = params.constraints.beta_s 

    for epoch in range(params.training.num_epochs):
        V_net.train()

        optimizer.zero_grad()
        if params.compute_V:
            # Compute bounds for ALL V cells at once (matching original!)
            if total_cells_V > 0:
                v_lowers_all, v_uppers_all = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all = torch.tensor([], device=device)
                v_uppers_all = torch.tensor([], device=device)

            # Split bounds by region (matching original's split_bounds_by_region)
            bounds = {}
            cell_idx = 0
            for name in region_order_V:
                num_cells = cell_counts_V[name]
                if num_cells > 0:
                    bounds[name] = (
                        v_lowers_all[cell_idx:cell_idx + num_cells],
                        v_uppers_all[cell_idx:cell_idx + num_cells]
                    )
                    cell_idx += num_cells
                else:
                    bounds[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        if params.compute_GV: 
            # Compute generator bounds if enabled
            needs_cache_rebuild = False
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None):
                phi_lowers, phi_uppers = crown_cache_phi.compute_bounds(input_lowers_gen, input_uppers_gen)
                current_gen_weight = params.training.generator_weight

                # Track failing cells for adaptive refinement
                phi_upper_failing_mask = phi_uppers > 0.0
                num_total_failing = phi_upper_failing_mask.sum().item()
                
            else:
                phi_lowers = torch.tensor([], device=device)
                phi_uppers = torch.tensor([], device=device)
                current_gen_weight = 0.0
                num_total_failing = 0

        # Get current beta_s value (learnable or constant)
        if learnable_beta_s is not None:
            current_beta_s = learnable_beta_s.value
        else:
            current_beta_s = beta_s_value

        # Compute total loss from bounds
        loss_kwargs = {
            'model': V_net,
            'goal_region': regions.goal,
            'beta_s': current_beta_s,
            'beta_ra': params.constraints.beta_ra,
            'device': device,
            'compute_V': params.compute_V,
            'compute_GV': params.compute_GV,
            'epoch': epoch,
            'w_soft': 2000.0,
        }

        # Add V bounds if computing V
        if params.compute_V:
            loss_kwargs.update({
                'V_goal_lower': bounds['goal'][0],
                'V_goal_upper': bounds['goal'][1],
                'V_unsafe_lower': bounds['unsafe'][0],
                'V_unsafe_upper': bounds['unsafe'][1],
                'V_init_lower': bounds['init'][0],
                'V_init_upper': bounds['init'][1],
                'V_outside_lower': bounds['outside'][0],
                'V_outside_upper': bounds['outside'][1],
                'V_boundary_lower': bounds['boundary'][0],
                'V_boundary_upper': bounds['boundary'][1]
            })

        # Add GV bounds if computing GV
        if params.compute_GV:
            loss_kwargs.update({
                'Phi_lower': phi_lowers,
                'Phi_upper': phi_uppers,
                'generator_weight': current_gen_weight
            })

        total_loss, loss_dict = compute_total_loss_bounds(**loss_kwargs)

        # Backward pass
        total_loss.backward()

        # Recompute bounds after optimizer step for verification
        V_net.eval()
        with torch.no_grad():
            if total_cells_V > 0:
                v_lowers_all_updated, v_uppers_all_updated = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all_updated = torch.tensor([], device=device)
                v_uppers_all_updated = torch.tensor([], device=device)

            # Split updated bounds by region
            bounds_updated = {}
            cell_idx = 0
            for name in region_order_V:
                num_cells = cell_counts_V[name]
                if num_cells > 0:
                    bounds_updated[name] = (
                        v_lowers_all_updated[cell_idx:cell_idx + num_cells],
                        v_uppers_all_updated[cell_idx:cell_idx + num_cells]
                    )
                    cell_idx += num_cells
                else:
                    bounds_updated[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        # Get current beta_s value for constraint checks
        beta_s_check = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s

        # Adaptive refinement for V outside region cells
        needs_cache_rebuild = False
        if params.compute_V and len(bounds_updated['outside'][0]) > 0:
            # Track failing cells in outside region
            outside_failing_mask = bounds_updated['outside'][0] <= 0.0
            num_outside_failing = outside_failing_mask.sum().item()

            # Ensure bounds match current cell count
            if len(outside_failing_mask) == len(region_cells['outside']) and num_outside_failing > 0:
                REFINE_INTERVAL = 200
                REFINE_FACTOR = 2
                MAX_CELLS = 100000

                # if epoch > 2500:
                #     REFINE_INTERVAL = 50

                if ((epoch + 1) % REFINE_INTERVAL == 0 and
                    len(region_cells['outside']) < MAX_CELLS):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['outside'],
                        outside_failing_mask,
                        REFINE_FACTOR,
                        scores=-bounds_updated['outside'][0]
                    )
                    region_cells['outside'] = new_cells
                    print(f"[Refine-Outside] Epoch {epoch+1}: {num_outside_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['outside'].append(epoch + 1)

            if((epoch + 1) % 501 == 0):
                outside_failing_mask_relax = bounds_updated['outside'][0] <= 8.0 + v_eq
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['outside'],
                    outside_failing_mask_relax,
                    max_passes=4,
                    max_merges=None,   # cap work; set None for full greedy
                    seed=0,
                    eps=1e-6,
                )
                region_cells['outside'] = merged_cells
                print(f"[Merge-Outside] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Unsafe refinement
        # if params.compute_V and len(bounds_updated['unsafe'][0]) > 0:
        #     # Track failing cells in outside region
        #     unsafe_failing_mask = bounds_updated['unsafe'][0] <= params.constraints.beta_ra
        #     num_unsafe_failing = unsafe_failing_mask.sum().item()
        #     # Ensure bounds match current cell count
        #     if num_unsafe_failing > 0:
        #         REFINE_INTERVAL = 200
        #         REFINE_FACTOR = 2
        #         MAX_CELLS = 100000
        #         if (epoch == 1 or (epoch + 1) % REFINE_INTERVAL == 0) and \
        #             len(region_cells['unsafe']) < MAX_CELLS:
        #             new_cells, num_refined = refine_failing_cells(
        #                 region_cells['unsafe'],
        #                 unsafe_failing_mask,
        #                 REFINE_FACTOR,
        #                 scores=-bounds_updated['unsafe'][0],
        #                 N_to_refine=100,
        #             )
        #             region_cells['unsafe'] = new_cells
        #             print(f"[Refine-Unsafe] Epoch {epoch+1}: {num_unsafe_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
        #             needs_cache_rebuild = True

        #     if((epoch + 1) % 501 == 0):
        #         unsafe_failing_mask_relax = bounds_updated['outside'][0] <= params.constraints.beta_ra*10.0
        #         merged_cells, num_merges = merge_passing_neighbor_cells(
        #             region_cells['unsafe'],
        #             unsafe_failing_mask_relax,
        #             max_passes=4,
        #             max_merges=None,   # cap work; set None for full greedy
        #             seed=0,
        #             eps=1e-6,
        #         )
        #         region_cells['unsafe'] = merged_cells
        #         print(f"[Merge-Unsafe] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
        #         needs_cache_rebuild = True

        # Adaptive refinement for generator cells (before optimizer step)
        if params.compute_GV:
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):

                # Refinement parameters (matching testing_simple3.py)
                REFINE_INTERVAL = 500  # Refine every 100k epochs
                REFINE_FACTOR = 2  # Split into 2x2 subcells
                MAX_CELLS = 100000  # Don't refine if we already have too many cells
                N_TO_REFINE = 100

                # # Adjust interval for later epochs
                # if epoch > 2500:
                #     REFINE_INTERVAL = 250

                # Check if it's time to refine
                if ((epoch + 1) % REFINE_INTERVAL == 0 and
                    len(region_cells['generator']) < MAX_CELLS):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['generator'],
                        phi_upper_failing_mask,
                        REFINE_FACTOR,
                        N_to_refine=N_TO_REFINE,
                        scores=phi_uppers,
                    )
                    region_cells['generator'] = new_cells
                    print(f"[Refine-Generator] Epoch {epoch+1}: {num_total_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['generator'].append(epoch + 1)
            
            if((epoch + 1) % 501 == 0):
                phi_upper_failing_mask_relax = phi_uppers > -500.0
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['generator'],
                    phi_upper_failing_mask_relax,
                    max_passes=8,
                    max_merges=None,   # cap work; set None for full greedy
                    seed=0,
                    eps=1e-6,
                )
                region_cells['generator'] = merged_cells
                print(f"[Merge-Generator] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Logging
        if epoch % 10 == 0 or epoch == params.training.num_epochs - 1 or epoch == 0:
            # Add beta_s to loss dict for logging
            if learnable_beta_s is not None:
                beta_s_log = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s
                print(f"Epoch [{epoch}/{params.training.num_epochs}]: Loss={total_loss.item():.4f}, β_s={beta_s_log:.4f}")
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV)
            # if control_net is not None:
            #     for name, param in control_net.named_parameters():
            #         if param.requires_grad:
            #             print(f" [Controller Params] {name} = {param.data}")
            loss_dict['epoch'] = epoch
            if learnable_beta_s is not None:
                loss_dict['beta_s'] = beta_s_log
            loss_history.append(loss_dict.copy())

        # Early stopping check
        with torch.no_grad():
            # Check V constraints
            all_satisfied = True
            if params.compute_V:
                show = (epoch % 10 == 0)
                _, goal_satisfied = compute_loss_goal_bounds(V_net, regions.goal, bounds_updated["goal"][0], bounds_updated["goal"][1], beta_s_check, 
                                                             bounds_updated['outside'][0], device=device, show=show, check=True, n_samples=10000)
                unsafe_satisfied = (bounds_updated['unsafe'][0].min() >= params.constraints.beta_ra)
                init_satisfied = (bounds_updated['init'][1].max() <= 1.0)
                outside_satisfied = (bounds_updated['outside'][0].min() >= 0.0)
                all_satisfied = all_satisfied and goal_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied

            # Check GV constraints
            if params.compute_GV:
                generator_satisfied = (phi_uppers.max() < 0.0)
                all_satisfied = all_satisfied and generator_satisfied

            # Early stop if all active constraints are satisfied
            if all_satisfied:
                print("\n" + "="*80)
                print("ALL CONSTRAINTS SATISFIED - EARLY STOPPING!")
                print("="*80)
                print(f"Training converged at epoch {epoch}")

                # Print relevant losses
                loss_parts = []
                if params.compute_V:
                    loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}, Boundary={loss_dict['boundary']:.4f}")
                if params.compute_GV:
                    loss_parts.append(f"Gen={loss_dict['generator']:.4f}")
                print(f"Final losses: {', '.join(loss_parts)}")

                if control_net is not None:
                    for name, param in control_net.named_parameters():
                        if param.requires_grad:
                            print(f" [Controller Params] {name} = {param.data}")

                final_beta_s = bounds_updated['outside'][0].min()
                print("final beta_s: {:.4f}".format(final_beta_s))

                break

        # Detailed evaluation and visualization
        if (epoch % 500 == 0) or epoch == params.training.num_epochs - 1:
            print(f"\nEpoch {epoch} - Detailed Evaluation:")
            # For evaluation, we can just create temporary caches (not in the hot path)
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                crown_cache_all=crown_cache_all,
                crown_cache_phi=crown_cache_phi,
                input_bounds_all=(input_lowers_all, input_uppers_all),
                cell_counts_V=cell_counts_V,
                input_bounds_gen=(input_lowers_gen, input_uppers_gen) if input_lowers_gen is not None else None,
                beta_s=current_beta_s,
                beta_ra=params.constraints.beta_ra,
                device=device,
                n_samples=1000
            )
            print_constraint_summary(results, prefix="  ")

            # Print bound statistics
            if params.compute_V:
                if len(bounds['goal'][0]) > 0:
                    print(f"  Goal bounds: V ∈ [{bounds['goal'][0].min().item():.3f}, {bounds['goal'][1].max().item():.3f}]")
                if len(bounds['unsafe'][0]) > 0:
                    print(f"  Unsafe bounds: V ∈ [{bounds['unsafe'][0].min().item():.3f}, {bounds['unsafe'][1].max().item():.3f}]")
            if params.compute_GV:
                if len(phi_uppers) > 0:
                    print(f"  Generator bounds: Φ ∈ [{phi_uppers.min().item():.6e}, {phi_uppers.max().item():.6e}]")
                    failing_cells = (phi_uppers > 0).sum().item()
                    print(f"  Generator failing cells: {failing_cells}/{len(phi_uppers)} ({100*failing_cells/len(phi_uppers):.1f}%)")
            print()

            # Visualize progress
            if visualize_interval > 0 and epoch % visualize_interval == 0:
                print(f"  Creating visualization for epoch {epoch}...")
                visualize_training_progress(
                    V_net, GV_net, regions, region_cells,
                    epoch=epoch,
                    output_dir="training_progress",
                    value_beta_ra=params.constraints.beta_ra
                )
        
        # _log_V_zero_at_offset(V_net, epoch, device, every=10, atol=1e-6)  # before step
        # Optimizer step
        optimizer.step()
        scheduler.step()
        _log_V_zero_at_offset(V_net, epoch, device, every=10, atol=1e-6)  # before step

        # Rebuild CROWN caches after optimizer step if needed (after adaptive refinement)
        # if params.compute_GV:
        if True:
            if needs_cache_rebuild:
                # with torch.no_grad():
                print(f"  Rebuilding CROWN caches with new generator cells...")

                # Rebuild V cache with all cells (in order: init, goal, unsafe, outside)
                # Note: generator cells are NOT included in V cache
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])

                total_cells_V = len(all_cells_V)
                input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

                # Rebuild V CROWN cache
                print(f"    Rebuilding V cache with {total_cells_V} cells...")
                crown_cache_all = SymbolicCROWNCache(
                    model=V_net,
                    num_cells=total_cells_V,
                    input_dim=params.network.n_inputs,
                    device=device
                )

                # Rebuild Phi CROWN cache (only for generator region)
                num_generator_cells = len(region_cells['generator'])
                print(f"    Rebuilding Phi cache with {num_generator_cells} cells...")
                crown_cache_phi = SymbolicCROWNCache_Phi(
                    phi_module=GV_net,
                    num_cells=num_generator_cells,
                    input_dim=params.network.n_inputs,
                    device=device
                )

                # Rebuild generator input bounds
                input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

                # Update cell counts after refinement
                for name in region_order_V:
                    cell_counts_V[name] = len(region_cells[name])
                
                # Rebuild concatenated input bounds for V
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])
                    input_lowers_all, input_uppers_all =  prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

                print(f"  Caches rebuilt successfully!")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Return final beta_s value along with loss history
    # final_beta_s = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s
    # if learnable_beta_s is not None:
    #     print(f"\nFinal learned β_s = {final_beta_s:.4f}")

    return loss_history, final_beta_s, refinement_epochs


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    args = parser.parse_args()

    print("=" * 80)
    print("Cartpole CERTIFICATE-BASED CONTROL SYNTHESIS (V + Phi)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1) Hyperparameters (clone your Lorentz defaults, then adjust)
    # -------------------------------------------------------------------------
    params = Hyperparameters.default()

    params.network.n_inputs = 4
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64

    # Scaling: pick something roughly comparable to variable ranges
    # v ~ [20,100] (span 80), gamma ~ [-0.26,0.26], beta ~ [0,1.57]
    # scale choices affect training conditioning; tune if needed.
    params.network.input_scale = [2*pi, 20.0, 10.0, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 30000
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0
    params.training.learnable_beta_s = False
    params.constraints.beta_s = 0.00
    params.constraints.beta_ra = 3.0

    params.compute_V = True
    params.compute_GV = True

    device = params.training.device

    # -------------------------------------------------------------------------
    # 3) Regions (init, goal, unsafe union, full)
    # -------------------------------------------------------------------------
    full_range = np.array([
        [-2*pi, 2*pi],
        [-20.0, 20.0],
        [-10.0, 10.0],
        [-20.0, 20.0],
    ], dtype=np.float32)

    init_range = np.array([
        [pi-15*DEG, pi+15*DEG],
        [-0.1, 0.1],
        [-1.0, 1.0],
        [-0.1, 0.1],
    ], dtype=np.float32)

    goal_range = np.array([
        [-0.4*pi, 0.4*pi],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-1.0, 1.0],
    ], dtype=np.float32)

    unsafe_min_z = np.array([
        full_range[0, :],
        full_range[1, :],
        [full_range[2, 0], full_range[2, 0]+0.5],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_max_z = np.array([
        full_range[0, :],
        full_range[1, :],
        [full_range[2, 1]-0.5, full_range[2, 1]],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_min_theta = np.array([
        [full_range[0, 0], full_range[0, 0]+0.5*DEG],
        full_range[1, :],
        full_range[2, :],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_max_theta = np.array([
        [full_range[0, 1]-0.5*DEG, full_range[0, 1]],
        full_range[1, :],
        full_range[2, :],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_min_z_dot = np.array([
        full_range[0, :],
        full_range[1, :],
        full_range[2, :],
        [full_range[3, 0], full_range[3,0]+0.5],
    ], dtype=np.float32)

    unsafe_max_z_dot = np.array([
        full_range[0, :],
        full_range[1, :],
        full_range[2, :],
        [full_range[3, 1]-0.5, full_range[3, 1]],
    ], dtype=np.float32)

    unsafe_min_theta_dot = np.array([
        full_range[0, :],
        [full_range[1, 0], full_range[1, 0]+0.5],
        full_range[2, :],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_max_theta_dot = np.array([
        full_range[0, :],
        [full_range[1, 1]-0.5, full_range[1, 1]],
        full_range[2, :],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_range = np.vstack((unsafe_min_z, unsafe_max_z,
                              unsafe_min_theta, unsafe_max_theta,
                              unsafe_min_z_dot, unsafe_max_z_dot,
                              unsafe_min_theta_dot, unsafe_max_theta_dot
                             ))
    init = Region(init_range)
    goal = Region(goal_range)
    full = Region(full_range)

    unsafe_min_z_reg = Region(unsafe_min_z)
    unsafe_max_z_reg = Region(unsafe_max_z)
    unsafe_min_theta_reg = Region(unsafe_min_theta)
    unsafe_max_theta_reg = Region(unsafe_max_theta)
    unsafe_min_z_dot_reg = Region(unsafe_min_z_dot)
    unsafe_max_z_dot_reg = Region(unsafe_max_z_dot)
    unsafe_min_theta_dot_reg = Region(unsafe_min_theta_dot)
    unsafe_max_theta_dot_reg = Region(unsafe_max_theta_dot)

    unsafe = Region.union(unsafe_min_z_reg, unsafe_max_z_reg, 
                          unsafe_min_theta_reg, unsafe_max_theta_reg,
                          unsafe_min_z_dot_reg, unsafe_max_z_dot_reg,
                          unsafe_min_theta_dot_reg, unsafe_max_theta_dot_reg
                          )
    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # Discretization: start modest; refine happens during training
    params.discretization.n_goal = 15
    params.discretization.n_outside_goal = 4
    params.discretization.n_generator = 2
    params.discretization.n_unsafe = 10
    params.discretization.n_init = 15

    # -------------------------------------------------------------------------
    # 2) Dynamics (closed-loop, torch, differentiable)
    # -------------------------------------------------------------------------
    # Create diffusion coefficients as a persistent tensor to avoid TracerWarnings
    g_coeffs = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term g(x) that is constant: [1.0, 1.0, 1.0] for every x.

        If x has shape (D,), returns (D,).
        If x has shape (N, D), returns (N, D) with each row [1.0, 1.0, 1.0].
        """
        base = g_coeffs.to(device=x.device, dtype=x.dtype)

        if x.dim() == 1:
            # x is shape (D,)
            return base
        elif x.dim() == 2:
            # x is shape (N, D)
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)  # (N, D)
        else:
            raise ValueError(f"g(x) expects x of shape (D,) or (N, D), got {tuple(x.shape)}")

    u_nn = CartpoleControlNN()

    f_cl_module = ClosedLoopCartPole(controller=u_nn).to(device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=params.network.n_inputs)

    # do animation
    animate_cartpole_state_control(
        f_cl_module=f_cl_module, g_fn=g,
        init_range=init_range, goal_range=goal_range,
        full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
        device=device,
        dt=0.01, T=100.0,
        seed=0, save_path=None,
        show=True,
        controller_label="open-loop"
    )

    # -------------------------------------------------------------------------
    # 4) Networks (V and GV)
    # -------------------------------------------------------------------------
    input_offset = [0.0, 0.0, 0.0, 0.0]
    output_offset = np.float32(0.1)
    V_net = create_V(params.network, input_offset=input_offset, output_offset=output_offset)
    V_net.verify_zero_at_offset(atol=1e-6, rtol=1e-6)

    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        training_config=params.training,
        input_offset=input_offset
    )

    res = check_gv_matches_autograd_full_range(
        V_net, GV_net, dynamics,
        full_range=full_range,
        num_points=1024,
        batch_size=64,
        device=device,
    )

    # -------------------------------------------------------------------------
    # 5) Discretize regions
    # -------------------------------------------------------------------------
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=False,
        boundary_n_partitions=1
    )

    # -------------------------------------------------------------------------
    # 6) Pretrain + train (or load)
    # -------------------------------------------------------------------------
    params.constraints.pretrain_goal_target = params.constraints.beta_s
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = 1.0
    params.constraints.pretrain_phi_target = 0.0
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")
        print_training_config(params)
        print(f"\n{dynamics}")
        print(f"\nUsing device: {device}")

        ENABLE_PRETRAINING = True
        PRETRAIN_EPOCHS = 20000
        PRETRAIN_LR = 0.01

        if ENABLE_PRETRAINING:
            pretrain_network_samples(
                model=V_net,
                x_goal_range=goal_range,
                x_unsafe_range=unsafe_range,     # pass first unsafe piece
                x_init_range=init_range,
                x_range=full_range,
                params=params,
                GV_net=GV_net,
                num_epochs=PRETRAIN_EPOCHS,
                lr=PRETRAIN_LR,
                device=device,
                control_net=u_nn,
                lambda_w=1e-3,
                n_each=1200,
                save_v_path= OUTPUT_DIR / "V_pretrained.pth",
                save_control_path= OUTPUT_DIR / "controller_pretrained.pth"
            )
            print("Pretraining completed.\n")

        V_net.load_state_dict(torch.load(OUTPUT_DIR / "V_pretrained.pth", map_location=device))
        u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
        
        animate_cartpole_state_control(
            f_cl_module=f_cl_module, g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
            device=device,
            dt=0.01, T=100.0,
            seed=0, save_path=None,
            show=True,
            controller_label="open-loop"
        )
        
        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            visualize_interval=1000,
            control_net=u_nn,  # CONTROL SYNTHESIS
        )

        print("\n" + "=" * 80)
        print("FINAL EVALUATION")
        print("=" * 80)

        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_s=final_beta_s,
            beta_ra=params.constraints.beta_ra,
            device=device,
            n_samples=5000
        )
        print_constraint_summary(results)

        print("\n" + "=" * 80)
        print("CREATING FINAL VISUALIZATIONS")
        print("=" * 80)

        create_summary_plots(
            V_net=V_net,
            GV_net=GV_net,
            regions=regions,
            region_cells=region_cells,
            beta_s=final_beta_s,
            beta_ra=params.constraints.beta_ra,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
            output_dir="results"
        )

        print("\n" + "=" * 80)
        print("SAVING EVAL BUNDLE")
        print("=" * 80)

        save_eval_bundle(
            OUTPUT_DIR,
            V_net=V_net,
            GV_net=GV_net,
            control_net=u_nn,
            params=params,
            regions=regions,
            region_cells=region_cells,
            final_beta_s=final_beta_s,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
        )
    # else:
    #     print("\n" + "=" * 80)
    #     print("LOADING SAVED BUNDLE (skip training)")
    #     print("=" * 80)

    #     bundle = load_eval_bundle(bundle_path, map_location="cpu")

    #     params = Hyperparameters.from_dict(bundle["hyperparameters"])
    #     device = params.training.device

    #     # rebuild region_cells (already cpu tensors)
    #     region_cells = bundle["region_cells"]

    #     # rebuild networks & load
    #     V_net.load_state_dict(bundle["V_state_dict"])

    #     # load controller
    #     if bundle["control_state_dict"] is not None:
    #         print("load u_nn")
    #         u_nn.load_state_dict(bundle["control_state_dict"])

    #     # rebuild dynamics and GV
    #     u_nn.verify_u_at_equilibrium()
    #     f_cl_module = ClosedLoopDrift(aero=aero, controller=u_nn).to(device)

    #     # create a constant u_eq open-loop control for comparison
    #     u_eq_device = u_eq.to(device=device, dtype=torch.float32)  # (3,)
    #     u_const = ConstantControl(u_eq_device).to(device)
    #     f_cl_module_no_control = ClosedLoopDrift(aero=aero, controller=u_const).to(device)

    #     # create the controller from pretraining
    #     u_nn_pretrain = XV15EqMLPControl(
    #         x_eq=x_eq,          # (3,) torch tensor
    #         u_eq=u_eq,          # (3,) torch tensor
    #         T_min=XV15Constants.MASS * 9.81 * 0.1,
    #         T_max=XV15Constants.MASS * 9.81 * 1.8,
    #         alpha_max=XV15Constants.AOA_MAX,
    #         delta_max=XV15Constants.MAX_TILT_ANGLE_RATE,
    #         hidden_dim=64,
    #         act="tanh",
    #     ).to(x_eq.device)
    #     u_nn_pretrain.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
    #     u_nn_pretrain.verify_u_at_equilibrium()
    #     f_cl_module_pretrain = ClosedLoopDrift(aero=aero, controller=u_nn_pretrain).to(device)

    #     # animation
    #     # open-loop constant u_eq animation
    #     animate_xv15_aircraft_state_control(
    #         f_cl_module=f_cl_module_no_control, g_fn=g,
    #         init_range=init_range, goal_range=goal_range,
    #         full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
    #         device=device,
    #         dt=0.02, T=120.0,
    #         seed=0, save_path=None,
    #         show=True,
    #         controller_label="open-loop"
    #     )
    #     # pretrain animation
    #     animate_xv15_aircraft_state_control(
    #         f_cl_module=f_cl_module_pretrain, g_fn=g,
    #         init_range=init_range, goal_range=goal_range,
    #         full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
    #         device=device,
    #         dt=0.02, T=120.0,
    #         seed=0, save_path=None,
    #         show=True,
    #         controller_label="pre-train"
    #     )
    #     # control-synthesis animation
    #     animate_xv15_aircraft_state_control(
    #         f_cl_module=f_cl_module, g_fn=g,
    #         init_range=init_range, goal_range=goal_range,
    #         full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
    #         device=device,
    #         dt=0.02, T=120.0,
    #         seed=0, save_path=None,
    #         show=True,
    #         controller_label="certified-synthesis"
    #     )

    #     print("\n" + "=" * 80)
    #     print("Monte Carlo results")
    #     print("=" * 80)

    #     # monte-carlo (open-loop)
    #     mc = mc_reach_avoid(
    #         f_cl_module=f_cl_module_no_control,
    #         g_fn=g,  # or None
    #         init_range=init_range,
    #         goal_range=goal_range,
    #         full_range=full_range,
    #         unsafe_boxes=unsafe_range,
    #         N_trials=100,
    #         dt=0.02,
    #         T=120.0,
    #         seed=0,
    #         device=device,
    #     )
    #     print("Open Loop Control")
    #     print(mc)

    #     # monte-carlo (pretrain)
    #     mc = mc_reach_avoid(
    #         f_cl_module=f_cl_module_pretrain,
    #         g_fn=g,  # or None
    #         init_range=init_range,
    #         goal_range=goal_range,
    #         full_range=full_range,
    #         unsafe_boxes=unsafe_range,
    #         N_trials=100,
    #         dt=0.02,
    #         T=120.0,
    #         seed=0,
    #         device=device,
    #     )
    #     print("Pretrain Control")
    #     print(mc)

    #     # monte-carlo (synthesis)
    #     mc = mc_reach_avoid(
    #         f_cl_module=f_cl_module,
    #         g_fn=g,  # or None
    #         init_range=init_range,
    #         goal_range=goal_range,
    #         full_range=full_range,
    #         unsafe_boxes=unsafe_range,
    #         N_trials=100,
    #         dt=0.02,
    #         T=120.0,
    #         seed=0,
    #         device=device,
    #     )
    #     print("Synthesized Control")
    #     print(mc)

    #     print("\n" + "=" * 80)
    #     print("Recreating Plots")
    #     print("=" * 80)

    #     # re-render plots
    #     dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=3)
    #     GV_net = create_GV(
    #         V_net=V_net,
    #         dynamics=dynamics,
    #         network_config=params.network,
    #         training_config=params.training,
    #         input_offset=input_offset
    #     ).to(device)
    #     if bundle["GV_state_dict"] is not None:
    #         GV_net.load_state_dict(bundle["GV_state_dict"])

    #     # move cells to device
    #     region_cells = {
    #         k: [(lo.to(device), hi.to(device)) for (lo, hi) in v]
    #         for k, v in region_cells.items()
    #     }

    #     final_beta_s = bundle["final_beta_s"]
    #     loss_history = bundle["loss_history"]
    #     refinement_epochs = bundle["refinement_epochs"]

    #     results = bundle.get("final_results", None)
    #     if results is None:
    #         print("No saved results found in bundle, recomputing evaluation...")
    #         results = evaluate_constraints(
    #             V_net, GV_net, region_cells,
    #             beta_s=final_beta_s,
    #             beta_ra=params.constraints.beta_ra,
    #             device=device,
    #             n_samples=5000
    #         )

    #     print("\n" + "=" * 80)
    #     print("FINAL EVALUATION (LOADED)")
    #     print("=" * 80)
    #     print_constraint_summary(results)

    #     print("\n" + "=" * 80)
    #     print("CREATING FINAL VISUALIZATIONS (LOADED)")
    #     print("=" * 80)
    #     log_loaded_training_epochs(loss_history)

    #     create_summary_plots(
    #         V_net=V_net,
    #         GV_net=GV_net,
    #         regions=regions,
    #         region_cells=region_cells,
    #         beta_s=final_beta_s,
    #         beta_ra=params.constraints.beta_ra,
    #         loss_history=loss_history,
    #         refinement_epochs=refinement_epochs,
    #         results=results,
    #         output_dir="results"
    #     )


if __name__ == "__main__":
    main()
