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
from matplotlib import animation
from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------------
# Repo paths (match your Lorentz script)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # repo_root
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
from src.save_load_utils import save_eval_bundle, load_eval_bundle, enable_terminal_logging
from src.utils import cleanup_and_setup_directories
from src.visualization import (
    create_summary_plots
)

torch.manual_seed(0)
np.random.seed(0)

DEG = np.pi / 180.0
pi = np.pi


# class UnicycleControlNN(nn.Module):
#     def __init__(
#         self,
#         input_dim=3,
#         hidden_dim=64,
#         input_scale=(8.0, 8.0, 0.9 * pi),
#         w_max=1.0,
#     ):
#         super().__init__()
#         if w_max <= 0.0:
#             raise ValueError(f"w_max must be positive, got {w_max}")
#         in_scale = torch.as_tensor(input_scale, dtype=torch.float32)
#         if in_scale.numel() != input_dim:
#             raise ValueError(f"input_scale must have length {input_dim}, got {in_scale.numel()}")
#
#         self.register_buffer("input_scale", in_scale)
#         self.w_max = float(w_max)
#         self.W1_raw = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
#         self.W2_raw = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         W1 = self.w_max * torch.tanh(self.W1_raw)
#         W2 = self.w_max * torch.tanh(self.W2_raw)
#         x_norm = x / self.input_scale.to(device=x.device, dtype=x.dtype)
#         h1 = torch.tanh(F.linear(x_norm, W1, bias=None))
#         omega = torch.tanh(F.linear(h1, W2, bias=None))
#         return omega


class UnicycleControlNN(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=32,
        input_scale=(8.0, 8.0, 0.9 * pi),
    ):
        super().__init__()
        in_scale = torch.as_tensor(input_scale, dtype=torch.float32)
        if in_scale.numel() != input_dim:
            raise ValueError(f"input_scale must have length {input_dim}, got {in_scale.numel()}")
        self.register_buffer("input_scale", in_scale)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / self.input_scale.to(device=x.device, dtype=x.dtype)
        h1 = torch.tanh(self.fc1(x_norm))
        omega = torch.tanh(self.fc2(h1))
        return omega


class ClosedLoopDrift(nn.Module):
    """
    Closed-loop unicycle drift with fixed forward speed:
      controller outputs angular rate omega
      drift = [v*cos(theta), v*sin(theta), omega]
    """
    def __init__(self, u_nn: nn.Module, forward_speed: float = 1.0):
        super().__init__()
        self.controller = u_nn
        self.forward_speed = float(forward_speed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega = self.controller(x).squeeze(-1)
        theta = x[:, 2]
        return torch.stack(
            [
                self.forward_speed * torch.cos(theta),
                self.forward_speed * torch.sin(theta),
                omega,
            ],
            dim=1,
        )


def animate_unicycle_xy_theta(
    dynamics_model: Dynamics,
    init_range: np.ndarray,
    goal_range: np.ndarray,
    unsafe_range: np.ndarray,
    full_range: np.ndarray,
    n_unicycles: int = 6,
    steps: int = 240,
    dt: float = 0.05,
) -> animation.FuncAnimation:
    """
    Animate unicycle trajectory in the x-y plane with heading theta.
    """
    if dynamics_model is None or dynamics_model.f is None:
        raise ValueError("dynamics_model.f must be provided.")
    if dynamics_model.g is None:
        raise ValueError("dynamics_model.g must be provided.")
    if not callable(dynamics_model.f) or not callable(dynamics_model.g):
        raise ValueError("dynamics_model.f and dynamics_model.g must be callable.")

    if init_range.shape != (3, 2) or goal_range.shape != (3, 2) or full_range.shape != (3, 2):
        raise ValueError("init_range, goal_range, and full_range must each be shape (3, 2).")

    if unsafe_range.ndim == 2 and unsafe_range.shape[1] == 2 and unsafe_range.shape[0] % 3 == 0:
        unsafe_boxes = unsafe_range.reshape(-1, 3, 2)
    elif unsafe_range.ndim == 3 and unsafe_range.shape[1:] == (3, 2):
        unsafe_boxes = unsafe_range
    else:
        raise ValueError("unsafe_range must be (K*3, 2) or (K, 3, 2).")

    if n_unicycles <= 0:
        raise ValueError(f"n_unicycles must be positive, got {n_unicycles}")

    initial_states = np.random.uniform(
        init_range[:, 0],
        init_range[:, 1],
        size=(n_unicycles, 3),
    ).astype(np.float32)
    states = np.zeros((steps + 1, n_unicycles, 3), dtype=np.float32)
    controls = np.zeros((steps + 1, n_unicycles, 2), dtype=np.float32)
    states[0] = initial_states

    f_cl_module = dynamics_model.f.eval() if isinstance(dynamics_model.f, nn.Module) else dynamics_model.f
    g_fn = dynamics_model.g
    with torch.no_grad():
        for k in range(steps):
            xk = torch.from_numpy(states[k]).float()
            drift_t = f_cl_module(xk)
            if drift_t.ndim != 2 or drift_t.shape[1] != 3:
                raise ValueError(f"dynamics_model.f(x) must return shape (N,3), got {tuple(drift_t.shape)}")
            drift = drift_t.cpu().numpy()  # [x_dot, y_dot, theta_dot]
            controls[k, :, 0] = np.sqrt(drift[:, 0] ** 2 + drift[:, 1] ** 2).astype(np.float32)  # forward speed
            controls[k, :, 1] = drift[:, 2].astype(np.float32)  # angular rate
            diffusion = g_fn(xk).cpu().numpy()
            if diffusion.ndim == 1:
                diffusion = np.broadcast_to(diffusion, drift.shape)
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=drift.shape).astype(np.float32)
            next_state = states[k] + dt * drift + diffusion * dW
            next_state[:, 2] = np.arctan2(np.sin(next_state[:, 2]), np.cos(next_state[:, 2]))
            states[k + 1] = next_state.astype(np.float32)
        drift_t = f_cl_module(torch.from_numpy(states[steps]).float())
        drift = drift_t.cpu().numpy()
        controls[steps, :, 0] = np.sqrt(drift[:, 0] ** 2 + drift[:, 1] ** 2).astype(np.float32)
        controls[steps, :, 1] = drift[:, 2].astype(np.float32)

    fig, (ax_xy, ax_u) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    ax_xy.set_xlim(full_range[0, 0], full_range[0, 1])
    ax_xy.set_ylim(full_range[1, 0], full_range[1, 1])
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("Unicycle x-y Animation with Heading")
    ax_xy.grid(True, alpha=0.3)

    t = np.arange(steps + 1, dtype=np.float32) * dt
    ax_u.set_xlim(0.0, float(t[-1]))
    u_min = float(np.min(controls))
    u_max = float(np.max(controls))
    u_pad = 0.1 * max(1e-6, (u_max - u_min))
    ax_u.set_ylim(u_min - u_pad, u_max + u_pad)
    ax_u.set_xlabel("time [s]")
    ax_u.set_ylabel("control")
    ax_u.set_title("Control Signals vs Time")
    ax_u.grid(True, alpha=0.3)

    ax_xy.add_patch(
        Rectangle(
            (init_range[0, 0], init_range[1, 0]),
            init_range[0, 1] - init_range[0, 0],
            init_range[1, 1] - init_range[1, 0],
            edgecolor="tab:blue",
            facecolor="tab:blue",
            alpha=0.18,
            linewidth=1.5,
            label="init",
        )
    )
    ax_xy.add_patch(
        Rectangle(
            (goal_range[0, 0], goal_range[1, 0]),
            goal_range[0, 1] - goal_range[0, 0],
            goal_range[1, 1] - goal_range[1, 0],
            edgecolor="tab:green",
            facecolor="tab:green",
            alpha=0.18,
            linewidth=1.5,
            label="goal",
        )
    )
    for box in unsafe_boxes:
        x0b, x1b = box[0, 0], box[0, 1]
        y0b, y1b = box[1, 0], box[1, 1]
        full_xy = (
            np.isclose(x0b, full_range[0, 0]) and np.isclose(x1b, full_range[0, 1]) and
            np.isclose(y0b, full_range[1, 0]) and np.isclose(y1b, full_range[1, 1])
        )
        if full_xy:
            continue
        ax_xy.add_patch(
            Rectangle(
                (x0b, y0b),
                x1b - x0b,
                y1b - y0b,
                edgecolor="tab:red",
                facecolor="tab:red",
                alpha=0.16,
                linewidth=1.2,
            )
        )
    ax_xy.plot([], [], color="tab:red", linewidth=5, alpha=0.2, label="unsafe")

    cmap = plt.cm.get_cmap("tab10", n_unicycles)
    traj_lines = []
    body_pts = []
    heading_lines = []
    v_lines = []
    omega_lines = []
    for idx in range(n_unicycles):
        color = cmap(idx)
        traj_line, = ax_xy.plot([], [], color=color, linewidth=1.7, alpha=0.95, label=f"traj {idx + 1}")
        body_pt, = ax_xy.plot([], [], marker="o", color=color, markersize=5)
        heading_line, = ax_xy.plot([], [], color=color, linewidth=1.8)
        v_line, = ax_u.plot([], [], color=color, linewidth=1.4, alpha=0.95, label=f"v-{idx + 1}")
        omega_line, = ax_u.plot([], [], color=color, linewidth=1.4, linestyle="--", alpha=0.95, label=f"omega-{idx + 1}")
        traj_lines.append(traj_line)
        body_pts.append(body_pt)
        heading_lines.append(heading_line)
        v_lines.append(v_line)
        omega_lines.append(omega_line)

    heading_len = 0.8

    def _init():
        artists = []
        for idx in range(n_unicycles):
            traj_lines[idx].set_data([], [])
            body_pts[idx].set_data([], [])
            heading_lines[idx].set_data([], [])
            v_lines[idx].set_data([], [])
            omega_lines[idx].set_data([], [])
            artists.extend([traj_lines[idx], body_pts[idx], heading_lines[idx], v_lines[idx], omega_lines[idx]])
        return tuple(artists)

    def _update(i: int):
        artists = []
        for idx in range(n_unicycles):
            xy = states[: i + 1, idx, :2]
            traj_lines[idx].set_data(xy[:, 0], xy[:, 1])
            x, y, th = states[i, idx]
            body_pts[idx].set_data([x], [y])
            hx = x + heading_len * math.cos(float(th))
            hy = y + heading_len * math.sin(float(th))
            heading_lines[idx].set_data([x, hx], [y, hy])
            v_lines[idx].set_data(t[: i + 1], controls[: i + 1, idx, 0])
            omega_lines[idx].set_data(t[: i + 1], controls[: i + 1, idx, 1])
            artists.extend([traj_lines[idx], body_pts[idx], heading_lines[idx], v_lines[idx], omega_lines[idx]])
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=steps + 1,
        init_func=_init,
        interval=40,
        blit=True,
        repeat=False,
    )
    ax_xy.legend(loc="upper right")
    ax_u.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()
    return anim


def pretrain_controller_rollout(
    control_net: nn.Module,
    dynamics_model: Dynamics,
    init_range: np.ndarray,
    goal_range: np.ndarray,
    unsafe_range: np.ndarray,
    full_range: np.ndarray,
    device: str = "cpu",
    num_epochs: int = 2000,
    batch_size: int = 256,
    horizon: int = 80,
    dt: float = 0.05,
    lr: float = 1e-3,
    save_control_path: Optional[Path] = None,
):
    """
    Pre-train controller directly from trajectory rollouts, then keep it fixed
    for certificate training.
    """
    print("\n" + "=" * 80)
    print("CONTROLLER PRE-TRAINING (ROLLOUT-BASED)")
    print("=" * 80)

    if dynamics_model is None or dynamics_model.f is None or dynamics_model.g is None:
        raise ValueError("dynamics_model must provide both drift f and diffusion g.")
    if not callable(dynamics_model.f) or not callable(dynamics_model.g):
        raise ValueError("dynamics_model.f and dynamics_model.g must be callable.")

    control_net = control_net.to(device)
    control_net.train()
    f_cl_module = dynamics_model.f
    g_fn = dynamics_model.g
    optimizer = torch.optim.Adam(control_net.parameters(), lr=lr)

    init_t = torch.as_tensor(init_range, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(goal_range, dtype=torch.float32, device=device)
    full_t = torch.as_tensor(full_range, dtype=torch.float32, device=device)
    goal_center = 0.5 * (goal_t[:, 0] + goal_t[:, 1])

    D = init_t.shape[0]

    unsafe_t = torch.as_tensor(unsafe_range, dtype=torch.float32, device=device)
    if unsafe_t.dim() == 2 and unsafe_t.shape[1] == 2 and unsafe_t.shape[0] % D == 0:
        unsafe_boxes = unsafe_t.view(-1, D, 2)
    elif unsafe_t.dim() == 3 and unsafe_t.shape[1:] == (D, 2):
        unsafe_boxes = unsafe_t
    else:
        raise ValueError("unsafe_range must be (K*D,2) or (K,D,2).")

    def _sample_init(n: int) -> torch.Tensor:
        lo = init_t[:, 0]
        hi = init_t[:, 1]
        return torch.rand(n, D, device=device) * (hi - lo) + lo

    def _unsafe_penalty(x: torch.Tensor) -> torch.Tensor:
        # Penalize states inside any unsafe box.
        penalties = []
        for k in range(unsafe_boxes.shape[0]):
            lo = unsafe_boxes[k, :, 0]
            hi = unsafe_boxes[k, :, 1]
            margin = torch.minimum(x - lo, hi - x)        # >0 inside this box
            inside_margin = torch.min(margin, dim=1).values
            penalties.append(F.relu(inside_margin) ** 2)
        return torch.stack(penalties, dim=1).max(dim=1).values.mean()

    def _outside_penalty(x: torch.Tensor) -> torch.Tensor:
        lo = full_t[:, 0]
        hi = full_t[:, 1]
        v_lo = F.relu(lo - x)
        v_hi = F.relu(x - hi)
        return (v_lo + v_hi).sum(dim=1).mean()

    def _goal_cost(x: torch.Tensor) -> torch.Tensor:
        pos_err = x[:, :2] - goal_center[:2]
        th_err = torch.atan2(
            torch.sin(x[:, 2] - goal_center[2]),
            torch.cos(x[:, 2] - goal_center[2]),
        )
        return (pos_err.pow(2).sum(dim=1) + 0.2 * th_err.pow(2)).mean()

    best_loss = float("inf")
    best_control_state = None

    sqrt_dt = math.sqrt(dt)
    for epoch in range(num_epochs):
        x = _sample_init(batch_size)
        rollout_goal = torch.tensor(0.0, device=device)
        rollout_unsafe = torch.tensor(0.0, device=device)
        rollout_outside = torch.tensor(0.0, device=device)
        rollout_u = torch.tensor(0.0, device=device)

        for _ in range(horizon):
            u = control_net(x)
            rollout_u = rollout_u + (u.pow(2).sum(dim=1).mean())
            rollout_goal = rollout_goal + _goal_cost(x)
            rollout_unsafe = rollout_unsafe + _unsafe_penalty(x)
            rollout_outside = rollout_outside + _outside_penalty(x)

            drift = f_cl_module(x)
            diffusion = g_fn(x)
            if diffusion.dim() == 1:
                diffusion = diffusion.unsqueeze(0).expand_as(drift)
            noise = torch.randn_like(drift)
            x_next = x + dt * drift + sqrt_dt * diffusion * noise
            theta_wrapped = torch.atan2(torch.sin(x_next[:, 2]), torch.cos(x_next[:, 2])).unsqueeze(1)
            x = torch.cat([x_next[:, :2], theta_wrapped], dim=1)

        terminal_goal = _goal_cost(x)
        loss = (
            1.0 * rollout_goal / horizon
            + 20.0 * terminal_goal
            + 50.0 * rollout_unsafe / horizon
            + 5.0 * rollout_outside / horizon
            + 0.02 * rollout_u / horizon
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(
                f"  Epoch [{epoch}/{num_epochs}] "
                f"Loss={loss.item():.4f} "
                f"(goal={float((rollout_goal / horizon).item()):.4f}, "
                f"term={float(terminal_goal.item()):.4f}, "
                f"unsafe={float((rollout_unsafe / horizon).item()):.4f}, "
                f"out={float((rollout_outside / horizon).item()):.4f}, "
                f"u={float((rollout_u / horizon).item()):.4f})"
            )

    if best_control_state is not None:
        control_net.load_state_dict(best_control_state)
        print(f"\n  Best rollout pretrain loss: {best_loss:.6f}")
        if save_control_path is not None:
            torch.save(best_control_state, save_control_path)
            print(f"  Saved rollout-pretrained Controller_net to: {save_control_path}")

    print("=" * 80 + "\n")


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
        # v_loss_others = F.relu(v_eq - v_others).sum()
        v_loss_others = F.relu(0.1 - v_others).sum()

        # Total V loss (v_loss_inside_goal and v_loss_others are not used anymore)
        loss_v = (v_loss_full
            + v_loss_init
            + v_loss_unsafe
            + v_loss_inside_goal
            + v_loss_others
        )

        # Phi loss on SAME x_others -> enforce phi(x) <= -1.0
        loss_phi = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_phi = x_others.detach().clone().requires_grad_(True)
            phi_output = GV_net(x_phi).squeeze(-1)
            loss_phi = F.relu(phi_output + 1.0).sum()

        total_loss = loss_v + loss_phi

        # Add regularization for V network
        reg_w = _l2_weight_penalty(model, exclude_bias=True)
        reg_w_control = torch.tensor(0.0, device=device)
        if control_net is not None:
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
                # print(model(model.input_offset))
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
    control_net: nn.Module = None,
    save_control_path: Optional[Path] = None,
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
        save_control_path: if set and control_net is optimized, save controller
            whenever total bound-training loss reaches a new best.
    """
    print("\n" + "="*80)
    print("BOUND-BASED TRAINING (using CROWN)")
    print("="*80)

    # Move models to device
    V_net = V_net.to(device)

    # Collect ALL cells in the same order as original (init, goal, unsafe, outside, generator)
    print("\nCollecting all cells for V network...")
    all_cells_V = []
    cell_counts_V = {}
    region_order_V = ['init', 'goal', 'unsafe', 'outside']

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
    best_total_loss = float("inf")
    start_time = time.time()
    # Compute total loss from bounds
    loss_kwargs = {
        'beta_ra': params.constraints.beta_ra,
        'device': params.training.device,
        'compute_V': params.compute_V,
        'compute_GV': params.compute_GV,
    }

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
                phi_uppers = crown_cache_phi.compute_bounds(input_lowers_gen, input_uppers_gen)
                current_gen_weight = params.training.generator_weight

                # Track failing cells for adaptive refinement
                phi_upper_failing_mask = phi_uppers > 0.0
                num_total_failing = phi_upper_failing_mask.sum().item()
                
            else:
                phi_uppers = torch.tensor([], device=device)
                current_gen_weight = 0.0
                num_total_failing = 0

        # Update loss kwargs with current bounds (reuse pre-allocated dict)
        if params.compute_V:
            loss_kwargs['beta_ra'] = params.constraints.beta_ra
            loss_kwargs['V_goal_lower'] = bounds['goal'][0]
            loss_kwargs['V_unsafe_lower'] = bounds['unsafe'][0]
            loss_kwargs['V_init_upper'] = bounds['init'][1]
            loss_kwargs['V_outside_lower'] = bounds['outside'][0]

        if params.compute_GV:
            loss_kwargs['Phi_upper'] = phi_uppers
            loss_kwargs['generator_weight'] = current_gen_weight

        total_loss, loss_dict, _ = compute_total_loss_bounds(**loss_kwargs)

        if (
            control_net is not None
            and save_control_path is not None
            and (epoch + 1) % 200 == 0
        ):
            control_state = {
                k: v.detach().cpu().clone()
                for k, v in control_net.state_dict().items()
            }
            torch.save(control_state, save_control_path)

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
                MAX_CELLS = 40000

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
                outside_failing_mask_relax = bounds_updated['outside'][0] <= 20.0 #8.0 + v_eq
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
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV)
            # if control_net is not None:
            #     for name, param in control_net.named_parameters():
            #         if param.requires_grad:
            #             print(f" [Controller Params] {name} = {param.data}")
            loss_dict['epoch'] = epoch
            loss_history.append(loss_dict.copy())

        # Early stopping check
        with torch.no_grad():
            # Check V constraints
            all_satisfied = True
            if params.compute_V:
                show = (epoch % 10 == 0)
                _, goal_satisfied = compute_loss_goal_bounds(bounds_updated["goal"][0])
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
                    loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
                if params.compute_GV:
                    loss_parts.append(f"Gen={loss_dict['generator']:.4f}")
                print(f"Final losses: {', '.join(loss_parts)}")

                if control_net is not None:
                    for name, param in control_net.named_parameters():
                        if param.requires_grad:
                            print(f" [Controller Params] {name} = {param.data}")
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
                beta_ra=params.constraints.beta_ra,
                device=device,
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
        
        # Optimizer step
        optimizer.step()
        scheduler.step()

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

    return loss_history, refinement_epochs


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    args = parser.parse_args()

    print("=" * 80)
    print("XV-15 CERTIFICATE-BASED CONTROL SYNTHESIS (V + Phi)")
    print("=" * 80)

        # === Hyperparameters ===
    params = Hyperparameters.default()

    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    pi = np.pi
    xy_limit = 8.0
    theta_limit = 0.9*pi
    unsafe_xy_band = 0.5
    unsafe_theta_band = 0.1
    params.network.input_scale = [xy_limit, xy_limit, theta_limit]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0
    params.constraints.beta_ra = 2.0

    params.compute_V = True
    params.compute_GV = True

    device = params.training.device

    # -------------------------------------------------------------------------
    # 3) Regions (init, goal, unsafe union, full)
    # -------------------------------------------------------------------------
    init_range = np.array([[-4.0, -3.0], [-4.0, -3.0], [-0.3, 0.3]], dtype=np.float32)
    goal_range = np.array([[2.0, xy_limit], [2.0, xy_limit], [0.0, 0.5*pi]], dtype=np.float32)
    unsafe_xmin = np.array([[-xy_limit, -xy_limit + unsafe_xy_band], [-xy_limit, xy_limit], [-theta_limit, theta_limit]], dtype=np.float32)
    unsafe_xmax = np.array([[xy_limit - unsafe_xy_band, xy_limit], [-xy_limit, 2.0], [-theta_limit, theta_limit]], dtype=np.float32)
    unsafe_ymin = np.array([[-xy_limit, xy_limit], [-xy_limit, -xy_limit + unsafe_xy_band], [-theta_limit, theta_limit]], dtype=np.float32)
    unsafe_ymax = np.array([[-xy_limit, 2.0], [xy_limit - unsafe_xy_band, xy_limit], [-theta_limit, theta_limit]], dtype=np.float32)
    unsafe_tmin = np.array([[-xy_limit, xy_limit], [-xy_limit, xy_limit], [-theta_limit, -theta_limit + unsafe_theta_band]], dtype=np.float32)
    unsafe_tmax = np.array([[-xy_limit, xy_limit], [-xy_limit, xy_limit], [theta_limit - unsafe_theta_band, theta_limit]], dtype=np.float32)
    unsafe_range = np.vstack((unsafe_xmin, unsafe_xmax, unsafe_ymin, unsafe_ymax, unsafe_tmin, unsafe_tmax))
    full_range = np.array([[-xy_limit, xy_limit], [-xy_limit, xy_limit], [-theta_limit, theta_limit]], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe_xmin = Region(unsafe_xmin)
    unsafe_xmax = Region(unsafe_xmax)
    unsafe_ymin = Region(unsafe_ymin)
    unsafe_ymax = Region(unsafe_ymax)
    unsafe_tmin = Region(unsafe_tmin)
    unsafe_tmax = Region(unsafe_tmax)
    unsafe = Region.union(unsafe_xmin, unsafe_xmax, unsafe_ymin, unsafe_ymax, unsafe_tmin, unsafe_tmax)
    full = Region(full_range)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # Discretization: start modest; refine happens during training
    params.discretization.n_goal = 32
    params.discretization.n_outside_goal = 8
    params.discretization.n_generator = 4
    params.discretization.n_unsafe = 16
    params.discretization.n_init = 28

    # -------------------------------------------------------------------------
    # 2) Dynamics (closed-loop, torch, differentiable)
    # -------------------------------------------------------------------------
    # Create diffusion coefficients as a persistent tensor to avoid TracerWarnings
    g_coeffs = torch.tensor([0.02, 0.02, 0.005], dtype=torch.float32)
    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term g(x) that is constant
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
        
    # === Dynamics ===
    u_nn = UnicycleControlNN()
    f_cl_module = ClosedLoopDrift(u_nn)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=3)

    # -------------------------------------------------------------------------
    # 4) Networks (V and GV)
    # -------------------------------------------------------------------------
    input_offset = [5.0, 5.0, 0.25*pi]
    output_offset = np.float32(0.1)
    V_net = create_V(params.network, 
        input_offset=input_offset, output_offset=output_offset
    )
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        input_offset=input_offset,
        verify=True
    )

    # -------------------------------------------------------------------------
    # 5) Discretize regions
    # -------------------------------------------------------------------------
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=False,
    )

    # -------------------------------------------------------------------------
    # 6) Pretrain + train (or load)
    # -------------------------------------------------------------------------
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = 1.0
    params.constraints.pretrain_phi_target = 0.0
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")
        print(f"\n{dynamics}")
        print(f"\nUsing device: {device}")

        ENABLE_PRETRAINING = True
        PRETRAIN_EPOCHS = 3000
        PRETRAIN_LR = 0.01

        if ENABLE_PRETRAINING:
            # Stage 1: train controller from trajectory rollout.
            # pretrain_controller_rollout(
            #     control_net=u_nn,
            #     dynamics_model=dynamics,
            #     init_range=init_range,
            #     goal_range=goal_range,
            #     unsafe_range=unsafe_range,
            #     full_range=full_range,
            #     device=device,
            #     num_epochs=3000,
            #     batch_size=256,
            #     horizon=80,
            #     dt=0.05,
            #     lr=1e-3,
            #     save_control_path=OUTPUT_DIR / "controller_pretrained.pth",
            # )
            # u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))

            # Stage 2: pre-train V (certificate shaping) with controller fixed.
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
                lambda_w=1e-4,
                n_each=1200,
                save_v_path= OUTPUT_DIR / "V_pretrained.pth",
                save_control_path=OUTPUT_DIR / "controller_pretrained.pth",
            )
            print("Pretraining completed.\n")
        else:
            u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
            V_net.load_state_dict(torch.load(OUTPUT_DIR / "V_pretrained.pth", map_location=device))
        
        loss_history, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            visualize_interval=1000,
            control_net=u_nn,  # freeze controller; train certificate V/GV only
            save_control_path=OUTPUT_DIR / "controller_best_bounds.pth",
        )

        print("\n" + "=" * 80)
        print("FINAL EVALUATION")
        print("=" * 80)

        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_ra=params.constraints.beta_ra,
            device=device,
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
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
        )

    else:
        print("\n" + "=" * 80)
        print("LOADING SAVED BUNDLE (skip training)")
        print("=" * 80)

        pretrain_u_nn = UnicycleControlNN()
        pretrain_u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
        pretrain_f_cl_module = ClosedLoopDrift(pretrain_u_nn)
        pretrain_dynamics = Dynamics.dynamics(f=pretrain_f_cl_module, g=g, state_dim=3)
        animate_unicycle_xy_theta(
            dynamics_model=pretrain_dynamics,
            init_range=init_range,
            goal_range=goal_range,
            unsafe_range=unsafe_range,
            full_range=full_range,
            n_unicycles=8,
        )

        bound_u_nn = UnicycleControlNN()
        bound_u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_best_bounds.pth", map_location=device))
        bound_f_cl_module = ClosedLoopDrift(bound_u_nn)
        bound_dynamics = Dynamics.dynamics(f=bound_f_cl_module, g=g, state_dim=3)
        animate_unicycle_xy_theta(
            dynamics_model=bound_dynamics,
            init_range=init_range,
            goal_range=goal_range,
            unsafe_range=unsafe_range,
            full_range=full_range,
            n_unicycles=16,
        )
        # bundle = load_eval_bundle(bundle_path, map_location="cpu")
        # params = Hyperparameters.from_dict(bundle["hyperparameters"])
        # device = params.training.device
        # # rebuild region_cells (already cpu tensors)
        # region_cells = bundle["region_cells"]
        # # rebuild networks & load
        # V_net.load_state_dict(bundle["V_state_dict"])
        # # load controller
        # if bundle["control_state_dict"] is not None:
        #     print("load u_nn")
        #     u_nn.load_state_dict(bundle["control_state_dict"])



if __name__ == "__main__":
    main()
