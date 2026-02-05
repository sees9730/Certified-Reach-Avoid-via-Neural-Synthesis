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
from matplotlib.patches import Polygon, Arc

# -----------------------------------------------------------------------------
# Repo paths (match your Lorentz script)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # repo_root
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.hyperparameters import Hyperparameters
from src.dynamics import Dynamics, ClosedLoopDrift
from src.regions import Regions, Region
from src.network import create_V
from src.phi_module import create_GV
from src.control_network import TanhPolicy, Wrapper4DConterlNN
from src.discretization import discretize_regions
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.trainer import train_network_bounds
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
from src.utils import cleanup_and_setup_directories
from src.visualization import (
    create_summary_plots
)

torch.manual_seed(0)
np.random.seed(0)


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
    lr=1e-4,
    device='cpu',
    control_net=None,
    n_each: int = 400,
    lambda_w = 0.1,
    save_v_path=None,
    save_control_path=None,
):
    """
    Pre-train V and GV networks using sampled points.
    """
    print("\n" + "="*20)
    print("Pre-training using samples")
    print("="*20)

    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    if GV_net is not None:
        print("GV pre-training enabled")
    else:
        print("GV pre-training disabled")

    best_loss = float('inf')
    best_model_state = None
    best_control_state = None

    # Convert ranges to torch once: each is (D,2)
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    init_t = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
    low  = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low

    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        b_low = box[:, 0]
        b_high = box[:, 1]
        return torch.rand(N, D, device=device) * (b_high - b_low) + b_low

    def _in_box(x_batch: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return ((x_batch >= box[:, 0]) & (x_batch <= box[:, 1])).all(dim=1)

    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
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

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # full-range samples -> enforce v(x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # init-range samples -> enforce v(x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # unsafe-range samples -> enforce v(x) >= beta_ra
        x_unsafe = _sample_in_unsafe_union(int(n_each/8))
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.beta_ra - v_unsafe).sum()

        # samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= 0.0
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

        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
        )

        loss_gv = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_gv = x_others.detach().clone().requires_grad_(True)
            gv_output = GV_net(x_gv).squeeze(-1)
            loss_gv = F.relu(gv_output).sum()

        total_loss = loss_v + loss_gv

        # L2 weight penalty
        reg_w = _l2_weight_penalty(model, exclude_bias=True)
        total_loss = total_loss + lambda_w * reg_w

        # Track best
        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        if epoch % 100 == 0:
            if GV_net is not None:
                print(f"Epoch {epoch} | V_loss={loss_v.item():8.4f} | GV_loss={loss_gv.item():8.4f}")
            else:
                print(f"Epoch {epoch} | V_loss={loss_v.item():8.4f}")

        # Optimize/update networks
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\nBest loss: {best_loss:.6f}")

        # Save best pretrained weights (state_dict)
        if save_v_path is not None:
            torch.save(best_model_state, save_v_path)
            print(f"Saved pretrained V_net to: {save_v_path}")
        if (control_net is not None) and (best_control_state is not None) and (save_control_path is not None):
            torch.save(best_control_state, save_control_path)
            print(f"Saved pretrained Controller_net to: {save_control_path}")

    networks_trained = ["V"]
    if GV_net is not None:
        networks_trained.append("GV")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")


def main(benchmark_mode=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    parser.add_argument("--benchmark", type=int, default=0, choices=[0, 1],
                        help="1: run benchmark (5 runs), 0: normal run")
    args = parser.parse_args()

    if benchmark_mode:
        args.benchmark = 1

    print("="*20)
    print("2D Inverted Pendulum Synthesis")
    print("="*20)

    training_time_result = None

    # === Hyperparameters ===
    params = Hyperparameters.default()

    params.network.n_inputs = 4
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [10.0, 10.0, 10.0, 10.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 12
    params.discretization.n_outside_goal = 4
    params.discretization.n_generator = 2
    params.discretization.n_unsafe = 8
    params.discretization.n_init = 12

    params.constraints.beta_ra = 5.0

    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 5000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1600

    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_interval = 500
    params.refinement.v_outside.late_epoch_threshold = 999999999
    params.refinement.v_outside.refine_interval_late = 100
    params.refinement.v_outside.refine_factor = 2
    params.refinement.v_outside.max_cells = 100000
    params.refinement.v_outside.N_to_refine = 100

    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 501
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 6.0

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_interval = 500
    params.refinement.gv_generator.late_epoch_threshold = 999999999
    params.refinement.gv_generator.refine_interval_late = 100
    params.refinement.gv_generator.refine_factor = 2
    params.refinement.gv_generator.max_cells = 100000
    params.refinement.gv_generator.N_to_refine = 100

    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -500.0

    # === Dynamics ===
    policy_net = TanhPolicy(n_in=4, n_hidden=64, n_out=2)
    u_nn = Wrapper4DConterlNN(policy_net, U_max=10.0)

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # terrain that induced gravity acceleration
        g = 3.71          # Mars (use 1.62 for Moon)
        a = 0.05
        kx = ky = 3.1415926535 / 10.0   # ~0.314 => 1 period across [-10,10]
        dhdx = a * kx * torch.cos(kx * x1) * torch.sin(ky * x3)
        dhdy = a * ky * torch.sin(kx * x1) * torch.cos(ky * x3)

        f1 = x2
        f2 = -g * dhdx
        f3 = x4
        f4 = -g * dhdy
        return torch.stack([f1, f2, f3, f4], dim=1)

    g_coeffs = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        base = g_coeffs.to(device=x.device, dtype=x.dtype)
        if x.dim() == 1:
            return base
        elif x.dim() == 2:
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)
        else:
            raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")

    f_cl_module = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    # === Regions ===
    full_range = np.array([
        [-10.0, 10.0], 
        [-10.0, 10.0],
        [-10.0, 10.0], 
        [-10.0, 10.0]
    ], dtype=np.float32)

    init_range = np.array([
        [-4.0, -3.0], 
        [-1.0, 1.0],
        [-4.0, -3.0], 
        [-1.0, 1.0]
    ], dtype=np.float32)

    goal_range = np.array([
        [3.0, 4.0], 
        [-1.0, 1.0],
        [3.0, 4.0], 
        [-1.0, 1.0]
    ], dtype=np.float32)

    unsafe_min_x1 = np.array([
        [full_range[0,0], full_range[0,0]+0.1],
        full_range[1,:],
        full_range[2,:],
        full_range[3,:],
    ], dtype=np.float32)

    unsafe_min_x2 = np.array([
        full_range[0,:],
        [full_range[1,0], full_range[1,0]+0.1],
        full_range[2,:],
        full_range[3,:],
    ], dtype=np.float32)

    unsafe_min_x3 = np.array([
        full_range[0,:],
        full_range[1,:],
        [full_range[2,0], full_range[1,0]+0.1],
        full_range[3,:],
    ], dtype=np.float32)

    unsafe_min_x4 = np.array([
        full_range[0,:],
        full_range[1,:],
        full_range[2,:],
        [full_range[3,0], full_range[3,0]+0.1],
    ], dtype=np.float32)

    unsafe_max_x1 = np.array([
        [full_range[0,1]-0.1, full_range[0,1]],
        full_range[1,:],
        full_range[2,:],
        full_range[3,:],
    ], dtype=np.float32)

    unsafe_max_x2 = np.array([
        full_range[0,:],
        [full_range[1,1]-0.1, full_range[1,1]],
        full_range[2,:],
        full_range[3,:],
    ], dtype=np.float32)

    unsafe_max_x3 = np.array([
        full_range[0,:],
        full_range[1,:],
        [full_range[2,1]-0.1, full_range[2,1]],
        full_range[3,:],
    ], dtype=np.float32)

    unsafe_max_x4 = np.array([
        full_range[0,:],
        full_range[1,:],
        full_range[2,:],
        [full_range[3,1]-0.1, full_range[3,1]],
    ], dtype=np.float32)

    unsafe_range = np.vstack((
        unsafe_min_x1, unsafe_min_x2, unsafe_min_x3, unsafe_min_x4,
        unsafe_max_x1, unsafe_max_x2, unsafe_max_x3, unsafe_max_x4
        ))

    init = Region(init_range)
    goal = Region(goal_range)
    us_min_x1_rg = Region(unsafe_min_x1)
    us_min_x2_rg = Region(unsafe_min_x2)
    us_min_x3_rg = Region(unsafe_min_x3)
    us_min_x4_rg = Region(unsafe_min_x4)
    us_max_x1_rg = Region(unsafe_max_x1)
    us_max_x2_rg = Region(unsafe_max_x2)
    us_max_x3_rg = Region(unsafe_max_x3)
    us_max_x4_rg = Region(unsafe_max_x4)
    unsafe = Region.union(
        us_min_x1_rg, us_min_x2_rg, us_min_x3_rg, us_min_x4_rg,
        us_max_x1_rg, us_max_x2_rg, us_max_x3_rg, us_max_x4_rg
    )
    full = Region(full_range)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # === Networks ===
    input_offset = [3.5, 0.0, 3.5, 0.0]
    output_offset = 0.1
    V_net = create_V(params.network, input_offset=input_offset, output_offset=output_offset)
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        input_offset=input_offset,
        verify=False
    )
    res = check_gv_matches_autograd_full_range(
        V_net, GV_net, dynamics,
        full_range=full_range,
        num_points=1024,
        batch_size=64,
        device=params.training.device,
    )

    # === Discretization ===
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=False
    )

    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")

        training_start_time = time.time()

        if params.training.enable_pretraining:
            pretrain_network_samples(
                model=V_net,
                x_goal_range=goal_range,
                x_unsafe_range=unsafe_range,
                x_init_range=init_range,
                x_range=full_range,
                params=params,
                GV_net=GV_net,
                num_epochs=params.training.pretrain_epochs,
                lr=params.training.pretrain_lr,
                device=params.training.device,
                control_net=u_nn,
                n_each=params.training.pretrain_n_samples,
                save_v_path=OUTPUT_DIR / "V_pretrained.pth",
                save_control_path=OUTPUT_DIR / "controller_pretrained.pth"
            )

        V_net.load_state_dict(torch.load(OUTPUT_DIR / "V_pretrained.pth", map_location=params.training.device))
        u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=params.training.device))
        
        def create_scheduler(optimizer):
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=2000,
                gamma=0.95
            )

        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            control_net=u_nn,
            create_scheduler=create_scheduler,
            start_time=training_start_time
        )

        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        print("\n" + "="*20)
        print("Final Evaluation")
        print("="*20)

        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_ra=params.constraints.beta_ra,
            device=params.training.device
        )
        print_constraint_summary(results)

        print("\n" + "="*20)
        print("Visualizations")
        print("="*20)

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

        print("\n" + "="*20)
        print("Saving Bundle")
        print("="*20)
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

        training_time_result = total_training_time

    else:
        print("\n" + "="*20)
        print("Loading Bundle")
        print("="*20)

        bundle = load_eval_bundle(bundle_path, map_location="cpu")

        params = Hyperparameters.from_dict(bundle["hyperparameters"])
        region_cells = bundle["region_cells"]

        V_net = create_V(params.network).to(params.training.device)
        V_net.load_state_dict(bundle["V_state_dict"])

        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network,
            training_config=params.training
        ).to(params.training.device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"])

        if bundle["control_state_dict"] is not None:
            u_nn.load_state_dict(bundle["control_state_dict"])

        region_cells = {
            k: [(lo.to(params.training.device), hi.to(params.training.device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }

        final_beta_s = bundle["final_beta_s"]
        loss_history = bundle["loss_history"]
        refinement_epochs = bundle["refinement_epochs"]

        results = bundle.get("final_results", None)
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_ra=params.constraints.beta_ra,
                device=params.training.device
            )

        print("\n" + "="*20)
        print("Final Evaluation (loaded)")
        print("="*20)
        print_constraint_summary(results)

        print("\n" + "="*20)
        print("Visualizations (loaded)")
        print("="*20)
        log_loaded_training_epochs(loss_history)

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

    return training_time_result


if __name__ == '__main__':
    import sys
    is_benchmark = '--benchmark=1' in sys.argv or '--benchmark' in sys.argv and '1' in sys.argv

    if is_benchmark:
        n_runs = 2
        times = []
        cells = []

        print("="*20)
        print(f"Benchmark: {n_runs} runs")
        print("="*20)

        for i in range(n_runs):
            print(f"\n*** Run {i+1}/{n_runs} ***")
            training_time = main(benchmark_mode=True)
            if training_time is not None:
                times.append(training_time)

            bundle_path = OUTPUT_DIR / "eval_bundle.pth"
            if bundle_path.exists():
                import torch
                bundle = torch.load(bundle_path, map_location='cpu')
                cell_counts = {
                    'init': len(bundle['region_cells']['init']),
                    'goal': len(bundle['region_cells']['goal']),
                    'unsafe': len(bundle['region_cells']['unsafe']),
                    'outside': len(bundle['region_cells']['outside']),
                    'generator': len(bundle['region_cells']['generator']),
                }
                cell_counts['total_v'] = cell_counts['init'] + cell_counts['goal'] + cell_counts['unsafe'] + cell_counts['outside']
                cell_counts['total'] = cell_counts['total_v'] + cell_counts['generator']
                cells.append(cell_counts)

        print("\n" + "="*20)
        print("Benchmark Results")
        print("="*20)
        avg_time = stats.mean(times)
        std_time = stats.stdev(times)
        print(f"Training time (pretrain+train): {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"  Individual times: {[f'{t:.2f}s' for t in times]}")

        if cells:
            categories = ['init', 'goal', 'unsafe', 'outside', 'generator', 'total_v', 'total']
            print("\nCell counts:")
            for cat in categories:
                values = [c[cat] for c in cells]
                avg = stats.mean(values)
                std = stats.stdev(values)
                print(f"  {cat:12s}: {avg:.0f} ± {std:.0f}")
    else:
        main()
