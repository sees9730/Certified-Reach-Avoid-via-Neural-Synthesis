from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Union, Optional, Any, Dict, Tuple
import numpy as np
import torch
import csv
import os


from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
import sys
sys.path.insert(0, str(ROOT))

from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.dynamics import Dynamics, ClosedLoopDrift
from src.control_network import LinearControlNN
from src.save_load_utils import load_eval_bundle

torch.set_default_dtype(torch.float32)


# =============================================================================
# Autograd utilities
# =============================================================================
def grad_and_hessdiag_scalar(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    y: (N,) scalar per sample
    x: (N,D) with requires_grad=True

    returns:
      grad_y: (N,D)
      hess_diag: (N,D) diagonal of Hessian
    """
    grad_y = torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]
    hdiag = []
    for d in range(x.shape[1]):
        gdd = torch.autograd.grad(grad_y[:, d].sum(), x, create_graph=True, retain_graph=True)[0][:, d]
        hdiag.append(gdd)
    return grad_y, torch.stack(hdiag, dim=1)


def G_of_scalar(y: torch.Tensor, x: torch.Tensor, dynamics: Dynamics) -> torch.Tensor:
    """
    Infinitesimal generator for diagonal or full-matrix diffusion.

    Phi(x) = f(x)·∇y(x) + 0.5 * Tr(g(x)g(x)^T H_y(x))

    Supports:
      gx: (N,D)      -> diagonal diffusion entries
      gx: (N,D,K)    -> full diffusion matrix
    """
    f_fn = dynamics.get_f()
    g_fn = dynamics.get_g()
    if not callable(f_fn) or not callable(g_fn):
        raise ValueError("dynamics.f and dynamics.g must be callable.")

    fx = f_fn(x)  # (N,D)
    gx = g_fn(x)  # (N,D) or (N,D,K)

    grad_y, hess_diag = grad_and_hessdiag_scalar(y, x)
    drift = (fx * grad_y).sum(dim=1)

    if gx.ndim == 2:
        # diagonal diffusion: Tr(g g^T H) = sum_d g_d^2 * d2y/dx_d2
        diff = 0.5 * ((gx * gx) * hess_diag).sum(dim=1)
        return drift + diff

    if gx.ndim == 3:
        # full diffusion: need full Hessian
        D = x.shape[1]
        H_rows = []
        for a in range(D):
            Ha = torch.autograd.grad(grad_y[:, a].sum(), x, create_graph=True, retain_graph=True)[0]  # (N,D)
            H_rows.append(Ha)
        H = torch.stack(H_rows, dim=1)  # (N,D,D)

        GGt = gx @ gx.transpose(1, 2)  # (N,D,D)
        diff = 0.5 * (GGt * H).sum(dim=(1, 2))
        return drift + diff

    raise ValueError(f"Unsupported g(x) shape {tuple(gx.shape)}; expected (N,D) or (N,D,K).")


# =============================================================================
# V-net feature helpers (consistent with src.network.V.forward)
# =============================================================================
def V_last_hidden(V_net, x: torch.Tensor) -> torch.Tensor:
    """
    Scaled last-hidden features passed into output():
        h_tilde(x) = scale_factor * act(layer2(act(layer1(x / input_scale))))
    Shape: (N, L)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    z = x / V_net.input_scale
    z = V_net.activation_fn(V_net.layer1(z))
    h = V_net.activation_fn(V_net.layer2(z))
    return h * V_net.scale_factor


def phi_features(V_net, x: torch.Tensor, dynamics: Dynamics) -> torch.Tensor:
    """
    phi_j(x) = G[h_tilde_j](x), stacked -> (N, L)

    Since V(x) = w^T h_tilde(x) + b and G is linear:
        G[V](x) = w^T G[h_tilde](x)
    """
    h_tilde = V_last_hidden(V_net, x)  # (N, L)
    cols = [G_of_scalar(h_tilde[:, j], x, dynamics) for j in range(h_tilde.shape[1])]
    return torch.stack(cols, dim=1)  # (N, L)


def verify_V_decomposition(V_net, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Verifies:
      V_net(x) == (V_last_hidden(x) * scale_factor) @ w + b
    Returns: (max_abs_err, mean_abs_err)
    """
    V_direct = V_net(x).squeeze(-1)
    h = V_last_hidden(V_net, x)
    w = V_net.output.weight[0]
    b = V_net.output.bias[0]
    V_recon = h @ w + b
    err = (V_direct - V_recon).abs()
    return err.max(), err.mean()


def verify_G_decomposition(V_net, x: torch.Tensor, dynamics: Dynamics) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Verifies:
      G[V](x) == phi(x) @ w
    Returns: (max_abs_err, mean_abs_err)
    """
    Vx = V_net(x).squeeze(-1)
    GV_direct = G_of_scalar(Vx, x, dynamics)
    phi = phi_features(V_net, x, dynamics)
    w = V_net.output.weight[0]
    GV_decomp = phi @ w
    err = (GV_direct - GV_decomp).abs()
    print("max GV (on x_gen samples): ", torch.max(GV_direct).item())
    return err.max(), err.mean()


# =============================================================================
# Sampling: sample once from full_range, then filter
# =============================================================================
def _as_torch(a, device="cpu", dtype=torch.float32):
    return torch.as_tensor(a, device=device, dtype=dtype)

def _in_box(x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    return ((x >= box[:, 0]) & (x <= box[:, 1])).all(dim=1)

def _unsafe_to_boxes(unsafe_range, D: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    t = _as_torch(unsafe_range, device=device, dtype=dtype)
    if t.dim() == 2:
        if t.shape == (D, 2):
            return t.unsqueeze(0)
        if t.shape[1] == 2 and (t.shape[0] % D == 0):
            return t.view(int(t.shape[0] // D), D, 2)
        raise ValueError(f"unsafe_range must be (D,2) or (K*D,2); got {tuple(t.shape)}")
    if t.dim() == 3:
        if t.shape[1:] != (D, 2):
            raise ValueError(f"unsafe_range must be (K,D,2) with D={D}; got {tuple(t.shape)}")
        return t
    raise ValueError(f"unsafe_range must be 2D or 3D; got {t.dim()}D")

def _in_unsafe_union(x: torch.Tensor, unsafe_boxes: torch.Tensor) -> torch.Tensor:
    m = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for k in range(unsafe_boxes.shape[0]):
        m |= _in_box(x, unsafe_boxes[k])
    return m

def sample_and_partition(
    N_samples: int,
    full_range: np.ndarray,
    init_range: np.ndarray,
    goal_range: np.ndarray,
    unsafe_range: np.ndarray,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
):
    # NEW: local RNG for reproducibility (doesn't affect global state)
    gen_rand = torch.Generator(device=device)
    gen_rand.manual_seed(int(seed))

    full_t = _as_torch(full_range, device=device, dtype=dtype)
    init_t = _as_torch(init_range, device=device, dtype=dtype)
    goal_t = _as_torch(goal_range, device=device, dtype=dtype)

    D = full_t.shape[0]
    low, high = full_t[:, 0], full_t[:, 1]

    x_full = torch.rand(N_samples, D, device=device, dtype=dtype, generator=gen_rand) * (high - low) + low

    init_mask = _in_box(x_full, init_t)

    unsafe_boxes = _unsafe_to_boxes(unsafe_range, D=D, device=device, dtype=dtype)
    unsafe_mask = _in_unsafe_union(x_full, unsafe_boxes)

    goal_mask = _in_box(x_full, goal_t)
    gen_mask = ~(goal_mask | unsafe_mask)

    return x_full, x_full[init_mask], x_full[unsafe_mask], x_full[gen_mask].requires_grad_(True)


def describe_samples_rows(
    x: Union[np.ndarray, torch.Tensor],
    *,
    V_net,
    dynamics,
    init_range: np.ndarray,
    unsafe_range: np.ndarray,
    goal_range: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    as_dict: bool = False,
) -> Union[List[list], Dict[str, Any], list]:
    """
    Works for one sample (D,) or (1,D) OR multiple samples (N,D).

    Returns (default):
      - if input is a single sample: one row -> [regions, V_last, GV_last_or_None]
      - if input is multiple samples: list of rows

    If as_dict=True, returns a dict-of-lists (always length N, where N=1 for single sample):
        {
          "regions": [List[str], ...],
          "V_last":  [np.ndarray, ...],
          "GV_last": [np.ndarray or None, ...],
        }

    Row format:
        [regions_belong_to, V_value_of_last_layer, GV_value_of_last_layer_or_None]
    """
    # -------------------------
    # Normalize input to torch (N, D)
    # -------------------------
    if isinstance(x, np.ndarray):
        x_t = torch.as_tensor(x, device=device, dtype=dtype)
    else:
        x_t = x.to(device=device, dtype=dtype)

    single_input = False
    if x_t.ndim == 1:
        single_input = True
        x_t = x_t.unsqueeze(0)  # (1, D)
    elif x_t.ndim == 2 and x_t.shape[0] == 1:
        single_input = True
    elif x_t.ndim != 2:
        raise ValueError(f"x must be shape (D,), (1,D), or (N,D). Got {tuple(x_t.shape)}")

    N, D = x_t.shape

    # -------------------------
    # Region membership masks
    # -------------------------
    init_t = _as_torch(init_range, device=device, dtype=dtype)
    goal_t = _as_torch(goal_range, device=device, dtype=dtype)
    unsafe_boxes = _unsafe_to_boxes(unsafe_range, D=D, device=device, dtype=dtype)

    init_mask = _in_box(x_t, init_t)                  # (N,)
    goal_mask = _in_box(x_t, goal_t)                  # (N,)
    unsafe_mask = _in_unsafe_union(x_t, unsafe_boxes) # (N,)
    gen_mask = ~(goal_mask | unsafe_mask)             # (N,) matches your sample_and_partition

    # regions list per sample (allow overlaps)
    regions_per_sample: List[List[str]] = []
    init_b = init_mask.detach().cpu().numpy().astype(bool)
    goal_b = goal_mask.detach().cpu().numpy().astype(bool)
    unsafe_b = unsafe_mask.detach().cpu().numpy().astype(bool)
    gen_b = gen_mask.detach().cpu().numpy().astype(bool)

    for i in range(N):
        r: List[str] = []
        if init_b[i]:
            r.append("init")
        if unsafe_b[i]:
            r.append("unsafe")
        if goal_b[i]:
            r.append("goal")
        if gen_b[i]:
            r.append("gen")
        regions_per_sample.append(r)

    # -------------------------
    # V last-hidden for all samples (batched)
    # -------------------------
    with torch.no_grad():
        v_last_all = V_last_hidden(V_net, x_t)  # (N, H)
    v_last_all_np = v_last_all.detach().cpu().numpy()

    # -------------------------
    # GV/phi features only for gen samples (batched), else None
    # -------------------------
    gv_last_list: List[Optional[np.ndarray]] = [None] * N
    if bool(gen_mask.any().item()):
        idx = torch.nonzero(gen_mask, as_tuple=False).squeeze(1)  # (K,)
        x_gen = x_t[idx].detach().requires_grad_(True)            # (K, D)

        gv_gen = phi_features(V_net, x_gen, dynamics=dynamics)    # (K, H)
        gv_gen_np = gv_gen.detach().cpu().numpy()

        for j, i in enumerate(idx.detach().cpu().tolist()):
            gv_last_list[i] = gv_gen_np[j]

    # -------------------------
    # Package outputs
    # -------------------------
    if as_dict:
        out = {
            "regions": regions_per_sample,
            "V_last": [v_last_all_np[i] for i in range(N)],
            "GV_last": gv_last_list,
        }
        return out

    rows = [[regions_per_sample[i], v_last_all_np[i], gv_last_list[i]] for i in range(N)]
    return rows[0] if single_input else rows


def write_describe_dict_to_csv(data: dict, csv_path: str) -> None:
    """
    data is the output of describe_samples_rows(..., as_dict=True):
      {
        "regions": [List[str], ...],
        "V_last":  [np.ndarray(H), ...],
        "GV_last": [np.ndarray(H) or None, ...],
      }
    Appends rows to csv_path. Writes header only if file is empty or missing.
    """

    regions_list = data["regions"]
    v_list = data["V_last"]
    gv_list = data["GV_last"]

    N = len(regions_list)
    if N == 0:
        raise ValueError("Empty dataset.")

    # infer H from V_last (always present)
    H = int(np.asarray(v_list[0]).shape[0])

    header = (
        ["regions"]
        + [f"Vlast{j}" for j in range(H)]
        + [f"GVlast{j}" for j in range(H)]
    )

    file_exists = os.path.exists(csv_path)
    file_is_empty = (not file_exists) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # write header only once
        if file_is_empty:
            writer.writerow(header)

        for i in range(N):
            regions_str = "|".join(regions_list[i])  # e.g. "init|gen"
            v = np.asarray(v_list[i]).reshape(-1)

            gv = gv_list[i]
            if gv is None:
                gv_row = [""] * H
            else:
                gv_row = np.asarray(gv).reshape(-1).tolist()

            row = [regions_str] + v.tolist() + gv_row
            writer.writerow(row)


def save_x_full_to_csv(x_full: torch.Tensor, csv_path: str) -> None:
    x_np = x_full.detach().cpu().numpy()
    D = x_np.shape[1]

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        if write_header:
            header = ",".join(f"x{j}" for j in range(D))
            f.write(header + "\n")

        np.savetxt(f, x_np, delimiter=",")