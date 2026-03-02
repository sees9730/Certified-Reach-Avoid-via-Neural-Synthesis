"""
Last-layer LP solver for certificate synthesis.

This module solves for the last-layer parameters of V (output weights + scalar offset)
using robust IBP bounds on:
  - V last-layer features a(x)
  - GV last-layer features phi(x)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from src.crown_bounds import prepare_cell_bounds


TensorPairList = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class LPSolveResult:
    solved: bool
    message: str
    t_opt: float | None = None
    status: int | None = None
    debug_summary: str | None = None


class VLastLayerFeature(nn.Module):
    """
    Feature map a(x) so that V(x) = a(x)^T w + c.
    """

    def __init__(self, v_net: nn.Module):
        super().__init__()
        self.v_net = v_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        if hasattr(self.v_net, "input_offset"):
            x_norm = (x_in - self.v_net.input_offset) / self.v_net.input_scale
        else:
            x_norm = x_in / self.v_net.input_scale

        h = self.v_net.activation_fn(self.v_net.layer1(x_norm))
        h = self.v_net.activation_fn(self.v_net.layer2(h))
        scale = self.v_net.scale_factor

        if hasattr(self.v_net, "input_offset"):
            # For V_offset: baseline is evaluated at x_norm = 0, not x = 0.
            x_norm0 = torch.zeros_like(x_norm)
            h0 = self.v_net.activation_fn(self.v_net.layer1(x_norm0))
            h0 = self.v_net.activation_fn(self.v_net.layer2(h0))
            return (h - h0) * scale

        return h * scale


class GVLastLayerFeature(nn.Module):
    """
    Feature map phi(x) so that GV(x) = phi(x)^T w.
    """

    def __init__(self, gv_net: nn.Module):
        super().__init__()
        self.gv = gv_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.gv, "include_time_derivative", False):
            raise NotImplementedError("LP last-layer features currently support include_time_derivative=False only.")

        N, D = x.shape
        W0, b0, W1, b1, _ = self.gv._get_V_params()

        inv_scale, inv_scale_sq = self.gv._get_inv_scales(D, x.device, x.dtype)
        if hasattr(self.gv, "input_offset"):
            offset = self.gv._expand_scale(self.gv.input_offset.to(device=x.device, dtype=x.dtype), D)
            x_norm = (x - offset) * inv_scale
        else:
            x_norm = x * inv_scale

        z0 = F.linear(x_norm, W0, b0)
        h0, d0, q0 = self.gv._sigmoid_derivs(z0)

        z1 = F.linear(h0, W1, b1)
        _, d1, q1 = self.gv._sigmoid_derivs(z1)

        # (N, L, m0)
        S = W1.unsqueeze(0) * d0.unsqueeze(1)
        # (N, L, D)
        sum_over_k_all = torch.matmul(S, W0)

        fx = self.gv._evaluate_f_fast(x)             # (N, D)
        g_diag_sq = self.gv._compute_gg_diag_fast(x) # (N, D)

        scale = self.gv.scale_factor
        inv_scale_row = inv_scale.view(1, 1, D)
        inv_scale_sq_row = inv_scale_sq.view(1, 1, D)

        # Gradient contribution per last-layer channel j.
        grad_terms = scale * d1.unsqueeze(2) * sum_over_k_all * inv_scale_row  # (N, L, D)
        drift_part = (grad_terms * fx.unsqueeze(1)).sum(dim=2)                  # (N, L)

        cross = q1.unsqueeze(2) * sum_over_k_all.square()  # (N, L, D)
        direct = d1.unsqueeze(2) * torch.matmul(
            W1.unsqueeze(0) * q0.unsqueeze(1),  # (N, L, m0)
            W0.square(),                        # (m0, D)
        )  # (N, L, D)
        hdiag_terms = scale * (cross + direct) * inv_scale_sq_row               # (N, L, D)
        diff_part = 0.5 * (hdiag_terms * g_diag_sq.unsqueeze(1)).sum(dim=2)     # (N, L)

        return drift_part + diff_part


def _compute_feature_bounds(
    feature_model: nn.Module,
    cells: TensorPairList,
    input_dim: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(cells)
    if n == 0:
        return (
            torch.empty(0, 0, dtype=torch.float32, device=device),
            torch.empty(0, 0, dtype=torch.float32, device=device),
        )
    lowers, uppers = prepare_cell_bounds(cells, device=device, input_dim=input_dim)
    dummy = torch.zeros(n, input_dim, dtype=torch.float32, device=device)
    lirpa_model = BoundedModule(feature_model, dummy, device=device)
    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=lowers, x_U=uppers)
    bounded_input = BoundedTensor(dummy, ptb)
    lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method="IBP")
    return lb, ub


def _apply_last_layer_solution(v_net: nn.Module, w: np.ndarray, c: float, device: str) -> None:
    with torch.no_grad():
        w_t = torch.tensor(w, dtype=torch.float32, device=device).view(1, -1)
        v_net.output.weight.copy_(w_t)
        if hasattr(v_net, "output_offset"):
            off = torch.tensor(c, dtype=v_net.output_offset.dtype, device=v_net.output_offset.device)
            v_net.output_offset.copy_(off.reshape_as(v_net.output_offset))
        elif v_net.output.bias is not None:
            v_net.output.bias.fill_(float(c))


def _get_current_last_layer_params(v_net: nn.Module) -> Tuple[np.ndarray, float]:
    w = v_net.output.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if hasattr(v_net, "output_offset"):
        c = float(v_net.output_offset.detach().cpu().item())
    elif v_net.output.bias is not None:
        c = float(v_net.output.bias.detach().cpu().item())
    else:
        c = 0.0
    return w, c


def _summarize_current_constraint_violations(
    w: np.ndarray,
    c: float,
    a_goal_lo_np: np.ndarray,
    a_goal_hi_np: np.ndarray,
    a_out_lo_np: np.ndarray,
    a_out_hi_np: np.ndarray,
    a_unsafe_lo_np: np.ndarray,
    a_unsafe_hi_np: np.ndarray,
    a_init_lo_np: np.ndarray,
    a_init_hi_np: np.ndarray,
    p_lo_np: np.ndarray,
    p_hi_np: np.ndarray,
    beta_ra_target: float,
    eps_gen: float,
    include_v_constraints: bool,
    include_generator_constraints: bool,
) -> str:
    w_pos = np.maximum(w, 0.0)
    w_neg = np.maximum(-w, 0.0)

    def _min_affine(a_lo: np.ndarray, a_hi: np.ndarray) -> np.ndarray:
        if a_lo.size == 0:
            return np.zeros((0,), dtype=np.float64)
        return a_lo @ w_pos - a_hi @ w_neg + c

    def _max_affine(a_lo: np.ndarray, a_hi: np.ndarray) -> np.ndarray:
        if a_lo.size == 0:
            return np.zeros((0,), dtype=np.float64)
        return a_hi @ w_pos - a_lo @ w_neg + c

    parts = []
    if include_v_constraints:
        goal_min = _min_affine(a_goal_lo_np, a_goal_hi_np)
        goal_viol = np.maximum(0.0, 0.0 - goal_min)
        parts.append(
            f"goal_lb(max_viol={goal_viol.max(initial=0.0):.3e}, n_viol={int((goal_viol > 0).sum())})"
        )

        outside_min = _min_affine(a_out_lo_np, a_out_hi_np)
        outside_viol = np.maximum(0.0, 0.0 - outside_min)
        parts.append(
            f"outside_lb(max_viol={outside_viol.max(initial=0.0):.3e}, n_viol={int((outside_viol > 0).sum())})"
        )

        unsafe_min = _min_affine(a_unsafe_lo_np, a_unsafe_hi_np)
        unsafe_viol = np.maximum(0.0, float(beta_ra_target) - unsafe_min)
        parts.append(
            f"unsafe_lb(max_viol={unsafe_viol.max(initial=0.0):.3e}, n_viol={int((unsafe_viol > 0).sum())})"
        )

        init_max = _max_affine(a_init_lo_np, a_init_hi_np)
        init_viol = np.maximum(0.0, init_max - 1.0)
        parts.append(
            f"init_ub(max_viol={init_viol.max(initial=0.0):.3e}, n_viol={int((init_viol > 0).sum())})"
        )

    if include_generator_constraints:
        if p_lo_np.size == 0:
            gen_viol = np.zeros((0,), dtype=np.float64)
        else:
            gen_max = p_hi_np @ w_pos - p_lo_np @ w_neg
            gen_viol = np.maximum(0.0, gen_max - (-float(eps_gen)))
        parts.append(
            f"gen_ub(max_viol={gen_viol.max(initial=0.0):.3e}, n_viol={int((gen_viol > 0).sum())})"
        )

    return " | ".join(parts) if len(parts) > 0 else "no_active_constraints"


def solve_last_layer_lp(
    v_net: nn.Module,
    gv_net: nn.Module,
    region_cells: Dict[str, TensorPairList],
    beta_ra_target: float,
    device: str,
    generator_threshold: float = 0.0,
    eps_gen: float = 1e-4,
    include_v_constraints: bool = True,
    include_generator_constraints: bool = True,
    apply_solution: bool = True,
    timeout_sec: float = 30.0,
    verbose: bool = False,
) -> LPSolveResult:
    try:
        from gurobipy import GRB, Model
    except Exception as exc:
        return LPSolveResult(False, f"gurobipy unavailable: {exc}")

    # IMPORTANT: use model snapshots for LP bound propagation so the LP probe
    # cannot mutate/autograd-touch the live training graph tensors.
    v_net_lp = copy.deepcopy(v_net).to(device).eval()
    gv_net_lp = copy.deepcopy(gv_net).to(device).eval()
    input_dim = int(v_net_lp.layer1.weight.shape[1])
    feature_v = VLastLayerFeature(v_net_lp).to(device).eval()
    feature_g = GVLastLayerFeature(gv_net_lp).to(device).eval()

    if not include_v_constraints and not include_generator_constraints:
        return LPSolveResult(False, "No active LP constraints for this epoch.")

    with torch.no_grad():
        if include_v_constraints:
            a_goal_lo, a_goal_hi = _compute_feature_bounds(feature_v, region_cells.get("goal", []), input_dim, device)
            a_out_lo, a_out_hi = _compute_feature_bounds(feature_v, region_cells.get("outside", []), input_dim, device)
            a_unsafe_lo, a_unsafe_hi = _compute_feature_bounds(feature_v, region_cells.get("unsafe", []), input_dim, device)
            a_init_lo, a_init_hi = _compute_feature_bounds(feature_v, region_cells.get("init", []), input_dim, device)
        else:
            empty = torch.empty(0, 0, dtype=torch.float32, device=device)
            a_goal_lo = a_goal_hi = empty
            a_out_lo = a_out_hi = empty
            a_unsafe_lo = a_unsafe_hi = empty
            a_init_lo = a_init_hi = empty

        if include_generator_constraints:
            p_lo, p_hi = _compute_feature_bounds(feature_g, region_cells.get("generator", []), input_dim, device)
        else:
            p_lo = p_hi = torch.empty(0, 0, dtype=torch.float32, device=device)

    L = int(v_net.output.weight.shape[1])
    if L <= 0:
        return LPSolveResult(False, "Invalid output layer width.")

    # Convert to numpy once.
    a_goal_lo_np, a_goal_hi_np = a_goal_lo.detach().cpu().numpy(), a_goal_hi.detach().cpu().numpy()
    a_out_lo_np, a_out_hi_np = a_out_lo.detach().cpu().numpy(), a_out_hi.detach().cpu().numpy()
    a_unsafe_lo_np, a_unsafe_hi_np = a_unsafe_lo.detach().cpu().numpy(), a_unsafe_hi.detach().cpu().numpy()
    a_init_lo_np, a_init_hi_np = a_init_lo.detach().cpu().numpy(), a_init_hi.detach().cpu().numpy()
    p_lo_np, p_hi_np = p_lo.detach().cpu().numpy(), p_hi.detach().cpu().numpy()

    def _build_and_solve():
        m = Model("last_layer_lp")
        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.TimeLimit = float(timeout_sec)

        w_plus = m.addVars(L, lb=0.0, name="w_plus")
        w_minus = m.addVars(L, lb=0.0, name="w_minus")
        c_var = m.addVar(lb=-GRB.INFINITY, name="c")
        t_var = m.addVar(lb=0.0, name="t")

        def _add_lower_bound_constraints(a_lo_np, a_hi_np, target: float, prefix: str):
            # min_a a^T w + c >= target  <=>  -(a_lo*w+ - a_hi*w- + c) <= -target
            for i in range(a_lo_np.shape[0]):
                m.addConstr(
                    -(sum(a_lo_np[i, j] * w_plus[j] for j in range(L))
                      - sum(a_hi_np[i, j] * w_minus[j] for j in range(L))
                      + c_var)
                    <= -float(target),
                    name=f"{prefix}_{i}",
                )

        def _add_upper_bound_constraints(a_lo_np, a_hi_np, target: float, prefix: str):
            # max_a a^T w + c <= target
            for i in range(a_lo_np.shape[0]):
                m.addConstr(
                    sum(a_hi_np[i, j] * w_plus[j] for j in range(L))
                    - sum(a_lo_np[i, j] * w_minus[j] for j in range(L))
                    + c_var
                    <= float(target),
                    name=f"{prefix}_{i}",
                )

        if include_v_constraints:
            _add_lower_bound_constraints(a_goal_lo_np, a_goal_hi_np, 0.0, "goal_lb")
            _add_lower_bound_constraints(a_out_lo_np, a_out_hi_np, 0.0, "outside_lb")
            _add_lower_bound_constraints(a_unsafe_lo_np, a_unsafe_hi_np, float(beta_ra_target), "unsafe_lb")
            _add_upper_bound_constraints(a_init_lo_np, a_init_hi_np, 1.0, "init_ub")

        if include_generator_constraints:
            for i in range(p_lo_np.shape[0]):
                m.addConstr(
                    sum(p_hi_np[i, j] * w_plus[j] for j in range(L))
                    - sum(p_lo_np[i, j] * w_minus[j] for j in range(L))
                    <= -float(eps_gen),
                    name=f"gen_ub_{i}",
                )

        for j in range(L):
            m.addConstr(w_plus[j] - w_minus[j] <= t_var, name=f"inf_pos_{j}")
            m.addConstr(-(w_plus[j] - w_minus[j]) <= t_var, name=f"inf_neg_{j}")

        # Minimize L1 norm of last-layer weights: ||w||_1 = sum_j (w_plus_j + w_minus_j).
        # Keep a tiny t regularizer as a tie-breaker so t_opt remains meaningful.
        m.setObjective(
            sum(w_plus[j] + w_minus[j] for j in range(L)) + 1e-9 * t_var,
            GRB.MINIMIZE,
        )
        m.optimize()

        if m.SolCount <= 0:
            return None, None, None, int(m.Status)

        w_sol = np.array([w_plus[j].X - w_minus[j].X for j in range(L)], dtype=np.float64)
        c_sol = float(c_var.X)
        t_opt = float(t_var.X)
        return w_sol, c_sol, t_opt, int(m.Status)

    w_sol, c_sol, t_opt, status = _build_and_solve()
    if w_sol is not None:
        if apply_solution:
            _apply_last_layer_solution(v_net, w_sol, c_sol, device)
        return LPSolveResult(
            True,
            f"LP feasible solution applied to V last layer (status={status}).",
            t_opt=t_opt,
            status=status,
        )

    w_curr, c_curr = _get_current_last_layer_params(v_net)
    debug_summary = _summarize_current_constraint_violations(
        w=w_curr,
        c=c_curr,
        a_goal_lo_np=a_goal_lo_np,
        a_goal_hi_np=a_goal_hi_np,
        a_out_lo_np=a_out_lo_np,
        a_out_hi_np=a_out_hi_np,
        a_unsafe_lo_np=a_unsafe_lo_np,
        a_unsafe_hi_np=a_unsafe_hi_np,
        a_init_lo_np=a_init_lo_np,
        a_init_hi_np=a_init_hi_np,
        p_lo_np=p_lo_np,
        p_hi_np=p_hi_np,
        beta_ra_target=beta_ra_target,
        eps_gen=eps_gen,
        include_v_constraints=include_v_constraints,
        include_generator_constraints=include_generator_constraints,
    )
    return LPSolveResult(
        False,
        f"LP has no feasible incumbent. status={status}. current_param_violations: {debug_summary}",
        status=status,
        debug_summary=debug_summary,
    )
