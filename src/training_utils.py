"""
Training utilities for loss computation and sampling.

This module provides:
- Loss computation functions
- Sampling from regions and cells
- Constraint evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
import itertools
from typing import List, Tuple, Union, Optional
from types import SimpleNamespace

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
OUTPUT_DIR = ROOT / "gbm_veri"/ "outputs"
import sys
sys.path.insert(0, str(ROOT))
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.discretization import discretize_region
from src.regions import Region
from src.dynamics import Dynamics
from src.phi_module import create_GV
from src.set_values import InvertedPendulumSetDrift, ClosedLoopSetValuedDrift


def sample_from_cells(
    cells: List[Tuple[torch.Tensor, torch.Tensor]],
    n_samples: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample uniformly from a list of cells.

    Args:
        cells: List of (lower, upper) cell bounds
        n_samples: Number of samples to draw
        device: Device to place samples on

    Returns:
        Samples of shape (n_samples, state_dim)
    """
    if len(cells) == 0:
        return torch.empty(0, 2, device=device)

    # Sample cell indices uniformly
    cell_indices = torch.randint(0, len(cells), (n_samples,))

    samples = []
    for idx in cell_indices:
        lower, upper = cells[idx]
        # Sample uniformly within cell
        sample = lower + torch.rand(lower.shape) * (upper - lower)
        samples.append(sample)

    return torch.stack(samples).to(device)

# ============================================================================
# BOUND-BASED LOSS FUNCTIONS
# ============================================================================

def compute_loss_goal_bounds(
    V_lower: torch.Tensor
) -> torch.Tensor:
    # FIXME: Add function signature

    # Want V >= 0.0, so penalize V_lower < 0.0
    loss = torch.relu(0.0 - V_lower).sum()

    # Check if goal is satisfied
    if (loss <= 0.0):
        sat = True
    else:
        sat = False
    return loss, sat

def compute_loss_unsafe_bounds(
    V_lower: torch.Tensor,
    beta_ra: float
) -> torch.Tensor:
    # FIXME: Add function signature

    # Want V >= beta_ra, so penalize V_lower < beta_ra
    loss = F.relu(beta_ra - V_lower).sum()

    # Check if unsafe is satisfied
    if (loss <= 0.0):
        sat = True
    else:
        sat = False
    return loss, sat

def compute_loss_init_bounds(
    V_upper: torch.Tensor
) -> torch.Tensor:
    # FIXME: Add function signature

    # Want V <= 1.0, so penalize V_upper > 1.0
    loss = F.relu(V_upper - 1.0).sum()

    # Check if init is satisfied
    if (loss <= 0.0):
        sat = True
    else:
        sat = False
    return loss, sat

def compute_loss_outside_bounds(
    V_lower: torch.Tensor
) -> torch.Tensor:
    # FIXME: Add function signature

    # Want V >= 0.0, so penalize V_lower < 0.0
    loss = F.relu(0.0 - V_lower).sum()

    # Check if outside is satisfied
    if (loss <= 0.0):
        sat = True
    else:
        sat = False
    return loss, sat

def compute_loss_generator_bounds(
    Phi_upper: torch.Tensor
) -> torch.Tensor:
    # FIXME: Add function signature

    # Want Phi < 0, so penalize Phi_upper > 0
    delta = 1e-4
    loss = F.relu(Phi_upper + delta).sum() # Add delta to encourage extra push

    # Check if generator is satisfied
    if (Phi_upper.max() < 0.0):
        sat = True
    else:
        sat = False
    return loss, sat

def compute_total_loss_bounds(
    beta_ra: float,
    V_goal_lower: torch.Tensor = None,
    V_unsafe_lower: torch.Tensor = None,
    V_init_upper: torch.Tensor = None,
    V_outside_lower: torch.Tensor = None,
    Phi_upper: torch.Tensor = None,
    generator_weight: float = 0.0,
    loss_weights: dict = None,
    device: str = 'cpu',
    compute_V: bool = True,
    compute_GV: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute total training loss from CROWN bounds.

    Args:
        model: V network
        goal_region: Goal region object
        V_goal_lower/upper: Value bounds in goal region
        V_unsafe_lower/upper: Value bounds in unsafe region
        V_init_lower/upper: Value bounds in init region
        V_outside_lower/upper: Value bounds outside goal
        Phi_lower/upper: Generator bounds outside (goal ∪ unsafe)
        beta_s: Separation threshold (float or torch.Tensor)
        beta_ra: Unsafe threshold
        generator_weight: Weight for generator loss
        loss_weights: Optional dictionary of weights for each loss component
        device: Device

    Returns:
        (total_loss, loss_dict) where loss_dict contains individual losses
    """
    if loss_weights is None:
        loss_weights = {
            'goal': 1.0,
            'unsafe': 1.0,
            'init': 1.0,
            'outside': 1.0,
            'boundary': 1.0,
            'generator': 1.0
        }

    # Compute individual losses and satisfactions
    if compute_V:
        loss_unsafe, sat_unsafe = compute_loss_unsafe_bounds(V_unsafe_lower, beta_ra)
        loss_goal, sat_goal = compute_loss_goal_bounds(V_goal_lower)
        loss_init, sat_init = compute_loss_init_bounds(V_init_upper)
        loss_outside, sat_outside = compute_loss_outside_bounds(V_outside_lower)
    if compute_GV:
        loss_generator, sat_generator = compute_loss_generator_bounds(Phi_upper)

    # Combine losses
    total_loss = torch.tensor(0.0, device=device)
    loss_dict = {}
    sat_dict = {}

    if compute_V:
        total_loss += (
            loss_weights['goal'] * loss_goal +
            loss_weights['unsafe'] * loss_unsafe +
            loss_weights['init'] * loss_init +
            loss_weights['outside'] * loss_outside
        )
        loss_dict.update({
            'goal': loss_goal.item(),
            'unsafe': loss_unsafe.item(),
            'init': loss_init.item(),
            'outside': loss_outside.item()
        })
        sat_dict.update({
            'goal': sat_goal,
            'unsafe': sat_unsafe,
            'init': sat_init,
            'outside': sat_outside
        })

    if compute_GV:
        total_loss += generator_weight * loss_generator
        loss_dict['generator'] = loss_generator.item()
        sat_dict['generator'] = sat_generator

    loss_dict['total'] = total_loss.item()

    return total_loss, loss_dict, sat_dict

def evaluate_constraints(
    V_net,
    GV_net,
    region_cells: dict,
    beta_ra: float,
    device: str = 'cpu',
    crown_cache_all=None,
    crown_cache_phi=None,
    input_bounds_all=None,
    cell_counts_V: dict = None,
    input_bounds_gen=None,
    theta_ranges: dict = None,
    theta_grid_splits=None,
    adv_delta: float = 1e-4,
) -> dict:
    """
    Evaluate constraint satisfaction using CROWN bounds.

    Dimension-generic: infers input_dim from provided bounds or region_cells.
    """

    if theta_ranges is not None and theta_grid_splits is not None:
        return evaluate_constraints_theta_grid(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            beta_ra=beta_ra,
            theta_ranges=theta_ranges,
            theta_grid_splits=theta_grid_splits,
            device=device,
            adv_delta=adv_delta,
        )

    V_net.eval()

    # -----------------------------
    # Infer input_dim (D) robustly
    # -----------------------------
    input_dim = None
    if input_bounds_all is not None:
        input_lowers_all, _ = input_bounds_all
        if isinstance(input_lowers_all, torch.Tensor) and input_lowers_all.ndim == 2:
            input_dim = int(input_lowers_all.shape[1])

    if input_dim is None:
        # Try infer from any non-empty cell list
        for name in ['goal', 'unsafe', 'init', 'outside', 'boundary', 'generator']:
            if name in region_cells and len(region_cells[name]) > 0:
                cell0 = region_cells[name][0]          # (lower, upper)
                input_dim = int(cell0[0].numel())       # lower is (D,)
                break

    if input_dim is None:
        raise ValueError("Could not infer input_dim: provide input_bounds_all or non-empty region_cells.")

    with torch.no_grad():
        region_bounds = {}

        if crown_cache_all is not None and input_bounds_all is not None and cell_counts_V is not None:
            # Use pre-computed single cache and split results
            input_lowers_all, input_uppers_all = input_bounds_all
            v_lowers_all, v_uppers_all = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)

            region_order = ['init', 'goal', 'unsafe', 'outside']
            start_idx = 0
            for name in region_order:
                count = cell_counts_V[name]
                if count > 0:
                    end_idx = start_idx + count
                    region_bounds[name] = (v_lowers_all[start_idx:end_idx], v_uppers_all[start_idx:end_idx])
                    start_idx = end_idx
                else:
                    region_bounds[name] = (torch.tensor([], device=device), torch.tensor([], device=device))
        else:
            # Fallback: create separate cache for each region (dimension-generic)
            for name in ['goal', 'unsafe', 'init', 'outside']:
                if len(region_cells[name]) > 0:
                    cache = SymbolicCROWNCache(V_net, len(region_cells[name]), input_dim=input_dim, device=device)
                    input_lowers, input_uppers = prepare_cell_bounds(region_cells[name], device=device, input_dim=input_dim)
                    v_lowers, v_uppers = cache.compute_bounds(input_lowers, input_uppers)
                    region_bounds[name] = (v_lowers, v_uppers)
                else:
                    region_bounds[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        # Goal: bounds only (universal)
        if len(region_bounds['goal'][0]) > 0:
            goal_satisfied = (region_bounds['goal'][0] >= 0).all().item()
            v_goal_min = region_bounds['goal'][0].min().item()
            v_goal_max = region_bounds['goal'][1].max().item()
        else:
            goal_satisfied = True
            v_goal_min = v_goal_max = 0.0

        # Outside: bounds only (universal)
        if len(region_bounds['outside'][0]) > 0:
            outside_satisfied = (region_bounds['outside'][0] >= 0.0).all().item()
            v_outside_min = region_bounds['outside'][0].min().item()
            v_outside_max = region_bounds['outside'][1].max().item()
        else:
            outside_satisfied = True
            v_outside_min = v_outside_max = 0.0

        # Unsafe: bounds only (universal)
        if len(region_bounds['unsafe'][0]) > 0:
            unsafe_satisfied = (region_bounds['unsafe'][0] >= beta_ra).all().item()
            v_unsafe_min = region_bounds['unsafe'][0].min().item()
            v_unsafe_max = region_bounds['unsafe'][1].max().item()
        else:
            unsafe_satisfied = True
            v_unsafe_min = v_unsafe_max = 0.0

        # Init: bounds only (universal)
        if len(region_bounds['init'][0]) > 0:
            init_lower_ok = (region_bounds['init'][0] >= 0.0).all().item()
            init_upper_ok = (region_bounds['init'][1] <= 1.0).all().item()
            init_satisfied = init_lower_ok and init_upper_ok
            v_init_min = region_bounds['init'][0].min().item()
            v_init_max = region_bounds['init'][1].max().item()
        else:
            init_satisfied = True
            v_init_min = v_init_max = 0.0

        # Generator: bounds only (universal)  GV(x) <= 0
        if len(region_cells['generator']) > 0:
            if crown_cache_phi is not None:
                if input_bounds_gen is not None:
                    input_lowers, input_uppers = input_bounds_gen
                else:
                    input_lowers, input_uppers = prepare_cell_bounds(
                        region_cells['generator'], device=device, input_dim=input_dim
                    )
                phi_uppers = crown_cache_phi.compute_bounds(input_lowers, input_uppers)
            else:
                cache_phi = SymbolicCROWNCache_Phi(GV_net, len(region_cells['generator']), input_dim=input_dim, device=device)
                input_lowers, input_uppers = prepare_cell_bounds(
                    region_cells['generator'], device=device, input_dim=input_dim
                )
                phi_uppers = cache_phi.compute_bounds(input_lowers, input_uppers)

            num_failing = (phi_uppers > 0.0).sum().item()
            generator_satisfied = (num_failing == 0)
            phi_min = phi_uppers.min().item()
            phi_max = phi_uppers.max().item()
            phi_mean = phi_uppers.mean().item()
        else:
            generator_satisfied = True
            num_failing = 0
            phi_min = phi_max = phi_mean = 0.0

    results = {
        'goal_satisfied': goal_satisfied,
        'unsafe_satisfied': unsafe_satisfied,
        'init_satisfied': init_satisfied,
        'outside_satisfied': outside_satisfied,
        'generator_satisfied': generator_satisfied,

        'V_goal_min': v_goal_min,
        'V_goal_max': v_goal_max,
        'V_unsafe_min': v_unsafe_min,
        'V_unsafe_max': v_unsafe_max,
        'V_init_min': v_init_min,
        'V_init_max': v_init_max,
        'V_outside_min': v_outside_min,
        'V_outside_max': v_outside_max,
        'Phi_min': phi_min,
        'Phi_max': phi_max,
        'Phi_mean': phi_mean,
        'num_failing_cells': num_failing
    }

    V_net.train()
    return results


def build_theta_grid_cells(theta_ranges: dict, theta_grid_splits, device: str = "cpu"):
    """
    Build axis-aligned theta cells from parameter ranges.

    theta_ranges keys: "g", "L", "b", "m"
    theta_grid_splits: iterable of length 4
    """
    keys = ("g", "L", "b", "m")
    if len(theta_grid_splits) != 4:
        raise ValueError(f"theta_grid_splits must be length 4, got {len(theta_grid_splits)}")

    edges = {}
    for i, k in enumerate(keys):
        lo = float(theta_ranges[k][0])
        hi = float(theta_ranges[k][1])
        n = int(theta_grid_splits[i])
        if n <= 1 or hi <= lo:
            edges[k] = torch.tensor([lo, hi], dtype=torch.float32, device=device)
        else:
            edges[k] = torch.linspace(lo, hi, steps=n + 1, dtype=torch.float32, device=device)

    boxes = []
    ranges = [range(edges[k].numel() - 1) for k in keys]
    for idx in itertools.product(*ranges):
        lo_box, hi_box = {}, {}
        valid = True
        for d, k in enumerate(keys):
            l = float(edges[k][idx[d]].item())
            h = float(edges[k][idx[d] + 1].item())
            if h < l:
                valid = False
                break
            lo_box[k], hi_box[k] = l, h
        if valid:
            boxes.append((lo_box, hi_box))
    return boxes


def evaluate_constraints_theta_grid(
    V_net,
    GV_net,
    region_cells: dict,
    beta_ra: float,
    theta_ranges: dict,
    theta_grid_splits,
    device: str = "cpu",
    adv_delta: float = 1e-4,
) -> dict:
    """
    Evaluate constraints, with generator evaluated over theta-grid cells:
      max_j sum_i ReLU(Phi_upper_i(theta_cell_j) + delta)
    """
    # Keep V-constraint evaluation identical to existing path.
    region_cells_v = dict(region_cells)
    region_cells_v["generator"] = []
    results = evaluate_constraints(
        V_net=V_net,
        GV_net=GV_net,
        region_cells=region_cells_v,
        beta_ra=beta_ra,
        device=device,
    )

    gen_cells = region_cells.get("generator", [])
    if len(gen_cells) == 0:
        results.update({
            "generator_satisfied": True,
            "Phi_min": 0.0,
            "Phi_max": 0.0,
            "Phi_mean": 0.0,
            "num_failing_cells": 0,
            "theta_cell_worst_idx": -1,
            "theta_cell_worst_lo": None,
            "theta_cell_worst_hi": None,
            "theta_cell_worst_obj": 0.0,
            "theta_grid_n_eval": 0,
        })
        return results

    input_dim = int(gen_cells[0][0].numel())
    in_lo_gen, in_hi_gen = prepare_cell_bounds(gen_cells, device=device, input_dim=input_dim)
    theta_boxes = build_theta_grid_cells(theta_ranges, theta_grid_splits, device=device)

    # Rebuild a Phi module per theta cell (constant-theta-in-cell interpretation).
    g_fn = GV_net.dynamics.get_g()
    f_orig = GV_net.dynamics.get_f()
    net_cfg = SimpleNamespace(
        scale_factor=float(GV_net.scale_factor.detach().cpu().item()),
        input_scale=GV_net.input_scale.detach().cpu().tolist(),
    )

    best_idx = -1
    best_obj = float("-inf")
    best_phi_upper = None
    best_lo = None
    best_hi = None

    with torch.no_grad():
        for j, (lo, hi) in enumerate(theta_boxes):
            f_ol = InvertedPendulumSetDrift(
                g_range=(lo["g"], hi["g"]),
                L_range=(lo["L"], hi["L"]),
                b_range=(lo["b"], hi["b"]),
                m_range=(lo["m"], hi["m"]),
            ).to(device)
            if hasattr(f_orig, "controller"):
                f_cl = ClosedLoopSetValuedDrift(f_ol, f_orig.controller).to(device)
            else:
                f_cl = f_ol

            dyn = Dynamics.dynamics(f=f_cl, g=g_fn, state_dim=input_dim)
            GV_theta = create_GV(V_net=V_net, dynamics=dyn, network_config=net_cfg, verify=False).to(device)
            cache_phi = SymbolicCROWNCache_Phi(GV_theta, len(gen_cells), input_dim=input_dim, device=device)
            phi_upper_j = cache_phi.compute_bounds(in_lo_gen, in_hi_gen)
            obj_j = float(F.relu(phi_upper_j + float(adv_delta)).sum().item())

            if obj_j > best_obj:
                best_obj = obj_j
                best_idx = j
                best_phi_upper = phi_upper_j.detach().clone()
                best_lo, best_hi = lo, hi

    if best_phi_upper is None:
        results.update({
            "generator_satisfied": True,
            "Phi_min": 0.0,
            "Phi_max": 0.0,
            "Phi_mean": 0.0,
            "num_failing_cells": 0,
            "theta_cell_worst_idx": -1,
            "theta_cell_worst_lo": None,
            "theta_cell_worst_hi": None,
            "theta_cell_worst_obj": 0.0,
            "theta_grid_n_eval": int(len(theta_boxes)),
        })
        return results

    num_failing = int((best_phi_upper > 0.0).sum().item())
    results["generator_satisfied"] = bool(best_obj <= 0.0)
    results["Phi_min"] = float(best_phi_upper.min().item())
    results["Phi_max"] = float(best_phi_upper.max().item())
    results["Phi_mean"] = float(best_phi_upper.mean().item())
    results["num_failing_cells"] = num_failing
    results["theta_cell_worst_idx"] = int(best_idx)
    results["theta_cell_worst_lo"] = best_lo
    results["theta_cell_worst_hi"] = best_hi
    results["theta_cell_worst_obj"] = float(best_obj)
    results["theta_grid_n_eval"] = int(len(theta_boxes))
    return results


def print_loss_summary(epoch: int, loss_dict: dict, prefix: str = "", compute_V: bool = True, compute_GV: bool = True, elapsed_time: float = None):
    """
    Print formatted loss summary.

    Args:
        epoch: Current epoch
        loss_dict: Dictionary of losses
        prefix: Optional prefix for print
        elapsed_time: Optional elapsed time in seconds
    """
    time_str = f" | Time: {elapsed_time:.1f}s" if elapsed_time is not None else ""

    if compute_V and compute_GV:
        print(f"{prefix}Epoch {epoch:5d} | "
            f"Total: {loss_dict['total']:9.4f} | "
            f"Goal: {loss_dict['goal']:8.4f} | "
            f"Unsafe: {loss_dict['unsafe']:8.4f} | "
            f"Init: {loss_dict['init']:8.4f} | "
            f"Outside: {loss_dict['outside']:8.4f} | "
            f"Gen: {loss_dict['generator']:8.4f}{time_str}")
    elif compute_V and not compute_GV:
        print(f"{prefix}Epoch {epoch:5d} | "
            f"Total: {loss_dict['total']:9.4f} | "
            f"Goal: {loss_dict['goal']:8.4f} | "
            f"Unsafe: {loss_dict['unsafe']:8.4f} | "
            f"Init: {loss_dict['init']:8.4f} | "
            f"Outside: {loss_dict['outside']:8.4f}{time_str}")
    else:
        print(f"{prefix}Epoch {epoch:5d} | "
            f"Gen: {loss_dict['generator']:8.4f}{time_str}")


def print_constraint_summary(results: dict, prefix: str = "", bounds: dict = None, phi_uppers: torch.Tensor = None, region_cells: dict = None, beta_ra: float = 20.0):
    """
    Print formatted constraint satisfaction summary with cell pass/fail counts.

    Args:
        results: Dictionary from evaluate_constraints
        prefix: Optional prefix for print
        bounds: Optional dict with bounds for each region (from training loop)
        phi_uppers: Optional tensor of phi upper bounds (from training loop)
        region_cells: Optional dict with cells for each region
        beta_ra: Unsafe region threshold (default 20.0)
    """
    def status(satisfied):
        return "PASS" if satisfied else "FAIL"

    def cell_stats(region_name, constraint_fn, total_cells):
        """Helper to compute passing/failing cell counts"""
        if total_cells == 0:
            return ""
        passing = constraint_fn.sum().item()
        failing = total_cells - passing
        pct = 100 * failing / total_cells
        return f" | {passing}/{total_cells} pass ({100 - pct:.1f}%)"

    # Goal
    stats = ""
    if bounds and 'goal' in bounds and len(bounds['goal'][0]) > 0:
        stats = cell_stats('goal', bounds['goal'][0] >= 0, len(bounds['goal'][0]))
    print(f"{prefix}Goal:      {status(results['goal_satisfied']):4s}, V bounds: [{results['V_goal_min']:7.3f}, {results['V_goal_max']:7.3f}]{stats}")

    # Unsafe
    stats = ""
    if bounds and 'unsafe' in bounds and len(bounds['unsafe'][0]) > 0:
        stats = cell_stats('unsafe', bounds['unsafe'][0] >= beta_ra, len(bounds['unsafe'][0]))
    print(f"{prefix}Unsafe:    {status(results['unsafe_satisfied']):4s}, V bounds: [{results['V_unsafe_min']:7.3f}, {results['V_unsafe_max']:7.3f}]{stats}")

    # Init
    stats = ""
    if bounds and 'init' in bounds and len(bounds['init'][1]) > 0:
        stats = cell_stats('init', bounds['init'][1] <= 1.0, len(bounds['init'][1]))
    print(f"{prefix}Init:      {status(results['init_satisfied']):4s}, V bounds: [{results['V_init_min']:7.3f}, {results['V_init_max']:7.3f}]{stats}")

    # Outside
    stats = ""
    if bounds and 'outside' in bounds and len(bounds['outside'][0]) > 0:
        stats = cell_stats('outside', bounds['outside'][0] >= 0.0, len(bounds['outside'][0]))
    print(f"{prefix}Outside:   {status(results['outside_satisfied']):4s}, V bounds: [{results['V_outside_min']:7.3f}, {results['V_outside_max']:7.3f}]{stats}")

    # Generator
    stats = ""
    if phi_uppers is not None and len(phi_uppers) > 0:
        stats = cell_stats('generator', phi_uppers < 0.0, len(phi_uppers))
    elif region_cells and 'generator' in region_cells:
        total_gen = len(region_cells['generator'])
        failing = results['num_failing_cells']
        if total_gen > 0:
            passing = total_gen - failing
            pct = 100 * failing / total_gen
            stats = f" | {passing}/{total_gen} pass ({failing} fail, {pct:.1f}%)"
    print(f"{prefix}Generator: {status(results['generator_satisfied']):4s}, GV bounds: [{results['Phi_min']:7.3f}, {results['Phi_max']:7.3f}]{stats}")

from typing import List, Tuple, Optional
import numpy as np
import torch

def refine_failing_cells(
    region_cells: List[Tuple[torch.Tensor, torch.Tensor]],
    failing_mask: torch.Tensor,
    refine_factor: int = 2,
    N_to_refine: int = 100,
    seed: Optional[int] = 0,
    scores: Optional[torch.Tensor] = None,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
    """
    Refine (split) only a subset of failing cells.

    If scores is provided, refine the top-N_to_refine failing cells with the
    largest scores (e.g., phi_uppers). Otherwise, refine a random subset.

    Args:
        region_cells: List of (lower, upper) cell bounds, each (D,)
        failing_mask: Boolean mask indicating which cells failed
        refine_factor: Factor to subdivide cells (2 = split into refine_factor^D subcells)
        N_to_refine: Max number of failing cells to refine
        seed: Optional RNG seed for reproducibility (only used when scores is None)
        scores: Optional 1D tensor aligned with region_cells giving priority (higher = refine first)

    Returns:
        Tuple of (new_cells, num_refined)
    """
    # Make sure mask is 1D on CPU for indexing
    failing_mask = failing_mask.reshape(-1).to(dtype=torch.bool).cpu()

    n_cells = len(region_cells)
    mask_len = min(len(failing_mask), n_cells)

    # indices of failing cells that are eligible (within mask range)
    failing_idxs_t = torch.nonzero(failing_mask[:mask_len], as_tuple=False).reshape(-1)

    # choose subset to refine
    if N_to_refine is None or N_to_refine <= 0 or failing_idxs_t.numel() == 0:
        refine_set = set()
    else:
        k = min(int(N_to_refine), int(failing_idxs_t.numel()))

        if scores is not None:
            # Top-k selection by score among failing cells (largest first)
            scores_cpu = scores.reshape(-1).detach().cpu()
            scores_cpu = scores_cpu[:mask_len]
            failing_scores = scores_cpu[failing_idxs_t]

            # Be safe with NaNs/Infs
            failing_scores = torch.nan_to_num(failing_scores, nan=-float("inf"))

            topk = torch.topk(failing_scores, k=k, largest=True).indices
            refine_idxs = failing_idxs_t[topk].tolist()
            refine_set = set(refine_idxs)
        else:
            # Random subset (original behavior)
            rng = np.random.default_rng(seed)
            refine_idxs = rng.choice(failing_idxs_t.tolist(), size=k, replace=False).tolist()
            refine_set = set(refine_idxs)

    # Pre-allocate: estimate size (upper bound = unrefined + refined*refine_factor^D)
    n_cells = len(region_cells)
    n_refine = len(refine_set)
    if n_refine > 0:
        D = region_cells[0][0].reshape(-1).numel()
        max_new_cells = (n_cells - n_refine) + n_refine * (refine_factor ** D)
    else:
        max_new_cells = n_cells

    new_cells: List[Tuple[torch.Tensor, torch.Tensor]] = [None] * max_new_cells
    insert_idx = 0
    num_refined = 0

    for i, (cell_lower, cell_upper) in enumerate(region_cells):
        if i in refine_set:
            cell_lower = cell_lower.reshape(-1)
            cell_upper = cell_upper.reshape(-1)
            D = int(cell_lower.numel())

            cell_bounds = np.array(
                [[cell_lower[d].item(), cell_upper[d].item()] for d in range(D)],
                dtype=np.float32
            )

            cell_region = Region(cell_bounds)
            refined = discretize_region(cell_region, refine_factor)
            for cell in refined:
                new_cells[insert_idx] = cell
                insert_idx += 1
            num_refined += 1
        else:
            new_cells[insert_idx] = (cell_lower, cell_upper)
            insert_idx += 1

    return new_cells[:insert_idx], num_refined


def merge_passing_neighbor_cells(
    region_cells: List[Tuple[torch.Tensor, torch.Tensor]],
    failing_mask: torch.Tensor,
    max_passes: int = 1,
    max_merges: Optional[int] = None,
    seed: int = 0,
    eps: float = 1e-6,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
    """
    Merge adjacent passing cells (hyper-rectangles) to reduce cell count.

    Two cells can merge if:
      - both are passing (not failing)
      - same bounds in all dims except one dim d
      - they touch along dim d (upper_d == other.lower_d)
      - same size along dim d (to keep grid alignment)
      - same size in all other dims implicitly enforced by identical bounds

    Args:
        region_cells: List of (lower, upper), each (D,)
        failing_mask: Boolean mask (len == #cells) where True means failing
        max_passes: How many coarsening sweeps to try (each pass attempts merges along all dims)
        max_merges: Optional cap on how many pair-merges to do (for speed)
        seed: RNG seed if max_merges is used (randomly subsamples merges)
        eps: Quantization step for robust float comparisons (bounds -> ints via round(val/eps))

    Returns:
        (new_cells, num_pair_merges)
    """
    failing_mask = failing_mask.reshape(-1).to(dtype=torch.bool).cpu()
    n = len(region_cells)
    if len(failing_mask) != n:
        # safer: only merge when mask matches cell count
        return region_cells, 0

    # passing indices are eligible to merge
    passing = (~failing_mask).tolist()

    def quantize(v: torch.Tensor) -> Tuple[int, ...]:
        v = v.detach().cpu().reshape(-1).to(torch.float64)
        q = torch.round(v / eps).to(torch.int64)
        return tuple(q.tolist())

    rng = np.random.default_rng(seed)
    num_merges_total = 0
    cells = region_cells

    for _ in range(max_passes):
        n = len(cells)
        if n == 0:
            break

        # assume all cells have same D
        D = int(cells[0][0].numel())

        # precompute quantized bounds for robust hashing
        lo_q = [quantize(lo) for (lo, _) in cells]
        hi_q = [quantize(hi) for (_, hi) in cells]

        merged_flag = [False] * n
        new_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []

        merges_this_pass = 0

        for d in range(D):
            # build lookup: (signature_except_d, lower_d, size_d) -> index
            lookup = {}
            for i in range(n):
                if merged_flag[i] or (not passing[i]):
                    continue
                size_d = hi_q[i][d] - lo_q[i][d]
                sig = tuple((lo_q[i][k], hi_q[i][k]) for k in range(D) if k != d)
                key = (sig, lo_q[i][d], size_d)
                lookup[key] = i

            # attempt merges i -> neighbor on +d side
            # optionally cap merges for speed
            indices = list(range(n))
            if max_merges is not None:
                rng.shuffle(indices)

            for i in indices:
                if merged_flag[i] or (not passing[i]):
                    continue

                size_d = hi_q[i][d] - lo_q[i][d]
                sig = tuple((lo_q[i][k], hi_q[i][k]) for k in range(D) if k != d)
                neighbor_key = (sig, hi_q[i][d], size_d)  # neighbor starts where i ends
                j = lookup.get(neighbor_key, None)
                if j is None or j == i or merged_flag[j] or (not passing[j]):
                    continue

                # merge i and j
                lo_i, hi_i = cells[i]
                lo_j, hi_j = cells[j]
                merged_lo = torch.minimum(lo_i, lo_j)
                merged_hi = torch.maximum(hi_i, hi_j)

                merged_flag[i] = True
                merged_flag[j] = True
                new_cells.append((merged_lo, merged_hi))

                merges_this_pass += 1
                num_merges_total += 1

                if max_merges is not None and num_merges_total >= int(max_merges):
                    break

            if max_merges is not None and num_merges_total >= int(max_merges):
                break

        # append anything not merged
        for i in range(n):
            if not merged_flag[i]:
                new_cells.append(cells[i])

        if len(new_cells) == len(cells):
            # no progress
            break
        cells = new_cells

    return cells, num_merges_total
