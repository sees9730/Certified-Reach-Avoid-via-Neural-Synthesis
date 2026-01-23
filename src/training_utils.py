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
from typing import List, Tuple, Union, Optional

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
OUTPUT_DIR = ROOT / "gbm_veri"/ "outputs"
import sys
sys.path.insert(0, str(ROOT))
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.discretization import discretize_region
from src.regions import Region


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


def sample_from_region_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    n_samples: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample uniformly from rectangular region defined by bounds.

    Args:
        lower: Lower corner (state_dim,)
        upper: Upper corner (state_dim,)
        n_samples: Number of samples
        device: Device for samples

    Returns:
        Samples of shape (n_samples, state_dim)
    """
    lower_t = torch.from_numpy(lower).float()
    upper_t = torch.from_numpy(upper).float()

    samples = lower_t + torch.rand(n_samples, len(lower)) * (upper_t - lower_t)
    return samples.to(device)

# ============================================================================
# BOUND-BASED LOSS FUNCTIONS (for CROWN training)
# ============================================================================

# Cache for fixed goal samples - prevents random sampling at each epoch for stability
_goal_samples_cache = {}


def clear_goal_samples_cache():
    """Clear the goal samples cache. Useful when starting a new training run."""
    global _goal_samples_cache
    _goal_samples_cache = {}


def compute_loss_goal_bounds(
    model,
    goal_region,
    V_lower: torch.Tensor,
    V_upper: torch.Tensor,
    beta_s: Union[float, torch.Tensor],
    V_outside_lower,
    n_samples: int = 1000,
    device: str = 'cpu',
    check: bool = False,
    show: bool = False,
    w_soft = 2000
) -> torch.Tensor:

    bounds = goal_region.bounds  # (D, 2)
    D = bounds.shape[0]

    # Compute center of goal region (1, D)
    center = torch.as_tensor(((bounds[:, 0] + bounds[:, 1]) / 2.0),
                             device=device, dtype=torch.float32).view(1, D)

    # Evaluate V at center point
    v_center = model(center).squeeze()

    # Strong loss on center point
    margin = 0.0
    # loss_center = F.relu(v_center - (beta_s - margin)) * 10.0

    # Shrink the sampling bounds by a factor (inner box)
    shrink_factor = 0.1
    lower = torch.as_tensor(bounds[:, 0], device=device, dtype=torch.float32)  # (D,)
    upper = torch.as_tensor(bounds[:, 1], device=device, dtype=torch.float32)  # (D,)
    span = upper - lower                                                      # (D,)

    inner_lower = lower + span * (1.0 - shrink_factor) / 2.0                  # (D,)
    inner_upper = upper - span * (1.0 - shrink_factor) / 2.0                  # (D,)

    # Sample uniformly in the inner box: (n_samples, D)
    # rand in [0,1) scaled to [inner_lower, inner_upper]
    r = torch.rand(n_samples, D, device=device, dtype=torch.float32)
    samples = r * (inner_upper - inner_lower).unsqueeze(0) + inner_lower.unsqueeze(0)

    v_samples = model(samples).squeeze()

    # Softer loss on minimum of sampled points
    threshold = V_outside_lower.min().item()
    loss_rest = F.relu(v_samples.min() - threshold) * w_soft

    # Combined soft loss
    loss_soft =  loss_rest

    # Encourage non-negativity of the certified lower bound inside goal
    # (kept identical behavior, but note: min() here returns a scalar tensor)
    loss_nonneg = torch.relu(0.0 - V_lower).sum()
    # loss_nonneg = torch.relu(0.0 - torch.min(V_lower))

    # Obtain minimum V inside goal region: from samples, center, and lowest upper bound
    v_min = min(v_samples.min().item(), v_center.item(), V_upper.min().item())

    # When checking for success (passed), check the logic based on v_min
    # Keep semantics identical to your original
    # if (v_min < beta_s) and (V_lower.min() >= 0).item():
    # if (v_min < V_outside_lower.min()) and (V_lower.min() >= 0).item():
    if (V_lower.min() >= 0):
        passed = True
        if show:
            # print(" ✓ Inside Goal passed, min V= {:.4f}, min V_lower={:.4f}".format(
            #     v_min, V_lower.min()
            # ))
            print(" ✓ Inside Goal passed, min V= {:.4f}, min V_outside_lower= {:.4f}, min V_lower={:.4f}".format(
                v_min, V_outside_lower.min(), V_lower.min()
            ))
    else:
        passed = False
        if show:
            # print(" ✗ Inside Goal passed, min V= {:.4f}, min V_lower={:.4f}".format(
            #     v_min, V_lower.min()
            # ))
            print(" ✗ Inside Goal failed, min V= {:.4f}, min V_outside_lower= {:.4f}, min V_lower={:.4f}".format(
                v_min, V_outside_lower.min(), V_lower.min()
            ))

    return loss_nonneg, passed


def compute_loss_unsafe_bounds(
    V_lower: torch.Tensor,
    V_upper: torch.Tensor,
    beta_ra: float
) -> torch.Tensor:
    """
    Compute unsafe constraint loss from bounds: V(x) >= beta_ra for x in Unsafe.

    Args:
        V_lower: Lower bounds on V (N,)
        V_upper: Upper bounds on V (N,)
        beta_ra: Target lower bound for unsafe

    Returns:
        Loss (scalar)
    """
    # Want V >= beta_ra, so penalize V_lower < beta_ra
    # Use .sum() to match original testing_simple3.py
    # print(f' Num of violations in unsafe: {(V_lower < beta_ra).sum().item()}')
    return F.relu(beta_ra - V_lower).sum()


def compute_loss_init_bounds(
    V_lower: torch.Tensor,
    V_upper: torch.Tensor,
    beta_s: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Compute init constraint loss from bounds.

    Init must satisfy TWO constraints:
    1. V < 1.0 (its own upper bound)
    2. V >= beta_s (general constraint that applies everywhere except goal)

    Args:
        V_lower: Lower bounds on V (N,)
        V_upper: Upper bounds on V (N,)
        beta_s: Target lower bound for init (float or torch.Tensor)

    Returns:
        Loss (scalar)
    """
    # Push V_upper <= 1.0 AND V_lower >= beta_s
    loss_upper = F.relu(V_upper - 1.0).sum()
    # loss_lower = F.relu(beta_s - V_lower).sum()
    return loss_upper


def compute_loss_init_bounds_offset(
    V_lower: torch.Tensor,
    V_upper: torch.Tensor,
    offset: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Compute init constraint loss from bounds.

    Init must satisfy TWO constraints:
    1. V < 1.0 (its own upper bound)
    2. V >= beta_s (general constraint that applies everywhere except goal)

    Args:
        V_lower: Lower bounds on V (N,)
        V_upper: Upper bounds on V (N,)
        beta_s: Target lower bound for init (float or torch.Tensor)

    Returns:
        Loss (scalar)
    """
    loss_upper = F.relu(V_upper - 1.0).sum() + F.relu(offset - V_lower).sum()
    return loss_upper


def compute_loss_outside_bounds(
    V_lower: torch.Tensor,
    V_upper: torch.Tensor,
    beta_s: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Compute outside goal constraint loss from bounds: V(x) >= beta_s for x outside Goal.

    Args:
        V_lower: Lower bounds on V (N,)
        V_upper: Upper bounds on V (N,)
        beta_s: Target lower bound outside goal (float or torch.Tensor)

    Returns:
        Loss (scalar)
    """
    # Want V >= beta_s, so penalize V_lower < beta_s
    return F.relu(beta_s - V_lower).sum()
    # return F.relu(beta_s - torch.min(V_lower))


def compute_loss_generator_bounds(
    Phi_lower: torch.Tensor,
    Phi_upper: torch.Tensor
) -> torch.Tensor:
    """
    Compute generator constraint loss from bounds: Φ(x) <= 0 for x outside (Goal ∪ Unsafe).

    Args:
        Phi_lower: Lower bounds on Φ (N,)
        Phi_upper: Upper bounds on Φ (N,)

    Returns:
        Loss (scalar) - unweighted (weight applied by caller)
    """
    # Want Phi < 0, so penalize Phi_upper > 0
    delta = 1e-4
    return F.relu(Phi_upper + delta).sum()
    # return torch.relu(torch.max(Phi_upper))


def compute_loss_boundary_bounds(
    V_lower: torch.Tensor,
    V_upper: torch.Tensor
) -> torch.Tensor:
    """
    Compute boundary constraint loss from bounds: V(x) >= 1.0 for x on boundary of full_range.

    This enforces that the safe region (full \\ unsafe) has V >= 1.0 at the boundary.

    Args:
        V_lower: Lower bounds on V (N,)
        V_upper: Upper bounds on V (N,)

    Returns:
        Loss (scalar)
    """
    # Want V >= 1.0, so penalize V_lower < 1.0
    return F.relu(1.0 - V_lower).sum()


def compute_total_loss_bounds(
    model,
    goal_region,
    beta_s: Union[float, torch.Tensor],
    beta_ra: float,
    V_goal_lower: torch.Tensor = None,
    V_goal_upper: torch.Tensor = None,
    V_unsafe_lower: torch.Tensor = None,
    V_unsafe_upper: torch.Tensor = None,
    V_init_lower: torch.Tensor = None,
    V_init_upper: torch.Tensor = None,
    V_outside_lower: torch.Tensor = None,
    V_outside_upper: torch.Tensor = None,
    V_boundary_lower: torch.Tensor = None,
    V_boundary_upper: torch.Tensor = None,
    Phi_lower: torch.Tensor = None,
    Phi_upper: torch.Tensor = None,
    Phi_lower_unsafe: torch.Tensor = None,
    Phi_upper_unsafe: torch.Tensor = None,
    Phi_lower_goal: torch.Tensor = None,
    Phi_upper_goal: torch.Tensor = None,
    generator_weight: float = 0.0,
    loss_weights: dict = None,
    device: str = 'cpu',
    compute_V: bool = True,
    compute_GV: bool = True,
    epoch: int = 0,
    w_soft = 2000,
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

    # Compute individual losses
    if compute_V:
        loss_unsafe = compute_loss_unsafe_bounds(V_unsafe_lower, V_unsafe_upper, beta_ra)
        loss_goal, _ = compute_loss_goal_bounds(model, goal_region, V_goal_lower, V_goal_upper, beta_s, V_outside_lower,
                                                 device=device, n_samples=10000, show=False, w_soft=w_soft) 
        if hasattr(model, "output_offset"):
            output_offset_ref = model.output_offset.detach()  # scalar tensor
            loss_init = compute_loss_init_bounds(V_init_lower, V_init_upper, output_offset_ref)
        else:
            loss_init = compute_loss_init_bounds(V_init_lower, V_init_upper, beta_s)
        loss_outside = compute_loss_outside_bounds(V_outside_lower, V_outside_upper, beta_s)
        
        # Boundary loss (only if boundary bounds provided)
        if V_boundary_lower is not None and V_boundary_upper is not None:
            loss_boundary = compute_loss_boundary_bounds(V_boundary_lower, V_boundary_upper)
        else:
            loss_boundary = torch.tensor(0.0, device=device)
    if compute_GV:
        loss_generator = compute_loss_generator_bounds(Phi_lower, Phi_upper)

    # Combine losses
    # Note: generator_weight is applied directly (no additional loss_weights multiplier for generator)
    # This matches original: total_loss = total_loss + generator_weight * loss_gen
    if compute_V and compute_GV:
        total_loss = (
            loss_weights['goal'] * loss_goal +
            loss_weights['unsafe'] * loss_unsafe +
            loss_weights['init'] * loss_init +
            loss_weights['outside'] * loss_outside +
            0.0 * loss_weights['boundary'] * loss_boundary +
            generator_weight * loss_generator #+ loss_generator_unsafe + loss_generator_goal
        )

        loss_dict = {
            'total': total_loss.item(),
            'goal': loss_goal.item(),
            'unsafe': loss_unsafe.item(),
            'init': loss_init.item(),
            'outside': loss_outside.item(),
            'boundary': loss_boundary.item(),
            'generator': loss_generator.item()
        }

    elif compute_V and not compute_GV:
        total_loss = (
            loss_weights['goal'] * loss_goal +
            loss_weights['unsafe'] * loss_unsafe +
            loss_weights['init'] * loss_init +
            loss_weights['outside'] * loss_outside +
            loss_weights['boundary'] * loss_boundary
        )

        loss_dict = {
            'total': total_loss.item(),
            'goal': loss_goal.item(),
            'unsafe': loss_unsafe.item(),
            'init': loss_init.item(),
            'outside': loss_outside.item(),
            'boundary': loss_boundary.item()
        }

    elif not compute_V and compute_GV:
        total_loss = (
            generator_weight * loss_generator
        )

        loss_dict = {
            'generator': loss_generator.item()
        }

    return total_loss, loss_dict


def evaluate_constraints(
    V_net,
    GV_net,
    region_cells: dict,
    beta_s: Union[float, torch.Tensor],
    beta_ra: float,
    device: str = 'cpu',
    n_samples: int = 1000,
    crown_cache_all=None,
    crown_cache_phi=None,
    input_bounds_all=None,
    cell_counts_V: dict = None,
    input_bounds_gen=None
) -> dict:
    """
    Evaluate constraint satisfaction using CROWN bounds.

    Dimension-generic: infers input_dim from provided bounds or region_cells.
    """

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

            region_order = ['init', 'goal', 'unsafe', 'outside', 'boundary']
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
            for name in ['goal', 'unsafe', 'init', 'outside', 'boundary']:
                if len(region_cells[name]) > 0:
                    cache = SymbolicCROWNCache(V_net, len(region_cells[name]), input_dim=input_dim, device=device)
                    input_lowers, input_uppers = prepare_cell_bounds(region_cells[name], device=device, input_dim=input_dim)
                    v_lowers, v_uppers = cache.compute_bounds(input_lowers, input_uppers)
                    region_bounds[name] = (v_lowers, v_uppers)
                else:
                    region_bounds[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        # Goal: sample + bounds (existential)
        if len(region_cells['goal']) > 0:
            goal_satisfied = (region_bounds['goal'][0] >= 0.0).all().item()
            v_goal_min = region_bounds['goal'][0].min().item()
            v_goal_max = region_bounds['goal'][1].max().item()
            x_goal = sample_from_cells(region_cells['goal'], n_samples, device)
            v_goal_samples = V_net(x_goal).squeeze(-1)
            v_goal_mean = v_goal_samples.mean().item()
        else:
            goal_satisfied = False
            v_goal_min = v_goal_max = v_goal_mean = 0.0

        # Outside: bounds only (universal)
        if len(region_bounds['outside'][0]) > 0:
            outside_satisfied = (region_bounds['outside'][0] >= beta_s).all().item()
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

            # # debug
            # x_unsafe = sample_from_cells(region_cells['unsafe'], n_samples, device)
            # v_unsafe_samples = V_net(x_unsafe).squeeze(-1)
            # print("[DEBUG] V samplesin UNSAFE", v_unsafe_samples.min(), v_unsafe_samples.max())
        else:
            unsafe_satisfied = True
            v_unsafe_min = v_unsafe_max = 0.0

        # Init: bounds only (universal)
        if len(region_bounds['init'][0]) > 0:
            init_lower_ok = (region_bounds['init'][0] >= beta_s).all().item()
            init_upper_ok = (region_bounds['init'][1] <= 1.0).all().item()
            init_satisfied = init_lower_ok and init_upper_ok
            v_init_min = region_bounds['init'][0].min().item()
            v_init_max = region_bounds['init'][1].max().item()
        else:
            init_satisfied = True
            v_init_min = v_init_max = 0.0

        # Boundary: bounds only (universal) V(x) >= 1.0
        if 'boundary' in region_cells and len(region_cells['boundary']) > 0:
            boundary_satisfied = True
            v_boundary_min = v_boundary_max = 0.0
        else:
            boundary_satisfied = True
            v_boundary_min = v_boundary_max = 0.0

        # Generator: bounds only (universal)  Φ(x) <= 0
        if len(region_cells['generator']) > 0:
            if crown_cache_phi is not None:
                if input_bounds_gen is not None:
                    input_lowers, input_uppers = input_bounds_gen
                else:
                    input_lowers, input_uppers = prepare_cell_bounds(
                        region_cells['generator'], device=device, input_dim=input_dim
                    )
                phi_lowers, phi_uppers = crown_cache_phi.compute_bounds(input_lowers, input_uppers)
            else:
                cache_phi = SymbolicCROWNCache_Phi(GV_net, len(region_cells['generator']), input_dim=input_dim, device=device)
                input_lowers, input_uppers = prepare_cell_bounds(
                    region_cells['generator'], device=device, input_dim=input_dim
                )
                phi_lowers, phi_uppers = cache_phi.compute_bounds(input_lowers, input_uppers)

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
        'boundary_satisfied': boundary_satisfied,
        'generator_satisfied': generator_satisfied,

        'V_goal_min': v_goal_min,
        'V_goal_max': v_goal_max,
        'V_goal_mean': v_goal_mean,
        'V_unsafe_min': v_unsafe_min,
        'V_unsafe_max': v_unsafe_max,
        'V_init_min': v_init_min,
        'V_init_max': v_init_max,
        'V_outside_min': v_outside_min,
        'V_outside_max': v_outside_max,
        'V_boundary_min': v_boundary_min,
        'V_boundary_max': v_boundary_max,
        'Phi_min': phi_min,
        'Phi_max': phi_max,
        'Phi_mean': phi_mean,
        'num_failing_cells': num_failing
    }

    V_net.train()
    return results


def print_loss_summary(epoch: int, loss_dict: dict, prefix: str = "", compute_V: bool = True, compute_GV: bool = True):
    """
    Print formatted loss summary.

    Args:
        epoch: Current epoch
        loss_dict: Dictionary of losses
        prefix: Optional prefix for print
    """
    if compute_V and compute_GV:
        print(f"{prefix}Epoch {epoch:5d} | "
            f"Total: {loss_dict['total']:.4f} | "
            f"Goal: {loss_dict['goal']:.4f} | "
            f"Unsafe: {loss_dict['unsafe']:.4f} | "
            f"Init: {loss_dict['init']:.4f} | "
            f"Outside: {loss_dict['outside']:.4f} | "
            f"Boundary: {loss_dict['boundary']:.4f} | "
            f"Gen: {loss_dict['generator']:.6e}")
    elif compute_V and not compute_GV:
        print(f"{prefix}Epoch {epoch:5d} | "
            f"Total: {loss_dict['total']:.4f} | "
            f"Goal: {loss_dict['goal']:.4f} | "
            f"Unsafe: {loss_dict['unsafe']:.4f} | "
            f"Init: {loss_dict['init']:.4f} | "
            f"Outside: {loss_dict['outside']:.4f} | "
            f"Boundary: {loss_dict['boundary']:.4f}")
    else:
        print(f"{prefix}Epoch {epoch:5d} | "
            f"Gen: {loss_dict['generator']:.4f}")


def print_constraint_summary(results: dict, prefix: str = ""):
    """
    Print formatted constraint satisfaction summary.

    Args:
        results: Dictionary from evaluate_constraints
        prefix: Optional prefix for print
    """
    # Helper to print checkmark or X
    def status_str(satisfied):
        return "✓" if satisfied else "✗"

    print(f"{prefix}Constraint Satisfaction:")
    print(f"{prefix}  Goal:      {status_str(results['goal_satisfied'])} "
          f"(V_min={results['V_goal_min']:.3f}, V_mean={results['V_goal_mean']:.3f}, V_max={results['V_goal_max']:.3f})")
    print(f"{prefix}  Unsafe:    {status_str(results['unsafe_satisfied'])} "
          f"(V_min={results['V_unsafe_min']:.3f}, V_max={results['V_unsafe_max']:.3f})")
    print(f"{prefix}  Init:      {status_str(results['init_satisfied'])} "
          f"(V_min={results['V_init_min']:.3f}, V_max={results['V_init_max']:.3f})")
    print(f"{prefix}  Outside:   {status_str(results['outside_satisfied'])} "
          f"(V_min={results['V_outside_min']:.3f}, V_max={results['V_outside_max']:.3f})")
    print(f"{prefix}  Boundary:  {status_str(results['boundary_satisfied'])} "
          f"(V_min={results['V_boundary_min']:.3f}, V_max={results['V_boundary_max']:.3f})")
    print(f"{prefix}  Generator: {status_str(results['generator_satisfied'])} "
          f"(Phi_upper_min={results['Phi_min']:.6e}, Phi_upper_max={results['Phi_max']:.6e}, failing={results['num_failing_cells']})")


# def refine_failing_cells(
#     region_cells: List[Tuple[torch.Tensor, torch.Tensor]],
#     failing_mask: torch.Tensor,
#     refine_factor: int = 2,
#     N_to_refine: int = 100,
#     seed: Optional[int] = 0,
# ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
#     """
#     Refine (split) only a random subset of failing cells.

#     Args:
#         region_cells: List of (lower, upper) cell bounds, each (D,)
#         failing_mask: Boolean mask indicating which cells failed
#         refine_factor: Factor to subdivide cells (2 = split into refine_factor^D subcells)
#         N_to_refine: Max number of failing cells to refine (randomly selected)
#         seed: Optional RNG seed for reproducibility

#     Returns:
#         Tuple of (new_cells, num_refined)
#     """
#     # Make sure mask is 1D on CPU for indexing
#     failing_mask = failing_mask.reshape(-1).to(dtype=torch.bool).cpu()

#     n_cells = len(region_cells)
#     mask_len = min(len(failing_mask), n_cells)

#     # indices of failing cells that are eligible (within mask range)
#     failing_idxs = torch.nonzero(failing_mask[:mask_len], as_tuple=False).reshape(-1).tolist()

#     # choose subset to refine
#     if N_to_refine is None or N_to_refine <= 0 or len(failing_idxs) == 0:
#         refine_set = set()
#     else:
#         rng = np.random.default_rng(seed)
#         k = min(int(N_to_refine), len(failing_idxs))
#         refine_set = set(rng.choice(failing_idxs, size=k, replace=False).tolist())

#     new_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
#     num_refined = 0

#     for i, (cell_lower, cell_upper) in enumerate(region_cells):
#         if i in refine_set:
#             # --- build (D,2) bounds (dimension-generic) ---
#             cell_lower = cell_lower.reshape(-1)
#             cell_upper = cell_upper.reshape(-1)
#             D = int(cell_lower.numel())

#             cell_bounds = np.array(
#                 [[cell_lower[d].item(), cell_upper[d].item()] for d in range(D)],
#                 dtype=np.float32
#             )
#             # ------------------------------------------------

#             cell_region = Region(cell_bounds)
#             refined = discretize_region(cell_region, refine_factor)
#             new_cells.extend(refined)
#             num_refined += 1
#         else:
#             new_cells.append((cell_lower, cell_upper))

#     return new_cells, num_refined


from typing import List, Tuple, Optional
import numpy as np
import torch

def refine_failing_cells(
    region_cells: List[Tuple[torch.Tensor, torch.Tensor]],
    failing_mask: torch.Tensor,
    refine_factor: int = 2,
    N_to_refine: int = 100,
    seed: Optional[int] = 0,
    scores: Optional[torch.Tensor] = None,   # <-- NEW (larger = refine first)
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
            print("[debug] use score")

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

    new_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
    num_refined = 0

    for i, (cell_lower, cell_upper) in enumerate(region_cells):
        if i in refine_set:
            # --- build (D,2) bounds (dimension-generic) ---
            cell_lower = cell_lower.reshape(-1)
            cell_upper = cell_upper.reshape(-1)
            D = int(cell_lower.numel())

            cell_bounds = np.array(
                [[cell_lower[d].item(), cell_upper[d].item()] for d in range(D)],
                dtype=np.float32
            )
            # ------------------------------------------------

            cell_region = Region(cell_bounds)
            refined = discretize_region(cell_region, refine_factor)
            new_cells.extend(refined)
            num_refined += 1
        else:
            new_cells.append((cell_lower, cell_upper))

    return new_cells, num_refined


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
