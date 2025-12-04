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
from typing import List, Tuple, Union


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
    n_samples: int = 1000,
    device: str = 'cpu',
    check: bool = False,
    show: bool = False
) -> torch.Tensor:
    
    bounds = goal_region.bounds

    # Compute center of goal region
    center_x1 = (bounds[0, 0] + bounds[0, 1]) / 2.0
    center_x2 = (bounds[1, 0] + bounds[1, 1]) / 2.0
    center = torch.tensor([[center_x1, center_x2]], device=device, dtype=torch.float32)

    # Evaluate V at center point
    v_center = model(center).squeeze()

    # Strong loss on center point
    margin = 0.3
    loss_center = F.relu(v_center - (beta_s - margin)) * 10.0 

    # Shrink the sampling bounds by a factor
    # This creates a "safe buffer" where V is allowed to transition from >= beta_s to < beta_s
    shrink_factor = 0.3
    
    span_x1 = bounds[0, 1] - bounds[0, 0]
    span_x2 = bounds[1, 1] - bounds[1, 0]
    
    inner_lower_x1 = bounds[0, 0] + span_x1 * (1 - shrink_factor) / 2
    inner_upper_x1 = bounds[0, 1] - span_x1 * (1 - shrink_factor) / 2
    inner_lower_x2 = bounds[1, 0] + span_x2 * (1 - shrink_factor) / 2
    inner_upper_x2 = bounds[1, 1] - span_x2 * (1 - shrink_factor) / 2

    x1_samples = torch.rand(n_samples, device=device) * (inner_upper_x1 - inner_lower_x1) + inner_lower_x1
    x2_samples = torch.rand(n_samples, device=device) * (inner_upper_x2 - inner_lower_x2) + inner_lower_x2
    samples = torch.stack([x1_samples, x2_samples], dim=1)
    v_samples = model(samples).squeeze()

    # Softer loss on minimum of sampled points
    loss_rest = F.relu(v_samples.min() - (beta_s - 0.05))

    # Combined soft loss
    loss_soft = loss_center + loss_rest

    loss_nonneg = torch.relu(-V_lower).min()
    
    # When checking for success (passed), still check the logic based on the CENTER or MIN sample, not the edges.
    if ((v_samples.min() < beta_s).item() or v_center < beta_s) and (V_lower.min() >= 0).item():
        passed = True
        if show:
            print(f"  ✓ Goal sample passed: min V = {min(v_samples.min().item(), v_center.item()):.4f}, min V_lower = {V_lower.min().item():.4f}")
    else:
        passed = False
        if show:
            print(f"  ✗ Goal sample failed: min V = {min(v_samples.min().item(), v_center.item()):.4f}, min V_lower = {V_lower.min().item():.4f}")
        
    return loss_soft + loss_nonneg, passed

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
    loss_lower = F.relu(beta_s - V_lower).sum()
    return loss_upper + loss_lower


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
    # Want Phi <= 0, so penalize Phi_upper > 0
    return F.relu(Phi_upper).sum()

def compute_loss_generator_bounds_unsafe(
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
    # Want Phi <= 0, so penalize Phi_upper > 0
    return F.relu(Phi_upper + 1000).sum()

def compute_loss_generator_bounds_goal(
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
    # Want Phi <= 0, so penalize Phi_upper > 0
    return F.relu(Phi_upper).sum()


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
    epoch: int = 0
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
            'generator': 1.0
        }

    # Compute individual losses
    if compute_V:
        loss_unsafe = compute_loss_unsafe_bounds(V_unsafe_lower, V_unsafe_upper, beta_ra)
        loss_goal, _ = compute_loss_goal_bounds(model, goal_region, V_goal_lower, V_goal_upper, beta_s, device=device, n_samples=10000, show=False)
        loss_init = compute_loss_init_bounds(V_init_lower, V_init_upper, beta_s)
        loss_outside = compute_loss_outside_bounds(V_outside_lower, V_outside_upper, beta_s)
    if compute_GV:
        loss_generator = compute_loss_generator_bounds(Phi_lower, Phi_upper)
        # loss_generator_unsafe = compute_loss_generator_bounds_unsafe(Phi_lower_unsafe, Phi_upper_unsafe)
        # loss_generator_goal = compute_loss_generator_bounds_goal(Phi_lower_goal, Phi_upper_goal)

    # Combine losses
    # Note: generator_weight is applied directly (no additional loss_weights multiplier for generator)
    # This matches original: total_loss = total_loss + generator_weight * loss_gen
    if compute_V and compute_GV:
        total_loss = (
            loss_weights['goal'] * loss_goal +
            loss_weights['unsafe'] * loss_unsafe +
            loss_weights['init'] * loss_init +
            loss_weights['outside'] * loss_outside +
            generator_weight * loss_generator #+ loss_generator_unsafe + loss_generator_goal
        )

        loss_dict = {
            'total': total_loss.item(),
            'goal': loss_goal.item(),
            'unsafe': loss_unsafe.item(),
            'init': loss_init.item(),
            'outside': loss_outside.item(),
            'generator': loss_generator.item()
        }

    elif compute_V and not compute_GV:
        total_loss = (
            loss_weights['goal'] * loss_goal +
            loss_weights['unsafe'] * loss_unsafe +
            loss_weights['init'] * loss_init +
            loss_weights['outside'] * loss_outside
        )

        loss_dict = {
            'total': total_loss.item(),
            'goal': loss_goal.item(),
            'unsafe': loss_unsafe.item(),
            'init': loss_init.item(),
            'outside': loss_outside.item()
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
    crown_cache_all = None,
    crown_cache_phi = None,
    input_bounds_all = None,
    cell_counts_V: dict = None,
    input_bounds_gen = None
) -> dict:
    """
    Evaluate constraint satisfaction using CROWN bounds (matches testing_simple3.py).

    Uses bounds checking (rigorous) rather than just sampling:
    - Goal: samples + bounds (existential constraint)
    - Others: bounds only (universal constraints)

    Args:
        V_net: Value network
        GV_net: Generator network
        region_cells: Dictionary of region cells
        beta_s: Separation threshold (float or torch.Tensor)
        beta_ra: Unsafe threshold
        device: Device
        n_samples: Number of samples for goal region
        crown_cache_all: Single CROWN cache for all V cells (in order: init, goal, unsafe, outside)
        crown_cache_phi: Optional pre-computed CROWN cache for Phi (avoids recreation)
        input_bounds_all: Tuple of (input_lowers_all, input_uppers_all) for all V cells
        cell_counts_V: Dict with counts for each region to split bounds
        input_bounds_gen: Tuple of (input_lowers_gen, input_uppers_gen) for generator cells

    Returns:
        Dictionary with satisfaction (True/False) and statistics
    """
    from crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds

    V_net.eval()
    # GV_net.eval()

    with torch.no_grad():
        # Compute CROWN bounds for all regions using single cache
        region_bounds = {}

        if crown_cache_all is not None and input_bounds_all is not None and cell_counts_V is not None:
            # Use pre-computed single cache and split results
            input_lowers_all, input_uppers_all = input_bounds_all
            v_lowers_all, v_uppers_all = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)

            # Split bounds by region (in order: init, goal, unsafe, outside)
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
            # Fallback: create separate cache for each region
            for name in ['goal', 'unsafe', 'init', 'outside']:
                if len(region_cells[name]) > 0:
                    cache = SymbolicCROWNCache(V_net, len(region_cells[name]), input_dim=2, device=device)
                    input_lowers, input_uppers = prepare_cell_bounds(region_cells[name], device)
                    v_lowers, v_uppers = cache.compute_bounds(input_lowers, input_uppers)
                    region_bounds[name] = (v_lowers, v_uppers)
                else:
                    region_bounds[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        # Goal: sample + bounds (existential: ∃x ∈ goal s.t. V(x) < beta_s)
        # Match original: (v_samples.min() < BETA_S) AND (v_lowers.min() >= 0)
        if len(region_cells['goal']) > 0:
            x_goal = sample_from_cells(region_cells['goal'], n_samples, device)
            v_goal_samples = V_net(x_goal).squeeze()
            goal_satisfied = ((v_goal_samples.min() < beta_s).item() and
                            (region_bounds['goal'][0].min() >= 0).item())
            v_goal_min = v_goal_samples.min().item()
            v_goal_max = v_goal_samples.max().item()
            v_goal_mean = v_goal_samples.mean().item()
        else:
            goal_satisfied = False
            v_goal_min = v_goal_max = v_goal_mean = 0.0

        # Outside: bounds only (universal: ∀x ∉ goal, V(x) >= beta_s)
        # Check: ALL lower bounds >= beta_s
        if len(region_bounds['outside'][0]) > 0:
            outside_satisfied = (region_bounds['outside'][0] >= beta_s).all().item()
            v_outside_min = region_bounds['outside'][0].min().item()
            v_outside_max = region_bounds['outside'][1].max().item()
        else:
            outside_satisfied = True
            v_outside_min = v_outside_max = 0.0

        # Unsafe: bounds only (universal: ∀x ∈ unsafe, V(x) >= beta_ra)
        # Check: ALL lower bounds >= beta_ra
        if len(region_bounds['unsafe'][0]) > 0:
            unsafe_satisfied = (region_bounds['unsafe'][0] >= beta_ra).all().item()
            v_unsafe_min = region_bounds['unsafe'][0].min().item()
            v_unsafe_max = region_bounds['unsafe'][1].max().item()
        else:
            unsafe_satisfied = True
            v_unsafe_min = v_unsafe_max = 0.0

        # Init: bounds only (universal: ∀x ∈ init, beta_s <= V(x) <= 1.0)
        # Check: ALL lower bounds >= beta_s AND ALL upper bounds <= 1.0
        if len(region_bounds['init'][0]) > 0:
            init_lower_ok = (region_bounds['init'][0] >= beta_s).all().item()
            init_upper_ok = (region_bounds['init'][1] <= 1.0).all().item()
            init_satisfied = init_lower_ok and init_upper_ok
            v_init_min = region_bounds['init'][0].min().item()
            v_init_max = region_bounds['init'][1].max().item()
        else:
            init_satisfied = True
            v_init_min = v_init_max = 0.0

        # Generator: bounds only (universal: ∀x ∉ (goal ∪ unsafe), Φ(x) <= 0)
        # Check: ALL upper bounds <= 0 (i.e., num_failing == 0 where failing = phi_upper > 0)
        if len(region_cells['generator']) > 0:
            if crown_cache_phi is not None:
                # Use pre-computed cache
                if input_bounds_gen is not None:
                    input_lowers, input_uppers = input_bounds_gen
                else:
                    input_lowers, input_uppers = prepare_cell_bounds(region_cells['generator'], device)
                phi_lowers, phi_uppers = crown_cache_phi.compute_bounds(input_lowers, input_uppers)
            else:
                # Create cache on-the-fly (for standalone use)
                cache_phi = SymbolicCROWNCache_Phi(GV_net, len(region_cells['generator']), input_dim=2, device=device)
                input_lowers, input_uppers = prepare_cell_bounds(region_cells['generator'], device)
                phi_lowers, phi_uppers = cache_phi.compute_bounds(input_lowers, input_uppers)

            num_failing = (phi_uppers > 0.0).sum().item()
            generator_satisfied = (num_failing == 0)
            phi_min = phi_lowers.min().item()
            phi_max = phi_uppers.max().item()
            phi_mean = phi_uppers.mean().item()
        else:
            generator_satisfied = True
            num_failing = 0
            phi_min = phi_max = phi_mean = 0.0

    results = {
        # Satisfaction (True/False)
        'goal_satisfied': goal_satisfied,
        'unsafe_satisfied': unsafe_satisfied,
        'init_satisfied': init_satisfied,
        'outside_satisfied': outside_satisfied,
        'generator_satisfied': generator_satisfied,

        # Statistics
        'V_goal_min': v_goal_min,
        'V_goal_max': v_goal_max,
        'V_goal_mean': v_goal_mean,
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
    # GV_net.train()

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
            f"Gen: {loss_dict['generator']:.4f}")
    elif compute_V and not compute_GV:
        print(f"{prefix}Epoch {epoch:5d} | "
            f"Total: {loss_dict['total']:.4f} | "
            f"Goal: {loss_dict['goal']:.4f} | "
            f"Unsafe: {loss_dict['unsafe']:.4f} | "
            f"Init: {loss_dict['init']:.4f} | "
            f"Outside: {loss_dict['outside']:.4f}")
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
    print(f"{prefix}  Generator: {status_str(results['generator_satisfied'])} "
          f"(Phi_min={results['Phi_min']:.3f}, Phi_max={results['Phi_max']:.3f}, failing={results['num_failing_cells']})")
