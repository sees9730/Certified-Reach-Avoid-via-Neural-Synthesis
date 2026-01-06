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

# Cache for fixed goal samples to prevent random sampling at each epoch
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
    V_init_lower,
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
    # threshold = V_outside_lower.min().item()
    threshold = V_init_lower.min().item()
    loss_rest = F.relu(v_samples.min() - threshold) * w_soft

    # Combined soft loss
    loss_soft =  loss_rest

    # Encourage non-negativity of the certified lower bound inside goal
    # (kept identical behavior, but note: min() here returns a scalar tensor)
    loss_nonneg = torch.relu(0.0 - V_lower).sum()

    # Obtain minimum V inside goal region: from samples, center, and lowest upper bound
    v_min = min(v_samples.min().item(), v_center.item(), V_upper.min().item())

    # When checking for success (passed), check the logic based on v_min
    # Keep semantics identical to your original
    # if (v_min < beta_s) and (V_lower.min() >= 0).item():
    if (v_min < V_init_lower.min()) and (V_lower.min() >= 0).item():
        passed = True
        if show:
            # print(" ✓ Inside Goal passed, min V= {:.4f}, min V_lower={:.4f}".format(
            #     v_min, V_lower.min()
            # ))
            print(" ✓ Inside Goal passed, min V= {:.4f}, min V_init_lower= {:.4f}, min V_lower={:.4f}".format(
                v_min, V_init_lower.min(), V_lower.min()
            ))
    else:
        passed = False
        if show:
            # print(" ✗ Inside Goal passed, min V= {:.4f}, min V_lower={:.4f}".format(
            #     v_min, V_lower.min()
            # ))
            print(" ✗ Inside Goal failed, min V= {:.4f}, min V_init_lower= {:.4f}, min V_lower={:.4f}".format(
                v_min, V_init_lower.min(), V_lower.min()
            ))

    # return loss_nonneg + loss_soft, passed
    # return loss_soft, passed
    return loss_nonneg, passed


def compute_loss_unsafe_bounds(
    V_lower: torch.Tensor,
    V_upper: torch.Tensor,
    beta_ra: float
) -> torch.Tensor:
    """
    Compute unsafe constraint loss: V(x) >= beta_ra for all x in Unsafe region.

    Args:
        V_lower: Lower bounds on V (N,)
        V_upper: Upper bounds on V (N,)
        beta_ra: Target lower bound for unsafe

    Returns:
        Loss (scalar)
    """
    return F.relu(beta_ra - V_lower).sum()


def compute_loss_init_bounds(
    V_lower: torch.Tensor,
    V_upper: torch.Tensor,
    beta_s: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Compute init constraint loss.

    Init region must satisfy:
    1. V < 1.0 (upper bound)
    2. V >= beta_s (general lower bound that applies everywhere except goal)

    Args:
        V_lower: Lower bounds on V (N,)
        V_upper: Upper bounds on V (N,)
        beta_s: Target lower bound for init (float or torch.Tensor)

    Returns:
        Loss (scalar)
    """
    loss_upper = F.relu(V_upper - 1.0).sum()
    loss_lower = F.relu(beta_s - V_lower).sum()
    return loss_upper + loss_lower


def compute_loss_outside_bounds(
    V_lower: torch.Tensor,
    V_upper: torch.Tensor,
    beta_s: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Compute outside goal constraint loss: V(x) >= beta_s for x outside Goal.

    Args:
        V_lower: Lower bounds on V (N,)
        V_upper: Upper bounds on V (N,)
        beta_s: Target lower bound outside goal (float or torch.Tensor)

    Returns:
        Loss (scalar)
    """
    return F.relu(beta_s - V_lower).sum()


def compute_loss_generator_bounds(
    Phi_lower: torch.Tensor,
    Phi_upper: torch.Tensor
) -> torch.Tensor:
    """
    Compute generator constraint loss: Φ(x) <= 0 for x outside (Goal ∪ Unsafe).

    Args:
        Phi_lower: Lower bounds on Φ (N,)
        Phi_upper: Upper bounds on Φ (N,)

    Returns:
        Loss (scalar) - unweighted (weight applied by caller)
    """
    return F.relu(Phi_upper).sum()


def compute_loss_generator_bounds_unsafe(
    Phi_lower: torch.Tensor,
    Phi_upper: torch.Tensor
) -> torch.Tensor:
    """
    Compute generator constraint loss with strong penalty for unsafe region.

    Args:
        Phi_lower: Lower bounds on Φ (N,)
        Phi_upper: Upper bounds on Φ (N,)

    Returns:
        Loss (scalar) - unweighted (weight applied by caller)
    """
    return F.relu(Phi_upper + 1000).sum()


def compute_loss_generator_bounds_goal(
    Phi_lower: torch.Tensor,
    Phi_upper: torch.Tensor
) -> torch.Tensor:
    """
    Compute generator constraint loss for goal region.

    Args:
        Phi_lower: Lower bounds on Φ (N,)
        Phi_upper: Upper bounds on Φ (N,)

    Returns:
        Loss (scalar) - unweighted (weight applied by caller)
    """
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
    epoch: int = 0,
    locked_to_all_sum: bool = False,
    current_sum_constraint: str = None,
    constraint_largest_counts: dict = None,
    prev_total_loss: float = None
) -> Tuple[torch.Tensor, dict, bool, str, dict]:
    """
    Compute total training loss from CROWN bounds with adaptive constraint focusing.

    This function implements a dynamic loss aggregation strategy:
    - Uses .sum() for the constraint with the largest violation (provides stronger gradient)
    - Uses .mean() for other constraints (prevents them from dominating)
    - Tracks which constraint has been largest over time to avoid rapid switching
    - Locks to using .sum() for all constraints once loss is small enough

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
        locked_to_all_sum: If True, use sum for all constraints (never switch back)
        current_sum_constraint: Which constraint currently has .sum() focus
        constraint_largest_counts: Dict tracking how many epochs each constraint has been largest
        prev_total_loss: Previous epoch's total loss (for switching logic)

    Returns:
        (total_loss, loss_dict, locked_to_all_sum, current_sum_constraint, constraint_largest_counts)
    """
    if loss_weights is None:
        loss_weights = {
            'goal': 1.0,
            'unsafe': 1.0,
            'init': 1.0,
            'outside': 1.0,
            'generator': 1.0
        }

    # Compute raw violations for each constraint
    loss_components = {}

    if compute_V:
        violations_unsafe = F.relu(beta_ra - V_unsafe_lower)
        violations_init = F.relu(V_init_upper - 1.0)
        violations_outside = F.relu(0.0 - V_outside_lower)
        violations_goal = F.relu(V_goal_upper - beta_s)

        loss_components['unsafe'] = violations_unsafe
        loss_components['init'] = violations_init
        loss_components['outside'] = violations_outside
        loss_components['goal'] = violations_goal

    if compute_GV:
        if len(Phi_upper) > 0:
            violations_generator = F.relu(Phi_upper)
            loss_components['generator'] = violations_generator
        else:
            violations_generator = torch.tensor([], device=Phi_upper.device if hasattr(Phi_upper, 'device') else 'cpu')

    # Calculate sum of violations for each constraint to determine focus
    loss_sums = {name: viols.sum().item() for name, viols in loss_components.items()}
    total_sum_loss = sum(loss_sums.values())

    if locked_to_all_sum:
        # Use sum for all constraints near convergence to prevent oscillation
        largest_loss = 'ALL'
        if compute_V:
            loss_unsafe = violations_unsafe.sum()
            loss_init = violations_init.sum()
            loss_outside = violations_outside.sum()
            loss_goal = violations_goal.sum()
        if compute_GV:
            loss_generator = violations_generator.sum() if len(violations_generator) > 0 else torch.tensor(0.0, device=device)
    else:
        # Dynamic focusing: use sum for largest violation, mean for others
        candidate_largest = max(loss_sums, key=loss_sums.get) if loss_sums else None

        STABILITY_THRESHOLD = 500

        if constraint_largest_counts is None:
            constraint_largest_counts = {}

        # Track which constraint has been largest
        if candidate_largest is not None:
            constraint_largest_counts[candidate_largest] = constraint_largest_counts.get(candidate_largest, 0) + 1

        # Initialize focus on first epoch
        if current_sum_constraint is None:
            current_sum_constraint = candidate_largest
        else:
            # Only allow switching focus if loss is still high
            # If we're making progress, stick with current focus
            if prev_total_loss is not None:
                allow_focus_switch = (prev_total_loss >= (beta_ra - 1.0))
            else:
                allow_focus_switch = (total_sum_loss >= (beta_ra - 1.0))

            if allow_focus_switch:
                # Look for constraints that have consistently been largest
                candidates_to_switch = {
                    name: count for name, count in constraint_largest_counts.items()
                    if name != current_sum_constraint and count >= STABILITY_THRESHOLD
                }

                if candidates_to_switch:
                    # Switch to the constraint with the most accumulated evidence
                    new_constraint = max(candidates_to_switch, key=candidates_to_switch.get)
                    print(f"[SWITCH] Switching from {current_sum_constraint} to {new_constraint}")

                    # Reset both old and new constraint counts to prevent rapid oscillation
                    old_constraint = current_sum_constraint
                    constraint_largest_counts[old_constraint] = 0
                    constraint_largest_counts[new_constraint] = 0

                    current_sum_constraint = new_constraint

        # Use the stable sum constraint (not necessarily the current largest)
        largest_loss = current_sum_constraint

        if compute_V:
            loss_unsafe = violations_unsafe.sum() if largest_loss == 'unsafe' else violations_unsafe.mean()
            loss_init = violations_init.sum() if largest_loss == 'init' else violations_init.mean()
            loss_outside = violations_outside.sum() if largest_loss == 'outside' else violations_outside.mean()
            loss_goal = violations_goal.sum() if largest_loss == 'goal' else violations_goal.mean()

        if compute_GV:
            if len(violations_generator) > 0:
                loss_generator = violations_generator.sum() if largest_loss == 'generator' else violations_generator.mean()
            else:
                loss_generator = torch.tensor(0.0, device=device)

    # Combine losses
    if compute_V and compute_GV:
        total_loss = (
            loss_weights['unsafe'] * loss_unsafe +
            loss_weights['init'] * loss_init +
            loss_weights['outside'] * loss_outside +
            
            loss_weights['goal'] * loss_goal + 
            generator_weight * loss_generator
        )

        loss_dict = {
            'total': total_loss.item(),
            'unsafe': loss_unsafe.item(),
            'init': loss_init.item(),
            'outside': loss_outside.item(),
            'goal': violations_goal.mean().item(),
            'generator': loss_generator.item(),
            'largest_loss': largest_loss
        }

    elif compute_V and not compute_GV:
        total_loss = (
            loss_weights['unsafe'] * loss_unsafe +
            loss_weights['init'] * loss_init +
            loss_weights['outside'] * loss_outside
        )

        loss_dict = {
            'total': total_loss.item(),
            'unsafe': loss_unsafe.item(),
            'init': loss_init.item(),
            'outside': loss_outside.item(),
            'largest_loss': largest_loss
        }

    elif not compute_V and compute_GV:
        total_loss = generator_weight * loss_generator

        loss_dict = {
            'generator': loss_generator.item(),
            'largest_loss': largest_loss
        }

    # Switch to all sums once loss is small enough for final convergence
    if not locked_to_all_sum and total_loss.item() < 1.0:
        locked_to_all_sum = True
        print(f'[SWITCH] Switching to all sums at epoch {epoch} (total_loss={total_loss.item():.4f} < 1.0)')

    return total_loss, loss_dict, locked_to_all_sum, current_sum_constraint, constraint_largest_counts



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

    # Infer input dimension from bounds or cells
    input_dim = None
    if input_bounds_all is not None:
        input_lowers_all, _ = input_bounds_all
        if isinstance(input_lowers_all, torch.Tensor) and input_lowers_all.ndim == 2:
            input_dim = int(input_lowers_all.shape[1])

    if input_dim is None:
        # Try to infer from any non-empty cell list
        for name in ['goal', 'unsafe', 'init', 'outside', 'generator']:
            if name in region_cells and len(region_cells[name]) > 0:
                cell0 = region_cells[name][0]
                input_dim = int(cell0[0].numel())
                break

    if input_dim is None:
        raise ValueError("Could not infer input_dim: provide input_bounds_all or non-empty region_cells.")

    with torch.no_grad():
        region_bounds = {}

        if crown_cache_all is not None and input_bounds_all is not None and cell_counts_V is not None:
            # Use pre-computed cache and split results by region
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
            # Fallback: create separate cache for each region
            for name in ['goal', 'unsafe', 'init', 'outside']:
                if len(region_cells[name]) > 0:
                    cache = SymbolicCROWNCache(V_net, len(region_cells[name]), input_dim=input_dim, device=device)
                    input_lowers, input_uppers = prepare_cell_bounds(region_cells[name], device=device, input_dim=input_dim)
                    v_lowers, v_uppers = cache.compute_bounds(input_lowers, input_uppers)
                    region_bounds[name] = (v_lowers, v_uppers)
                else:
                    region_bounds[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        # Check outside constraint
        if len(region_bounds['outside'][0]) > 0:
            outside_satisfied = (region_bounds['outside'][0] >= beta_s).all().item()
            v_outside_min = region_bounds['outside'][0].min().item()
            v_outside_max = region_bounds['outside'][1].max().item()
        else:
            outside_satisfied = True
            v_outside_min = v_outside_max = 0.0

        # Check goal constraint
        if len(region_cells['goal']) > 0:
            goal_satisfied = ((region_bounds['goal'][0].min().item() >= 0) and 
                              (region_bounds['goal'][1].max().item() <= 1))
            v_goal_min = region_bounds['goal'][0].min().item()
            v_goal_max = region_bounds['goal'][1].max().item()
            # x_goal = sample_from_cells(region_cells['goal'], n_samples, device)
            # v_goal_samples = V_net(x_goal).squeeze(-1)
            # goal_satisfied = ((v_goal_samples.min() < beta_s).item() and
                            #   (region_bounds['goal'][0].min() >= 0).item())
            # v_goal_min = v_goal_samples.min().item()
            # v_goal_max = v_goal_samples.max().item()
            # v_goal_mean = v_goal_samples.mean().item()
        else:
            goal_satisfied = True
            # v_goal_min = v_goal_max = v_goal_mean = 0
            v_goal_min = v_goal_max = 0.0

        # Check unsafe constraint
        if len(region_bounds['unsafe'][0]) > 0:
            unsafe_satisfied = (region_bounds['unsafe'][0] >= beta_ra).all().item()
            v_unsafe_min = region_bounds['unsafe'][0].min().item()
            v_unsafe_max = region_bounds['unsafe'][1].max().item()

            # Show how many cells violate the constraint
            num_unsafe_cells = len(region_bounds['unsafe'][0])
            num_violated = (region_bounds['unsafe'][0] < beta_ra).sum().item()
            if num_violated > 0:
                print(f"  [DEBUG] Unsafe: {num_violated}/{num_unsafe_cells} cells violate (V_lower < {beta_ra})")
        else:
            unsafe_satisfied = True
            v_unsafe_min = v_unsafe_max = 0.0

        # Check init constraint
        if len(region_bounds['init'][0]) > 0:
            init_lower_ok = (region_bounds['init'][0] >= beta_s).all().item()
            init_upper_ok = (region_bounds['init'][1] <= 1.0).all().item()
            init_satisfied = init_lower_ok and init_upper_ok
            v_init_min = region_bounds['init'][0].min().item()
            v_init_max = region_bounds['init'][1].max().item()
        else:
            init_satisfied = True
            v_init_min = v_init_max = 0.0

        # Check generator constraint: Φ(x) <= 0
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
        'unsafe_satisfied': unsafe_satisfied,
        'init_satisfied': init_satisfied,
        'goal_satisfied': goal_satisfied,
        'outside_satisfied': outside_satisfied,
        'generator_satisfied': generator_satisfied,

        'V_unsafe_min': v_unsafe_min,
        'V_unsafe_max': v_unsafe_max,
        'V_init_min': v_init_min,
        'V_init_max': v_init_max,
        'V_outside_min': v_outside_min,
        'V_outside_max': v_outside_max,
        'V_goal_min': v_goal_min,
        'V_goal_max': v_goal_max,
        'Phi_min': phi_min,
        'Phi_max': phi_max,
        'Phi_mean': phi_mean,
        'num_failing_cells': num_failing
    }

    V_net.train()
    return results


def print_loss_summary(epoch: int, loss_dict: dict, prefix: str = "", compute_V: bool = True, compute_GV: bool = True, focus: str = ""):
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
            f"Gen: {loss_dict['generator']:.4f} | "
            f"Focus: {focus}")
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
    def status_str(satisfied):
        return "✓" if satisfied else "✗"

    print(f"{prefix}Constraint Satisfaction:")
    print(f"{prefix}  Goal:      {status_str(results['goal_satisfied'])} "
          f"(V_min={results['V_goal_min']:.3f}, V_max={results['V_goal_max']:.3f})")
    print(f"{prefix}  Unsafe:    {status_str(results['unsafe_satisfied'])} "
          f"(V_min={results['V_unsafe_min']:.3f}, V_max={results['V_unsafe_max']:.3f})")
    print(f"{prefix}  Init:      {status_str(results['init_satisfied'])} "
          f"(V_min={results['V_init_min']:.3f}, V_max={results['V_init_max']:.3f})")
    print(f"{prefix}  Outside:   {status_str(results['outside_satisfied'])} "
          f"(V_min={results['V_outside_min']:.3f}, V_max={results['V_outside_max']:.3f})")
    print(f"{prefix}  Generator: {status_str(results['generator_satisfied'])} "
          f"(Phi_upper_min={results['Phi_min']:.3f}, Phi_upper_max={results['Phi_max']:.3f}, failing={results['num_failing_cells']})")


def print_cell_counts(region_cells: dict, prefix: str = ""):
    """
    Print the number of cells for each region.

    Args:
        region_cells: Dictionary mapping region names to lists of cells
        prefix: Optional prefix for print
    """
    print(f"{prefix}Discretization cell counts:")
    total_v_cells = 0
    for name in ['init', 'goal', 'unsafe', 'outside']:
        if name in region_cells:
            count = len(region_cells[name])
            print(f"{prefix}  {name}: {count} cells")
            total_v_cells += count

    print(f"{prefix}  Total V cells: {total_v_cells}")

    if 'generator' in region_cells:
        gen_count = len(region_cells['generator'])
        print(f"{prefix}  generator: {gen_count} cells")


def refine_failing_cells(
    region_cells: List[Tuple[torch.Tensor, torch.Tensor]],
    failing_mask: torch.Tensor,
    refine_factor: int = 2,
    N_to_refine: int = 100,
    seed: Optional[int] = 0,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
    """
    Refine (split) a random subset of failing cells to improve accuracy.

    Args:
        region_cells: List of (lower, upper) cell bounds, each (D,)
        failing_mask: Boolean mask indicating which cells failed
        refine_factor: Factor to subdivide cells (2 = split into 2^D subcells)
        N_to_refine: Max number of failing cells to refine (randomly selected)
        seed: Optional RNG seed for reproducibility

    Returns:
        Tuple of (new_cells, num_refined)
    """
    failing_mask = failing_mask.reshape(-1).to(dtype=torch.bool).cpu()

    n_cells = len(region_cells)
    mask_len = min(len(failing_mask), n_cells)

    # Get indices of failing cells
    failing_idxs = torch.nonzero(failing_mask[:mask_len], as_tuple=False).reshape(-1).tolist()

    # Choose subset to refine
    if N_to_refine is None or N_to_refine <= 0 or len(failing_idxs) == 0:
        refine_set = set()
    else:
        rng = np.random.default_rng(seed)
        k = min(int(N_to_refine), len(failing_idxs))
        refine_set = set(rng.choice(failing_idxs, size=k, replace=False).tolist())

    new_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
    num_refined = 0

    for i, (cell_lower, cell_upper) in enumerate(region_cells):
        if i in refine_set:
            # Build bounds array (dimension-generic)
            cell_lower = cell_lower.reshape(-1)
            cell_upper = cell_upper.reshape(-1)
            D = int(cell_lower.numel())

            cell_bounds = np.array(
                [[cell_lower[d].item(), cell_upper[d].item()] for d in range(D)],
                dtype=np.float32
            )

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
    Merge adjacent passing cells to reduce cell count while maintaining accuracy.

    Two cells can merge if:
      - Both are passing (not failing)
      - Same bounds in all dimensions except one dimension d
      - They touch along dimension d (one's upper equals other's lower)
      - Same size along dimension d (maintains grid alignment)

    Args:
        region_cells: List of (lower, upper), each (D,)
        failing_mask: Boolean mask (len == #cells) where True means failing
        max_passes: How many coarsening sweeps to try
        max_merges: Optional cap on how many pair-merges to do (for speed)
        seed: RNG seed if max_merges is used (randomly subsamples merges)
        eps: Quantization step for robust float comparisons

    Returns:
        (new_cells, num_pair_merges)
    """
    failing_mask = failing_mask.reshape(-1).to(dtype=torch.bool).cpu()
    n = len(region_cells)
    if len(failing_mask) != n:
        return region_cells, 0

    # Only passing cells are eligible to merge
    passing = (~failing_mask).tolist()

    def quantize(v: torch.Tensor) -> Tuple[int, ...]:
        """Convert float bounds to integers for robust comparison."""
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

        D = int(cells[0][0].numel())

        # Precompute quantized bounds for robust hashing
        lo_q = [quantize(lo) for (lo, _) in cells]
        hi_q = [quantize(hi) for (_, hi) in cells]

        merged_flag = [False] * n
        new_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []

        merges_this_pass = 0

        for d in range(D):
            # Build lookup: (signature_except_d, lower_d, size_d) -> index
            lookup = {}
            for i in range(n):
                if merged_flag[i] or (not passing[i]):
                    continue
                size_d = hi_q[i][d] - lo_q[i][d]
                sig = tuple((lo_q[i][k], hi_q[i][k]) for k in range(D) if k != d)
                key = (sig, lo_q[i][d], size_d)
                lookup[key] = i

            # Attempt merges along this dimension
            indices = list(range(n))
            if max_merges is not None:
                rng.shuffle(indices)

            for i in indices:
                if merged_flag[i] or (not passing[i]):
                    continue

                size_d = hi_q[i][d] - lo_q[i][d]
                sig = tuple((lo_q[i][k], hi_q[i][k]) for k in range(D) if k != d)
                neighbor_key = (sig, hi_q[i][d], size_d)  # Neighbor starts where i ends
                j = lookup.get(neighbor_key, None)
                if j is None or j == i or merged_flag[j] or (not passing[j]):
                    continue

                # Merge cells i and j
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

        # Append anything not merged
        for i in range(n):
            if not merged_flag[i]:
                new_cells.append(cells[i])

        if len(new_cells) == len(cells):
            # No progress made, stop trying
            break
        cells = new_cells

    return cells, num_merges_total
