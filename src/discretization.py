"""
Discretization utilities for region decomposition.

This module provides functions to:
- Discretize rectangular regions into grid cells
- Handle region subtraction (set difference)
- Adaptive refinement based on distance or failure
- Compute rectangular partitions
"""

import numpy as np
import torch
from typing import List, Tuple
from itertools import product

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
import sys
sys.path.insert(0, str(ROOT))
from src.regions import Region


def debug_print_region_bounds(region_cells, label: str = "Region") -> None:
    """
    Print per-dimension [min, max] bounds for region cells.

    Accepted inputs:
      - list of cells: [(lower_tensor, upper_tensor), ...]
      - dict of cells: {"init": [...], "goal": [...], ...}
      - Region (kept for backward compatibility)
    """
    if isinstance(region_cells, Region):
        region = region_cells
        if region.is_union and len(region.components) > 0:
            lowers = np.stack([comp.bounds[:, 0] for comp in region.components], axis=0)
            uppers = np.stack([comp.bounds[:, 1] for comp in region.components], axis=0)
            mins = lowers.min(axis=0)
            maxs = uppers.max(axis=0)
        else:
            mins = region.bounds[:, 0]
            maxs = region.bounds[:, 1]
    elif isinstance(region_cells, dict):
        all_cells = []
        for cells in region_cells.values():
            all_cells.extend(cells)
        if len(all_cells) == 0:
            print(f"{label} bounds: no cells")
            return
        lowers = np.stack([lo.detach().cpu().numpy() for lo, _ in all_cells], axis=0)
        uppers = np.stack([hi.detach().cpu().numpy() for _, hi in all_cells], axis=0)
        mins = lowers.min(axis=0)
        maxs = uppers.max(axis=0)
    else:
        cells = list(region_cells)
        if len(cells) == 0:
            print(f"{label} bounds: no cells")
            return
        lowers = np.stack([lo.detach().cpu().numpy() for lo, _ in cells], axis=0)
        uppers = np.stack([hi.detach().cpu().numpy() for _, hi in cells], axis=0)
        mins = lowers.min(axis=0)
        maxs = uppers.max(axis=0)

    print(f"{label} bounds:")
    for d, (lo, hi) in enumerate(zip(mins, maxs)):
        print(f"  dim {d}: [{float(lo):.6g}, {float(hi):.6g}]")


def discretize_region_with_splits(
    region: Region,
    n_splits_per_dim: List[int]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize a (non-union) region with potentially different split counts per dimension.
    """
    if region.is_union:
        all_cells = []
        for comp in region.components:
            all_cells.extend(discretize_region_with_splits(comp, n_splits_per_dim))
        return all_cells

    bounds = region.bounds  # (D, 2)
    D = bounds.shape[0]
    if len(n_splits_per_dim) != D:
        raise ValueError(f"Expected {D} splits, got {len(n_splits_per_dim)}")

    splits = [max(1, int(n)) for n in n_splits_per_dim]
    edges = [
        np.linspace(bounds[d, 0], bounds[d, 1], splits[d] + 1, dtype=np.float32)
        for d in range(D)
    ]

    cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in product(*[range(s) for s in splits]):
        cell_lower = np.array([edges[d][idx[d]] for d in range(D)], dtype=np.float32)
        cell_upper = np.array([edges[d][idx[d] + 1] for d in range(D)], dtype=np.float32)
        cells.append((
            torch.tensor(cell_lower, dtype=torch.float32),
            torch.tensor(cell_upper, dtype=torch.float32)
        ))
    return cells


def _allocate_budgets(weights: np.ndarray, total_budget: int) -> np.ndarray:
    """
    Allocate integer budgets proportional to weights with exact sum.
    """
    if total_budget <= 0:
        raise ValueError(f"total_budget must be positive, got {total_budget}")

    K = int(weights.shape[0])
    if K == 0:
        return np.zeros((0,), dtype=np.int64)

    eff_budget = int(total_budget)
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if float(w.sum()) <= 0.0:
        w = np.ones_like(w)

    if eff_budget < K:
        # Not enough budget to give each component at least one cell.
        b = np.zeros((K,), dtype=np.int64)
        idx = np.argsort(-w)[:eff_budget]
        b[idx] = 1
        return b

    raw = (w / w.sum()) * eff_budget
    b = np.floor(raw).astype(np.int64)
    rem = eff_budget - int(b.sum())
    if rem > 0:
        frac_idx = np.argsort(-(raw - b))
        for i in frac_idx[:rem]:
            b[i] += 1

    # Ensure at least one per component when budget permits.
    zeros = np.where(b == 0)[0].tolist()
    for zi in zeros:
        j = int(np.argmax(b))
        if b[j] <= 1:
            break
        b[j] -= 1
        b[zi] += 1

    return b


def _flatten_region_components(region: Region) -> List[Region]:
    return region.components if region.is_union else [region]


def _splits_from_budget_and_lengths(lengths: np.ndarray, budget: int) -> List[int]:
    """
    Choose per-dimension integer splits with product <= budget, roughly proportional to lengths.
    """
    D = int(lengths.shape[0])
    B = max(1, int(budget))
    L = np.asarray(lengths, dtype=np.float64)
    L = np.maximum(L, 1e-8)

    c = (B / float(np.prod(L))) ** (1.0 / D)
    n = np.maximum(1, np.floor(c * L).astype(np.int64))

    def _prod(arr):
        return int(np.prod(arr.astype(np.int64)))

    # Grow until we would exceed budget.
    while True:
        cur = _prod(n)
        if cur >= B:
            break
        best_d = None
        best_gain = None
        for d in range(D):
            cand = n.copy()
            cand[d] += 1
            p = _prod(cand)
            if p <= B:
                # Favor dimensions with larger lengths while accounting current split count.
                gain = L[d] / float(n[d] + 1)
                if (best_gain is None) or (gain > best_gain):
                    best_gain = gain
                    best_d = d
        if best_d is None:
            break
        n[best_d] += 1

    return [int(v) for v in n.tolist()]


def discretize_region_list_with_budget(
    regions_list: List[Region],
    total_budget: int,
    label: str = "Region",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize a list of (possibly union) regions using a total cell budget.

    Budget allocation is volume-proportional across boxes, then per-dimension
    split counts are length-proportional within each box.
    """
    boxes: List[Region] = []
    for r in regions_list:
        boxes.extend(_flatten_region_components(r))

    if len(boxes) == 0:
        return []

    sizes = np.stack([np.maximum(b.bounds[:, 1] - b.bounds[:, 0], 0.0) for b in boxes], axis=0)  # (K,D)
    vols = np.prod(np.maximum(sizes, 1e-8), axis=1)
    box_budgets = _allocate_budgets(vols, int(total_budget))

    all_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
    used = 0
    for box, b in zip(boxes, box_budgets):
        if int(b) <= 0:
            continue
        lengths = np.maximum(box.bounds[:, 1] - box.bounds[:, 0], 1e-8)
        splits = _splits_from_budget_and_lengths(lengths, int(b))
        cells = discretize_region_with_splits(box, splits)
        all_cells.extend(cells)
        used += len(cells)

    print(f" {label} budget target: {int(total_budget)}")
    print(f" {label} budget actual: {used}")
    return all_cells


def _splits_from_budget_and_weights(weights: np.ndarray, budget: int) -> List[int]:
    """
    Choose per-dimension integer splits with product <= budget using directional weights.

    If weights are all ones, this yields approximately uniform splits across dimensions.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    D = int(w.shape[0])
    B = max(1, int(budget))
    w = np.maximum(w, 1e-8)

    c = (B / float(np.prod(w))) ** (1.0 / D)
    n = np.maximum(1, np.floor(c * w).astype(np.int64))

    def _prod(arr):
        return int(np.prod(arr.astype(np.int64)))

    while True:
        cur = _prod(n)
        if cur >= B:
            break
        best_d = None
        best_gain = None
        for d in range(D):
            cand = n.copy()
            cand[d] += 1
            p = _prod(cand)
            if p <= B:
                gain = w[d] / float(n[d] + 1)
                if (best_gain is None) or (gain > best_gain):
                    best_gain = gain
                    best_d = d
        if best_d is None:
            break
        n[best_d] += 1

    return [int(v) for v in n.tolist()]


def discretize_region_list_with_weights_budget(
    regions_list: List[Region],
    axis_weights: np.ndarray,
    total_budget: int,
    label: str = "Region",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize regions with:
      - total budget per region group,
      - directional split preference from axis_weights.

    Steps:
      1) allocate budget across boxes by box volume,
      2) split each box according to axis_weights.
    """
    boxes: List[Region] = []
    for r in regions_list:
        boxes.extend(_flatten_region_components(r))

    if len(boxes) == 0:
        return []

    D = int(boxes[0].bounds.shape[0])
    w = np.asarray(axis_weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != D:
        raise ValueError(f"{label}: expected {D} axis_weights, got {w.shape[0]}")
    if np.any(w <= 0.0):
        raise ValueError(f"{label}: axis_weights must be positive, got {w}")

    sizes = np.stack([np.maximum(b.bounds[:, 1] - b.bounds[:, 0], 0.0) for b in boxes], axis=0)  # (K,D)
    vols = np.prod(np.maximum(sizes, 1e-8), axis=1)
    box_budgets = _allocate_budgets(vols, int(total_budget))

    all_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
    used = 0
    for box, b in zip(boxes, box_budgets):
        if int(b) <= 0:
            continue
        splits = _splits_from_budget_and_weights(w, int(b))
        cells = discretize_region_with_splits(box, splits)
        all_cells.extend(cells)
        used += len(cells)

    print(f" {label} target budget: {int(total_budget)}")
    print(f" {label} actual cells: {used}")
    return all_cells


def _splits_from_unit_lengths(lengths: np.ndarray, unit_lengths: np.ndarray, scale: float = 1.0) -> List[int]:
    unit = np.maximum(unit_lengths * float(scale), 1e-8)
    splits = np.maximum(1, np.ceil(np.maximum(lengths, 0.0) / unit).astype(np.int64))
    return [int(v) for v in splits.tolist()]


def _count_cells_for_boxes_with_unit_lengths(boxes: List[Region], unit_lengths: np.ndarray, scale: float) -> int:
    total = 0
    for box in boxes:
        lengths = np.maximum(box.bounds[:, 1] - box.bounds[:, 0], 0.0)
        splits = _splits_from_unit_lengths(lengths, unit_lengths, scale=scale)
        total += int(np.prod(np.array(splits, dtype=np.int64)))
    return int(total)


def discretize_region_list_with_unit_lengths(
    regions_list: List[Region],
    unit_lengths: np.ndarray,
    max_budget: int = 0,
    label: str = "Region",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize a list of regions using per-dimension unit lengths.

    If max_budget > 0 and the resulting cell count exceeds max_budget, enlarge unit lengths
    by a global scale factor until the count is <= max_budget.
    """
    boxes: List[Region] = []
    for r in regions_list:
        boxes.extend(_flatten_region_components(r))

    if len(boxes) == 0:
        return []

    D = int(boxes[0].bounds.shape[0])
    u = np.asarray(unit_lengths, dtype=np.float64).reshape(-1)
    if u.shape[0] != D:
        raise ValueError(f"{label}: expected {D} unit lengths, got {u.shape[0]}")
    if np.any(u <= 0.0):
        raise ValueError(f"{label}: unit lengths must be positive, got {u}")

    base_count = _count_cells_for_boxes_with_unit_lengths(boxes, u, scale=1.0)
    scale = 1.0
    budget = int(max_budget)
    if budget > 0 and base_count > budget:
        min_count = len(boxes)  # one cell per box (can't go lower without dropping boxes)
        if min_count > budget:
            # Budget is too small to satisfy exactly; use one cell per box.
            scale = 1e12
            print(
                f" {label}: budget {budget} is below minimum feasible {min_count}; "
                f"using one cell per box."
            )
        else:
            lo, hi = 1.0, 1.0
            while _count_cells_for_boxes_with_unit_lengths(boxes, u, scale=hi) > budget:
                hi *= 2.0

            # Binary search minimal scale meeting budget.
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                cnt = _count_cells_for_boxes_with_unit_lengths(boxes, u, scale=mid)
                if cnt > budget:
                    lo = mid
                else:
                    hi = mid
            scale = hi

    all_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
    used = 0
    for box in boxes:
        lengths = np.maximum(box.bounds[:, 1] - box.bounds[:, 0], 0.0)
        splits = _splits_from_unit_lengths(lengths, u, scale=scale)
        cells = discretize_region_with_splits(box, splits)
        all_cells.extend(cells)
        used += len(cells)

    if budget > 0:
        print(f" {label}: unit-length mode with budget={budget}, cells={used}, scale={scale:.4f}")
    else:
        print(f" {label}: unit-length mode, cells={used}, scale={scale:.4f}")
    return all_cells


def discretize_unsafe_with_budget(
    unsafe_region: Region,
    total_budget: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize unsafe region using a total cell budget.

    Strategy:
      1) Split total budget across unsafe boxes by box volume.
      2) For each box, choose per-dimension splits proportional to side lengths,
         with product <= that box's budget.
    """
    return discretize_region_list_with_budget([unsafe_region], int(total_budget), label="Unsafe")


def discretize_region(region: Region, n_squares: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize a region into an n_squares-per-dimension grid of cells.

    For D dimensions, this yields n_squares^D cells per (non-union) rectangle.

    For union regions, discretizes each component separately and combines.
    """
    if region.is_union:
        all_cells = []
        for comp in region.components:
            all_cells.extend(discretize_region(comp, n_squares))
        return all_cells

    bounds = region.bounds  # (D, 2)
    D = bounds.shape[0]

    # Edges for each dimension
    edges = [
        np.linspace(bounds[d, 0], bounds[d, 1], n_squares + 1, dtype=np.float32)
        for d in range(D)
    ]

    cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in product(range(n_squares), repeat=D):
        cell_lower = np.array([edges[d][idx[d]] for d in range(D)], dtype=np.float32)
        cell_upper = np.array([edges[d][idx[d] + 1] for d in range(D)], dtype=np.float32)
        cells.append((
            torch.tensor(cell_lower, dtype=torch.float32),
            torch.tensor(cell_upper, dtype=torch.float32)
        ))
    return cells


def subtract_rectangle(
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    excl_lower: np.ndarray,
    excl_upper: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Subtract exclusion rectangle from cell rectangle (now supports D dims).
    Returns list of rectangles representing cell \\ exclusion.

    Produces a disjoint decomposition with up to 2D rectangles.
    """
    cell_lower = np.asarray(cell_lower, dtype=np.float32)
    cell_upper = np.asarray(cell_upper, dtype=np.float32)
    excl_lower = np.asarray(excl_lower, dtype=np.float32)
    excl_upper = np.asarray(excl_upper, dtype=np.float32)

    D = cell_lower.shape[0]

    # Compute intersection box between cell and exclusion
    inter_lower = np.maximum(cell_lower, excl_lower)
    inter_upper = np.minimum(cell_upper, excl_upper)

    # No overlap -> return the cell as-is
    if np.any(inter_lower >= inter_upper):
        return [(cell_lower, cell_upper)]

    # Complete containment: cell entirely inside exclusion -> removed
    if np.all(excl_lower <= cell_lower) and np.all(cell_upper <= excl_upper):
        return []

    # Partial overlap: carve "slabs" around the intersection region
    result: List[Tuple[np.ndarray, np.ndarray]] = []

    core_lower = cell_lower.copy()
    core_upper = cell_upper.copy()

    for d in range(D):
        # Lower slab along dim d
        if core_lower[d] < inter_lower[d]:
            low = core_lower.copy()
            up = core_upper.copy()
            up[d] = inter_lower[d]
            result.append((low, up))
            core_lower[d] = inter_lower[d]

        # Upper slab along dim d
        if inter_upper[d] < core_upper[d]:
            low = core_lower.copy()
            up = core_upper.copy()
            low[d] = inter_upper[d]
            result.append((low, up))
            core_upper[d] = inter_upper[d]

    # Remaining core equals the intersection (removed)
    return result


def clip_cell_against_exclusions(
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    exclusion_regions: List[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Clip a cell against multiple exclusion regions (supports D dims).
    Returns list of cells with all exclusions removed.
    """
    current_cells = [(cell_lower, cell_upper)]

    for excl_region in exclusion_regions:
        excl_lower = excl_region[:, 0]
        excl_upper = excl_region[:, 1]

        new_cells = []
        for lower, upper in current_cells:
            new_cells.extend(subtract_rectangle(lower, upper, excl_lower, excl_upper))

        current_cells = new_cells
        if not current_cells:
            break

    return current_cells


def discretize_region_radial(
    region: Region,
    radius_thresholds: List[float],
    n_splits: List[int],
    n_subdivide: int = 20
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize a region with adaptive refinement based on rectangular (L∞) distance from origin.

    For each macro-cell (a sub-rectangle of the region), define:
      closest = clamp(0 into each interval)
      rect_dist = max_i |closest_i|
    """
    assert len(n_splits) == len(radius_thresholds) + 1, \
        f"n_splits must have {len(radius_thresholds) + 1} elements"

    bounds = region.bounds  # (D, 2)
    D = bounds.shape[0]

    edges = [
        np.linspace(bounds[d, 0], bounds[d, 1], n_subdivide + 1, dtype=np.float32)
        for d in range(D)
    ]

    all_cells: List[Tuple[torch.Tensor, torch.Tensor]] = []
    zone_counts = {i: 0 for i in range(len(n_splits))}

    for idx in product(range(n_subdivide), repeat=D):
        sub_bounds = np.array(
            [[edges[d][idx[d]], edges[d][idx[d] + 1]] for d in range(D)],
            dtype=np.float32
        )
        sub_rect = Region(sub_bounds)

        # Closest point to origin per dimension: clamp 0 into [low, high]
        closest = np.clip(0.0, sub_rect.bounds[:, 0], sub_rect.bounds[:, 1])  # (D,)
        rect_dist = float(np.max(np.abs(closest)))  # L∞ distance

        # Determine n_split based on thresholds
        n_split = n_splits[-1]
        zone_idx = len(n_splits) - 1
        for k, threshold in enumerate(radius_thresholds):
            if rect_dist < threshold:
                n_split = n_splits[k]
                zone_idx = k
                break

        zone_counts[zone_idx] += 1
        all_cells.extend(discretize_region(sub_rect, n_split))

    total_macro = n_subdivide ** D
    print(f"  Zone distribution across {n_subdivide}^{D} = {total_macro} macro-cells:")
    for zone_idx, count in zone_counts.items():
        if zone_idx < len(radius_thresholds):
            print(f"    Zone {zone_idx} (dist < {radius_thresholds[zone_idx]}): {count} macro-cells with {n_splits[zone_idx]}^{D} subdivision")
        else:
            print(f"    Zone {zone_idx} (dist >= {radius_thresholds[-1]}): {count} macro-cells with {n_splits[zone_idx]}^{D} subdivision")

    return all_cells


def refine_cells_by_mask(
    cells: List[Tuple[torch.Tensor, torch.Tensor]],
    failing_mask: torch.Tensor,
    refinement_factor: int = 2
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Refine cells marked as failing.

    For D dims, each refined cell becomes refinement_factor^D subcells.
    """
    new_cells = []
    num_refined = 0

    D = int(cells[0][0].numel()) if len(cells) > 0 else 0

    for i, (cell_lower, cell_upper) in enumerate(cells):
        if failing_mask[i]:
            lower_np = cell_lower.detach().cpu().numpy().astype(np.float32)
            upper_np = cell_upper.detach().cpu().numpy().astype(np.float32)

            cell_region = Region(np.stack([lower_np, upper_np], axis=1))  # (D, 2)

            refined = discretize_region(cell_region, refinement_factor)
            new_cells.extend(refined)
            num_refined += 1
        else:
            new_cells.append((cell_lower, cell_upper))

    print(f"    Refined {num_refined} failing cells into {num_refined * (refinement_factor ** D)} subcells")
    return new_cells


def compute_rectangular_partition_outside_goal(
    full_region: Region,
    goal_region: Region
) -> List[Region]:
    """
    Compute rectangles covering full_region \\ goal_region (supports D dims).
    """
    full_bounds = full_region.bounds
    goal_bounds = goal_region.bounds
    D = full_bounds.shape[0]

    # Collect boundaries per dimension
    boundaries = [
        sorted(set([full_bounds[d, 0], full_bounds[d, 1], goal_bounds[d, 0], goal_bounds[d, 1]]))
        for d in range(D)
    ]

    rectangles: List[Region] = []
    index_ranges = [range(len(boundaries[d]) - 1) for d in range(D)]

    for idx in product(*index_ranges):
        rect_bounds = np.array(
            [[boundaries[d][idx[d]], boundaries[d][idx[d] + 1]] for d in range(D)],
            dtype=np.float32
        )
        center = rect_bounds.mean(axis=1)  # (D,)
        if not goal_region.contains(center):
            rectangles.append(Region(rect_bounds))

    return rectangles


def compute_rectangular_partition_outside_goal_and_unsafe(
    full_region: Region,
    goal_region: Region,
    unsafe_region: Region
) -> List[Region]:
    """
    Compute rectangles covering full_region \\ (goal_region ∪ unsafe_region) (supports D dims).

    Handles union unsafe regions properly.
    """
    full_bounds = full_region.bounds
    goal_bounds = goal_region.bounds
    D = full_bounds.shape[0]

    # Collect boundaries per dimension from full and goal
    boundaries = [set() for _ in range(D)]
    for d in range(D):
        boundaries[d].update([full_bounds[d, 0], full_bounds[d, 1], goal_bounds[d, 0], goal_bounds[d, 1]])

    # Add boundaries from unsafe region (handle union)
    unsafe_components = unsafe_region.components if unsafe_region.is_union else [unsafe_region]
    for comp in unsafe_components:
        for d in range(D):
            boundaries[d].update([comp.bounds[d, 0], comp.bounds[d, 1]])

    boundaries = [sorted(b) for b in boundaries]

    rectangles: List[Region] = []
    index_ranges = [range(len(boundaries[d]) - 1) for d in range(D)]

    for idx in product(*index_ranges):
        rect_bounds = np.array(
            [[boundaries[d][idx[d]], boundaries[d][idx[d] + 1]] for d in range(D)],
            dtype=np.float32
        )
        center = rect_bounds.mean(axis=1)  # (D,)
        if not goal_region.contains(center) and not unsafe_region.contains(center):
            rectangles.append(Region(rect_bounds))

    return rectangles

def discretize_regions(regions, discretization_config, use_radial_generator=True):
    """
    Discretize all regions according to configuration.

    NOTE: For D>2, cell counts grow as n^D. Keep n_* small.
    """
    print("\n" + "="*20)
    print("Region Discretization")
    print("="*20)

    region_cells = {}
    D = int(regions.full.bounds.shape[0])
    axis_weights_cfg = getattr(discretization_config, "axis_weights", None)
    use_weights = axis_weights_cfg is not None and len(axis_weights_cfg) > 0
    if use_weights:
        axis_weights = np.asarray(axis_weights_cfg, dtype=np.float64).reshape(-1)
        if axis_weights.shape[0] != D:
            raise ValueError(
                f"discretization_config.axis_weights must have length {D}, got {axis_weights.shape[0]}"
            )
        max_region_budget = int(getattr(discretization_config, "max_region_budget", 0))
        max_generator_budget = int(getattr(discretization_config, "max_generator_budget", 0))
        if max_region_budget <= 0:
            raise ValueError("axis_weights mode requires max_region_budget > 0")
        if max_generator_budget <= 0:
            raise ValueError("axis_weights mode requires max_generator_budget > 0")

        print(f"Using weights-based discretization with axis_weights={axis_weights.tolist()}")
        print(f"Shared region budget (init/goal/unsafe/outside): {max_region_budget}")
        print(f"Generator budget: {max_generator_budget}")

        # V-related regions (near-equal by shared budget and same weighting policy)
        print("Init region: weights-based discretization")
        region_cells['init'] = discretize_region_list_with_weights_budget(
            [regions.init], axis_weights, max_region_budget, label="Init"
        )
        print(f" {len(region_cells['init'])} cells")

        print("Goal region: weights-based discretization")
        region_cells['goal'] = discretize_region_list_with_weights_budget(
            [regions.goal], axis_weights, max_region_budget, label="Goal"
        )
        print(f" {len(region_cells['goal'])} cells")

        print("Unsafe region: weights-based discretization")
        region_cells['unsafe'] = discretize_region_list_with_weights_budget(
            [regions.unsafe], axis_weights, max_region_budget, label="Unsafe"
        )
        print(f" {len(region_cells['unsafe'])} cells")

        outside_goal_rects = compute_rectangular_partition_outside_goal(regions.full, regions.goal)
        print("Outside goal region: weights-based discretization")
        region_cells['outside'] = discretize_region_list_with_weights_budget(
            outside_goal_rects, axis_weights, max_region_budget, label="Outside"
        )
        print(f" {len(region_cells['outside'])} cells")

        generator_rects = compute_rectangular_partition_outside_goal_and_unsafe(
            regions.full, regions.goal, regions.unsafe
        )
        print("Generator region: weights-based discretization")
        region_cells['generator'] = discretize_region_list_with_weights_budget(
            generator_rects, axis_weights, max_generator_budget, label="Generator"
        )
        print(f" {len(region_cells['generator'])} cells")
        return region_cells

    unit_lengths_cfg = getattr(discretization_config, "unit_lengths", None)
    use_unit_lengths = unit_lengths_cfg is not None and len(unit_lengths_cfg) > 0
    if use_unit_lengths:
        unit_lengths = np.asarray(unit_lengths_cfg, dtype=np.float64).reshape(-1)
        if unit_lengths.shape[0] != D:
            raise ValueError(
                f"discretization_config.unit_lengths must have length {D}, got {unit_lengths.shape[0]}"
            )
        print(f"Using unit-length discretization with unit_lengths={unit_lengths.tolist()}")
    else:
        unit_lengths = None

    # Init region
    n_init_budget = int(getattr(discretization_config, "n_init_budget", 0))
    if use_unit_lengths:
        print(f"Init region: unit-length discretization")
        region_cells['init'] = discretize_region_list_with_unit_lengths(
            [regions.init], unit_lengths, max_budget=n_init_budget, label="Init"
        )
    elif n_init_budget > 0:
        print(f"Init region: budget-based discretization (budget={n_init_budget})")
        region_cells['init'] = discretize_region_list_with_budget([regions.init], n_init_budget, label="Init")
    else:
        print(f"Init region: {discretization_config.n_init} per-dim grid")
        region_cells['init'] = discretize_region(regions.init, discretization_config.n_init)
    print(f" {len(region_cells['init'])} cells")

    # Goal region
    n_goal_budget = int(getattr(discretization_config, "n_goal_budget", 0))
    if use_unit_lengths:
        print(f"Goal region: unit-length discretization")
        region_cells['goal'] = discretize_region_list_with_unit_lengths(
            [regions.goal], unit_lengths, max_budget=n_goal_budget, label="Goal"
        )
    elif n_goal_budget > 0:
        print(f"Goal region: budget-based discretization (budget={n_goal_budget})")
        region_cells['goal'] = discretize_region_list_with_budget([regions.goal], n_goal_budget, label="Goal")
    else:
        print(f"Goal region: {discretization_config.n_goal} per-dim grid")
        region_cells['goal'] = discretize_region(regions.goal, discretization_config.n_goal)
    print(f" {len(region_cells['goal'])} cells")

    # Unsafe region
    n_unsafe_budget = getattr(discretization_config, "n_unsafe_budget", 0)
    if use_unit_lengths:
        print(f"Unsafe region: unit-length discretization")
        region_cells['unsafe'] = discretize_region_list_with_unit_lengths(
            [regions.unsafe], unit_lengths, max_budget=int(n_unsafe_budget), label="Unsafe"
        )
    elif int(n_unsafe_budget) > 0:
        print(f"Unsafe region: budget-based discretization (budget={int(n_unsafe_budget)})")
        region_cells['unsafe'] = discretize_unsafe_with_budget(regions.unsafe, int(n_unsafe_budget))
    else:
        print(f"Unsafe region: {discretization_config.n_unsafe} per-dim grid")
        region_cells['unsafe'] = discretize_region(regions.unsafe, discretization_config.n_unsafe)
    print(f" {len(region_cells['unsafe'])} cells")

    # Outside goal (for V constraint)
    n_outside_budget = int(getattr(discretization_config, "n_outside_goal_budget", 0))
    outside_goal_rects = compute_rectangular_partition_outside_goal(regions.full, regions.goal)
    if use_unit_lengths:
        print(f"Outside goal region: unit-length discretization")
        region_cells['outside'] = discretize_region_list_with_unit_lengths(
            outside_goal_rects, unit_lengths, max_budget=n_outside_budget, label="Outside"
        )
    elif n_outside_budget > 0:
        print(f"Outside goal region: budget-based discretization (budget={n_outside_budget})")
        region_cells['outside'] = discretize_region_list_with_budget(
            outside_goal_rects, n_outside_budget, label="Outside"
        )
    else:
        print(f"Outside goal region: {discretization_config.n_outside_goal} per-dim per rectangle")
        # Pre-allocate: each rectangle -> n_outside_goal^D cells
        n_per_rect = discretization_config.n_outside_goal ** regions.full.bounds.shape[0]
        total_outside = n_per_rect * len(outside_goal_rects)
        region_cells['outside'] = [None] * total_outside
        idx = 0
        for rect in outside_goal_rects:
            for cell in discretize_region(rect, discretization_config.n_outside_goal):
                region_cells['outside'][idx] = cell
                idx += 1
        region_cells['outside'] = region_cells['outside'][:idx]
    print(f" {len(region_cells['outside'])} cells")

    # Generator region (outside goal and unsafe)
    if use_radial_generator and (not use_unit_lengths):
        print(f"Generator region: Radial adaptive discretization with clipping")
        print(f" Using adaptive refinement based on distance from origin")

        RADIUS_THRESHOLDS = [25.0, 40.0]
        N_SPLITS = [2, 1, 1]

        all_cells = discretize_region_radial(
            regions.full,
            radius_thresholds=RADIUS_THRESHOLDS,
            n_splits=N_SPLITS,
            n_subdivide=6
        )

        print(f' Clipping {len(all_cells)} generator cells against goal and unsafe regions...')

        exclusion_regions = [regions.goal.bounds]
        if regions.unsafe.is_union:
            for comp in regions.unsafe.components:
                exclusion_regions.append(comp.bounds)
            print(f' Excluding goal + {len(regions.unsafe.components)} unsafe components')
        else:
            exclusion_regions.append(regions.unsafe.bounds)
            print(f' Excluding goal + 1 unsafe region')

        # Pre-allocate with upper bound (clipping can increase cell count)
        max_clipped = len(all_cells) * (2 ** regions.full.bounds.shape[0])  # worst case: each cell splits
        clipped_cells = [None] * max_clipped
        idx = 0

        for cell_lower, cell_upper in all_cells:
            lower_np = cell_lower.numpy() if isinstance(cell_lower, torch.Tensor) else cell_lower
            upper_np = cell_upper.numpy() if isinstance(cell_upper, torch.Tensor) else cell_upper

            result_cells = clip_cell_against_exclusions(lower_np, upper_np, exclusion_regions)

            for lower, upper in result_cells:
                clipped_cells[idx] = (
                    torch.tensor(lower, dtype=torch.float32),
                    torch.tensor(upper, dtype=torch.float32)
                )
                idx += 1

        region_cells['generator'] = clipped_cells[:idx]
        print(f' After clipping: {len(region_cells["generator"])} cells (may have increased due to cell splitting)')
        if len(region_cells['generator']) > 0:
            first_cell = region_cells['generator'][0]
            print(f' First cell bounds: {first_cell[0].numpy()} to {first_cell[1].numpy()}')
    else:
        n_gen_budget = int(getattr(discretization_config, "n_generator_budget", 0))
        generator_rects = compute_rectangular_partition_outside_goal_and_unsafe(
            regions.full, regions.goal, regions.unsafe
        )
        if use_unit_lengths:
            print(f"Generator region: unit-length discretization")
            region_cells['generator'] = discretize_region_list_with_unit_lengths(
                generator_rects, unit_lengths, max_budget=n_gen_budget, label="Generator"
            )
        elif n_gen_budget > 0:
            print(f"Generator region: budget-based discretization (budget={n_gen_budget})")
            region_cells['generator'] = discretize_region_list_with_budget(
                generator_rects, n_gen_budget, label="Generator"
            )
        else:
            print(f"Generator region: {discretization_config.n_generator} per-dim per rectangle")
            # Pre-allocate: each rectangle -> n_generator^D cells
            n_per_rect = discretization_config.n_generator ** regions.full.bounds.shape[0]
            total_gen = n_per_rect * len(generator_rects)
            region_cells['generator'] = [None] * total_gen
            idx = 0
            for rect in generator_rects:
                for cell in discretize_region(rect, discretization_config.n_generator):
                    region_cells['generator'][idx] = cell
                    idx += 1
            region_cells['generator'] = region_cells['generator'][:idx]
        print(f" {len(region_cells['generator'])} cells")
        
    return region_cells
