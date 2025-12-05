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

from regions import Region


def discretize_region(region: Region, n_squares: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize a region into n_squares x n_squares grid cells.

    For union regions, discretizes each component separately and combines.

    Args:
        region: Region to discretize
        n_squares: Number of subdivisions per dimension

    Returns:
        List of (lower, upper) tuples representing cells
    """
    if region.is_union:
        # Discretize each component and combine
        all_cells = []
        for comp in region.components:
            comp_cells = discretize_region(comp, n_squares)
            all_cells.extend(comp_cells)
        return all_cells
    else:
        # Single rectangle - standard grid discretization
        bounds = region.bounds
        x1_edges = np.linspace(bounds[0, 0], bounds[0, 1], n_squares + 1)
        x2_edges = np.linspace(bounds[1, 0], bounds[1, 1], n_squares + 1)

        cells = []
        for i in range(n_squares):
            for j in range(n_squares):
                cell_lower = np.array([x1_edges[i], x2_edges[j]], dtype=np.float32)
                cell_upper = np.array([x1_edges[i+1], x2_edges[j+1]], dtype=np.float32)
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
    Subtract exclusion rectangle from cell rectangle.
    Returns list of rectangles representing cell \ exclusion.

    Args:
        cell_lower: Lower corner of cell
        cell_upper: Upper corner of cell
        excl_lower: Lower corner of exclusion
        excl_upper: Upper corner of exclusion

    Returns:
        List of (lower, upper) tuples representing the non-overlapping parts
    """
    # Extract bounds
    x1_min, x2_min = cell_lower
    x1_max, x2_max = cell_upper
    ex1_min, ex2_min = excl_lower
    ex1_max, ex2_max = excl_upper

    # Check for no overlap
    if (x1_max <= ex1_min or x1_min >= ex1_max or
        x2_max <= ex2_min or x2_min >= ex2_max):
        # No overlap - return cell as-is
        return [(cell_lower, cell_upper)]

    # Check for complete containment (cell entirely inside exclusion)
    if (ex1_min <= x1_min and x1_max <= ex1_max and
        ex2_min <= x2_min and x2_max <= ex2_max):
        # Cell completely inside exclusion - return empty list
        return []

    # Partial overlap - decompose into up to 4 rectangles around the exclusion
    result = []

    # Bottom strip: below the exclusion zone
    if x2_min < ex2_min:
        result.append((
            np.array([x1_min, x2_min], dtype=np.float32),
            np.array([x1_max, min(x2_max, ex2_min)], dtype=np.float32)
        ))

    # Top strip: above the exclusion zone
    if x2_max > ex2_max:
        result.append((
            np.array([x1_min, max(x2_min, ex2_max)], dtype=np.float32),
            np.array([x1_max, x2_max], dtype=np.float32)
        ))

    # Middle vertical range (overlaps with exclusion in x2 direction)
    y_mid_min = max(x2_min, ex2_min)
    y_mid_max = min(x2_max, ex2_max)

    # Left strip: left of exclusion zone, within middle vertical range
    if x1_min < ex1_min:
        result.append((
            np.array([x1_min, y_mid_min], dtype=np.float32),
            np.array([min(x1_max, ex1_min), y_mid_max], dtype=np.float32)
        ))

    # Right strip: right of exclusion zone, within middle vertical range
    if x1_max > ex1_max:
        result.append((
            np.array([max(x1_min, ex1_max), y_mid_min], dtype=np.float32),
            np.array([x1_max, y_mid_max], dtype=np.float32)
        ))

    return result


def clip_cell_against_exclusions(
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    exclusion_regions: List[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Clip a cell against multiple exclusion regions.
    Returns list of cells with all exclusions removed.

    Args:
        cell_lower: Lower corner of cell
        cell_upper: Upper corner of cell
        exclusion_regions: List of exclusion region bounds

    Returns:
        List of (lower, upper) cells with exclusions removed
    """
    current_cells = [(cell_lower, cell_upper)]

    for excl_region in exclusion_regions:
        excl_lower = excl_region[:, 0]
        excl_upper = excl_region[:, 1]

        new_cells = []
        for lower, upper in current_cells:
            # Subtract this exclusion from each current cell
            new_cells.extend(subtract_rectangle(lower, upper, excl_lower, excl_upper))

        current_cells = new_cells
        if not current_cells:  # All removed
            break

    return current_cells


def discretize_region_radial(
    region: Region,
    radius_thresholds: List[float],
    n_splits: List[int],
    n_subdivide: int = 20
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize a region with adaptive refinement based on rectangular distance from origin.

    Args:
        region: Region to discretize
        radius_thresholds: List of distance thresholds, e.g., [30.0, 70.0]
        n_splits: List of n_split values for each zone, e.g., [10, 30, 100]
        n_subdivide: Number of subdivisions for radial classification (default: 20)

    Returns:
        List of (lower, upper) cell tuples
    """
    assert len(n_splits) == len(radius_thresholds) + 1, \
        f"n_splits must have {len(radius_thresholds) + 1} elements"

    bounds = region.bounds
    x1_edges = np.linspace(bounds[0, 0], bounds[0, 1], n_subdivide + 1)
    x2_edges = np.linspace(bounds[1, 0], bounds[1, 1], n_subdivide + 1)

    all_cells = []
    zone_counts = {i: 0 for i in range(len(n_splits))}

    for i in range(n_subdivide):
        for j in range(n_subdivide):
            sub_rect = Region(np.array([
                [x1_edges[i], x1_edges[i+1]],
                [x2_edges[j], x2_edges[j+1]]
            ], dtype=np.float32))

            # Rectangular distance: max(|x|, |y|) of closest point to origin
            # Closest point to origin: clamp origin coordinates to rectangle bounds
            x1_closest = np.clip(0.0, sub_rect.bounds[0, 0], sub_rect.bounds[0, 1])
            x2_closest = np.clip(0.0, sub_rect.bounds[1, 0], sub_rect.bounds[1, 1])
            rect_dist = max(abs(x1_closest), abs(x2_closest))

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

    print(f"  Zone distribution across {n_subdivide}x{n_subdivide} = {n_subdivide**2} macro-cells:")
    for zone_idx, count in zone_counts.items():
        if zone_idx < len(radius_thresholds):
            print(f"    Zone {zone_idx} (dist < {radius_thresholds[zone_idx]}): {count} macro-cells with {n_splits[zone_idx]}x{n_splits[zone_idx]} subdivision")
        else:
            print(f"    Zone {zone_idx} (dist >= {radius_thresholds[-1]}): {count} macro-cells with {n_splits[zone_idx]}x{n_splits[zone_idx]} subdivision")

    return all_cells


def refine_cells_by_mask(
    cells: List[Tuple[torch.Tensor, torch.Tensor]],
    failing_mask: torch.Tensor,
    refinement_factor: int = 2
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Refine cells marked as failing.

    Args:
        cells: List of (lower, upper) tuples
        failing_mask: Boolean tensor indicating which cells are failing
        refinement_factor: Split each failing cell into refinement_factor^2 subcells

    Returns:
        List of refined cells
    """
    new_cells = []
    num_refined = 0

    for i, (cell_lower, cell_upper) in enumerate(cells):
        if failing_mask[i]:
            # Refine this failing cell
            cell_region = Region(np.array([
                [cell_lower[0].item(), cell_upper[0].item()],
                [cell_lower[1].item(), cell_upper[1].item()]
            ], dtype=np.float32))

            refined = discretize_region(cell_region, refinement_factor)
            new_cells.extend(refined)
            num_refined += 1
        else:
            # Keep original cell
            new_cells.append((cell_lower, cell_upper))

    print(f"    Refined {num_refined} failing cells into {num_refined * refinement_factor**2} subcells")
    return new_cells


def compute_rectangular_partition_outside_goal(
    full_region: Region,
    goal_region: Region
) -> List[Region]:
    """
    Compute rectangles covering full_region \ goal_region.

    Args:
        full_region: Full region
        goal_region: Goal region to exclude

    Returns:
        List of Region objects covering the complement
    """
    full_bounds = full_region.bounds
    goal_bounds = goal_region.bounds

    # Collect boundaries
    x_boundaries = sorted(set([
        full_bounds[0, 0], full_bounds[0, 1],
        goal_bounds[0, 0], goal_bounds[0, 1]
    ]))
    y_boundaries = sorted(set([
        full_bounds[1, 0], full_bounds[1, 1],
        goal_bounds[1, 0], goal_bounds[1, 1]
    ]))

    rectangles = []
    for i in range(len(x_boundaries) - 1):
        for j in range(len(y_boundaries) - 1):
            rect_bounds = np.array([
                [x_boundaries[i], x_boundaries[i+1]],
                [y_boundaries[j], y_boundaries[j+1]]
            ], dtype=np.float32)

            # Check if center is inside goal
            center_x = (rect_bounds[0, 0] + rect_bounds[0, 1]) / 2
            center_y = (rect_bounds[1, 0] + rect_bounds[1, 1]) / 2

            center = np.array([center_x, center_y])
            if not goal_region.contains(center):
                rectangles.append(Region(rect_bounds))

    return rectangles


def compute_rectangular_partition_outside_goal_and_unsafe(
    full_region: Region,
    goal_region: Region,
    unsafe_region: Region
) -> List[Region]:
    """
    Compute rectangles covering full_region \ (goal_region ∪ unsafe_region).

    Handles union unsafe regions properly.

    Args:
        full_region: Full region
        goal_region: Goal region to exclude
        unsafe_region: Unsafe region to exclude (can be a union)

    Returns:
        List of Region objects covering the complement
    """
    full_bounds = full_region.bounds
    goal_bounds = goal_region.bounds

    # Collect boundaries from full and goal
    x_boundaries = set([
        full_bounds[0, 0], full_bounds[0, 1],
        goal_bounds[0, 0], goal_bounds[0, 1]
    ])
    y_boundaries = set([
        full_bounds[1, 0], full_bounds[1, 1],
        goal_bounds[1, 0], goal_bounds[1, 1]
    ])

    # Add boundaries from unsafe region (handle union case)
    if unsafe_region.is_union:
        # Add boundaries from each component
        for comp in unsafe_region.components:
            x_boundaries.add(comp.bounds[0, 0])
            x_boundaries.add(comp.bounds[0, 1])
            y_boundaries.add(comp.bounds[1, 0])
            y_boundaries.add(comp.bounds[1, 1])
    else:
        # Single rectangle
        unsafe_bounds = unsafe_region.bounds
        x_boundaries.add(unsafe_bounds[0, 0])
        x_boundaries.add(unsafe_bounds[0, 1])
        y_boundaries.add(unsafe_bounds[1, 0])
        y_boundaries.add(unsafe_bounds[1, 1])

    x_boundaries = sorted(x_boundaries)
    y_boundaries = sorted(y_boundaries)

    rectangles = []
    for i in range(len(x_boundaries) - 1):
        for j in range(len(y_boundaries) - 1):
            rect_bounds = np.array([
                [x_boundaries[i], x_boundaries[i+1]],
                [y_boundaries[j], y_boundaries[j+1]]
            ], dtype=np.float32)

            # Check if center is inside goal or unsafe
            center_x = (rect_bounds[0, 0] + rect_bounds[0, 1]) / 2
            center_y = (rect_bounds[1, 0] + rect_bounds[1, 1]) / 2

            center = np.array([center_x, center_y])
            # unsafe_region.contains() handles union regions automatically
            if not goal_region.contains(center) and not unsafe_region.contains(center):
                rectangles.append(Region(rect_bounds))

    return rectangles


def discretize_regions(regions, discretization_config, use_radial_generator=True):
    """
    Discretize all regions according to configuration.

    Args:
        regions: Regions object with init, goal, unsafe, full
        discretization_config: DiscretizationConfig
        use_radial_generator: If True, use radial adaptive discretization with clipping
                             for generator region (matches original testing_simple3.py)

    Returns:
        Dictionary with discretized cells for each region
    """
    print("\n" + "="*80)
    print("DISCRETIZING REGIONS")
    print("="*80)

    region_cells = {}

    # Init region
    print(f"\nInit region: {discretization_config.n_init}x{discretization_config.n_init} grid")
    region_cells['init'] = discretize_region(regions.init, discretization_config.n_init)
    print(f"  → {len(region_cells['init'])} cells")

    # Goal region
    print(f"\nGoal region: {discretization_config.n_goal}x{discretization_config.n_goal} grid")
    region_cells['goal'] = discretize_region(regions.goal, discretization_config.n_goal)
    print(f"  → {len(region_cells['goal'])} cells")

    # Unsafe region
    print(f"\nUnsafe region: {discretization_config.n_unsafe}x{discretization_config.n_unsafe} grid")
    region_cells['unsafe'] = discretize_region(regions.unsafe, discretization_config.n_unsafe)
    print(f"  → {len(region_cells['unsafe'])} cells")

    # Outside goal (for V constraint)
    print(f"\nOutside goal region: {discretization_config.n_outside_goal}x{discretization_config.n_outside_goal} per rectangle")
    outside_goal_rects = compute_rectangular_partition_outside_goal(regions.full, regions.goal)
    region_cells['outside'] = []
    for rect in outside_goal_rects:
        region_cells['outside'].extend(discretize_region(rect, discretization_config.n_outside_goal))
    print(f"  → {len(region_cells['outside'])} cells")

    # Generator region (outside goal and unsafe)
    if use_radial_generator:
        # Use radial adaptive discretization with clipping (matches original testing_simple3.py)
        print(f"\nGenerator region: Radial adaptive discretization with clipping")
        print(f"  Using adaptive refinement based on distance from origin")

        # Radial discretization parameters (from original testing_simple3.py)
        RADIUS_THRESHOLDS = [25.0, 40.0]
        N_SPLITS = [2, 1, 1]

        # Discretize full region with adaptive refinement
        all_cells = discretize_region_radial(
            regions.full,
            radius_thresholds=RADIUS_THRESHOLDS,
            n_splits=N_SPLITS,
            n_subdivide=6  # Original uses n_subdivide=1
        )

        # Clip cells to remove overlaps with goal and unsafe regions
        print(f'  Clipping {len(all_cells)} generator cells against goal and unsafe regions...')

        # Build exclusion list: goal + all unsafe components (for union regions)
        exclusion_regions = [regions.goal.bounds]
        if regions.unsafe.is_union:
            # Add each component of the union separately
            for comp in regions.unsafe.components:
                exclusion_regions.append(comp.bounds)
            print(f'  Excluding goal + {len(regions.unsafe.components)} unsafe components')
        else:
            # Single unsafe region
            exclusion_regions.append(regions.unsafe.bounds)
            print(f'  Excluding goal + 1 unsafe region')

        clipped_cells = []
        for cell_lower, cell_upper in all_cells:
            # Convert from torch tensors to numpy for clipping
            lower_np = cell_lower.numpy() if isinstance(cell_lower, torch.Tensor) else cell_lower
            upper_np = cell_upper.numpy() if isinstance(cell_upper, torch.Tensor) else cell_upper

            # Clip this cell against all exclusion regions
            result_cells = clip_cell_against_exclusions(lower_np, upper_np, exclusion_regions)

            # Convert back to torch tensors
            for lower, upper in result_cells:
                clipped_cells.append((
                    torch.tensor(lower, dtype=torch.float32),
                    torch.tensor(upper, dtype=torch.float32)
                ))

        region_cells['generator'] = clipped_cells
        print(f'  After clipping: {len(region_cells["generator"])} cells (may have increased due to cell splitting)')
        if len(region_cells['generator']) > 0:
            first_cell = region_cells['generator'][0]
            print(f'  First cell bounds: {first_cell[0].numpy()} to {first_cell[1].numpy()}')
    else:
        # Simple rectangular partitioning (simpler but different from original)
        print(f"\nGenerator region: {discretization_config.n_generator}x{discretization_config.n_generator} per rectangle")
        generator_rects = compute_rectangular_partition_outside_goal_and_unsafe(
            regions.full, regions.goal, regions.unsafe
        )
        region_cells['generator'] = []
        for rect in generator_rects:
            region_cells['generator'].extend(discretize_region(rect, discretization_config.n_generator))
        print(f"  → {len(region_cells['generator'])} cells")

    print(f"\nRegion definitions:")
    print(f"  'outside': X \\ Goal (for V ≥ β_s constraint)")
    print(f"  'generator': X \\ (Goal ∪ Unsafe) (for Φ ≤ 0 constraint)")

    return region_cells
