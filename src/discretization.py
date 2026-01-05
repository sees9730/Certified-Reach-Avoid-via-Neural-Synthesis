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


def create_boundary_cells(
    full_region: Region,
    thickness: float = 0.1,
    n_partitions: int = 1,
    exclusion_regions: List[Region] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create thin border cells around the boundary of full_region.

    For D dimensions, creates 2*D thin slabs (one for each face).
    Each slab has thickness `thickness` along one dimension and can be
    subdivided into n_partitions^(D-1) smaller cells.

    Args:
        full_region: The full bounding region
        thickness: Thickness of the boundary cells
        n_partitions: Number of subdivisions per dimension (except the thickness dimension)
                     If n_partitions=1, creates single slab per face (original behavior)
                     If n_partitions>1, subdivides each slab into n_partitions^(D-1) cells
        exclusion_regions: Optional list of regions to exclude (e.g., unsafe)

    Returns:
        List of (lower, upper) boundary cell tuples
    """
    bounds = full_region.bounds  # (D, 2)
    D = bounds.shape[0]

    boundary_cells = []

    # Create 2*D slabs (min and max face for each dimension)
    for d in range(D):
        # Lower face along dimension d
        lower_slab_lower = bounds[:, 0].copy()
        lower_slab_upper = bounds[:, 1].copy()
        lower_slab_upper[d] = bounds[d, 0] + thickness

        # Upper face along dimension d
        upper_slab_lower = bounds[:, 0].copy()
        upper_slab_upper = bounds[:, 1].copy()
        upper_slab_lower[d] = bounds[d, 1] - thickness

        # Subdivide each slab if n_partitions > 1
        if n_partitions > 1:
            # Create partitions for all dimensions except d
            lower_slab_cells = _subdivide_slab(lower_slab_lower, lower_slab_upper, d, n_partitions)
            upper_slab_cells = _subdivide_slab(upper_slab_lower, upper_slab_upper, d, n_partitions)
        else:
            # Single cell per slab (original behavior)
            lower_slab_cells = [(lower_slab_lower, lower_slab_upper)]
            upper_slab_cells = [(upper_slab_lower, upper_slab_upper)]

        # Process each subdivided cell (clip against exclusions if needed)
        for cells in [lower_slab_cells, upper_slab_cells]:
            for cell_lower, cell_upper in cells:
                if exclusion_regions is not None:
                    excl_bounds_list = []
                    for excl_region in exclusion_regions:
                        if excl_region.is_union:
                            for comp in excl_region.components:
                                excl_bounds_list.append(comp.bounds)
                        else:
                            excl_bounds_list.append(excl_region.bounds)

                    # Clip cell against exclusions
                    clipped_cells = clip_cell_against_exclusions(
                        cell_lower, cell_upper, excl_bounds_list
                    )
                    for lower, upper in clipped_cells:
                        boundary_cells.append((
                            torch.tensor(lower, dtype=torch.float32),
                            torch.tensor(upper, dtype=torch.float32)
                        ))
                else:
                    boundary_cells.append((
                        torch.tensor(cell_lower, dtype=torch.float32),
                        torch.tensor(cell_upper, dtype=torch.float32)
                    ))

    return boundary_cells


def _subdivide_slab(
    slab_lower: np.ndarray,
    slab_upper: np.ndarray,
    fixed_dim: int,
    n_partitions: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Subdivide a slab into n_partitions^(D-1) smaller cells.

    The slab is thin along dimension `fixed_dim` and is subdivided
    along all other dimensions.

    Args:
        slab_lower: Lower corner of slab (D,)
        slab_upper: Upper corner of slab (D,)
        fixed_dim: Dimension that is thin (not subdivided)
        n_partitions: Number of partitions per free dimension

    Returns:
        List of (lower, upper) tuples for subdivided cells
    """
    D = len(slab_lower)

    # Get all dimensions except the fixed one
    free_dims = [i for i in range(D) if i != fixed_dim]

    # Create edge points for each free dimension
    edges = {}
    for dim in free_dims:
        edges[dim] = np.linspace(slab_lower[dim], slab_upper[dim], n_partitions + 1, dtype=np.float32)

    # Generate all combinations of subdivisions
    cells = []
    from itertools import product

    # Create index ranges for each free dimension
    index_ranges = [range(n_partitions) for _ in free_dims]

    for indices in product(*index_ranges):
        cell_lower = slab_lower.copy()
        cell_upper = slab_upper.copy()

        # Set bounds for each free dimension
        for i, dim in enumerate(free_dims):
            idx = indices[i]
            cell_lower[dim] = edges[dim][idx]
            cell_upper[dim] = edges[dim][idx + 1]

        cells.append((cell_lower, cell_upper))

    return cells


def discretize_regions(regions, discretization_config, use_radial_generator=True):
    """
    Discretize all regions according to configuration.

    NOTE: For D>2, cell counts grow as n^D. Keep n_* small.
    """
    print("\n" + "="*80)
    print("DISCRETIZING REGIONS")
    print("="*80)

    region_cells = {}

    # Init region
    print(f"\nInit region: {discretization_config.n_init} per-dim grid")
    region_cells['init'] = discretize_region(regions.init, discretization_config.n_init)
    print(f"  → {len(region_cells['init'])} cells")

    # Goal region
    print(f"\nGoal region: {discretization_config.n_goal} per-dim grid")
    region_cells['goal'] = discretize_region(regions.goal, discretization_config.n_goal)
    print(f"  → {len(region_cells['goal'])} cells")

    # Unsafe region
    print(f"\nUnsafe region: {discretization_config.n_unsafe} per-dim grid")
    region_cells['unsafe'] = discretize_region(regions.unsafe, discretization_config.n_unsafe)
    print(f"  → {len(region_cells['unsafe'])} cells")

    # Outside goal (for V constraint)
    print(f"\nOutside goal region: {discretization_config.n_outside_goal} per-dim per rectangle")
    outside_goal_rects = compute_rectangular_partition_outside_goal(regions.full, regions.goal)
    region_cells['outside'] = []
    for rect in outside_goal_rects:
        region_cells['outside'].extend(discretize_region(rect, discretization_config.n_outside_goal))
    print(f"  → {len(region_cells['outside'])} cells")

    # Generator region (outside goal and unsafe)
    if use_radial_generator:
        print(f"\nGenerator region: Radial adaptive discretization with clipping")
        print(f"  Using adaptive refinement based on distance from origin")

        RADIUS_THRESHOLDS = [25.0, 40.0]
        N_SPLITS = [2, 1, 1]

        all_cells = discretize_region_radial(
            regions.full,
            radius_thresholds=RADIUS_THRESHOLDS,
            n_splits=N_SPLITS,
            n_subdivide=6
        )

        print(f'  Clipping {len(all_cells)} generator cells against goal and unsafe regions...')

        exclusion_regions = [regions.goal.bounds]
        if regions.unsafe.is_union:
            for comp in regions.unsafe.components:
                exclusion_regions.append(comp.bounds)
            print(f'  Excluding goal + {len(regions.unsafe.components)} unsafe components')
        else:
            exclusion_regions.append(regions.unsafe.bounds)
            print(f'  Excluding goal + 1 unsafe region')

        clipped_cells = []
        for cell_lower, cell_upper in all_cells:
            lower_np = cell_lower.numpy() if isinstance(cell_lower, torch.Tensor) else cell_lower
            upper_np = cell_upper.numpy() if isinstance(cell_upper, torch.Tensor) else cell_upper

            result_cells = clip_cell_against_exclusions(lower_np, upper_np, exclusion_regions)

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
        print(f"\nGenerator region: {discretization_config.n_generator} per-dim per rectangle")
        generator_rects = compute_rectangular_partition_outside_goal_and_unsafe(
            regions.full, regions.goal, regions.unsafe
        )
        region_cells['generator'] = []
        for rect in generator_rects:
            region_cells['generator'].extend(discretize_region(rect, discretization_config.n_generator))
        print(f"  → {len(region_cells['generator'])} cells")

    # Boundary region (thin slabs on full_range boundary, excluding unsafe)
    print(f"\nBoundary region: Thin slabs on full_range boundary")
    boundary_thickness = 0.5  # Adjust thickness as needed
    boundary_n_partitions = 10  # Number of subdivisions per dimension (1 = single slab per face)
    region_cells['boundary'] = create_boundary_cells(
        regions.full,
        thickness=boundary_thickness,
        n_partitions=boundary_n_partitions#,
        # exclusion_regions=[regions.unsafe]  # Exclude unsafe region
    )
    D = regions.full.bounds.shape[0]
    expected_cells_per_face = boundary_n_partitions ** (D - 1)
    print(f"  → {len(region_cells['boundary'])} boundary cells "
          f"(thickness={boundary_thickness}, {boundary_n_partitions} partitions/dim, "
          f"~{expected_cells_per_face} cells/face × {2*D} faces)")
    if len(region_cells['boundary']) > 0:
        first_cell = region_cells['boundary'][0]
        print(f'  First boundary cell: {first_cell[0].numpy()} to {first_cell[1].numpy()}')

    print(f"\nRegion definitions:")
    print(f"  'outside': X \\ Goal (for V ≥ β_s constraint)")
    print(f"  'generator': X \\ (Goal ∪ Unsafe) (for Φ ≤ 0 constraint)")
    print(f"  'boundary': Thin slabs on ∂X \\ Unsafe (for V < 1.0 constraint)")

    return region_cells
