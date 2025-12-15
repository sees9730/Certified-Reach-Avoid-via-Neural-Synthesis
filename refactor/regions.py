"""
Regions module for defining spatial regions in the state space.

This module handles:
- Initial region (where the system starts)
- Goal region (target set)
- Unsafe region (avoid set)
- Full state space region
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class Region:
    """
    Rectangular region in 2D state space.

    Defined by bounds: [x1_min, x1_max] × [x2_min, x2_max]

    Can represent either:
    - A single rectangle (when bounds is a 2D array)
    - A union of rectangles (when is_union=True and components is a list)
    """
    bounds: np.ndarray  # Shape: (state_dim, 2) where (:, 0) is lower, (:, 1) is upper

    def __init__(self, bounds: np.ndarray, is_union: bool = False, components: Optional[list] = None):
        """
        Initialize region.

        Args:
            bounds: Array of shape (state_dim, 2) where:
                    bounds[i, 0] = lower bound for dimension i
                    bounds[i, 1] = upper bound for dimension i
                    For union regions, this is the bounding box of all components
            is_union: Whether this region is a union of multiple rectangles
            components: List of Region objects if is_union=True
        """
        self.bounds = np.array(bounds, dtype=np.float32)
        self.state_dim = self.bounds.shape[0]
        self.is_union = is_union
        self.components = components if components is not None else []

        # Validate
        # assert self.bounds.shape[1] == 2, "bounds must have shape (state_dim, 2)"
        assert np.all(self.bounds[:, 0] <= self.bounds[:, 1]), \
            "Lower bounds must be <= upper bounds"

        if is_union:
            assert len(self.components) > 0, "Union region must have at least one component"
            # Verify bounding box encompasses all components
            for comp in self.components:
                assert comp.state_dim == self.state_dim, "All components must have same state_dim"

    @property
    def lower(self) -> np.ndarray:
        """Get lower bounds (state_dim,)."""
        return self.bounds[:, 0]

    @property
    def upper(self) -> np.ndarray:
        """Get upper bounds (state_dim,)."""
        return self.bounds[:, 1]

    @property
    def center(self) -> np.ndarray:
        """Get center point (state_dim,)."""
        return (self.lower + self.upper) / 2

    @property
    def size(self) -> np.ndarray:
        """Get size along each dimension (state_dim,)."""
        return self.upper - self.lower

    def contains(self, x: Union[np.ndarray, torch.Tensor]) -> Union[bool, np.ndarray, torch.Tensor]:
        """
        Check if point(s) are inside region.

        For union regions, returns True if point is in ANY component.

        Args:
            x: Point or batch of points. Shape: (state_dim,) or (batch_size, state_dim)

        Returns:
            Boolean or boolean array indicating membership
        """
        if self.is_union:
            # Check membership in any component
            if x.ndim == 1:
                # Single point
                return any(comp.contains(x) for comp in self.components)
            else:
                # Batch of points - use OR across all components
                result = self.components[0].contains(x)
                for comp in self.components[1:]:
                    result = result | comp.contains(x)
                return result
        else:
            # Single rectangle - use standard bounds check
            if isinstance(x, torch.Tensor):
                lower = torch.from_numpy(self.lower).to(x.device)
                upper = torch.from_numpy(self.upper).to(x.device)
            else:
                lower = self.lower
                upper = self.upper

            if x.ndim == 1:
                # Single point
                return np.all((x >= lower) & (x <= upper)) if isinstance(x, np.ndarray) \
                       else torch.all((x >= lower) & (x <= upper))
            else:
                # Batch of points
                return np.all((x >= lower) & (x <= upper), axis=1) if isinstance(x, np.ndarray) \
                       else torch.all((x >= lower) & (x <= upper), dim=1)

    def to_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to torch tensors (lower, upper).

        Returns:
            (lower, upper) tensors of shape (state_dim,)
        """
        return (
            torch.tensor(self.lower, dtype=torch.float32),
            torch.tensor(self.upper, dtype=torch.float32)
        )

    def __repr__(self):
        """String representation."""
        if self.is_union:
            comp_strs = [str(comp) for comp in self.components]
            return f"UnionRegion({' ∪ '.join(comp_strs)})"
        else:
            ranges = [f"[{self.bounds[i, 0]:.2f}, {self.bounds[i, 1]:.2f}]"
                      for i in range(self.state_dim)]
            return f"Region({' × '.join(ranges)})"

    @classmethod
    def from_corners(cls, lower: np.ndarray, upper: np.ndarray):
        """
        Create region from corner points.

        Args:
            lower: Lower corner (state_dim,)
            upper: Upper corner (state_dim,)

        Returns:
            Region instance
        """
        lower = np.array(lower, dtype=np.float32)
        upper = np.array(upper, dtype=np.float32)
        bounds = np.stack([lower, upper], axis=1)
        return cls(bounds)

    @classmethod
    def union(cls, *regions):
        """
        Create a union of multiple rectangular regions.

        Args:
            *regions: Variable number of Region objects to union together

        Returns:
            Region instance representing the union

        Example:
            region1 = Region(bounds1)
            region2 = Region(bounds2)
            union_region = Region.union(region1, region2)
        """
        if len(regions) == 0:
            raise ValueError("Must provide at least one region")

        if len(regions) == 1:
            # Single region - no union needed
            return regions[0]

        # Compute bounding box of all regions
        all_lowers = np.stack([r.lower for r in regions])
        all_uppers = np.stack([r.upper for r in regions])

        bounding_lower = all_lowers.min(axis=0)
        bounding_upper = all_uppers.max(axis=0)
        bounding_bounds = np.stack([bounding_lower, bounding_upper], axis=1)

        # Create union region
        return cls(bounds=bounding_bounds, is_union=True, components=list(regions))


class Regions:
    """
    Collection of regions for the verification problem.

    Manages:
    - Initial region (where system starts)
    - Goal region (target set)
    - Unsafe region (avoid set)
    - Full state space region
    """

    def __init__(
        self,
        init: Region,
        goal: Region,
        unsafe: Region,
        full: Optional[Region] = None
    ):
        """
        Initialize regions.

        Args:
            init: Initial region
            goal: Goal region
            unsafe: Unsafe region
            full: Full state space region (optional, will be inferred if not provided)
        """
        self.init = init
        self.goal = goal
        self.unsafe = unsafe

        # Infer full region if not provided (bounding box of all regions)
        if full is None:
            all_bounds = np.stack([
                init.bounds,
                goal.bounds,
                unsafe.bounds
            ])  # (3, state_dim, 2)

            full_lower = all_bounds[:, :, 0].min(axis=0)  # (state_dim,)
            full_upper = all_bounds[:, :, 1].max(axis=0)  # (state_dim,)

            # Add some padding (10%)
            padding = 0.1 * (full_upper - full_lower)
            full_lower -= padding
            full_upper += padding

            self.full = Region.from_corners(full_lower, full_upper)
        else:
            self.full = full

        self.state_dim = self.init.state_dim

    def get_region(self, name: str) -> Region:
        """
        Get region by name.

        Args:
            name: One of 'init', 'goal', 'unsafe', 'full'

        Returns:
            Region instance
        """
        region_map = {
            'init': self.init,
            'goal': self.goal,
            'unsafe': self.unsafe,
            'full': self.full
        }
        if name not in region_map:
            raise ValueError(f"Unknown region '{name}'. Must be one of {list(region_map.keys())}")
        return region_map[name]

    def __repr__(self):
        """String representation."""
        return (
            f"Regions(\n"
            f"  init={self.init},\n"
            f"  goal={self.goal},\n"
            f"  unsafe={self.unsafe},\n"
            f"  full={self.full}\n"
            f")"
        )

    @classmethod
    def from_numpy_ranges(
        cls,
        init_range: np.ndarray,
        goal_range: np.ndarray,
        unsafe_range: np.ndarray,
        full_range: Optional[np.ndarray] = None
    ):
        """
        Create regions from numpy arrays.

        Args:
            init_range: Initial region bounds (state_dim, 2)
            goal_range: Goal region bounds (state_dim, 2)
            unsafe_range: Unsafe region bounds (state_dim, 2)
            full_range: Full region bounds (state_dim, 2), optional

        Returns:
            Regions instance
        """
        init = Region(init_range)
        goal = Region(goal_range)
        unsafe = Region(unsafe_range)
        full = Region(full_range) if full_range is not None else None

        return cls(init=init, goal=goal, unsafe=unsafe, full=full)

    def to_dict(self):
        """Convert to dictionary (for saving/loading)."""
        return {
            'init': self.init.bounds,
            'goal': self.goal.bounds,
            'unsafe': self.unsafe.bounds,
            'full': self.full.bounds
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            init=Region(data['init']),
            goal=Region(data['goal']),
            unsafe=Region(data['unsafe']),
            full=Region(data['full'])
        )


# Type alias for Union
Union = Union