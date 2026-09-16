"""Allocate cells across all unsafe boundary strips, including thin faces."""
import numpy as np

from src.discretization import discretize_region_with_splits
from src.regions import Region


def discretize_unsafe_boxes(boxes, input_scale, budget):
    """Spend each box's budget on its widest normalized cell directions.

    A thin boundary slab needs subdivisions mainly along its face. Equal
    per-box budgets also prevent a small unsafe component from disappearing.
    Each tensor-product grid covers its original box, including endpoints.
    """
    if len(boxes) == 0 or budget < len(boxes):
        raise ValueError("Unsafe discretization needs at least one cell per box")
    cells = []
    for i, box in enumerate(boxes):
        box_budget = budget // len(boxes) + int(i < budget % len(boxes))
        widths = (box[:, 1] - box[:, 0]) / np.asarray(input_scale)
        splits = np.ones(len(widths), dtype=int)
        while True:
            count = int(np.prod(splits))
            candidates = [d for d in range(len(widths))
                          if count // splits[d] * (splits[d] + 1) <= box_budget]
            if not candidates:
                break
            axis = max(candidates, key=lambda d: widths[d] / splits[d])
            splits[axis] += 1
        cells.extend(discretize_region_with_splits(Region(box), splits.tolist()))
    print(f"Unsafe face-aware grid: {len(cells)} cells (budget {budget})")
    return cells
