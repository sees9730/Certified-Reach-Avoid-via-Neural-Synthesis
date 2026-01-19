"""Provides a data class for reach-state-avoid specification"""
import dataclasses
from typing import Optional
from .aabb import AABBSet


@dataclasses.dataclass
class Specification:
    """
    Reach-avoid specification, optionally with stay requirement (reach-avoid-stay).
    """
    interest_set: AABBSet
    initial_set: AABBSet
    unsafe_set: AABBSet
    target_set: AABBSet
    reach_avoid_probability: float
    stay_probability: Optional[float] = None   # None => RA only
    require_stay: bool = True                  # if False => RA only

    def __post_init__(self):
        # If stay_probability is not provided, treat as reach-avoid only.
        if self.stay_probability is None:
            self.require_stay = False
        # If user explicitly disables stay, ignore stay_probability.
        if not self.require_stay:
            self.stay_probability = None
