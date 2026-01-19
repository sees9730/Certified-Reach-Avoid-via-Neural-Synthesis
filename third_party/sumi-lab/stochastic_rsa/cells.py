import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm, BoundedModule
import itertools


class CellVerificationSystem():
    def __init__(self, max_depth=10):
        super().__init__()
        self.max_depth = max_depth
        self.corners = None  # built lazily once we know N

    @staticmethod
    def _make_corners(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # (2**dim, dim) with entries in {-1, +1}
        return torch.tensor(
            list(itertools.product([-1.0, 1.0], repeat=dim)),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _ensure_mag_shape(magnitude: torch.Tensor, B: int, dim: int) -> torch.Tensor:
        """
        Return magnitude as shape (B, dim), broadcasted appropriately.
        Accepts:
          - (dim,)
          - (1, dim)
          - (B, dim)
          - (B, 1) or (1, 1)  (isotropic)
        """
        if magnitude.ndim == 1:
            if magnitude.numel() == dim:
                magnitude = magnitude.view(1, dim).expand(B, dim)
            elif magnitude.numel() == 1:
                magnitude = magnitude.view(1, 1).expand(B, dim)
            else:
                raise ValueError(f"magnitude 1D must have length 1 or {dim}, got {magnitude.numel()}")
        elif magnitude.ndim == 2:
            if magnitude.shape == (B, dim):
                pass
            elif magnitude.shape == (1, dim):
                magnitude = magnitude.expand(B, dim)
            elif magnitude.shape[1] == 1 and magnitude.shape[0] in (1, B):
                magnitude = magnitude.expand(B, 1).expand(B, dim)
            else:
                raise ValueError(f"magnitude 2D must be (B,{dim}), (1,{dim}), (B,1), or (1,1); got {tuple(magnitude.shape)}")
        else:
            raise ValueError(f"magnitude must be 1D or 2D, got {magnitude.ndim}D")

        return magnitude


    def verify(
        self,
            verifier: BoundedModule,
            locations: torch.Tensor,
            magnitude: torch.Tensor,
            depth: int = 0
    ):
        if locations.ndim != 2:
            raise ValueError(f"locations must be (B,dim), got {tuple(locations.shape)}")
        B, dim = locations.shape

        # Build corners once per dimension/device/dtype
        if (self.corners is None) or (self.corners.shape[1] != dim) or (self.corners.device != locations.device) or (self.corners.dtype != locations.dtype):
            self.corners = self._make_corners(dim, device=locations.device, dtype=locations.dtype)

        magnitude = self._ensure_mag_shape(magnitude, B=B, dim=dim)

        bounded_cells = BoundedTensor(
            locations,
            PerturbationLpNorm(
                x_L=locations - magnitude,
                x_U=locations + magnitude,
            ),
        )
        _, ub = verifier.compute_bounds(
            bounded_cells,
            bound_lower=False,
            method="IBP"
        )
        mask = ub.squeeze() >= 0.0
        counterexamples = locations[mask]

        if torch.numel(counterexamples) > 0:
            print(
                f"Could not verify decrease at {counterexamples.shape[0]} "
                "cells. Splitting further"
            )
            if depth < self.max_depth:
                # get per-failing-cell magnitudes
                mag_fail = magnitude[mask]          # (K, dim)
                half_m = 0.5 * mag_fail             # (K, dim)

                # Vectorized split: each cell -> 2**dim subcells
                # new_cells: (K, 2**dim, dim) -> flatten to (K*2**dim, dim)
                new_cells = (
                    counterexamples[:, None, :] + half_m[:, None, :] * self.corners[None, :, :]
                ).reshape(-1, dim)

                counterexamples = self.verify(verifier, new_cells, half_m.repeat_interleave(self.corners.shape[0], dim=0), depth + 1)
        return counterexamples
