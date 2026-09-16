"""
CROWN bounds computation for neural networks.

This module provides symbolic CROWN bound computation that can be used
for training (with gradients) and verification.
"""

import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from src.network import bound_body

# auto_LiRPA bound methods usable for training: every one of these is
# differentiable and returns both bounds. 'IBP' is cheapest and loosest;
# 'CROWN-IBP' adds one backward pass over the final layer; 'CROWN' is a full
# backward pass, tightest and slowest.
BOUND_METHODS = ('IBP', 'CROWN-IBP', 'CROWN')


def validate_bound_method(method):
    """Normalize and check a bound method name."""
    if method not in BOUND_METHODS:
        raise ValueError(f"Unknown bound method {method!r}; expected one of {list(BOUND_METHODS)}")
    return method


class SymbolicCROWNCache:
    """
    Cache for symbolic CROWN computation on V network.

    Computes symbolic backward bounds once, then numerically evaluates
    during training by plugging in new bound values. Much faster than
    creating a new BoundedModule every iteration!

    The bounds are DIFFERENTIABLE and can be used in training loss.
    """

    def __init__(self, model, num_cells, input_dim=2, device='cpu', method='IBP'):
        """
        Initialize CROWN cache.

        Args:
            model: V network
            num_cells: Number of cells to compute bounds for
            input_dim: Input dimension
            device: Device
            method: auto_LiRPA bound method, one of BOUND_METHODS
        """
        self.model = model
        self.num_cells = num_cells
        self.device = device
        self.method = validate_bound_method(method)

        # Backward methods cannot propagate through V_offset's two-branch graph,
        # so bound the single-branch body and shift the result by the constant
        # baseline it leaves out.
        self.body, self.shift_fn = bound_body(model)

        # Create dummy batch input
        dummy_batch = torch.zeros(num_cells, input_dim, dtype=torch.float32, device=device)

        # Create BoundedModule ONCE - this builds the symbolic computation graph
        # print(f"[SymbolicCROWNCache] Creating BoundedModule for {num_cells} cells...")
        self.lirpa_model = BoundedModule(self.body, dummy_batch, device=device)

        # Initialize with dummy bounds
        dummy_lower = torch.zeros(num_cells, input_dim, device=device)
        dummy_upper = torch.ones(num_cells, input_dim, device=device)

        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=dummy_lower,
            x_U=dummy_upper
        )
        bounded_input = BoundedTensor(dummy_batch, ptb)

        # Forward pass to build symbolic computation graph
        _ = self.lirpa_model(bounded_input)

        # Cache a dummy batch (zeros) for IBP - actual values don't matter, only bounds
        self.dummy_batch_cache = torch.zeros(num_cells, input_dim, dtype=torch.float32, device=device)

    def compute_bounds(self, input_lowers, input_uppers):
        """
        Compute differentiable CROWN bounds using cached symbolic structure.

        Args:
            input_lowers: (N, D) lower bounds on inputs
            input_uppers: (N, D) upper bounds on inputs

        Returns:
            v_lowers: (N,) lower bounds on V(x)
            v_uppers: (N,) upper bounds on V(x)
        """
        assert input_lowers.shape[0] == self.num_cells

        # Use cached dummy batch (for IBP, actual values don't matter)
        # Ensure bounds are on correct device (avoid redundant .to() if already on device)
        if input_lowers.device != self.device:
            input_lowers = input_lowers.to(self.device)
            input_uppers = input_uppers.to(self.device)

        # Create new perturbation with updated bounds
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=input_lowers,
            x_U=input_uppers
        )
        bounded_input = BoundedTensor(self.dummy_batch_cache, ptb)

        # Compute bounds
        lb, ub = self.lirpa_model.compute_bounds(
            x=(bounded_input,),
            method=self.method
        )

        v_lowers = lb.squeeze(-1)  # (N,)
        v_uppers = ub.squeeze(-1)  # (N,)

        if self.shift_fn is not None:
            # A constant in x, so it shifts both bounds without loosening them.
            shift = self.shift_fn()
            v_lowers = v_lowers + shift
            v_uppers = v_uppers + shift

        return v_lowers, v_uppers

class SymbolicCROWNCache_Phi:
    """
    Cache for symbolic CROWN computation on Phi (GV) network.

    Computes differentiable bounds on GV(x) = f·∇V + 0.5·Tr(g·g^T·H_V)
    that can be used in training loss!
    """

    def __init__(self, phi_module, num_cells, input_dim=2, device='cpu', method='IBP'):
        """
        Initialize CROWN cache for Phi.

        Args:
            phi_module: GV module (PhiModuleTrainable/GV)
            num_cells: Number of cells
            input_dim: Input dimension
            device: Device
            method: auto_LiRPA bound method, one of BOUND_METHODS
        """
        self.num_cells = num_cells
        self.device = device
        self.method = validate_bound_method(method)

        # Create dummy batch input
        dummy_batch = torch.zeros(num_cells, input_dim, dtype=torch.float32, device=device)

        # Create BoundedModule ONCE
        # print(f"[SymbolicCROWNCache_Phi] Creating BoundedModule for {num_cells} cells...")
        self.lirpa_model = BoundedModule(phi_module, dummy_batch, device=device)

        # Initialize with dummy bounds
        dummy_lower = torch.zeros(num_cells, input_dim, device=device)
        dummy_upper = torch.ones(num_cells, input_dim, device=device)

        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=dummy_lower,
            x_U=dummy_upper
        )
        bounded_input = BoundedTensor(dummy_batch, ptb)

        # Forward pass to build symbolic computation graph
        _ = self.lirpa_model(bounded_input)

        # Cache a dummy batch (zeros) for IBP - actual values don't matter, only bounds
        self.dummy_batch_cache = torch.zeros(num_cells, input_dim, dtype=torch.float32, device=device)

    def compute_bounds(self, input_lowers, input_uppers):
        """
        Compute differentiable CROWN bounds on GV(x).

        Args:
            input_lowers: (N, D) lower bounds on inputs
            input_uppers: (N, D) upper bounds on inputs

        Returns:
            phi_lowers: (N,) lower bounds on GV(x)
            phi_uppers: (N,) upper bounds on GV(x)
        """
        # assert input_lowers.shape[0] == self.num_cells

        # Use cached dummy batch (for IBP, actual values don't matter)
        # Ensure bounds are on correct device (avoid redundant .to() if already on device)
        if input_lowers.device != self.device:
            input_lowers = input_lowers.to(self.device)
            input_uppers = input_uppers.to(self.device)

        # Create new perturbation
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=input_lowers,
            x_U=input_uppers
        )
        bounded_input = BoundedTensor(self.dummy_batch_cache, ptb)

        # Compute bounds (only the upper bound is needed for the generator loss,
        # but bound_lower=False saves little here, so we compute both)
        _, ub = self.lirpa_model.compute_bounds(
            x=(bounded_input,),
            method=self.method
        )

        # phi_lowers = lb.squeeze(-1)  # (N,) - computed but unused in loss
        phi_uppers = ub.squeeze(-1)  # (N,)

        return phi_uppers


def prepare_cell_bounds(cells, device='cpu', input_dim=2):
    """
    Prepare input bounds tensors from list of cells.

    Args:
        cells: List of (lower, upper) cell tuples
        device: Device

    Returns:
        (input_lowers, input_uppers) tensors of shape (N, input_dim)
    """
    n = len(cells)
    if n == 0:
        return torch.empty(0, input_dim, device=device), torch.empty(0, input_dim, device=device)

    # Pre-allocate tensors and fill directly (avoid list comprehension + stack)
    input_lowers = torch.empty(n, input_dim, dtype=torch.float32, device=device)
    input_uppers = torch.empty(n, input_dim, dtype=torch.float32, device=device)

    for i, (cell_lower, cell_upper) in enumerate(cells):
        input_lowers[i] = cell_lower.to(device) if cell_lower.device != device else cell_lower
        input_uppers[i] = cell_upper.to(device) if cell_upper.device != device else cell_upper

    return input_lowers, input_uppers
