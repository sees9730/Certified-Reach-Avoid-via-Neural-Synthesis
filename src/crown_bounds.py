"""
CROWN bounds computation for neural networks.

This module provides symbolic CROWN bound computation that can be used
for training (with gradients) and verification.
"""

import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class SymbolicCROWNCache:
    """
    Cache for symbolic CROWN computation on V network.

    Computes symbolic backward bounds once, then numerically evaluates
    during training by plugging in new bound values. Much faster than
    creating a new BoundedModule every iteration!

    The bounds are DIFFERENTIABLE and can be used in training loss.
    """

    def __init__(self, model, num_cells, input_dim=2, device='cpu'):
        """
        Initialize CROWN cache.

        Args:
            model: V network
            num_cells: Number of cells to compute bounds for
            input_dim: Input dimension
            device: Device
        """
        self.model = model
        self.num_cells = num_cells
        self.input_dim = input_dim
        self.device = device

        # Save original training mode
        self.was_training = model.training
        # model.eval()

        # Create dummy batch input
        dummy_batch = torch.zeros(num_cells, input_dim, dtype=torch.float32, device=device)

        # Create BoundedModule ONCE - this builds the symbolic computation graph
        print(f"[SymbolicCROWNCache] Creating BoundedModule for {num_cells} cells...")
        self.lirpa_model = BoundedModule(model, dummy_batch, device=device)

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

        # Create dummy batch input (center of each cell)
        dummy_batch = input_lowers.clone().to(self.device)
        dummy_batch.add_(input_uppers.to(self.device)).mul_(0.5)

        # Create new perturbation with updated bounds
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=input_lowers.detach().to(self.device),
            x_U=input_uppers.detach().to(self.device)
        )
        bounded_input = BoundedTensor(dummy_batch, ptb)

        # Compute bounds
        lb, ub = self.lirpa_model.compute_bounds(
            x=(bounded_input,),
            method='IBP',
            forward=True,
            bound_lower=True,
            bound_upper=True
        )

        v_lowers = lb.squeeze(-1)  # (N,)
        v_uppers = ub.squeeze(-1)  # (N,)

        return v_lowers, v_uppers

    def __del__(self):
        """Restore model training mode on cleanup"""
        if hasattr(self, 'was_training') and self.was_training:
            self.model.train()


class SymbolicCROWNCache_Phi:
    """
    Cache for symbolic CROWN computation on Phi (GV) network.

    Computes differentiable bounds on Φ(x) = f·∇V + 0.5·Tr(g·g^T·H_V)
    that can be used in training loss!
    """

    def __init__(self, phi_module, num_cells, input_dim=2, device='cpu'):
        """
        Initialize CROWN cache for Phi.

        Args:
            phi_module: GV module (PhiModuleTrainable/GV)
            num_cells: Number of cells
            input_dim: Input dimension
            device: Device
        """
        self.phi_module = phi_module
        self.num_cells = num_cells
        self.input_dim = input_dim
        self.device = device

        # Save original training mode
        self.was_training_V = phi_module.V_net.training
        # phi_module.V_net.eval()
        # phi_module.eval()

        # Create dummy batch input
        dummy_batch = torch.zeros(num_cells, input_dim, dtype=torch.float32, device=device)

        # Create BoundedModule ONCE
        print(f"[SymbolicCROWNCache_Phi] Creating BoundedModule for {num_cells} cells...")
        self.lirpa_model = BoundedModule(phi_module, dummy_batch, device=device)

    def compute_bounds(self, input_lowers, input_uppers):
        """
        Compute differentiable CROWN bounds on Φ(x).

        Args:
            input_lowers: (N, D) lower bounds on inputs
            input_uppers: (N, D) upper bounds on inputs

        Returns:
            phi_lowers: (N,) lower bounds on Φ(x)
            phi_uppers: (N,) upper bounds on Φ(x)
        """
        # assert input_lowers.shape[0] == self.num_cells

        # Create dummy batch input
        dummy_batch = input_lowers.clone().to(self.device)
        dummy_batch.add_(input_uppers.to(self.device)).mul_(0.5)

        # Create new perturbation
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=input_lowers.detach().to(self.device),
            x_U=input_uppers.detach().to(self.device)
        )
        bounded_input = BoundedTensor(dummy_batch, ptb)

        # Compute bounds
        lb, ub = self.lirpa_model.compute_bounds(
            x=(bounded_input,),
            method='IBP',
            forward=True,
            bound_lower=True,
            bound_upper=True
        )

        phi_lowers = lb.squeeze(-1)  # (N,)
        phi_uppers = ub.squeeze(-1)  # (N,)

        return phi_lowers, phi_uppers

    def __del__(self):
        """Restore training mode"""
        if hasattr(self, 'was_training_V') and self.was_training_V:
            self.phi_module.V_net.train()


def prepare_cell_bounds(cells, device='cpu', input_dim=2):
    """
    Prepare input bounds tensors from list of cells.

    Args:
        cells: List of (lower, upper) cell tuples
        device: Device

    Returns:
        (input_lowers, input_uppers) tensors of shape (N, input_dim)
    """
    if len(cells) == 0:
        return torch.empty(0, input_dim, device=device), torch.empty(0, input_dim, device=device)

    input_lowers = torch.stack([cell[0] for cell in cells]).to(device)
    input_uppers = torch.stack([cell[1] for cell in cells]).to(device)

    return input_lowers, input_uppers
