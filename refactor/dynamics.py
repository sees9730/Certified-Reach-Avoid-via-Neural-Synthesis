"""
Dynamics module for stochastic differential equations.

This module defines the drift (F) and diffusion (G) terms for the SDE:
    dx = f(x)dt + g(x)dW

where:
    - f(x) = F @ x  (linear drift)
    - f(x) = f_fn(x) (nonlinear drift)
    - g(x) = G(x)   (potentially state-dependent diffusion)

The infinitesimal generator for a value function V is:
    Φ(x) = f(x) · ∇V + 0.5 * Tr(g(x)g(x)^T @ H_V)

where H_V is the Hessian of V.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Callable, Optional


class Dynamics:
    """
    System dynamics for stochastic differential equations.

    Supports:
    - Linear drift: f(x) = F @ x
    - Nonlinear drift: f(x) = f_fn(x) (via custom function)
    - Constant diffusion: g(x) = G (constant matrix)
    - State-dependent diffusion: g(x) = G(x) (via custom function)
    """

    def __init__(
        self,
        f: Union[np.ndarray, torch.Tensor, Callable, None] = None,
        g: Union[np.ndarray, torch.Tensor, Callable, None] = None,
        state_dim: int = 2
    ):
        """
        Initialize system dynamics.

        Args:
            F: Drift matrix or function. Can be:
                - np.ndarray or torch.Tensor: linear drift f(x) = F @ x
                - Callable: nonlinear drift f(x) = f_fn(x)
                - None: must provide drift_fn instead
            G: Diffusion matrix or function. Can be:
                - np.ndarray or torch.Tensor: constant diffusion g(x) = G
                - Callable: function g(x) = G(x) that takes state and returns diffusion
                - None: no diffusion (deterministic system)
            state_dim: State space dimension (default: 2)
            drift_fn: Alternative way to specify nonlinear drift (overrides F if both provided)
            B: Control input matrix (state_dim, control_dim) for controlled systems.
               If provided, closed-loop drift becomes: f(x) + B @ u(x)
            controller: Controller function u(x) (e.g., TrainableFeedbackControl instance).
                       If both B and controller are provided, they're applied to drift.
        """
        self.state_dim = state_dim

        self.f = f
        self.g = g

    def _setup_diffusion(self, G):
        """Setup diffusion term (constant matrix or state-dependent function)."""
        if G is None:
            # No diffusion (deterministic system)
            self.G_constant = torch.zeros(self.state_dim, self.state_dim)
            self.G_fn = None
            self.is_constant_diffusion = True
            self.sigma_diag = None
        elif callable(G):
            # State-dependent diffusion: G(x)
            self.G_fn = G
            self.G_constant = None
            self.is_constant_diffusion = False
            # Check if diffusion function is diagonal (marked by helper functions)
            self.is_diagonal_state_diffusion = getattr(G, '_is_diagonal', False)
            # Store sigma_diag if available (for CROWN compatibility)
            self.sigma_diag = getattr(G, '_sigma_diag', None)
        else:
            # Constant diffusion matrix
            if isinstance(G, np.ndarray):
                G = torch.from_numpy(G).float()
            self.G_constant = G.float() if G.dtype != torch.float32 else G

            assert self.G_constant.shape == (self.state_dim, self.state_dim), \
                f"G must be {self.state_dim}x{self.state_dim}, got {self.G_constant.shape}"

            self.G_fn = None
            self.is_constant_diffusion = True
            self.sigma_diag = None

    def drift(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute drift term f(x), optionally with control.

        For linear drift: f(x) = F @ x
        For nonlinear drift: f(x) = drift_fn(x)
        With control: f_cl(x) = f(x) + B @ u(x)

        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Drift f(x) of same shape as x
        """
        # Compute base drift
        if self.is_linear_drift:
            # Linear drift: f(x) = F @ x
            if x.dim() == 1:
                # Single state: (state_dim,)
                f_base = self.F @ x
            else:
                # Batch: (batch_size, state_dim)
                f_base = x @ self.F.T
        else:
            # Nonlinear drift: call drift_fn(x)
            f_base = self.drift_fn(x, u)

        # Apply control if controller and B matrix are provided
        if self.controller is not None and self.B is not None:
            u = self.controller(x)  # (batch_size, control_dim) or (control_dim,)

            if x.dim() == 1:
                # Single state
                f_base = f_base + self.B @ u
            else:
                # Batch
                f_base = f_base + (u @ self.B.T)

        # print(f"[Dynamics] f(x) = {f_base}")

        return f_base

    def diffusion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion matrix g(x).

        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Diffusion g(x) of shape (batch_size, state_dim, state_dim) or (state_dim, state_dim)
        """
        if self.is_constant_diffusion:
            # Constant diffusion: return G for all states
            if x.dim() == 1:
                return self.G_constant
            else:
                # Expand to batch: (batch_size, state_dim, state_dim)
                batch_size = x.shape[0]
                return self.G_constant.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # State-dependent diffusion: call G_fn(x)
            return self.G_fn(x)

    def diffusion_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute g(x) @ g(x)^T efficiently.

        This is used in the generator: 0.5 * Tr(g(x)g(x)^T @ H_V)

        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            g(x) @ g(x)^T of shape (batch_size, state_dim, state_dim) or (state_dim, state_dim)
        """
        G = self.diffusion(x)

        if x.dim() == 1:
            # Single state
            return G @ G.T
        else:
            # Batch: use bmm (batch matrix multiply)
            return torch.bmm(G, G.transpose(1, 2))

    def get_f(self) -> Optional[torch.Tensor]:
        return self.f

    def get_g(self) -> Optional[torch.Tensor]:
        return self.g

    def is_diffusion_diagonal(self) -> bool:
        """
        Check if diffusion is diagonal (constant or state-dependent).

        Returns:
            True if diffusion is diagonal, False otherwise
        """
        if self.is_constant_diffusion:
            # Check if constant G is diagonal
            if self.G_constant is None:
                return True  # Zero diffusion is considered diagonal
            return torch.allclose(
                self.G_constant,
                torch.diag(torch.diag(self.G_constant)),
                atol=1e-6
            )
        else:
            # Check if state-dependent diffusion is marked as diagonal
            return getattr(self, 'is_diagonal_state_diffusion', False)

    def get_diagonal_diffusion_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get diagonal elements of g(x) @ g(x)^T efficiently (CROWN-compatible).

        For diagonal diffusion g(x) = diag(g_1(x), g_2(x), ...), returns:
            [g_1(x)^2, g_2(x)^2, ...]

        This avoids creating full matrices and uses only CROWN-compatible operations.

        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Diagonal squared elements of shape (batch_size, state_dim) or (state_dim,)
        """
        if self.is_constant_diffusion:
            # Constant diffusion: extract diagonal of G @ G^T
            ggt_diag = torch.diag(self.G_constant @ self.G_constant.T)
            if x.dim() == 1:
                return ggt_diag
            else:
                # Broadcast to batch
                return ggt_diag.unsqueeze(0).expand(x.shape[0], -1)
        else:
            # State-dependent diffusion: use function's diagonal_squared method
            if hasattr(self.G_fn, 'get_diagonal_squared'):
                return self.G_fn.get_diagonal_squared(x)
            else:
                # Fallback: extract from full matrix (not CROWN-compatible!)
                G = self.G_fn(x)
                if x.dim() == 1:
                    return torch.diag(G @ G.T)
                else:
                    # Extract diagonal from batch
                    GGT = torch.bmm(G, G.transpose(1, 2))
                    return torch.diagonal(GGT, dim1=1, dim2=2)

    @classmethod
    def from_matrices(cls, F: np.ndarray, G: np.ndarray):
        """
        Create dynamics from numpy matrices with STATE-DEPENDENT diagonal diffusion.

        This method interprets G as the diagonal coefficients for state-dependent diffusion:
            g(x) = diag(G[0,0], G[1,1], ...) * x

        This matches the behavior of testing_simple3.py where g(x) = sigma * x.

        Args:
            F: Drift matrix (state_dim x state_dim)
            G: Diffusion matrix (state_dim x state_dim) - only diagonal elements are used

        Returns:
            Dynamics instance with state-dependent diffusion
        """
        state_dim = F.shape[0]

        # Extract diagonal coefficients from G
        # For diagonal G = [[sigma1, 0], [0, sigma2]], we want g(x) = diag(sigma1*x1, sigma2*x2)
        if isinstance(G, np.ndarray):
            sigma_diag = np.diag(G).copy()  # Extract diagonal as [sigma1, sigma2, ...]
        else:
            sigma_diag = torch.diag(G).clone()

        print(f"[Dynamics.from_matrices] Extracted sigma_diag: {sigma_diag}")

        # Create state-dependent diffusion function
        # g(x) = diag(sigma * x) where sigma is the diagonal of G
        diffusion_fn = diagonal_state_diffusion_general(sigma_diag)

        return cls.state_dependent_diffusion(F=F, diffusion_fn=diffusion_fn)

    @classmethod
    def scalar_diffusion(cls, F: np.ndarray, sigma: float, state_dim: int = 2):
        """
        Create dynamics with scalar diffusion: g(x) = sigma * I.

        Args:
            F: Drift matrix
            sigma: Diffusion coefficient
            state_dim: State dimension

        Returns:
            Dynamics instance
        """
        G = sigma * np.eye(state_dim, dtype=np.float32)
        return cls(F=F, G=G, state_dim=state_dim)

    @classmethod
    def state_dependent_diffusion(
        cls,
        F: np.ndarray,
        diffusion_fn: Callable,
        state_dim: int = 2
    ):
        """
        Create dynamics with state-dependent diffusion: g(x) = G(x).

        Args:
            F: Drift matrix
            diffusion_fn: Function that takes state x and returns diffusion matrix G(x)
            state_dim: State dimension

        Returns:
            Dynamics instance
        """
        return cls(F=F, G=diffusion_fn, state_dim=state_dim)

    def __repr__(self):
        """String representation."""
        # if self.is_linear_drift:
        #     F_str = f"F=\n{self.F.numpy()}"
        # else:
        #     F_str = "f(x)=<nonlinear>"

        # if self.is_constant_diffusion:
        #     G_str = f"G=\n{self.G_constant.numpy()}"
        # else:
        #     G_str = "G=<state-dependent>"
        # return f"Dynamics(state_dim={self.state_dim},\n{F_str},\n{G_str})"
        return "FIX ME"

    @classmethod
    def dynamics(
        cls,
        f: Union[np.ndarray, torch.tensor, Callable, None] = None,
        g: Union[np.ndarray, torch.Tensor, Callable, None] = None
    ):
        if isinstance(f, np.ndarray) and isinstance(g, np.ndarray):
            if g.shape[0] == f.shape[0]:
                state_dim = g.shape[0]
            else:
                raise ValueError("State dimension mismatch between f and g")
        elif isinstance(f, torch.Tensor) and isinstance(g, torch.Tensor):
            if g.shape[0] == f.shape[0]:
                state_dim = g.shape[0]
            else:
                raise ValueError("State dimension mismatch between f and g")
        else:
            # Default state dimension
            state_dim = 2
        return cls(f=f, g=g, state_dim=state_dim)


# Example helper functions for common diffusion patterns
def diagonal_state_diffusion_general(sigma_diag):
    """
    Create state-dependent diagonal diffusion with different coefficients per dimension.

    g(x) = diag(sigma[0] * x[0], sigma[1] * x[1], ...)

    Args:
        sigma_diag: Array/tensor of diffusion coefficients, shape (state_dim,)

    Returns:
        Function that computes G(x) = diag(sigma_diag * x)
    """
    # Convert to tensor if needed
    if isinstance(sigma_diag, np.ndarray):
        sigma_diag = torch.from_numpy(sigma_diag).float()
    else:
        sigma_diag = sigma_diag.float()

    def G_fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            # Single state: (state_dim,) -> (state_dim, state_dim)
            return torch.diag(sigma_diag * x)
        else:
            # Batch: (batch_size, state_dim) -> (batch_size, state_dim, state_dim)
            batch_size, state_dim = x.shape
            # Create diagonal matrices for batch
            G_batch = torch.zeros(batch_size, state_dim, state_dim, device=x.device, dtype=x.dtype)
            for i in range(state_dim):
                G_batch[:, i, i] = sigma_diag[i] * x[:, i]
            return G_batch

    def get_diagonal_squared(x: torch.Tensor) -> torch.Tensor:
        """
        CROWN-compatible: directly compute [g_1(x)^2, g_2(x)^2, ...] without full matrix.

        For g(x) = diag(sigma * x), we have g_i(x)^2 = (sigma_i * x_i)^2
        """
        # (sigma_diag * x)^2 = sigma_diag^2 * x^2
        # This uses only element-wise operations (CROWN-compatible)
        return (sigma_diag * x) ** 2

    # Mark as diagonal for CROWN compatibility
    G_fn._is_diagonal = True
    G_fn._sigma_diag = sigma_diag  # Store for access by Dynamics
    G_fn.get_diagonal_squared = get_diagonal_squared
    return G_fn


def diagonal_state_diffusion(sigma: float):
    """
    Create state-dependent diagonal diffusion: g(x) = diag(sigma * x).

    Args:
        sigma: Diffusion coefficient

    Returns:
        Function that computes G(x) = diag(sigma * x)
    """
    def G_fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            # Single state: (state_dim,) -> (state_dim, state_dim)
            return torch.diag(sigma * x)
        else:
            # Batch: (batch_size, state_dim) -> (batch_size, state_dim, state_dim)
            batch_size, state_dim = x.shape
            # Create diagonal matrices for batch
            G_batch = torch.zeros(batch_size, state_dim, state_dim, device=x.device, dtype=x.dtype)
            for i in range(state_dim):
                G_batch[:, i, i] = sigma * x[:, i]
            return G_batch

    def get_diagonal_squared(x: torch.Tensor) -> torch.Tensor:
        """
        CROWN-compatible: directly compute [g_1(x)^2, g_2(x)^2, ...] without full matrix.

        For g(x) = diag(sigma * x), we have g_i(x)^2 = (sigma * x_i)^2
        """
        return (sigma * x) ** 2

    # Mark as diagonal for CROWN compatibility
    G_fn._is_diagonal = True
    G_fn.get_diagonal_squared = get_diagonal_squared
    return G_fn


class ClosedLoopDrift(nn.Module):
    """
    Closed-loop drift dynamics: f_cl(x) = f_ol(x) + u(x).

    Combines open-loop drift with a controller. The controller is registered
    as a submodule, allowing its parameters to be trained.
    """
    def __init__(self, f_ol: Callable, controller: nn.Module):
        """
        Args:
            f_ol: Open-loop drift function f_ol(x) -> torch.Tensor
            controller: Controller network u(x) -> torch.Tensor
        """
        super().__init__()
        self.f_ol = f_ol
        self.controller = controller

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute closed-loop drift: f_cl(x) = f_ol(x) + u(x)."""
        return self.f_ol(x) + self.controller(x)
