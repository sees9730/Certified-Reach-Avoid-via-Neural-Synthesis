"""
Dynamics module for stochastic differential equations.

This module defines the drift (F) and diffusion (G) terms for the SDE:
    dx = f(x)dt + g(x)dW

where:
    - f(x) = F @ x  (linear drift)
    - g(x) = G(x)   (potentially state-dependent diffusion)

The infinitesimal generator for a value function V is:
    Φ(x) = f(x) · ∇V + 0.5 * Tr(g(x)g(x)^T @ H_V)

where H_V is the Hessian of V.
"""

import torch
import numpy as np
from typing import Union, Callable, Optional


class Dynamics:
    """
    System dynamics for stochastic differential equations.

    Supports:
    - Linear drift: f(x) = F @ x
    - Constant diffusion: g(x) = G (constant matrix)
    - State-dependent diffusion: g(x) = G(x) (via custom function)
    """

    def __init__(
        self,
        F: Union[np.ndarray, torch.Tensor],
        G: Optional[Union[np.ndarray, torch.Tensor, Callable]] = None,
        state_dim: int = 2
    ):
        """
        Initialize system dynamics.

        Args:
            F: Drift matrix (state_dim x state_dim). Defines f(x) = F @ x
            G: Diffusion matrix or function. Can be:
                - np.ndarray or torch.Tensor: constant diffusion g(x) = G
                - Callable: function g(x) = G(x) that takes state and returns diffusion
                - None: no diffusion (deterministic system)
            state_dim: State space dimension (default: 2)
        """
        self.state_dim = state_dim

        # Convert F to tensor and validate
        if isinstance(F, np.ndarray):
            F = torch.from_numpy(F).float()
        self.F = F.float() if F.dtype != torch.float32 else F

        assert self.F.shape == (state_dim, state_dim), \
            f"F must be {state_dim}x{state_dim}, got {self.F.shape}"

        # Handle diffusion matrix/function
        self._setup_diffusion(G)

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

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute drift term f(x) = F @ x.

        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Drift f(x) of same shape as x
        """
        if x.dim() == 1:
            # Single state: (state_dim,)
            return self.F @ x
        else:
            # Batch: (batch_size, state_dim)
            return x @ self.F.T

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

    def get_drift_matrix(self) -> torch.Tensor:
        """Get the drift matrix F."""
        return self.F

    def get_diffusion_matrix(self) -> Optional[torch.Tensor]:
        """
        Get the diffusion matrix G (if constant).

        Returns:
            G if diffusion is constant, None if state-dependent
        """
        return self.G_constant

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
        F_str = f"F=\n{self.F.numpy()}"
        if self.is_constant_diffusion:
            G_str = f"G=\n{self.G_constant.numpy()}"
        else:
            G_str = "G=<state-dependent>"
        return f"Dynamics(state_dim={self.state_dim},\n{F_str},\n{G_str})"


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
