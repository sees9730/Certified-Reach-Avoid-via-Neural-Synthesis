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

    def get_f(self) -> Optional[torch.Tensor]:
        return self.f

    def get_g(self) -> Optional[torch.Tensor]:
        return self.g

    def __repr__(self):
        """String representation."""
        # Drift type
        if callable(self.f):
            f_str = "f(x)=<callable>"
        elif isinstance(self.f, (torch.Tensor, np.ndarray)):
            f_str = "f(x)=F@x"
        else:
            f_str = "f=None"

        # Diffusion type
        if callable(self.g):
            g_str = "g(x)=<callable>"
        elif isinstance(self.g, (torch.Tensor, np.ndarray)):
            g_str = "g(x)=G"
        else:
            g_str = "g=None"

        return f"Dynamics(dim={self.state_dim}, {f_str}, {g_str})"

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
