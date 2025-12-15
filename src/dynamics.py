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
    """

    def __init__(
        self,
        f: Union[np.ndarray, torch.Tensor, Callable, None] = None,
        g: Union[np.ndarray, torch.Tensor, Callable, None] = None,
        state_dim: int = 2
    ):
        self.state_dim = state_dim
        self.f = f
        self.g = g

    def get_f(self) -> Optional[torch.Tensor]:
        return self.f

    def get_g(self) -> Optional[torch.Tensor]:
        return self.g

    def __repr__(self):
        if callable(self.f):
            f_str = "f(x)=<callable>"
        elif isinstance(self.f, (torch.Tensor, np.ndarray)):
            f_str = "f(x)=F@x"
        else:
            f_str = "f=None"

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
        f: Union[np.ndarray, torch.Tensor, Callable, None] = None,
        g: Union[np.ndarray, torch.Tensor, Callable, None] = None
    ):
        def infer_dim(obj) -> Optional[int]:
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                # Works for (N,N), (N,m), (N,), etc.
                return int(obj.shape[0])
            return None

        f_dim = infer_dim(f)
        g_dim = infer_dim(g)

        if f_dim is not None and g_dim is not None:
            if f_dim != g_dim:
                raise ValueError(f"State dimension mismatch between f ({f_dim}) and g ({g_dim})")
            state_dim = f_dim
        elif f_dim is not None:
            state_dim = f_dim
        elif g_dim is not None:
            state_dim = g_dim
        else:
            state_dim = 2  # fallback when both are callables/None

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
