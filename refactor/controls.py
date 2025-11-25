"""
Controls module for applying feedback control to system dynamics.

This module provides utilities for transforming the drift matrix F using
state feedback control laws of the form u = K @ x.

For a system dx = F @ x dt + G(x) dW with control u, the closed-loop dynamics are:
    dx = (F + K) @ x dt + G(x) dW

where K is the control gain matrix.
"""

import torch
import numpy as np
from typing import Union


def apply_control(
    F: Union[np.ndarray, torch.Tensor],
    K: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply state feedback control u = K @ x to drift matrix F.

    Returns the closed-loop drift matrix: F_cl = F + K

    Args:
        F: Open-loop drift matrix (state_dim x state_dim)
        K: Control gain matrix (state_dim x state_dim)
            For controller u = [k1*x1, k2*x2], use K = diag(k1, k2)

    Returns:
        F_cl: Closed-loop drift matrix (same type as F)

    Example:
        >>> F = np.array([[-0.5, 1.0], [-1.0, -0.5]])
        >>> K = np.array([[-1.0, 0.0], [0.0, -1.0]])  # u = [-x1, -x2]
        >>> F_cl = apply_control(F, K)
        >>> # F_cl = [[-1.5, 1.0], [-1.0, -1.5]]
    """
    # Ensure same type
    if isinstance(F, np.ndarray):
        if isinstance(K, torch.Tensor):
            K = K.cpu().numpy()
        return F + K
    else:
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).to(F.device).to(F.dtype)
        return F + K


def diagonal_control(coefficients: Union[list, np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Create a diagonal control gain matrix.

    For u = [k1*x1, k2*x2, ...], returns K = diag(k1, k2, ...)

    Args:
        coefficients: List or array of control gains [k1, k2, ...]

    Returns:
        K: Diagonal control gain matrix

    Example:
        >>> K = diagonal_control([-1.0, -1.0])  # u = [-x1, -x2]
        >>> # K = [[-1.0, 0.0], [0.0, -1.0]]
    """
    if isinstance(coefficients, (list, tuple)):
        coefficients = np.array(coefficients, dtype=np.float32)
    elif isinstance(coefficients, torch.Tensor):
        coefficients = coefficients.cpu().numpy()

    return np.diag(coefficients).astype(np.float32)
