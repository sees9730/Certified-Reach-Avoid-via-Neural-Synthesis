"""
Controls module for generic feedback control.

This module provides a generic control function that can be:
- Linear feedback: u = K @ x
- Neural network: u = NN(x)
- Any callable: u = u_control(x)
"""

import torch
import numpy as np
from typing import Union, Callable


def u_control(K: Union[np.ndarray, torch.Tensor, Callable]) -> Callable:
    """
    Create a generic control function u(x).

    Args:
        K: Can be:
            - np.ndarray or torch.Tensor: Linear control u = K @ x
            - Callable (e.g., neural network): u = K(x)

    Returns:
        Control function u(x) that takes state x and returns control input

    Example:
        >>> # Linear control
        >>> K = np.array([[-1.0, 0.0], [0.0, -1.0]])
        >>> u = u_control(K)
        >>> u(x)  # Returns K @ x

        >>> # Neural network control
        >>> u_nn = SomeNeuralNetwork()
        >>> u = u_control(u_nn)
        >>> u(x)  # Returns u_nn(x)
    """
    if callable(K):
        # K is already a callable (neural network, custom function, etc.)
        return K
    else:
        # K is a matrix, create linear control function
        if isinstance(K, np.ndarray):
            K_torch = torch.from_numpy(K).float()
        else:
            K_torch = K.float()

        def linear_control(x: torch.Tensor) -> torch.Tensor:
            """Linear control: u = K @ x"""
            if x.dim() == 1:
                return K_torch @ x
            else:
                return x @ K_torch.T

        return linear_control
