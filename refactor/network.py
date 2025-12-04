"""
Neural network architectures for value function approximation.

This module provides flexible network architectures with:
- Configurable hidden layers
- Multiple activation function options
- Input normalization
- Output scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import NetworkConfig


class V(nn.Module):
    """
    Value function neural network.

    Architecture:
        x -> normalize -> layer1 -> activation -> layer2 -> activation -> output * scale -> V(x)

    Supports configurable:
    - Number of hidden layers and units
    - Activation functions (sigmoid, ReLU, GELU, arctan)
    - Input normalization scale
    - Output scaling
    """

    def __init__(self, config: NetworkConfig):
        """
        Initialize value network.

        Args:
            config: Network configuration
        """
        super(V, self).__init__()

        self.config = config

        # Input normalization for each dimension [input_scale_i, input_scale_j] -> [x_orig_i/input_scale_i, x_orig_j/input_scale_j]
        self.register_buffer('input_scale', torch.tensor(config.input_scale, dtype=torch.float32))

        # Network layers
        self.layer1 = nn.Linear(config.n_inputs, config.n_hidden_1)
        self.layer2 = nn.Linear(config.n_hidden_1, config.n_hidden_2)
        self.output = nn.Linear(config.n_hidden_2, config.n_outputs)

        # Select activation function
        self.activation_fn = self._get_activation_fn(config)

        # Output scaling (applied before final linear layer)
        self.register_buffer('scale_factor', torch.tensor(config.scale_factor))

    def _get_activation_fn(self, config: NetworkConfig):
        """Select activation function."""
        return torch.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input state (batch_size, n_inputs) or (n_inputs,)

        Returns:
            Value V(x) of shape (batch_size, n_outputs) or (n_outputs,)
        """
        # Normalize input: [-input_scale, input_scale] -> [-1, 1]
        x = x / self.input_scale

        # Hidden layers
        x = self.layer1(x)
        x = self.activation_fn(x)

        x = self.layer2(x)
        x = self.activation_fn(x)

        # Output layer with scaling
        x = self.output(x * self.scale_factor)

        return x

    def get_weights(self):
        """
        Get network weights for manual bound computation.

        Returns:
            Dictionary with layer weights and biases
        """
        return {
            'layer1': {
                'weight': self.layer1.weight,
                'bias': self.layer1.bias
            },
            'layer2': {
                'weight': self.layer2.weight,
                'bias': self.layer2.bias
            },
            'output': {
                'weight': self.output.weight,
                'bias': self.output.bias
            }
        }

    def __repr__(self):
        """String representation."""
        activation_name = self.activation_fn.__class__.__name__ if hasattr(self.activation_fn, '__class__') else str(self.activation_fn)
        return (
            f"ValueNetwork(\n"
            f"  input_dim={self.config.n_inputs},\n"
            f"  hidden=[{self.config.n_hidden_1}, {self.config.n_hidden_2}],\n"
            f"  output_dim={self.config.n_outputs},\n"
            f"  activation={activation_name},\n"
            f"  input_scale={self.config.input_scale},\n"
            f"  scale_factor={self.config.scale_factor}\n"
            f")"
        )


def create_V(config: NetworkConfig) -> nn.Module:
    """
    Factory function to create value network from config.

    Args:
        config: Network configuration

    Returns:
        ValueNetwork instance
    """
    return V(config)
