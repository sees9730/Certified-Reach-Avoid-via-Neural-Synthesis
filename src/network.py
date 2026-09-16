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

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
import sys
sys.path.insert(0, str(ROOT))

from src.hyperparameters import NetworkConfig


class V(nn.Module):
    """Value network architecture."""

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
        self.activation_fn = self._get_activation_fn()

        # Output scaling (applied before final linear layer)
        self.register_buffer('scale_factor', torch.tensor(config.scale_factor))

        print("V initialized with:")
        print(" input_scale:", self.input_scale.detach().cpu().tolist())
        print(" scale_factor:", self.scale_factor.detach().cpu())
        print(" activation_fn:", self.activation_fn)
        print(" n_inputs:", config.n_inputs)
        print(" n_hidden_1:", config.n_hidden_1)
        print(" n_hidden_2:", config.n_hidden_2)
        print(" n_outputs:", config.n_outputs)
        print("")

    def _get_activation_fn(self):
        """Select activation function."""
        return torch.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor of shape (batch_size, n_inputs)

        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        # Normalize input
        x = x / self.input_scale

        # Hidden layers
        x = self.layer1(x)
        x = self.activation_fn(x)

        x = self.layer2(x)
        x = self.activation_fn(x)

        # Scale hidden activations before output layer
        x = self.output(x * self.scale_factor)

        return x

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


class V_offset(nn.Module):
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

    def __init__(self, config: NetworkConfig, input_offset, output_offset):
        """
        Initialize value network.

        Args:
            config: Network configuration
        """
        super(V_offset, self).__init__()

        self.config = config

        # Input normalization for each dimension [input_scale_i, input_scale_j] -> [x_orig_i/input_scale_i, x_orig_j/input_scale_j]
        self.register_buffer('input_scale', torch.tensor(config.input_scale, dtype=torch.float32))
        self.register_buffer("input_offset", torch.tensor(input_offset, dtype=torch.float32))
        self.register_buffer("output_offset", torch.tensor(output_offset, dtype=torch.float32))

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
        Guarantee: V(x = input_offset) = 0 by subtracting the full-network baseline at x_norm = 0.
        """
        was_1d = (x.dim() == 1)
        if was_1d:
            x = x.unsqueeze(0)  # (1,D)

        # Normalize: (x - offset)/scale
        x_norm = (x - self.input_offset.unsqueeze(0)) / self.input_scale.unsqueeze(0)

        # ---- main forward ----
        h = self.layer1(x_norm)
        h = self.activation_fn(h)
        h = self.layer2(h)
        h = self.activation_fn(h)
        y = self.output(h * self.scale_factor)  # (N,out)

        # ---- baseline at x_norm = 0 (i.e., x = input_offset) ----
        x_norm0 = torch.zeros_like(x_norm)      # (N,D) but all zeros
        h0 = self.layer1(x_norm0)
        h0 = self.activation_fn(h0)
        h0 = self.layer2(h0)
        h0 = self.activation_fn(h0)
        y0 = self.output(h0 * self.scale_factor)  # (N,out), constant across N

        out = y - y0 + self.output_offset

        return out.squeeze(0) if was_1d else out


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
    
    @torch.no_grad()
    def verify_zero_at_offset(
        self,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        verbose: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> bool:
        """
        Verify numerically that V(input_offset) == 0.
        Returns True if all outputs are close to zero within tolerances.
        """
        if device is None:
            device = self.input_offset.device
        if dtype is None:
            dtype = self.input_offset.dtype

        x0 = self.input_offset.to(device=device, dtype=dtype)

        y0 = self.forward(x0)  # shape (n_outputs,) or scalar-like
        target = torch.zeros_like(y0)

        ok = torch.allclose(y0, target, atol=atol, rtol=rtol)

        if verbose:
            print("x = input_offset:", x0.detach().cpu().numpy())
            print("V(input_offset):", y0.detach().cpu().numpy())
            print(f"allclose_to_zero={bool(ok)} (atol={atol}, rtol={rtol})")

        return bool(ok)
    

class _FeedForwardBody(nn.Module):
    """Single-branch view of V_offset, sharing its layers.

    V_offset.forward evaluates the network twice (at x and at input_offset) and
    subtracts. auto_LiRPA's backward passes ('CROWN', 'CROWN-IBP') cannot
    propagate through that two-branch graph, so bounds are taken on this body
    and the constant baseline is applied to the resulting bounds instead.
    """

    def __init__(self, source: "V_offset"):
        super().__init__()
        # The same Linear objects, so this body shares parameters (and therefore
        # gradients) with the network it was built from.
        self.layer1, self.layer2, self.output = source.layer1, source.layer2, source.output
        self.activation_fn = source.activation_fn
        for name in ("input_scale", "input_offset", "scale_factor"):
            self.register_buffer(name, getattr(source, name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_offset) / self.input_scale
        h = self.activation_fn(self.layer1(x_norm))
        h = self.activation_fn(self.layer2(h))
        return self.output(h * self.scale_factor)


def bound_body(model: nn.Module):
    """Split a value network into a body whose bounds shift by a constant.

    Returns (body, shift_fn) with ``model(x) == body(x) + shift_fn()`` for every
    x, so bounds on the body plus the shift are exact bounds on the model. For a
    plain feed-forward network the body is the model itself and shift_fn is None.
    """
    if not isinstance(model, V_offset):
        return model, None

    body = _FeedForwardBody(model)

    def shift_fn():
        # Recomputed per call: it moves as the shared parameters are trained.
        return model.output_offset - body(model.input_offset.unsqueeze(0)).reshape(())

    return body, shift_fn


def create_V(config: NetworkConfig, input_offset=None, output_offset=None) -> nn.Module:
    """
    Factory function to create value network from config.

    Args:
        config: Network configuration

    Returns:
        ValueNetwork instance
    """
    if(input_offset is not None and output_offset is not None):
        return V_offset(config, input_offset=input_offset, output_offset=output_offset)
    else:
        return V(config)
