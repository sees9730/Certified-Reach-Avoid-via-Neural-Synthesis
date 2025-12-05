"""
Phi module for computing the infinitesimal generator of the value function.

The infinitesimal generator for a stochastic system dx = f(x,u)dt + g(x,u)dW is:
    Φ(x) = f(x,u) · ∇V + 0.5 * Tr(g(x,u)g(x,u)^T @ H_V)

This module handles ANY form of drift f and diffusion g:
- f: constant matrix, f(x), or f(x, u)
- g: scalar, vector, constant matrix, g(x), or g(x, u)
- g can return diagonal vector (N, 2) or full matrix (N, 2, 2)
- Automatically computes diagonal of g@g^T for diffusion term
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dynamics import Dynamics


class GV(nn.Module):
    """
    Computes Φ(x) = f(x,u)·∇V + 0.5·Tr(g(x,u)g(x,u)^T @ H_V) using closed-form derivatives.

    Supports ANY form of drift and diffusion (2D systems):
    - f: constant matrix F, f(x), or f(x, u)
    - g: scalar σ, vector [σ₁, σ₂], constant matrix G, g(x), or g(x, u)
    - g(x) or g(x,u) can return diagonal (N,2) or full matrix (N,2,2)

    Key features:
    - Uses REFERENCES to V_net's layers (not detached copies)
    - Gradients flow back to V_net's parameters during training
    - Automatically handles diagonal extraction from g@g^T
    - Compatible with auto_LiRPA's BoundedModule
    """

    def __init__(
        self,
        V_net: nn.Module,
        dynamics: Dynamics,
        scale_factor: float = 1.0,
        learnable_scale: bool = False,
        learnable_input_scale: bool = False,
        input_scale_init: float = None
    ):
        """
        Initialize Phi module.

        Args:
            V_net: Value function network
            dynamics: System dynamics (drift F and diffusion G)
            scale_factor: Output scaling factor
            learnable_scale: Whether scale_factor is learnable
            learnable_input_scale: Whether input_scale is learnable
            input_scale_init: Initial input scale (default: use V_net's input_scale)
        """
        super().__init__()
        self.V_net = V_net
        self.dynamics = dynamics

        # Make scale_factor learnable or constant based on flag
        if learnable_scale:
            self.scale_factor = nn.Parameter(torch.tensor(scale_factor, dtype=torch.float32))
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor, dtype=torch.float32))

        # Input normalization scale
        if input_scale_init is None:
            input_scale_init = V_net.input_scale

        self.learnable_input_scale = learnable_input_scale
        if learnable_input_scale:
            self.input_scale = nn.Parameter(torch.tensor(input_scale_init, dtype=torch.float32))
        else:
            self.register_buffer('input_scale', torch.tensor(input_scale_init, dtype=torch.float32))
            input_scale_tensor = torch.tensor(input_scale_init, dtype=torch.float32)
            self.register_buffer('input_scale_sq', input_scale_tensor ** 2)

        # Register masks as buffers to avoid TracerWarnings
        self.register_buffer('mask1', torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        self.register_buffer('mask2', torch.tensor([[0.0, 1.0]], dtype=torch.float32))

        # Get drift and diffusion from dynamics
        self.f = dynamics.get_f()
        self.g = dynamics.get_g()

        print(f"[PhiModule] Initialized with:")
        print(f"  scale_factor: {self.scale_factor.item() if hasattr(self.scale_factor, 'item') else self.scale_factor}")
        print(f"  input_scale: {self.input_scale.tolist() if hasattr(self.input_scale, 'tolist') else self.input_scale}")
        print(f"  f type: {type(self.f)}")
        print(f"  g type: {type(self.g)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Φ(x) using closed-form derivatives.

        For network: h0 = σ(W0·x_norm + b0), h1 = σ(W1·h0 + b1), V = W2·(SCALE·h1) + b2
        where x_norm = x / input_scale

        Args:
            x: State tensor (batch_size, state_dim) or (state_dim,)

        Returns:
            Φ(x) of shape (batch_size, 1) or (1,)
        """
        # Store original x for dynamics computation (f(x) = F·x uses unnormalized coords)
        x_orig = x.clone()

        # Normalize input
        x_norm = x / self.input_scale

        # Get weights from V_net (trainable - not detached!)
        W0 = self.V_net.layer1.weight  # (m0, state_dim)
        b0 = self.V_net.layer1.bias    # (m0,)
        W1 = self.V_net.layer2.weight  # (m1, m0)
        b1 = self.V_net.layer2.bias    # (m1,)
        W2 = self.V_net.output.weight  # (1, m1)
        b2 = self.V_net.output.bias    # (1,)

        state_dim = W0.shape[1]

        # Forward pass through first hidden layer
        z0 = F.linear(x_norm, W0, b0)     # (N, m0)
        h0 = torch.sigmoid(z0)             # (N, m0)
        d0 = h0 * (1.0 - h0)              # σ'(z0): (N, m0)
        q0 = (1.0 - 2.0 * h0) * d0        # σ''(z0): (N, m0)

        # Forward pass through second hidden layer
        z1 = F.linear(h0, W1, b1)         # (N, m1)
        h1 = torch.sigmoid(z1)             # (N, m1)
        d1 = h1 * (1.0 - h1)              # σ'(z1): (N, m1)
        q1 = (1.0 - 2.0 * h1) * d1        # σ''(z1): (N, m1)

        # Compute ∇V w.r.t normalized coordinates
        # ∂V/∂x_norm_i = SCALE · Σ_j W2[j]·σ'(z1[j])·(Σ_k W1[j,k]·σ'(z0[k])·W0[k,i])

        # Pre-compute common terms to avoid redundant operations
        # W1_scaled[j,k] = W1[j,k] · σ'(z0[k])  (shared for both x1 and x2)
        W1_scaled = W1.unsqueeze(0) * d0.unsqueeze(1)  # (N, m1, m0)

        # For x1 (i=0):
        sum_over_k1 = (W1_scaled * W0[:, 0].view(1, 1, -1)).sum(dim=2)  # (N, m1)
        dVdx1_norm = self.scale_factor * (W2 * d1 * sum_over_k1).sum(dim=1, keepdim=True)  # (N, 1)

        # For x2 (i=1):
        sum_over_k2 = (W1_scaled * W0[:, 1].view(1, 1, -1)).sum(dim=2)  # (N, m1)
        dVdx2_norm = self.scale_factor * (W2 * d1 * sum_over_k2).sum(dim=1, keepdim=True)  # (N, 1)

        # Apply chain rule: ∇V w.r.t x = (1/input_scale) · ∇V w.r.t x_norm
        dVdx1 = dVdx1_norm / self.input_scale[0]  # (N, 1)
        dVdx2 = dVdx2_norm / self.input_scale[1]  # (N, 1)

        # Compute Hessian diagonal w.r.t normalized coordinates
        # Pre-compute common term for Hessian: W1[j,k] · σ''(z0[k])
        W1_q0 = W1.unsqueeze(0) * q0.unsqueeze(1)  # (N, m1, m0)

        # For x1 (i=0):
        # Cross-term: σ''(z1[j]) · (Σ_k W1[j,k]·σ'(z0[k])·W0[k,0])²
        cross_term1 = q1 * (sum_over_k1 ** 2)  # (N, m1)

        # Direct term: σ'(z1[j]) · Σ_k W1[j,k]·σ''(z0[k])·W0[k,0]²
        W0_0_sq = W0[:, 0] ** 2  # Pre-compute squared weights
        direct_term1 = d1 * (W1_q0 * W0_0_sq.view(1, 1, -1)).sum(dim=2)  # (N, m1)

        H11_norm = self.scale_factor * (W2 * (cross_term1 + direct_term1)).sum(dim=1, keepdim=True)  # (N, 1)

        # For x2 (i=1):
        cross_term2 = q1 * (sum_over_k2 ** 2)  # (N, m1)
        W0_1_sq = W0[:, 1] ** 2  # Pre-compute squared weights
        direct_term2 = d1 * (W1_q0 * W0_1_sq.view(1, 1, -1)).sum(dim=2)  # (N, m1)

        H22_norm = self.scale_factor * (W2 * (cross_term2 + direct_term2)).sum(dim=1, keepdim=True)  # (N, 1)

        # Compute input_scale_sq
        if self.learnable_input_scale:
            input_scale_sq = self.input_scale ** 2
        else:
            input_scale_sq = self.input_scale_sq

        # Apply chain rule: H_ii w.r.t x = (1/input_scale²) · H_ii w.r.t x_norm
        H11 = H11_norm / input_scale_sq[0]  # (N, 1)
        H22 = H22_norm / input_scale_sq[1]  # (N, 1)

        # Compute f(x) or f(x, u) using ORIGINAL UNNORMALIZED coordinates
        fx = self._evaluate_f(x_orig)  # (N, 2)
        # Use registered buffer masks to avoid TracerWarnings
        f1 = (fx * self.mask1.to(x.dtype)).sum(dim=1, keepdim=True)  # (N, 1)
        f2 = (fx * self.mask2.to(x.dtype)).sum(dim=1, keepdim=True)  # (N, 1)

        # Compute diagonal of g(x) @ g(x)^T for any form of g
        g_diag_sq = self._compute_gg_diag(x_orig)  # (N, 2)
        # Use registered buffer masks
        g11_sq = (g_diag_sq * self.mask1.to(x.dtype)).sum(dim=1, keepdim=True)  # (N, 1)
        g22_sq = (g_diag_sq * self.mask2.to(x.dtype)).sum(dim=1, keepdim=True)  # (N, 1)
        # print(f"DEBUG: g11_sq.shape = {g11_sq.shape}, g22_sq.shape = {g22_sq.shape}")

        # Φ(x) = f·∇V + 0.5·(g²·H_diag)
        drift = f1 * dVdx1 + f2 * dVdx2
        diff = 0.5 * (g11_sq * H11 + g22_sq * H22)

        # print(f" Drift: {drift}")
        # print(f" Diffusion: {diff}")
        # exit(0)

        return drift + diff  # (N, 1)

    def _evaluate_f(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluate drift f for any form: constant, f(x), or f(x, u).

        Args:
            x: State tensor (N, 2)
            u: Control tensor (optional)

        Returns:
            Drift tensor (N, 2)
        """
        if callable(self.f):
            # Check if f accepts u parameter
            import inspect
            sig = inspect.signature(self.f)
            num_params = len([p for p in sig.parameters.values()
                            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)])

            if num_params >= 2 and u is not None:
                f_result = self.f(x, u)  # f(x, u)
            else:
                f_result = self.f(x)  # f(x)

            # Convert numpy to torch if needed
            if isinstance(f_result, np.ndarray):
                f_result = torch.from_numpy(f_result).float().to(x.device)

            return f_result
        elif isinstance(self.f, (torch.Tensor, np.ndarray)):
            # Convert numpy to torch if needed
            if isinstance(self.f, np.ndarray):
                f_tensor = torch.from_numpy(self.f).float().to(x.device)
            else:
                f_tensor = self.f
            # Constant matrix: f = F @ x
            return x @ f_tensor.T  # (N, 2)
        else:
            raise ValueError(f"Unsupported f type: {type(self.f)}")

    def _compute_gg_diag(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Compute diagonal of g(x) @ g(x)^T for any form of g.

        Handles:
        - Constant g (scalar, vector, or matrix)
        - g(x) returning vector (diagonal)
        - g(x) returning matrix (full 2x2, can have cross terms)
        - g(x, u) returning vector or matrix

        Args:
            x: State tensor (N, 2)
            u: Control tensor (optional)

        Returns:
            Diagonal of G @ G^T: (N, 2)
        """
        # print(f"DEBUG _compute_gg_diag: x.shape = {x.shape}, type(self.g) = {type(self.g)}")
        if self.g is None:
            # No diffusion
            return torch.zeros_like(x)

        if callable(self.g):
            # Check if g has a CROWN-compatible get_diagonal_squared method
            if hasattr(self.g, 'get_diagonal_squared'):
                # Use the optimized CROWN-compatible method directly
                g_diag_sq = self.g.get_diagonal_squared(x)  # (N, 2)
                # Convert numpy to torch if needed
                if isinstance(g_diag_sq, np.ndarray):
                    g_diag_sq = torch.from_numpy(g_diag_sq).float().to(x.device)
                return g_diag_sq

            # Fallback: call g and extract diagonal
            # Check if g accepts u parameter
            import inspect
            sig = inspect.signature(self.g)
            num_params = len([p for p in sig.parameters.values()
                            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)])

            if num_params >= 2 and u is not None:
                g_result = self.g(x, u)  # g(x, u)
            else:
                g_result = self.g(x)  # g(x)

            # Convert numpy to torch if needed
            if isinstance(g_result, np.ndarray):
                g_result = torch.from_numpy(g_result).float().to(x.device)

            # Determine what g returned based on shape
            ndim = len(g_result.shape)
            if ndim == 2:
                # Returns diagonal vector (N, 2): treat as diag(g)
                # (G @ G^T)_ii = g_i^2
                return g_result ** 2
            elif ndim == 3:
                # Returns full matrix (N, 2, 2)
                # Compute G @ G^T and extract diagonal (CROWN-compatible)
                GGT = torch.bmm(g_result, g_result.transpose(1, 2))  # (N, 2, 2)
                # Extract diagonal using sum trick (avoid ScatterND from indexing)
                # Create mask for diagonal elements
                diag_mask = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, 2, 2)
                g_diag_sq = (GGT * diag_mask).sum(dim=2)  # (N, 2)
                return g_diag_sq
            else:
                raise ValueError(f"Unexpected g output shape: {g_result.shape}")

        elif isinstance(self.g, (torch.Tensor, np.ndarray)):
            # Convert numpy to torch if needed
            if isinstance(self.g, np.ndarray):
                g_tensor = torch.from_numpy(self.g).float().to(x.device)
            else:
                g_tensor = self.g

            # print(f"DEBUG: g_tensor.shape = {g_tensor.shape}, g_tensor.dim() = {g_tensor.dim()}")

            # Constant g - use broadcasting with x to get batch size
            if g_tensor.dim() == 0:
                # Scalar: g = σ * I
                g_sq = g_tensor ** 2
                # Broadcast to (N, 2) using x as template
                return g_sq * torch.ones_like(x)
            elif g_tensor.dim() == 1:
                # Vector (2,): diagonal diffusion g = diag(g)
                g_sq = g_tensor ** 2  # (2,)
                # Broadcast: (2,) -> (N, 2) by adding zeros_like
                # This ensures proper batch dimension
                result = g_sq.view(1, 2) + torch.zeros_like(x)
                # print(f"DEBUG: dim==1, g_sq.shape = {g_sq.shape}, result.shape = {result.shape}")
                return result
            elif g_tensor.dim() == 2:
                # Matrix (2, 2): compute diagonal of G @ G^T (CROWN-compatible)
                GGT = g_tensor @ g_tensor.T  # (2, 2)
                # print(f"DEBUG: dim==2, GGT.shape = {GGT.shape}")
                # Extract diagonal using sum trick (avoid ScatterND)
                diag_mask = torch.eye(2, device=x.device, dtype=x.dtype)  # (2, 2)
                g_diag = (GGT * diag_mask).sum(dim=1)  # (2,)
                # print(f"DEBUG: g_diag.shape = {g_diag.shape}, x.shape = {x.shape}")
                # Broadcast: (2,) -> (N, 2)
                result = g_diag.view(1, 2) + torch.zeros_like(x)
                # print(f"DEBUG: result.shape = {result.shape}")
                return result
            else:
                raise ValueError(f"Unexpected g shape: {g_tensor.shape}")

        elif isinstance(self.g, (int, float)):
            # Scalar constant: g = σ
            g_sq = self.g ** 2
            # Broadcast to (N, 2)
            return g_sq * torch.ones_like(x)

        else:
            raise ValueError(f"Unsupported g type: {type(self.g)}")


def create_GV(
    V_net: nn.Module,
    dynamics: Dynamics,
    network_config,  # NetworkConfig
    training_config  # TrainingConfig
) -> GV:
    """
    Factory function to create GV (Phi) module.

    Args:
        V_net: Value function network (GV-specific)
        dynamics: System dynamics (GV-specific)
        network_config: NetworkConfig with scale_factor and input_scale (shared with V)
        training_config: TrainingConfig with learnable flags (shared with V)

    Returns:
        GV instance
    """
    return GV(
        V_net=V_net,
        dynamics=dynamics,
        scale_factor=network_config.scale_factor,
        learnable_scale=training_config.learnable_scale,
        learnable_input_scale=training_config.learnable_input_scale,
        input_scale_init=network_config.input_scale
    )
