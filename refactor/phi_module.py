"""
Phi module for computing the infinitesimal generator of the value function.

The infinitesimal generator for a stochastic system dx = f(x)dt + g(x)dW is:
    Φ(x) = f(x) · ∇V + 0.5 * Tr(g(x)g(x)^T @ H_V)

This module properly handles:
- General drift matrices F (not just hardcoded A)
- General diffusion matrices G (not just R[0,0])
- Diagonal and non-diagonal diffusion
- State-dependent diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dynamics import Dynamics


class GV(nn.Module):
    """
    Computes Φ(x) = f(x)·∇V + 0.5·Tr(g(x)g(x)^T @ H_V) using closed-form derivatives.

    Supports:
    - General linear drift f(x) = F @ x
    - General constant diffusion g(x) = G (constant matrix)
    - Diagonal diffusion (optimized path using only diagonal Hessian)
    - Non-diagonal diffusion (requires full Hessian computation)

    Key features:
    - Uses REFERENCES to V_net's layers (not detached copies)
    - Gradients flow back to V_net's parameters during training
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
            input_scale_init = V_net.input_scale.item()

        self.learnable_input_scale = learnable_input_scale
        if learnable_input_scale:
            self.input_scale = nn.Parameter(torch.tensor(input_scale_init, dtype=torch.float32))
        else:
            self.register_buffer('input_scale', torch.tensor(input_scale_init, dtype=torch.float32))
            self.register_buffer('input_scale_sq', torch.tensor(input_scale_init ** 2, dtype=torch.float32))

        # Register drift and diffusion matrices
        self.register_buffer('F', dynamics.get_drift_matrix())

        G_constant = dynamics.get_diffusion_matrix()
        if G_constant is not None:
            print('FFFFFFFFFFFFFFFFFFFFFFFF')
            raise ValueError('FFFFFFFFFFFFFFFFFFFFFFFF')
            # self.register_buffer('G', G_constant)
            # # Compute G @ G^T for constant diffusion
            # self.register_buffer('GGT', G_constant @ G_constant.T)
            # self.sigma_diag = None
        else:
            # State-dependent diffusion - extract sigma values as Python floats (like original)
            self.G = None
            self.GGT = None
            # Extract sigma diagonal as Python list/floats for CROWN compatibility
            if dynamics.sigma_diag is not None:
                # Convert to Python floats (not tensors) - matches original pattern
                self.sigma_diag = [float(s) for s in dynamics.sigma_diag]
                # raise ValueError('aaaaaa')
            else:
                self.sigma_diag = None
                # raise ValueError('vvvvvvv')

        # Check if diffusion is diagonal (works for both constant and state-dependent)
        self.is_diagonal_diffusion = dynamics.is_diffusion_diagonal()

        print(f"[PhiModule] Initialized with:")
        print(f"  scale_factor: {self.scale_factor.item() if hasattr(self.scale_factor, 'item') else self.scale_factor}")
        print(f"  input_scale: {self.input_scale.item() if hasattr(self.input_scale, 'item') else self.input_scale}")
        print(f"  Drift F:\n{self.F.numpy()}")
        if self.G is not None:
            raise ValueError('ggggggg')
            # print(f"  Diffusion G:\n{self.G.numpy()}")
            # print(f"  G@G^T:\n{self.GGT.numpy()}")
            # print(f"  Diagonal diffusion: {self.is_diagonal_diffusion}")
        else:
            print(f"  Diffusion: state-dependent")
            print(f"  Diagonal diffusion: {self.is_diagonal_diffusion}")
            if self.sigma_diag is not None:
                print(f"  Sigma diagonal: {self.sigma_diag}")

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
        dVdx1 = dVdx1_norm / self.input_scale  # (N, 1)
        dVdx2 = dVdx2_norm / self.input_scale  # (N, 1)

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
        H11 = H11_norm / input_scale_sq  # (N, 1)
        H22 = H22_norm / input_scale_sq  # (N, 1)

        # Compute f(x) = F·x using ORIGINAL UNNORMALIZED coordinates
        # (Dynamics operate on the physical state space, not the normalized NN input)
        fx = self.dynamics.drift(x_orig)  # (N, 2)
        f1 = fx[:, 0:1]    # (N, 1)
        f2 = fx[:, 1:2]    # (N, 1)

        # State-dependent diffusion: g(x) = σ·x using ORIGINAL UNNORMALIZED coordinates
        x1 = x_orig[:, 0:1]  # (N, 1)
        x2 = x_orig[:, 1:2]  # (N, 1)

        if self.sigma_diag is not None:
            g11_sq = (self.sigma_diag[0] * x1) ** 2  # (N, 1)
            g22_sq = (self.sigma_diag[1] * x2) ** 2  # (N, 1)
        else:
            raise ValueError('hhhhhh')
            # Constant diffusion
            g11_sq = self.GGT[0, 0]
            g22_sq = self.GGT[1, 1]

        # Φ(x) = f·∇V + 0.5·(g²·H_diag)
        drift = f1 * dVdx1 + f2 * dVdx2
        diff = 0.5 * (g11_sq * H11 + g22_sq * H22)

        # print(f" Drift: {drift}")
        # print(f" Diffusion: {diff}")
        # exit(0)

        return drift + diff  # (N, 1)

    def _compute_diffusion_diagonal(
        self,
        x: torch.Tensor,
        W0, W1, W2,
        d0, d1, q0, q1,
        W1_scaled,
        state_dim: int
    ) -> torch.Tensor:
        """
        Compute diffusion term for diagonal diffusion: 0.5 * Σ_i [GG^T]_{ii} * H_{ii}.

        Args:
            x: Original state (N, state_dim)
            W0, W1, W2: Network weights
            d0, d1: First derivatives of activation
            q0, q1: Second derivatives of activation
            W1_scaled: Pre-computed W1 * d0
            state_dim: State dimension

        Returns:
            Diffusion term (N, 1)
        """
        # Compute input_scale_sq
        if self.learnable_input_scale:
            input_scale_sq = self.input_scale ** 2
        else:
            input_scale_sq = self.input_scale_sq

        # Compute Hessian diagonal elements
        W1_q0 = W1.unsqueeze(0) * q0.unsqueeze(1)  # (N, m1, m0)

        # Compute diffusion term using CROWN-compatible operations
        diffusion_sum = 0.0
        for i in range(state_dim):
            # Compute H_{ii} (Hessian diagonal element)
            sum_over_k = (W1_scaled * W0[:, i].view(1, 1, -1)).sum(dim=2)  # (N, m1)

            # Cross term: σ''(z1) * (Σ_k W1·σ'(z0)·W0)^2
            cross_term = q1 * (sum_over_k ** 2)  # (N, m1)

            # Direct term: σ'(z1) * Σ_k W1·σ''(z0)·W0^2
            W0_i_sq = W0[:, i] ** 2
            direct_term = d1 * (W1_q0 * W0_i_sq.view(1, 1, -1)).sum(dim=2)  # (N, m1)

            H_ii_norm = self.scale_factor * (W2 * (cross_term + direct_term)).sum(dim=1, keepdim=True)  # (N, 1)
            H_ii = H_ii_norm / input_scale_sq  # (N, 1)

            # Compute g_ii(x)^2 for this dimension (matching original's pattern)
            if self.sigma_diag is not None:
                # State-dependent: g_ii(x) = sigma_i * x_i (sigma_i is Python float)
                x_i = x[:, i:i+1]  # (N, 1)
                g_ii_sq = (self.sigma_diag[i] * x_i) ** 2  # (N, 1), CROWN-compatible

                # Debug on first call
                if not hasattr(self, '_debug_printed'):
                    if i == 0:
                        print(f"DEBUG diffusion: sigma_diag[{i}]={self.sigma_diag[i]}, x_i sample={x_i[0].item():.3f}, g_ii_sq sample={g_ii_sq[0].item():.6f}")
            else:
                # Constant diffusion: use pre-computed GGT diagonal
                g_ii_sq = self.GGT[i, i]  # scalar

            # Add contribution: g_{ii}^2 * H_{ii}
            diffusion_sum = diffusion_sum + g_ii_sq * H_ii

        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True

        return 0.5 * diffusion_sum  # (N, 1)

    def _compute_diffusion_full(
        self,
        x: torch.Tensor,
        W0, W1, W2,
        d0, d1, q0, q1,
        W1_scaled,
        state_dim: int
    ) -> torch.Tensor:
        """
        Compute diffusion term for non-diagonal diffusion: 0.5 * Tr(GG^T @ H_V).

        This requires computing the full Hessian matrix, not just the diagonal.

        Args:
            x: Original state (N, state_dim)
            W0, W1, W2: Network weights
            d0, d1: First derivatives of activation
            q0, q1: Second derivatives of activation
            W1_scaled: Pre-computed W1 * d0
            state_dim: State dimension

        Returns:
            Diffusion term (N, 1)
        """
        # Compute input_scale_sq
        if self.learnable_input_scale:
            input_scale_sq = self.input_scale ** 2
        else:
            input_scale_sq = self.input_scale_sq

        batch_size = x.shape[0]

        # Compute full Hessian: H_V[i,j] = ∂²V / ∂x_i ∂x_j
        # This is more expensive but necessary for non-diagonal diffusion

        # Pre-compute terms for efficiency
        W1_q0 = W1.unsqueeze(0) * q0.unsqueeze(1)  # (N, m1, m0)

        # Compute Hessian matrix for each batch element
        # For efficiency, we'll compute it in a vectorized way
        H_full = torch.zeros(batch_size, state_dim, state_dim, device=x.device, dtype=x.dtype)

        for i in range(state_dim):
            for j in range(state_dim):
                # Compute H_{ij}
                if i == j:
                    # Diagonal: use same formula as before
                    sum_over_k_i = (W1_scaled * W0[:, i].view(1, 1, -1)).sum(dim=2)
                    cross_term = q1 * (sum_over_k_i ** 2)
                    W0_i_sq = W0[:, i] ** 2
                    direct_term = d1 * (W1_q0 * W0_i_sq.view(1, 1, -1)).sum(dim=2)
                    H_ij_norm = self.scale_factor * (W2 * (cross_term + direct_term)).sum(dim=1)
                else:
                    # Off-diagonal: only cross term (no direct term for i≠j)
                    sum_over_k_i = (W1_scaled * W0[:, i].view(1, 1, -1)).sum(dim=2)
                    sum_over_k_j = (W1_scaled * W0[:, j].view(1, 1, -1)).sum(dim=2)
                    cross_term = q1 * sum_over_k_i * sum_over_k_j
                    H_ij_norm = self.scale_factor * (W2 * cross_term).sum(dim=1)

                H_full[:, i, j] = H_ij_norm / input_scale_sq

        # Compute Tr(GG^T @ H_V) for each batch element
        if self.GGT is not None:
            # Constant diffusion: GG^T is the same for all batch elements
            # Tr(GG^T @ H) = Σ_i Σ_j [GG^T]_{ij} * H_{ji}
            diffusion_term = torch.einsum('ij,bij->b', self.GGT, H_full).unsqueeze(1)  # (N, 1)
        else:
            # State-dependent diffusion: compute GG^T for each state
            GGT_batch = self.dynamics.diffusion_squared(x)  # (N, state_dim, state_dim)
            # Tr(GG^T @ H) = Σ_i Σ_j [GG^T]_{ij} * H_{ji}
            diffusion_term = torch.einsum('bij,bij->b', GGT_batch, H_full).unsqueeze(1)  # (N, 1)

        return 0.5 * diffusion_term  # (N, 1)


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
