import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
import sys
sys.path.insert(0, str(ROOT))
from src.dynamics import Dynamics


class GV(nn.Module):
    """
    Infinitesimal generator G applied to value function V
    """

    def __init__(
        self,
        V_net: nn.Module,
        dynamics: Dynamics,
        scale_factor: float = 1.0,
        input_scale_init: float = None
    ):
        super().__init__()
        self.V_net = V_net
        self.dynamics = dynamics

        # Scale factor
        self.register_buffer("scale_factor", torch.tensor(scale_factor, dtype=torch.float32))

        # Input scale
        if input_scale_init is None:
            input_scale_init = V_net.input_scale

        s = torch.tensor(input_scale_init, dtype=torch.float32)
        self.register_buffer("input_scale", s)
        self.register_buffer("input_scale_sq", s ** 2)

        # Cached values
        self._cached_scale_D = None
        self._cached_inv_scale = None
        self._cached_inv_scale_sq = None

        # Dynamics
        self.f = dynamics.get_f()
        self.g = dynamics.get_g()

        self._init_f_dispatch()
        self._init_g_dispatch()

        print("Phi initialized with:")
        print(f" input_scale: {self.input_scale.detach().cpu().tolist() if self.input_scale.numel() > 1 else float(self.input_scale.detach().cpu())}")
        print(f" scale_factor: {self.scale_factor.detach().cpu()}")
        print(f" f type: {type(self.f)}")
        print(f" g type: {type(self.g)}")

    # -------------------------
    # forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (D,) or (N,D)
        Returns:
            (1,) if input was (D,), else (N,1)
        """
        x, was_1d = self._as_batch(x)
        N, D = x.shape

        W0, b0, W1, b1, W2 = self._get_V_params()  # W2: (1,m1)
        W2v = W2.view(-1)  # (m1,)

        inv_scale, inv_scale_sq = self._get_inv_scales(D, x.device, x.dtype)  # (D,), (D,)

        # normalize (multiplication is a bit cheaper than division)
        x_norm = x * inv_scale

        # layer 1
        z0 = F.linear(x_norm, W0, b0)     # (N,m0)
        h0, d0, q0 = self._sigmoid_derivs(z0)

        # layer 2
        z1 = F.linear(h0, W1, b1)         # (N,m1)
        h1, d1, q1 = self._sigmoid_derivs(z1)

        # S = W1[j,k] * d0[n,k] -> (N,m1,m0)
        S = W1.unsqueeze(0) * d0.unsqueeze(1)

        # sum_over_k_all = S @ W0  -> (N,m1,D)   (broadcasted matmul; avoids expand + bmm)
        sum_over_k_all = torch.matmul(S, W0)  # (N,m1,D)

        # Compute gradient and Hessian in one pass to reuse computations
        # Pre-compute W2v * d1 once for both gradient and direct Hessian term
        W2v_d1 = W2v.unsqueeze(0) * d1  # (N,m1)
        W2v_d1_3d = W2v_d1.unsqueeze(2)  # (N,m1,1) for broadcasting

        # gradient wrt normalized coords:
        # dVdx_norm[n,d] = scale * sum_j W2[j] * d1[n,j] * sum_over_k_all[n,j,d]
        dVdx_norm = self.scale_factor * (W2v_d1_3d * sum_over_k_all).sum(dim=1)  # (N,D)

        # chain rule back to x: dVdx = dVdx_norm / input_scale = dVdx_norm * inv_scale
        dVdx = dVdx_norm * inv_scale

        # Hessian diagonal wrt normalized coords:
        # cross term: W2v * q1 * sum_over_k_all^2
        cross = (W2v.unsqueeze(0) * q1).unsqueeze(2) * sum_over_k_all.square()  # (N,m1,D)

        # direct term: W2v * d1 * ( (W1 * q0) @ (W0^2) ) - reuses W2v_d1_3d
        direct = W2v_d1_3d * torch.matmul(
            W1.unsqueeze(0) * q0.unsqueeze(1),  # (N,m1,m0)
            W0.square()                          # (m0,D)
        )  # (N,m1,D)

        # Fuse final operations: scale * sum * inv_scale_sq
        Hdiag = (self.scale_factor * inv_scale_sq) * (cross + direct).sum(dim=1)  # (N,D)

        # drift and diffusion terms - compute and fuse in one pass
        # fx = self._evaluate_f_fast(x)  # (N,D)
        # g_diag_sq = self._compute_gg_diag_fast(x)  # (N,D)

        # # Fuse: out = (fx * dVdx).sum() + 0.5 * (g_diag_sq * Hdiag).sum()
        # out = ((fx * dVdx) + (0.5 * g_diag_sq * Hdiag)).sum(dim=1, keepdim=True)  # (N,1)
        # --- drift contribution ---
        if hasattr(self.f, "support") and callable(getattr(self.f, "support")):
            # worst-case drift dot grad: (N,1)
            drift_term = self.f.support(x, dVdx)
        else:
            fx = self._evaluate_f_fast(x)  # (N,D)
            drift_term = (fx * dVdx).sum(dim=1, keepdim=True)

        # --- diffusion contribution (same as before) ---
        g_diag_sq = self._compute_gg_diag_fast(x)  # (N,D)
        diff_term = (0.5 * g_diag_sq * Hdiag).sum(dim=1, keepdim=True)

        out = drift_term + diff_term

        return out.squeeze(0) if was_1d else out

    # -------------------------
    # cached-scale helpers
    # -------------------------
    def _get_inv_scales(self, D: int, device, dtype):
        """
        Returns:
            inv_scale: (D,)
            inv_scale_sq: (D,)
        """
        # check cache
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            cache_ok = (
                self._cached_scale_D == D
                and self._cached_inv_scale is not None
                and self._cached_inv_scale.device == device
                and self._cached_inv_scale.dtype == dtype
            )
        if cache_ok:
            return self._cached_inv_scale, self._cached_inv_scale_sq

        s = self._expand_scale(self.input_scale, D).to(device=device, dtype=dtype)
        s2 = self._expand_scale(self.input_scale_sq, D).to(device=device, dtype=dtype)

        inv = s.reciprocal()
        inv_sq = s2.reciprocal()

        self._cached_scale_D = D
        self._cached_inv_scale = inv
        self._cached_inv_scale_sq = inv_sq
        return inv, inv_sq

    # -------------------------
    # small internal helpers
    # -------------------------
    @staticmethod
    def _as_batch(x: torch.Tensor):
        was_1d = (x.dim() == 1)
        return (x.unsqueeze(0) if was_1d else x), was_1d

    @staticmethod
    def _expand_scale(scale: torch.Tensor, D: int) -> torch.Tensor:
        return scale.expand(D) if scale.numel() == 1 else scale.view(-1)

    def _activation_derivs(self, z: torch.Tensor):
        """Compute activation, first derivative, and second derivative using autograd."""
        with torch.enable_grad():
            z_leaf = z.detach().requires_grad_(True)
            h = self.V_net.activation_fn(z_leaf)

            # First derivative
            d = torch.autograd.grad(h.sum(), z_leaf, create_graph=True)[0]

            # Second derivative
            q = torch.autograd.grad(d.sum(), z_leaf)[0]

        return h.detach(), d.detach(), q.detach()
    
    @staticmethod
    def _sigmoid_derivs(z: torch.Tensor):
        h = torch.sigmoid(z)
        d = h * (1.0 - h)           # σ'
        q = (1.0 - 2.0 * h) * d     # σ''
        return h, d, q

    def _get_V_params(self):
        return (
            self.V_net.layer1.weight,
            self.V_net.layer1.bias,
            self.V_net.layer2.weight,
            self.V_net.layer2.bias,
            self.V_net.output.weight,  # (1,m1)
        )

    # -------------------------
    # drift / diffusion dispatch init (done once)
    # -------------------------
    def _init_f_dispatch(self):
        self._f_callable = callable(self.f)
        self._f_expects_u = False

        # convert constant arrays to buffers once (so .to(device) works)
        if isinstance(self.f, np.ndarray):
            f_t = torch.tensor(self.f, dtype=torch.float32)
            self.register_buffer("_f_const", f_t)
            self.f = self._f_const
            self._f_callable = False

        if self._f_callable:
            import inspect
            sig = inspect.signature(self.f)
            self._f_expects_u = (len(sig.parameters) >= 2)

        self._f_is_tensor = isinstance(self.f, torch.Tensor)
        self._f_is_numpy = isinstance(self.f, np.ndarray)

    def _init_g_dispatch(self):
        self._g_is_none = (self.g is None)
        self._g_callable = callable(self.g)
        self._g_has_diag_sq = (self._g_callable and hasattr(self.g, "get_diagonal_squared"))

        # convert constant arrays to buffers once
        if isinstance(self.g, np.ndarray):
            g_t = torch.tensor(self.g, dtype=torch.float32)
            self.register_buffer("_g_const", g_t)
            self.g = self._g_const
            self._g_callable = False
            self._g_has_diag_sq = False

        self._g_is_tensor = isinstance(self.g, torch.Tensor)
        self._g_is_float = isinstance(self.g, (int, float))
        self._g_expects_u = False
        if self._g_callable and (not self._g_has_diag_sq):
            import inspect
            sig = inspect.signature(self.g)
            self._g_expects_u = (len(sig.parameters) >= 2)

    # -------------------------
    # drift / diffusion (fast paths)
    # -------------------------
    def _evaluate_f_fast(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Returns f(x) in shape (N,D). Fast dispatch with cached info.
        """
        if self._f_callable:
            f_result = self.f(x, u) if (self._f_expects_u and u is not None) else self.f(x)
            if isinstance(f_result, np.ndarray):
                # still supported, but slower than returning torch directly
                f_result = torch.from_numpy(f_result).to(device=x.device, dtype=x.dtype)
            return f_result

        # linear drift matrix form: x @ A^T
        if isinstance(self.f, torch.Tensor):
            f_tensor = self.f.to(device=x.device, dtype=x.dtype)
            return x @ f_tensor.T

        raise ValueError(f"Unsupported f type: {type(self.f)}")

    def _compute_gg_diag_fast(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Returns diag(GG^T)(x) in shape (N,D). Fast paths:
          - If G is (N,D,m) or (N,D,D): diag(GG^T) = sum_k G^2 along last dim
          - If G is constant (D,D): diag(GG^T) = row-wise sum of squares of G
        """
        if self._g_is_none:
            return torch.zeros_like(x)

        N, D = x.shape

        if self._g_callable:
            if self._g_has_diag_sq:
                g_diag_sq = self.g.get_diagonal_squared(x)
                if isinstance(g_diag_sq, np.ndarray):
                    g_diag_sq = torch.from_numpy(g_diag_sq).to(device=x.device, dtype=x.dtype)
                return g_diag_sq

            g_out = self.g(x, u) if (self._g_expects_u and u is not None) else self.g(x)
            if isinstance(g_out, np.ndarray):
                g_out = torch.from_numpy(g_out).to(device=x.device, dtype=x.dtype)

            if g_out.dim() == 2:
                # (N,D) diagonal diffusion vector
                return g_out.square()

            if g_out.dim() == 3:
                # (N,D,m) or (N,D,D): diag(GG^T) = sum_k G_{i,k}^2
                if g_out.shape[1] != D:
                    raise ValueError(f"Expected g_out shape (N,D,*) but got {tuple(g_out.shape)} with D={D}")
                return g_out.square().sum(dim=2)

            raise ValueError(f"Unexpected g output shape: {tuple(g_out.shape)}")

        # constant tensor / scalar g
        if self._g_is_float:
            return (float(self.g) ** 2) * torch.ones_like(x)

        if isinstance(self.g, torch.Tensor):
            g_t = self.g.to(device=x.device, dtype=x.dtype)

            if g_t.dim() == 0:
                return g_t.square() * torch.ones_like(x)

            if g_t.dim() == 1:
                if g_t.numel() != D:
                    raise ValueError(f"g vector dim {g_t.numel()} != state dim {D}")
                return g_t.view(1, D).square().expand_as(x)

            if g_t.dim() == 2:
                if g_t.shape != (D, D):
                    raise ValueError(f"g matrix shape {tuple(g_t.shape)} not (D,D) with D={D}")
                # diag(GG^T) = row-wise sum of squares
                diag = g_t.square().sum(dim=1)  # (D,)
                return diag.view(1, D).expand_as(x)

            raise ValueError(f"Unexpected g shape: {tuple(g_t.shape)}")

        raise ValueError(f"Unsupported g type: {type(self.g)}")


class GV_offset(nn.Module):
    """
    Optimized GV:
      - caches callable signatures once
      - computes diag(GG^T) via sum of squares (no bmm, no eye mask)
      - avoids per-forward expand() + bmm(...) patterns by using matmul broadcasting
      - caches expanded input scales when not learnable
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
        super().__init__()
        self.V_net = V_net
        self.dynamics = dynamics

        # -------------------------
        # scale factor
        # -------------------------
        if learnable_scale:
            self.scale_factor = nn.Parameter(torch.tensor(scale_factor, dtype=torch.float32))
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor, dtype=torch.float32))

        # -------------------------
        # input scale
        # -------------------------
        if input_scale_init is None:
            input_scale_init = V_net.input_scale

        self.learnable_input_scale = learnable_input_scale
        if learnable_input_scale:
            self.input_scale = nn.Parameter(torch.tensor(input_scale_init, dtype=torch.float32))
            # no caching when learnable
            self._cached_scale_D = None
            self._cached_inv_scale = None
            self._cached_inv_scale_sq = None
        else:
            s = torch.tensor(input_scale_init, dtype=torch.float32)
            self.register_buffer("input_scale", s)
            self.register_buffer("input_scale_sq", s ** 2)

            # cache expanded (D,) inverse scales for speed
            self._cached_scale_D = None
            self._cached_inv_scale = None
            self._cached_inv_scale_sq = None

        # input offset: use V_net.input_offset if present, else zeros
        offset_init = getattr(V_net, "input_offset", torch.zeros_like(self.input_scale))
        self.register_buffer("input_offset",
                            torch.as_tensor(offset_init, dtype=torch.float32).detach().clone())


        # -------------------------
        # dynamics (cache dispatch decisions once)
        # -------------------------
        self.f = dynamics.get_f()
        self.g = dynamics.get_g()

        self._init_f_dispatch()
        self._init_g_dispatch()

        print("[PhiModule] Initialized with:")
        print(f"  scale_factor: {float(self.scale_factor.detach().cpu())}")
        print(f"  input_scale: {self.input_scale.detach().cpu().tolist() if self.input_scale.numel() > 1 else float(self.input_scale.detach().cpu())}")
        print(f"  f type: {type(self.f)}")
        print(f"  g type: {type(self.g)}")

    # -------------------------
    # forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (D,) or (N,D)
        Returns:
            (1,) if input was (D,), else (N,1)
        """
        x, was_1d = self._as_batch(x)
        N, D = x.shape

        W0, b0, W1, b1, W2 = self._get_V_params()  # W2: (1,m1)
        W2v = W2.view(-1)  # (m1,)

        inv_scale, inv_scale_sq = self._get_inv_scales(D, x.device, x.dtype)  # (D,), (D,)

        # normalize (multiplication is a bit cheaper than division)
        # x_norm = x * inv_scale
        # NEW: match V normalization: (x - offset)/scale
        offset = self._expand_scale(self.input_offset.to(device=x.device, dtype=x.dtype), D)  # (D,)
        x_norm = (x - offset) * inv_scale


        # layer 1
        z0 = F.linear(x_norm, W0, b0)     # (N,m0)
        h0, d0, q0 = self._sigmoid_derivs(z0)

        # layer 2
        z1 = F.linear(h0, W1, b1)         # (N,m1)
        h1, d1, q1 = self._sigmoid_derivs(z1)

        # S = W1[j,k] * d0[n,k] -> (N,m1,m0)
        S = W1.unsqueeze(0) * d0.unsqueeze(1)

        # sum_over_k_all = S @ W0  -> (N,m1,D)   (broadcasted matmul; avoids expand + bmm)
        sum_over_k_all = torch.matmul(S, W0)  # (N,m1,D)

        # Compute gradient and Hessian in one pass to reuse computations
        # Pre-compute W2v * d1 once for both gradient and direct Hessian term
        W2v_d1 = W2v.unsqueeze(0) * d1  # (N,m1)
        W2v_d1_3d = W2v_d1.unsqueeze(2)  # (N,m1,1) for broadcasting

        # gradient wrt normalized coords:
        # dVdx_norm[n,d] = scale * sum_j W2[j] * d1[n,j] * sum_over_k_all[n,j,d]
        dVdx_norm = self.scale_factor * (W2v_d1_3d * sum_over_k_all).sum(dim=1)  # (N,D)

        # chain rule back to x: dVdx = dVdx_norm / input_scale = dVdx_norm * inv_scale
        dVdx = dVdx_norm * inv_scale

        # Hessian diagonal wrt normalized coords:
        # cross term: W2v * q1 * sum_over_k_all^2
        cross = (W2v.unsqueeze(0) * q1).unsqueeze(2) * sum_over_k_all.square()  # (N,m1,D)

        # direct term: W2v * d1 * ( (W1 * q0) @ (W0^2) ) - reuses W2v_d1_3d
        direct = W2v_d1_3d * torch.matmul(
            W1.unsqueeze(0) * q0.unsqueeze(1),  # (N,m1,m0)
            W0.square()                          # (m0,D)
        )  # (N,m1,D)

        # Fuse final operations: scale * sum * inv_scale_sq
        Hdiag = (self.scale_factor * inv_scale_sq) * (cross + direct).sum(dim=1)  # (N,D)

        # drift and diffusion terms - compute and fuse in one pass
        # fx = self._evaluate_f_fast(x)  # (N,D)
        # g_diag_sq = self._compute_gg_diag_fast(x)  # (N,D)

        # # Fuse: out = (fx * dVdx).sum() + 0.5 * (g_diag_sq * Hdiag).sum()
        # out = ((fx * dVdx) + (0.5 * g_diag_sq * Hdiag)).sum(dim=1, keepdim=True)  # (N,1)

        if hasattr(self.f, "support") and callable(getattr(self.f, "support")):
            # worst-case drift dot grad: (N,1)
            drift_term = self.f.support(x, dVdx)
        else:
            fx = self._evaluate_f_fast(x)  # (N,D)
            drift_term = (fx * dVdx).sum(dim=1, keepdim=True)

        # --- diffusion contribution (same as before) ---
        g_diag_sq = self._compute_gg_diag_fast(x)  # (N,D)
        diff_term = (0.5 * g_diag_sq * Hdiag).sum(dim=1, keepdim=True)

        out = drift_term + diff_term
        
        return out.squeeze(0) if was_1d else out

    # -------------------------
    # cached-scale helpers
    # -------------------------
    def _get_inv_scales(self, D: int, device, dtype):
        """
        Returns:
            inv_scale: (D,)
            inv_scale_sq: (D,)
        """
        # learnable => recompute each call (cheap compared to rest, but avoids stale cache)
        if self.learnable_input_scale:
            s = self._expand_scale(self.input_scale.to(device=device, dtype=dtype), D)
            inv = s.reciprocal()
            inv_sq = (s * s).reciprocal()
            return inv, inv_sq

        # non-learnable => cache expanded vectors on the right device/dtype
        cache_ok = (
            self._cached_scale_D == D
            and self._cached_inv_scale is not None
            and self._cached_inv_scale.device == device
            and self._cached_inv_scale.dtype == dtype
        )
        if cache_ok:
            return self._cached_inv_scale, self._cached_inv_scale_sq

        s = self._expand_scale(self.input_scale, D).to(device=device, dtype=dtype)
        s2 = self._expand_scale(self.input_scale_sq, D).to(device=device, dtype=dtype)

        inv = s.reciprocal()
        inv_sq = s2.reciprocal()

        self._cached_scale_D = D
        self._cached_inv_scale = inv
        self._cached_inv_scale_sq = inv_sq
        return inv, inv_sq

    # -------------------------
    # small internal helpers
    # -------------------------
    @staticmethod
    def _as_batch(x: torch.Tensor):
        was_1d = (x.dim() == 1)
        return (x.unsqueeze(0) if was_1d else x), was_1d

    @staticmethod
    def _expand_scale(scale: torch.Tensor, D: int) -> torch.Tensor:
        return scale.expand(D) if scale.numel() == 1 else scale.view(-1)

    @staticmethod
    def _sigmoid_derivs(z: torch.Tensor):
        h = torch.sigmoid(z)
        d = h * (1.0 - h)           # σ'
        q = (1.0 - 2.0 * h) * d     # σ''
        return h, d, q

    def _get_V_params(self):
        return (
            self.V_net.layer1.weight,
            self.V_net.layer1.bias,
            self.V_net.layer2.weight,
            self.V_net.layer2.bias,
            self.V_net.output.weight,  # (1,m1)
        )

    # -------------------------
    # drift / diffusion dispatch init (done once)
    # -------------------------
    def _init_f_dispatch(self):
        self._f_callable = callable(self.f)
        self._f_expects_u = False

        # convert constant arrays to buffers once (so .to(device) works)
        if isinstance(self.f, np.ndarray):
            f_t = torch.tensor(self.f, dtype=torch.float32)
            self.register_buffer("_f_const", f_t)
            self.f = self._f_const
            self._f_callable = False

        if self._f_callable:
            import inspect
            sig = inspect.signature(self.f)
            self._f_expects_u = (len(sig.parameters) >= 2)

        self._f_is_tensor = isinstance(self.f, torch.Tensor)
        self._f_is_numpy = isinstance(self.f, np.ndarray)

    def _init_g_dispatch(self):
        self._g_is_none = (self.g is None)
        self._g_callable = callable(self.g)
        self._g_has_diag_sq = (self._g_callable and hasattr(self.g, "get_diagonal_squared"))

        # convert constant arrays to buffers once
        if isinstance(self.g, np.ndarray):
            g_t = torch.tensor(self.g, dtype=torch.float32)
            self.register_buffer("_g_const", g_t)
            self.g = self._g_const
            self._g_callable = False
            self._g_has_diag_sq = False

        self._g_is_tensor = isinstance(self.g, torch.Tensor)
        self._g_is_float = isinstance(self.g, (int, float))
        self._g_expects_u = False
        if self._g_callable and (not self._g_has_diag_sq):
            import inspect
            sig = inspect.signature(self.g)
            self._g_expects_u = (len(sig.parameters) >= 2)

    # -------------------------
    # drift / diffusion (fast paths)
    # -------------------------
    def _evaluate_f_fast(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Returns f(x) in shape (N,D). Fast dispatch with cached info.
        """
        if self._f_callable:
            f_result = self.f(x, u) if (self._f_expects_u and u is not None) else self.f(x)
            if isinstance(f_result, np.ndarray):
                # still supported, but slower than returning torch directly
                f_result = torch.from_numpy(f_result).to(device=x.device, dtype=x.dtype)
            return f_result

        # linear drift matrix form: x @ A^T
        if isinstance(self.f, torch.Tensor):
            f_tensor = self.f.to(device=x.device, dtype=x.dtype)
            return x @ f_tensor.T

        raise ValueError(f"Unsupported f type: {type(self.f)}")

    def _compute_gg_diag_fast(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Returns diag(GG^T)(x) in shape (N,D). Fast paths:
          - If G is (N,D,m) or (N,D,D): diag(GG^T) = sum_k G^2 along last dim
          - If G is constant (D,D): diag(GG^T) = row-wise sum of squares of G
        """
        if self._g_is_none:
            return torch.zeros_like(x)

        N, D = x.shape

        if self._g_callable:
            if self._g_has_diag_sq:
                g_diag_sq = self.g.get_diagonal_squared(x)
                if isinstance(g_diag_sq, np.ndarray):
                    g_diag_sq = torch.from_numpy(g_diag_sq).to(device=x.device, dtype=x.dtype)
                return g_diag_sq

            g_out = self.g(x, u) if (self._g_expects_u and u is not None) else self.g(x)
            if isinstance(g_out, np.ndarray):
                g_out = torch.from_numpy(g_out).to(device=x.device, dtype=x.dtype)

            if g_out.dim() == 2:
                # (N,D) diagonal diffusion vector
                return g_out.square()

            if g_out.dim() == 3:
                # (N,D,m) or (N,D,D): diag(GG^T) = sum_k G_{i,k}^2
                if g_out.shape[1] != D:
                    raise ValueError(f"Expected g_out shape (N,D,*) but got {tuple(g_out.shape)} with D={D}")
                return g_out.square().sum(dim=2)

            raise ValueError(f"Unexpected g output shape: {tuple(g_out.shape)}")

        # constant tensor / scalar g
        if self._g_is_float:
            return (float(self.g) ** 2) * torch.ones_like(x)

        if isinstance(self.g, torch.Tensor):
            g_t = self.g.to(device=x.device, dtype=x.dtype)

            if g_t.dim() == 0:
                return g_t.square() * torch.ones_like(x)

            if g_t.dim() == 1:
                if g_t.numel() != D:
                    raise ValueError(f"g vector dim {g_t.numel()} != state dim {D}")
                return g_t.view(1, D).square().expand_as(x)

            if g_t.dim() == 2:
                if g_t.shape != (D, D):
                    raise ValueError(f"g matrix shape {tuple(g_t.shape)} not (D,D) with D={D}")
                # diag(GG^T) = row-wise sum of squares
                diag = g_t.square().sum(dim=1)  # (D,)
                return diag.view(1, D).expand_as(x)

            raise ValueError(f"Unexpected g shape: {tuple(g_t.shape)}")

        raise ValueError(f"Unsupported g type: {type(self.g)}")
    

def compute_GV_autograd(
    V_net: nn.Module,
    dynamics: Dynamics,
    x: torch.Tensor,
    scale_factor: float = 1.0
) -> torch.Tensor:
    """
    Compute G[V] using PyTorch autograd for verification.

    Args:
        V_net: Value function network
        dynamics: Dynamics with f and g
        x: State tensor (N, D)
        scale_factor: GV module's scale factor

    Returns:
        G[V](x): (N, 1)
    """
    x = x.detach().requires_grad_(True)

    # Compute V
    x_norm = x / V_net.input_scale
    h = V_net.activation_fn(V_net.layer1(x_norm))
    h = V_net.activation_fn(V_net.layer2(h))
    V = V_net.output(h)

    # Compute gradient
    dVdx_unscaled = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    # Compute Hessian diagonal if diffusion exists
    g_fn = dynamics.get_g()
    if g_fn is not None:
        H_diag = torch.zeros_like(x)
        for i in range(x.shape[1]):
            H_diag[:, i] = torch.autograd.grad(dVdx_unscaled[:, i].sum(), x, retain_graph=True)[0][:, i]
        H_diag = scale_factor * H_diag

    # Apply scale_factor to gradient
    dVdx = scale_factor * dVdx_unscaled

    # Drift term
    f_fn = dynamics.get_f()
    if callable(f_fn):
        fx = f_fn(x)
        if isinstance(fx, np.ndarray):
            fx = torch.from_numpy(fx).to(x.device, x.dtype)
    else:
        if isinstance(f_fn, np.ndarray):
            f_fn = torch.from_numpy(f_fn).to(x.device, x.dtype)
        fx = x @ f_fn.T
    drift_term = (dVdx * fx).sum(dim=1, keepdim=True)

    if g_fn is None:
        return drift_term

    # Diffusion term
    if callable(g_fn):
        g_out = g_fn(x)
        if isinstance(g_out, np.ndarray):
            g_out = torch.from_numpy(g_out).to(x.device, x.dtype)
        g_diag_sq = g_out.square() if g_out.dim() == 2 else g_out.square().sum(dim=2)
    else:
        if isinstance(g_fn, np.ndarray):
            g_fn = torch.from_numpy(g_fn).to(x.device, x.dtype)
        if isinstance(g_fn, torch.Tensor) and g_fn.dim() == 2:
            g_diag_sq = g_fn.square().sum(dim=1).unsqueeze(0).expand_as(x)
        else:
            g_diag_sq = (g_fn ** 2) * torch.ones_like(x)

    diffusion_term = (0.5 * g_diag_sq * H_diag).sum(dim=1, keepdim=True)
    return drift_term + diffusion_term


def verify_GV(
    phi_module: GV,
    dynamics: Dynamics,
    x: torch.Tensor = None,
    n_samples: int = 10,
    tol: float = 1e-4,
    verbose: bool = True
) -> bool:
    """
    Verify analytical GV matches autograd G[V].

    Args:
        phi_module: Analytical GV module
        dynamics: Dynamics object
        x: Optional sample points (N, D). If None, random samples generated
        n_samples: Number of random samples if x is None
        tol: Numerical tolerance
        verbose: Print detailed comparison info

    Returns:
        True if match within tolerance, False otherwise
    """
    if x is None:
        D = phi_module.V_net.layer1.weight.shape[1]
        x = torch.randn(n_samples, D, dtype=torch.float32) * 10.0

    # Compute both versions
    with torch.no_grad():
        analytical = phi_module(x)
    autograd = compute_GV_autograd(phi_module.V_net, dynamics, x, scale_factor=float(phi_module.scale_factor))

    # Compare
    with torch.no_grad():
        diff = (analytical - autograd).abs()
        max_diff = diff.max().item()

    if verbose:
        print(f"Phi Verification {'PASSED' if max_diff <= tol else 'FAILED'}: max diff = {max_diff:.6e}")

    return max_diff <= tol


def create_GV(
    V_net: nn.Module,
    dynamics: Dynamics,
    network_config,
    verify: bool = True,
    input_offset=None,
) -> GV:
    if(input_offset is not None):
        phi = GV_offset(
            V_net=V_net,
            dynamics=dynamics,
            scale_factor=network_config.scale_factor,
            input_scale_init=network_config.input_scale
        )
    else:
        phi = GV(
            V_net=V_net,
            dynamics=dynamics,
            scale_factor=network_config.scale_factor,
            input_scale_init=network_config.input_scale
        )

    if verify:
        verify_GV(phi, dynamics)

    return phi