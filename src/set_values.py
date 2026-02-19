# src/set_values.py
import torch
import torch.nn as nn


# =============================================================================
# Existing (keep working for previous examples)
# =============================================================================

class AdditiveBoxSetDrift(nn.Module):
    """
    F(x) = f_nom(x) + w,  w in [-d, d] (elementwise box).

    support(x, p) = max_{w in box} <f_nom(x)+w, p>
                  = <f_nom(x), p> + sum_i d_i * |p_i|
    """
    def __init__(self, f_nominal, d: torch.Tensor):
        super().__init__()
        self.f_nominal = f_nominal
        self.register_buffer("d", torch.as_tensor(d, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_nominal(x)

    def support(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        fx = self.f_nominal(x)                    # (N,D)
        d  = self.d.to(x.device, x.dtype)         # (D,)
        return (fx * p).sum(dim=1, keepdim=True) + (d * p.abs()).sum(dim=1, keepdim=True)


class ClosedLoopSetValuedDrift(nn.Module):
    """
    Closed-loop wrapper for set-valued or single-valued open-loop drifts.

    forward(x) = f_ol(x) + u(x)
    support(x,p):
      - if f_ol has support: support_f_ol(x,p) + <u(x), p>
      - else: <f_ol(x)+u(x), p>
    """
    def __init__(self, f_ol: nn.Module, controller: nn.Module):
        super().__init__()
        self.f_ol = f_ol
        self.controller = controller

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_ol(x) + self.controller(x)

    def support(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        u = self.controller(x)
        u_dot_p = (u * p).sum(dim=1, keepdim=True)
        if hasattr(self.f_ol, "support") and callable(getattr(self.f_ol, "support")):
            return self.f_ol.support(x, p) + u_dot_p
        f = self.f_ol(x)
        return (f * p).sum(dim=1, keepdim=True) + u_dot_p


class LinearIntervalSetDrift(nn.Module):
    """
    Set-valued linear drift:
        f_set(x) = { A x : A_ij in [A_L_ij, A_U_ij] }.

    support(x,p) = max_{A in [A_L,A_U]} <A x, p>
                 = <A_mid x, p> + sum_{i,j} A_rad[i,j] * |p_i * x_j|.
    """
    def __init__(self, A_L: torch.Tensor, A_U: torch.Tensor):
        super().__init__()
        A_L = torch.as_tensor(A_L, dtype=torch.float32)
        A_U = torch.as_tensor(A_U, dtype=torch.float32)
        assert A_L.shape == A_U.shape and A_L.dim() == 2

        self.register_buffer("A_L", A_L)
        self.register_buffer("A_U", A_U)

        A_mid = 0.5 * (A_L + A_U)
        A_rad = 0.5 * (A_U - A_L)
        self.register_buffer("A_mid", A_mid)
        self.register_buffer("A_rad", A_rad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A_mid = self.A_mid.to(x.device, x.dtype)
        return x @ A_mid.T

    def support(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        A_mid = self.A_mid.to(x.device, x.dtype)
        A_rad = self.A_rad.to(x.device, x.dtype)

        Ax_mid = x @ A_mid.T
        base = (Ax_mid * p).sum(dim=1, keepdim=True)

        tmp = p.abs().unsqueeze(2) * x.abs().unsqueeze(1)      # (N,D,D)
        extra = (tmp * A_rad.unsqueeze(0)).sum(dim=(1, 2)).unsqueeze(1)

        return base + extra


class InvertedPendulumSetDrift(nn.Module):
    """
    Set-valued drift for inverted pendulum with independent parameter intervals.

    f1 = x2
    f2 = (g/L) * sin(x1) - (b/(m*L^2)) * x2

    with:
      g in [g_L, g_U], L in [L_L, L_U], b in [b_L, b_U], m in [m_L, m_U]

    This class computes induced intervals for:
      a = g/L
      c = b/(m*L^2)
    and uses:
      support(x,p) = <f_mid(x), p> + a_rad*|p2*sin(x1)| + c_rad*|p2*x2|
    """
    def __init__(
        self,
        g_range,
        L_range,
        b_range,
        m_range,
    ):
        super().__init__()

        def _bounds(name, r):
            if len(r) != 2:
                raise ValueError(f"{name}_range must have length 2, got {r}")
            lo = float(r[0])
            hi = float(r[1])
            if lo > hi:
                raise ValueError(f"Expected {name}_range[0] <= {name}_range[1], got {r}")
            return lo, hi

        g_L, g_U = _bounds("g", g_range)
        L_L, L_U = _bounds("L", L_range)
        b_L, b_U = _bounds("b", b_range)
        m_L, m_U = _bounds("m", m_range)

        if L_L <= 0.0 or m_L <= 0.0:
            raise ValueError("L_range and m_range must be strictly positive")

        a_L = g_L / L_U
        a_U = g_U / L_L
        c_L = b_L / (m_U * (L_U ** 2))
        c_U = b_U / (m_L * (L_L ** 2))

        if a_L > a_U or c_L > c_U:
            raise ValueError("Invalid induced intervals for a=g/L or c=b/(m*L^2)")

        self.register_buffer("a_mid", torch.tensor(0.5 * (a_L + a_U), dtype=torch.float32))
        self.register_buffer("a_rad", torch.tensor(0.5 * (a_U - a_L), dtype=torch.float32))
        self.register_buffer("c_mid", torch.tensor(0.5 * (c_L + c_U), dtype=torch.float32))
        self.register_buffer("c_rad", torch.tensor(0.5 * (c_U - c_L), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_mid = self.a_mid.to(x.device, x.dtype)
        c_mid = self.c_mid.to(x.device, x.dtype)
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x2
        f2 = a_mid * torch.sin(x1) - c_mid * x2
        return torch.stack([f1, f2], dim=1)

    def support(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        a_mid = self.a_mid.to(x.device, x.dtype)
        a_rad = self.a_rad.to(x.device, x.dtype)
        c_mid = self.c_mid.to(x.device, x.dtype)
        c_rad = self.c_rad.to(x.device, x.dtype)
        x1 = x[:, 0]
        x2 = x[:, 1]
        p1 = p[:, 0]
        p2 = p[:, 1]

        base = p1 * x2 + p2 * (a_mid * torch.sin(x1) - c_mid * x2)
        extra = a_rad * (p2 * torch.sin(x1)).abs() + c_rad * (p2 * x2).abs()
        return (base + extra).unsqueeze(1)
