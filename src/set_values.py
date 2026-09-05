import torch
import torch.nn as nn


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
