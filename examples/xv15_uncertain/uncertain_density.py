"""Set-valued XV-15 drift for one shared uncertain air-density parameter."""
import math

import torch
from torch import nn


def load_density_interval(config):
    """Read a positive closed interval in kg/m^3; a singleton is allowed."""
    values = config["uncertainty"]["air_density_kg_m3"]
    if len(values) != 2:
        raise ValueError("air_density_kg_m3 must contain [lower, upper]")
    lower, upper = map(float, values)
    if not all(math.isfinite(v) for v in (lower, upper)) or not 0 < lower <= upper:
        raise ValueError("Air density must satisfy 0 < lower <= upper, with finite endpoints")
    return lower, upper


class DensityIntervalDrift(nn.Module):
    """Closed-loop f(x,u(x),rho), rho in [rho_min,rho_max].

    forward(x) evaluates nominal density for point simulation/diagnostics.
    support(x,p) returns max_rho <f(x,u(x),rho),p> for robust training.
    Density is shared by lift and drag; it is not an independent disturbance
    in each state coordinate. The controller observes x only, not rho.
    """
    def __init__(self, aero, controller, density_interval):
        super().__init__()
        lower, upper = load_density_interval({
            "uncertainty": {"air_density_kg_m3": density_interval}
        })
        if not math.isfinite(aero.density) or aero.density <= 0:
            raise ValueError("Nominal density must be finite and positive")
        self.aero = aero
        self.controller = controller
        self.register_buffer("density_interval", torch.tensor([lower, upper], dtype=torch.float32))
        self.register_buffer("density_offset", torch.tensor((lower + upper) / 2 - aero.density, dtype=torch.float32))
        self.register_buffer("density_radius", torch.tensor((upper - lower) / 2, dtype=torch.float32))

    def forward(self, x):
        return self.aero(x, self.controller(x))

    def support(self, x, p):
        u = self.controller(x)
        nominal = self.aero(x, u)
        lift, drag = self.aero.forces(x[:, 0], u[:, 1], x[:, 2])
        # f_rho = [-D/(m*rho_nom), L/(m*v*rho_nom), 0].
        # Both terms belong to the SAME scalar parameter, so take abs only
        # after forming <f_rho,p>. Channel-wise abs would discard correlation.
        density_slope = (-drag * p[:, 0] + lift * p[:, 1] / x[:, 0]) / (self.aero.mass * self.aero.density)
        nominal_dot = (nominal * p).sum(dim=1, keepdim=True)
        correction = self.density_offset * density_slope + self.density_radius * density_slope.abs()
        return nominal_dot + correction.unsqueeze(1)
