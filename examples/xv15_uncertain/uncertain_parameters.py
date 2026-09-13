"""Exact XV-15 drift support over independent density and mass intervals."""
import math

import torch

from examples.xv15_uncertain.uncertain_density import DensityIntervalDrift


def load_mass_interval(config):
    """Read a positive interval in kg; older density-only configs stay nominal."""
    nominal = config["dynamics"]["mass"]
    values = config["uncertainty"].get("mass_kg", [nominal, nominal])
    if len(values) != 2:
        raise ValueError("mass_kg must contain [lower, upper]")
    lower, upper = map(float, values)
    if not all(math.isfinite(v) for v in (lower, upper)) or not 0 < lower <= upper:
        raise ValueError("Mass must satisfy 0 < lower <= upper, with finite endpoints")
    return lower, upper


class DensityMassIntervalDrift(DensityIntervalDrift):
    """Shared density in lift/drag and shared mass in both accelerations.

    The controller observes state only and outputs thrust in newtons using
    nominal control limits. Inherited forward(x) evaluates nominal physics;
    support(x,p) maximizes over the full density/mass rectangle.

    Three algebraically equal support graphs preserve different dependencies
    during interval propagation. Their pointwise minimum is the same exact
    support, and its IBP upper bound is the tightest of the three upper bounds.
    """
    def __init__(self, aero, controller, density_interval, mass_interval):
        super().__init__(aero, controller, density_interval)
        if not math.isfinite(aero.mass) or aero.mass <= 0:
            raise ValueError("Nominal mass must be finite and positive")
        lower, upper = load_mass_interval({
            "dynamics": {"mass": aero.mass}, "uncertainty": {"mass_kg": mass_interval},
        })
        self.register_buffer("mass_interval", torch.tensor([lower, upper], dtype=torch.float32))
        self.register_buffer("inverse_mass_center", torch.tensor((1 / lower + 1 / upper) / 2, dtype=torch.float32))
        self.register_buffer("inverse_mass_radius", torch.tensor((1 / lower - 1 / upper) / 2, dtype=torch.float32))
        self.register_buffer("inverse_mass_offset", torch.tensor(
            (1 / lower + 1 / upper) / 2 - 1 / aero.mass, dtype=torch.float32))

    def support_forms(self, x, p):
        """Return equivalent decomposed, nominal-centered, and corner supports.

        Native minimum/maximum operators have monotone IBP rules. Do not
        replace them with abs/ReLU identities: those lose dependencies.
        """
        u = self.controller(x)
        v, gamma, beta = x[:, 0], x[:, 1], x[:, 2]
        thrust, alpha, delta = u[:, 0], u[:, 1], u[:, 2]
        lift, drag = self.aero.forces(v, alpha, beta)
        aero_dot = -drag * p[:, 0] + lift * p[:, 1] / v
        thrust_dot = thrust * (torch.cos(alpha + beta) * p[:, 0]
                              + torch.sin(alpha + beta) * p[:, 1] / v)
        density_slope = aero_dot / self.aero.density
        # Positive mass means the maximizing density is independent of mass.
        # Sum correlated force contributions BEFORE taking absolute values.
        force_support = (thrust_dot + aero_dot + self.density_offset * density_slope
                         + self.density_radius * density_slope.abs())
        # Weight/mass cancels: gravity and tilt rate do not vary with mass.
        independent = (-self.aero.gravity * (torch.sin(gamma) * p[:, 0]
                                             + torch.cos(gamma) * p[:, 1] / v)
                       + delta * p[:, 2])
        decomposed = (independent + self.inverse_mass_center * force_support
                      + self.inverse_mass_radius * force_support.abs()).unsqueeze(1)

        # Keep the original density-only graph, including force/gravity
        # cancellation before multiplying by p, and add a small mass correction.
        nominal = self.aero(x, u)
        nominal_density_slope = aero_dot / (self.aero.mass * self.aero.density)
        density_correction = (self.density_offset * nominal_density_slope
                              + self.density_radius * nominal_density_slope.abs())
        density_support = (nominal * p).sum(dim=1) + density_correction
        centered_force = ((thrust * torch.cos(alpha + beta) - drag) * p[:, 0]
                          + (thrust * torch.sin(alpha + beta) + lift) * p[:, 1] / v
                          + self.aero.mass * density_correction)
        centered = (density_support + self.inverse_mass_offset * centered_force
                    + self.inverse_mass_radius * centered_force.abs()).unsqueeze(1)

        # p.f = c + a/m + b*rho/m is multi-affine in (rho, 1/m).
        # Its maximum over the rectangle occurs at one of its four corners.
        # Keep each physical acceleration intact before taking its dot product.
        corners = []
        for density in self.density_interval.unbind():
            for mass in self.mass_interval.unbind():
                weight = mass * self.aero.gravity
                density_scale = density / self.aero.density
                v_dot = (thrust * torch.cos(alpha + beta) - drag * density_scale
                         - weight * torch.sin(gamma)) / mass
                gamma_dot = (thrust * torch.sin(alpha + beta) + lift * density_scale
                             - weight * torch.cos(gamma)) / (mass * v)
                drift = torch.stack([v_dot, gamma_dot, delta], dim=1)
                corners.append((drift * p).sum(dim=1, keepdim=True))
        corner_support = torch.maximum(torch.maximum(corners[0], corners[1]),
                                       torch.maximum(corners[2], corners[3]))
        return decomposed, centered, corner_support

    def support(self, x, p):
        decomposed, centered, corners = self.support_forms(x, p)
        # Each operand equals the full robust support, not an individual
        # parameter case. Taking their minimum preserves all uncertain cases.
        return torch.minimum(torch.minimum(decomposed, centered), corners)
