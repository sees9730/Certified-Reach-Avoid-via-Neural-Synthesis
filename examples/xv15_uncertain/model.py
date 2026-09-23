"""XV-15 nominal physics and bounded neural feedback, shared by this example.

Equations and physical constants follow examples/synthesis/xv15aircraft_syn.
Runtime states are [v, gamma, beta] in [m/s, rad, rad]; controls are
[thrust, angle of attack, tilt rate] in [N, rad, rad/s].
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from torch import nn

DEG = np.pi / 180.0
HERE = Path(__file__).resolve().parent
UNSAFE_REGION_NAMES = (
    "unsafe_up", "unsafe_dn", "unsafe_min_beta", "unsafe_max_beta",
    "unsafe_min_vel", "unsafe_max_vel",
)


def load_config(path=HERE / "config.json"):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_region_arrays(config):
    """Convert human-readable degree intervals to the source's radian units."""
    regions = {}
    for name, rows in config["regions_mps_deg_deg"].items():
        arr = np.asarray(rows, dtype=np.float64)
        if arr.shape != (3, 2) or not np.isfinite(arr).all() or (arr[:, 0] >= arr[:, 1]).any():
            raise ValueError(f"Invalid XV-15 region: {name}")
        regions[name] = (arr * np.array([1.0, DEG, DEG])[:, None]).astype(np.float32)
    domain = regions["full_range"]
    if domain[0, 0] <= 0:
        raise ValueError("The XV-15 domain must have strictly positive airspeed")
    for name, arr in regions.items():
        if (arr[:, 0] < domain[:, 0]).any() or (arr[:, 1] > domain[:, 1]).any():
            raise ValueError(f"Region {name} extends outside full_range")
    regions["unsafe_ranges"] = np.stack([regions[name] for name in UNSAFE_REGION_NAMES])
    return regions


class XV15Aero(nn.Module):
    """Nominal lift/drag with linear interpolation between cruise and hover."""
    def __init__(self, config):
        super().__init__()
        p = config["dynamics"]
        self.mass = float(p["mass"])
        self.gravity = float(p["gravity"])
        self.density = float(p["density"])
        self.wing_area = float(p["wing_area"])
        for name in ("lift_cruise", "lift_hover", "drag_cruise", "drag_hover"):
            self.register_buffer(name, torch.tensor(p[name], dtype=torch.float32))

    def forces(self, v, alpha, beta, density=None):
        alpha_deg = alpha / DEG
        ratio = beta * (2.0 / np.pi)
        cl_cruise = self.lift_cruise[0] * alpha_deg + self.lift_cruise[1]
        cl_hover = self.lift_hover[0] * alpha_deg + self.lift_hover[1]
        cd_cruise = self.drag_cruise[0] * alpha_deg ** 2 + self.drag_cruise[1] * alpha_deg + self.drag_cruise[2]
        cd_hover = self.drag_hover[0] * alpha_deg ** 2 + self.drag_hover[1] * alpha_deg + self.drag_hover[2]
        density = self.density if density is None else density
        pressure = 0.5 * density * v * v * self.wing_area
        return (pressure * (cl_cruise * (1.0 - ratio) + cl_hover * ratio),
                pressure * (cd_cruise * (1.0 - ratio) + cd_hover * ratio))

    def forward(self, x, u, density=None, mass=None):
        """Evaluate nominal or realized physics; parameters may be batch vectors."""
        v, gamma, beta = x[:, 0], x[:, 1], x[:, 2]
        thrust, alpha, delta = u[:, 0], u[:, 1], u[:, 2]
        lift, drag = self.forces(v, alpha, beta, density=density)
        mass = self.mass if mass is None else mass
        weight = mass * self.gravity
        # Exactly the source dynamics: no velocity or tilt clamping.
        v_dot = (thrust * torch.cos(alpha + beta) - drag - weight * torch.sin(gamma)) / mass
        gamma_dot = (thrust * torch.sin(alpha + beta) + lift - weight * torch.cos(gamma)) / (mass * v)
        return torch.stack([v_dot, gamma_dot, delta], dim=1)


def find_goal_equilibrium(config, aero):
    """Solve the nominal trim equations at the goal center (no learned data)."""
    center = np.asarray(config["regions_mps_deg_deg"]["goal_range"], dtype=float).mean(axis=1)
    v, gamma, beta = center * np.array([1.0, DEG, DEG])
    weight = aero.mass * aero.gravity
    alpha_max = config["control"]["alpha_max_deg"] * DEG

    def forces(alpha):
        with torch.no_grad():
            return [float(t) for t in aero.forces(*[
                torch.tensor(a, dtype=torch.float64) for a in (v, alpha, beta)
            ])]

    def residual(alpha):
        lift, drag = forces(alpha)
        return (drag + weight * np.sin(gamma)) * np.tan(alpha + beta) + lift - weight * np.cos(gamma)

    alpha = brentq(residual, -alpha_max, alpha_max)
    _, drag = forces(alpha)
    thrust = (drag + weight * np.sin(gamma)) / np.cos(alpha + beta)
    lower, upper = np.asarray(config["control"]["thrust_over_weight"]) * weight
    if not lower < thrust < upper or not abs(alpha) < alpha_max:
        raise ValueError("Goal-center trim is outside the controller limits")
    return (torch.tensor([v, gamma, beta], dtype=torch.float32),
            torch.tensor([thrust, alpha, 0.0], dtype=torch.float32))


class XV15EqMLPControl(nn.Module):
    """Smooth bounded feedback anchored at nominal trim, as in the source.

    State errors use state scales; outputs use physical control scales.
    Bias-free layers preserve u(x_eq)=u_eq throughout training.
    """
    def __init__(self, config, x_eq, u_eq, input_scale):
        super().__init__()
        hidden_dim = int(config["controller_hidden_dim"])
        self.fc1 = nn.Linear(3, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 3, bias=False)
        self.act = nn.Tanh()
        c = config["control"]
        weight = config["dynamics"]["mass"] * config["dynamics"]["gravity"]
        t_min, t_max = np.asarray(c["thrust_over_weight"]) * weight
        a_max, d_max = c["alpha_max_deg"] * DEG, c["delta_max_deg_per_second"] * DEG
        for name, value in (
            ("x_eq", x_eq), ("u_eq", u_eq), ("input_scale", input_scale),
            ("T_min", t_min), ("T_max", t_max), ("alpha_max", a_max), ("delta_max", d_max),
            ("du_scale", [t_max, a_max, d_max]),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        # Normalizer for run_mc.py's normalized_effort: thrust by nominal
        # weight, angles by their limits. Non-persistent so checkpoints saved
        # before it still load with strict=True.
        self.register_buffer("effort_scale",
                             torch.tensor([weight, a_max, d_max], dtype=torch.float32),
                             persistent=False)
        p = (self.u_eq[0] - self.T_min) / (self.T_max - self.T_min)
        a, d = self.u_eq[1] / self.alpha_max, self.u_eq[2] / self.delta_max
        self.register_buffer("z_eq", torch.stack([torch.logit(p), torch.atanh(a), torch.atanh(d)]))
        slopes = torch.stack([(self.T_max - self.T_min) * p * (1.0 - p),
                              self.alpha_max * (1.0 - a * a), self.delta_max * (1.0 - d * d)])
        self.register_buffer("inv_slope_eq", 1.0 / slopes)

    def forward(self, x):
        error = (x - self.x_eq.unsqueeze(0)) / self.input_scale.unsqueeze(0)
        du = self.fc2(self.act(self.fc1(error))) * self.du_scale.unsqueeze(0)
        z = self.z_eq.unsqueeze(0) + du * self.inv_slope_eq.unsqueeze(0)
        thrust = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(z[:, 0])
        alpha = self.alpha_max * torch.tanh(z[:, 1])
        delta = self.delta_max * torch.tanh(z[:, 2])
        return torch.stack([thrust, alpha, delta], dim=1)

    def raw_control(self, x):
        """Normalized control; its squared norm is run_mc.py's effort rate.

        The augmented-energy generator reads this through
        phi_module._compute_energy_rate, so dE/dt matches the Monte-Carlo
        effort integrand exactly.
        """
        return self.forward(x) / self.effort_scale.unsqueeze(0)


class NominalClosedLoopDrift(nn.Module):
    """Evaluate f(x,u(x)); XV-15 control is not additive to the drift."""
    def __init__(self, aero, controller):
        super().__init__()
        self.aero = aero
        self.controller = controller

    def forward(self, x):
        return self.aero(x, self.controller(x))


class DiagonalDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        sigma = np.asarray(config["dynamics"]["diffusion_mps_deg_deg"]) * np.array([1.0, DEG, DEG])
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    def forward(self, x):
        sigma = self.sigma.to(x.device, x.dtype)
        return sigma if x.dim() == 1 else sigma.unsqueeze(0).expand(x.shape[0], -1)
