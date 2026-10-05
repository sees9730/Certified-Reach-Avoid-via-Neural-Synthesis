"""Planar inverse-square gravity with bounded radial/tangential acceleration.

State: [r, theta, r_dot, theta_dot]; angles are radians. The default example
uses normalized length/time units. Theta is unwrapped in a finite angular
sector; there is no modulo operation in the certificate graph.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
STATE_NAMES = ("r", "theta", "r_dot", "theta_dot")


def load_config(path=HERE / "config.json"):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def load_region_arrays(config):
    """Generate unsafe slabs on all eight faces of the four-dimensional box."""
    arrays = {}
    for name in ("full_range", "init_range", "goal_range"):
        box = np.asarray(config["regions"][name], dtype=np.float64)
        if box.shape != (4, 2) or not np.isfinite(box).all() or (box[:, 0] >= box[:, 1]).any():
            raise ValueError(f"{name} must be a finite (4,2) box with lower < upper")
        arrays[name] = box.astype(np.float32)
    full = arrays["full_range"]
    if full[0, 0] <= 0:
        raise ValueError("The full domain must exclude r=0: r_min > 0")
    if float(full[1, 1] - full[1, 0]) >= 2 * math.pi:
        raise ValueError("Use an unwrapped angular sector of width less than 2*pi")
    widths = np.asarray(config["regions"]["unsafe_boundary_width"], dtype=np.float32)
    if (widths.shape != (4,) or not np.isfinite(widths).all()
            or (widths <= 0).any() or (2 * widths >= full[:, 1] - full[:, 0]).any()):
        raise ValueError("Each boundary width must be positive and less than half the domain width")
    for name in ("init_range", "goal_range"):
        box = arrays[name]
        if (box[:, 0] <= full[:, 0] + widths).any() or (box[:, 1] >= full[:, 1] - widths).any():
            raise ValueError(f"{name} must lie strictly inside the safe interior")
    if np.all(np.maximum(arrays["init_range"][:, 0], arrays["goal_range"][:, 0]) <=
              np.minimum(arrays["init_range"][:, 1], arrays["goal_range"][:, 1])):
        raise ValueError("Initial and goal boxes must be disjoint")
    unsafe = []
    for axis in range(4):
        for side in range(2):
            box = full.copy()
            if side == 0:
                box[axis, 1] = full[axis, 0] + widths[axis]
            else:
                box[axis, 0] = full[axis, 1] - widths[axis]
            unsafe.append(box)
    arrays["unsafe_ranges"] = np.stack(unsafe)
    return arrays


def validate_config(config):
    if tuple(config.get("state_order", ())) != STATE_NAMES:
        raise ValueError(f"state_order must be {STATE_NAMES}")
    arrays = load_region_arrays(config)
    for name in ("mu", "radial_acceleration_noise", "tangential_acceleration_noise"):
        value = float(config["dynamics"][name])
        if not math.isfinite(value) or value < 0 or (name == "mu" and value == 0):
            raise ValueError(f"Invalid dynamics parameter: {name}")
    for name in ("radial_acceleration", "tangential_acceleration"):
        limits = np.asarray(config["control"][name], dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f"Invalid control limits: {name}")
    hidden = config["controller_hidden_dim"]
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
        raise ValueError("controller_hidden_dim must be a positive integer")
    if not math.isfinite(float(config["beta_ra"])) or config["beta_ra"] <= 1:
        raise ValueError("beta_ra must be finite and greater than 1")
    goal = arrays["goal_range"]
    if not np.all((goal[2:, 0] < 0) & (goal[2:, 1] > 0)):
        raise ValueError("The station-keeping goal must contain zero radial and angular velocity")


class PlanarKepler(nn.Module):
    """Acceleration controls, not forces or direct angular acceleration.

    r_ddot     = r * theta_dot^2 - mu/r^2 + u_r
    theta_ddot = (u_t - 2*r_dot*theta_dot)/r

    The radius floor equals the certified domain's positive r_min. It only
    extends the model outside that domain to make the bound engine's zero
    dummy inputs finite; it is the identity on every certified state.
    """
    def __init__(self, config):
        super().__init__()
        self.mu = float(config["dynamics"]["mu"])
        self.radius_floor = float(config["regions"]["full_range"][0][0])

    def radius(self, x):
        return self.radius_floor + torch.relu(x[:, 0] - self.radius_floor)

    def forward(self, x, u):
        r = self.radius(x)
        radial_velocity, angular_velocity = x[:, 2], x[:, 3]
        radial_acceleration = r * angular_velocity * angular_velocity - self.mu / (r * r) + u[:, 0]
        angular_acceleration = (u[:, 1] - 2 * radial_velocity * angular_velocity) / r
        return torch.stack([radial_velocity, angular_velocity, radial_acceleration, angular_acceleration], dim=1)


class AccelerationDiffusion(nn.Module):
    """Independent radial/tangential acceleration noise, expressed in polar state.

    The diagonal amplitude vector is [0, 0, sigma_r, sigma_t/r]. Positions
    have finite-variation paths, so this acceleration-noise model needs no
    additional Ito coordinate correction. This is not GBM in each state.
    """
    def __init__(self, config):
        super().__init__()
        self.radius_floor = float(config["regions"]["full_range"][0][0])
        self.register_buffer("sigma", torch.tensor([
            config["dynamics"]["radial_acceleration_noise"],
            config["dynamics"]["tangential_acceleration_noise"],
        ], dtype=torch.float32))

    def forward(self, x):
        r = self.radius_floor + torch.relu(x[:, 0] - self.radius_floor)
        zero = x[:, 0] * 0.0
        return torch.stack([zero, zero, zero + self.sigma[0], self.sigma[1] / r], dim=1)


def find_goal_equilibrium(config):
    """Hold the goal's position center at rest against inverse-square gravity."""
    goal = load_region_arrays(config)["goal_range"]
    x_eq = torch.tensor([*goal[:2].mean(axis=1), 0.0, 0.0], dtype=torch.float32)
    u_eq = torch.tensor([config["dynamics"]["mu"] / float(x_eq[0]) ** 2, 0.0], dtype=torch.float32)
    bounds = torch.tensor([config["control"]["radial_acceleration"],
                           config["control"]["tangential_acceleration"]], dtype=torch.float32)
    if not ((u_eq > bounds[:, 0]) & (u_eq < bounds[:, 1])).all():
        raise ValueError("Control limits must strictly contain the station-keeping acceleration")
    return x_eq, u_eq


class KeplerEqMLPControl(nn.Module):
    """Bias-free 4→hidden→2 tanh MLP with physical bounds and exact goal trim."""
    def __init__(self, config, x_eq, u_eq, input_scale):
        super().__init__()
        self.fc1 = nn.Linear(4, config["controller_hidden_dim"], bias=False)
        self.fc2 = nn.Linear(config["controller_hidden_dim"], 2, bias=False)
        self.act = nn.Tanh()
        bounds = torch.tensor([config["control"]["radial_acceleration"],
                               config["control"]["tangential_acceleration"]], dtype=torch.float32)
        for name, value in (("x_eq", x_eq), ("u_eq", u_eq), ("input_scale", input_scale),
                            ("control_low", bounds[:, 0]), ("control_high", bounds[:, 1])):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        if self.input_scale.shape != (4,) or not (self.input_scale > 0).all():
            raise ValueError("input_scale must contain four positive scales")
        self.register_buffer("control_center", (bounds[:, 0] + bounds[:, 1]) / 2)
        self.register_buffer("control_scale", (bounds[:, 1] - bounds[:, 0]) / 2)
        self.register_buffer("z_eq", torch.atanh((self.u_eq - self.control_center) / self.control_scale))

    def forward(self, x):
        error = (x - self.x_eq) / self.input_scale
        residual = self.fc2(self.act(self.fc1(error)))
        return self.control_center + self.control_scale * torch.tanh(self.z_eq + residual)


class NominalClosedLoopDrift(nn.Module):
    """Embed two acceleration outputs only in the two velocity equations."""
    def __init__(self, physics, controller):
        super().__init__()
        self.physics = physics
        self.controller = controller

    def forward(self, x):
        return self.physics(x, self.controller(x))
