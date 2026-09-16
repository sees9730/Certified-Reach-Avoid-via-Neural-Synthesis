"""Nominal 4D double-integrator SDE in dimensionless coordinates.

State: [px, py, vx, vy]; control: [ux, uy].
Drift: [vx, vy, ux, uy]. Independent Brownian noise acts on velocities.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROBLEM = "double_integrator_4d"
STATE_NAMES = ("px", "py", "vx", "vy")


def validate_config(config):
    if config.get("problem") != PROBLEM:
        raise ValueError("Expected a double_integrator_4d config; historical Kepler runs are incompatible")
    beta = config["beta_ra"]
    if not np.isfinite(beta) or beta <= 1:
        raise ValueError("beta_ra must be finite and greater than 1")
    limits = np.asarray(config["control"]["u_max"], dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("control.u_max must contain two positive finite axis limits")
    kind = config['control'].get('type', 'neural')
    if kind == 'learnable_pd':
        for name in ('kp_init', 'kd_init'):
            gain = config['control'][name]
            if not np.isscalar(gain) or not np.isfinite(gain) or gain <= 0:
                raise ValueError(f"control.{name} must be a positive finite scalar")
    elif kind == 'neural':
        hidden = config['control'].get('hidden_dim', 32)
        if type(hidden) is not int or hidden < 1:
            raise ValueError('control.hidden_dim must be a positive integer')
    else:
        raise ValueError(f"Unknown control.type: {kind}")
    rate = np.asarray(config["dynamics"]["velocity_diffusion_rate"], dtype=float)
    if rate.shape != (2,) or not np.isfinite(rate).all() or (rate < 0).any():
        raise ValueError("velocity_diffusion_rate must contain two finite nonnegative covariance rates")
    load_region_arrays(config)
    return config


def load_config(path=HERE / "config.json"):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(json.load(stream))


def _box(config, name):
    box = np.asarray([config[name][axis] for axis in STATE_NAMES], dtype=float)
    if box.shape != (4, 2) or not np.isfinite(box).all() or (box[:, 0] >= box[:, 1]).any():
        raise ValueError(f"{name} must specify four finite, increasing intervals")
    return box


def domain_box(config):
    return _box(config, "domain")


def boundary_slabs(full_range, margin_fraction):
    """Eight closed unsafe strips, one at each face of the full domain."""
    if not np.isfinite(margin_fraction) or not 0 < margin_fraction < 0.5:
        raise ValueError("boundary_margin_fraction must be between 0 and 0.5")
    widths = margin_fraction * (full_range[:, 1] - full_range[:, 0])
    slabs = []
    for dim in range(4):
        lower, upper = full_range.copy(), full_range.copy()
        lower[dim, 1] = full_range[dim, 0] + widths[dim]
        upper[dim, 0] = full_range[dim, 1] - widths[dim]
        slabs.extend([lower, upper])
    return np.asarray(slabs)


def _intersects(a, b):
    return bool(np.all(np.maximum(a[:, 0], b[:, 0]) <= np.minimum(a[:, 1], b[:, 1])))


def load_region_arrays(config):
    full, init, goal = domain_box(config), _box(config, "initial"), _box(config, "goal")
    for name, box in (("initial", init), ("goal", goal)):
        if (box[:, 0] < full[:, 0]).any() or (box[:, 1] > full[:, 1]).any():
            raise ValueError(f"{name} must lie inside the domain")
    if _intersects(init, goal):
        raise ValueError("Initial and goal boxes must be disjoint")
    unsafe = boundary_slabs(full, float(config["boundary_margin_fraction"]))
    if any(_intersects(box, init) or _intersects(box, goal) for box in unsafe):
        raise ValueError("Initial and goal boxes must not touch the unsafe boundary strips")
    return dict(full_range=full.astype(np.float32), init_range=init.astype(np.float32),
                goal_range=goal.astype(np.float32), unsafe_ranges=unsafe.astype(np.float32))


class DoubleIntegratorDrift(nn.Module):
    def forward(self, x, u):
        return torch.cat([x[:, 2:4], u], dim=1)


class NeuralControl(nn.Module):
    """Learned feedback with |ux| <= u_max[0] and |uy| <= u_max[1]."""
    def __init__(self, config, input_offset, input_scale):
        super().__init__()
        for name, data in (("input_offset", input_offset), ("input_scale", input_scale),
                           ("axis_limit", config["control"]["u_max"])):
            self.register_buffer(name, torch.as_tensor(data, dtype=torch.float32))
        self.fc1 = nn.Linear(4, int(config["control"].get("hidden_dim", 32)))
        self.fc2 = nn.Linear(int(config["control"].get("hidden_dim", 32)), 2)

    def forward(self, x):
        normalized = (x - self.input_offset) / self.input_scale
        return self.axis_limit * torch.tanh(self.fc2(torch.tanh(self.fc1(normalized))))


class LearnablePDControl(nn.Module):
    """Two shared positive gains, with the baseline's exact axis saturation.

    The optimizer updates log gains; kp=exp(log_kp), kd=exp(log_kd).
    Gains act on the raw dimensionless state errors, not normalized inputs.
    """
    def __init__(self, config, input_offset):
        super().__init__()
        control = config['control']
        self.register_buffer('input_offset', torch.as_tensor(input_offset, dtype=torch.float32))
        self.register_buffer('axis_limit', torch.as_tensor(control['u_max'], dtype=torch.float32))
        self.kp = nn.Parameter(torch.tensor([(control['kp_init'])], dtype=torch.float32))
        self.kd = nn.Parameter(torch.tensor([(control['kd_init'])], dtype=torch.float32))

    def forward(self, x):
        error = x - self.input_offset
        u = -self.kp * error[:, :2] - self.kd * error[:, 2:4]
        return u
        # return torch.maximum(torch.minimum(u, self.axis_limit), -self.axis_limit)

    def gains(self):
        return dict(kp=float(self.kp.detach().item()), kd=float(self.kd.detach().item()))


def create_control(config, input_offset, input_scale):
    kind = config['control'].get('type', 'neural')  # Older neural checkpoints.
    if kind == 'learnable_pd':
        return LearnablePDControl(config, input_offset)
    if kind == 'neural':
        return NeuralControl(config, input_offset, input_scale)
    raise ValueError(f"Unknown control.type: {kind}")


class NominalClosedLoopDrift(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.physics = DoubleIntegratorDrift()
        self.controller = controller

    def forward(self, x):
        return self.physics(x, self.controller(x))


class DiagonalDiffusion(nn.Module):
    """Diagonal coefficient [0, 0, sqrt(qx), sqrt(qy)] for standard Brownian noise.

    qx and qy are velocity covariance rates, not noise standard deviations.
    The four-entry diagonal form is equivalent to two velocity noise channels.
    """
    def __init__(self, config):
        super().__init__()
        rate = np.asarray(config["dynamics"]["velocity_diffusion_rate"], dtype=float)
        self.register_buffer("sigma", torch.tensor(np.r_[np.zeros(2), np.sqrt(rate)], dtype=torch.float32))

    def forward(self, x):
        return self.sigma.unsqueeze(0).expand(x.shape[0], -1)
