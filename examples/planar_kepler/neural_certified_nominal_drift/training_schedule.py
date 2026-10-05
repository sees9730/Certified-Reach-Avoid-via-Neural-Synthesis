"""Safety-first generator schedule for the Kepler bound-training phase."""
import math

import torch

from src.training_utils import UNSAFE_TRAINING_MARGIN_RATIO


@torch.no_grad()
def initialize_boundary_certificate(value, arrays, boundary_widths, beta_ra):
    """Initialize the existing dense MLP with eight smooth boundary features.

    Each feature rises toward one unsafe face and is almost flat in the safe
    interior. This gives interval propagation useful coordinate structure
    instead of immediately shrinking a highly coupled sampled certificate
    into its constant offset. All weights remain trainable, including small
    random extra features; no certificate condition is assumed satisfied.
    """
    if value.layer1.out_features < 8 or value.layer2.out_features < 8:
        raise ValueError("Boundary initialization requires at least eight neurons per hidden layer")
    value.layer1.weight.mul_(0.02)
    value.layer1.bias.zero_()
    value.layer2.weight.mul_(0.02)
    value.layer2.bias.zero_()
    value.output.weight.zero_()
    value.output.bias.zero_()
    for axis in range(4):
        for side in range(2):
            feature = 2 * axis + side
            sign = -1 if side == 0 else 1
            boundary = (arrays['full_range'][axis, side] +
                        (boundary_widths[axis] if side == 0 else -boundary_widths[axis]))
            normalized_boundary = (float(boundary) - float(value.input_offset[axis])) / float(value.input_scale[axis])
            value.layer1.weight[feature].zero_()
            value.layer1.weight[feature, axis] = sign * 8.0
            value.layer1.bias[feature] = -sign * 8.0 * normalized_boundary
            value.layer2.weight[feature].zero_()
            value.layer2.weight[feature, feature] = 8.0
            value.layer2.bias[feature] = -4.0
            value.output.weight[0, feature] = 2.0 * beta_ra / float(value.scale_factor)


class SafetyFirstGeneratorSchedule:
    """Ramp generator weight only when all value constraints are nearly met.

    Use worst-cell violations, so refinement cannot dilute the readiness
    check. Back off when safety regresses; each recovery resumes a gradual
    ramp rather than abruptly restoring the previous generator weight.
    This affects optimization only, never the certificate's SAT thresholds.
    """
    def __init__(self, beta_ra, warmup_epochs=1000, ramp_epochs=3000,
                 target_weight=1.0, safety_tolerance=0.1):
        if warmup_epochs < 0 or ramp_epochs < 1:
            raise ValueError("warmup_epochs must be nonnegative and ramp_epochs positive")
        if not all(math.isfinite(v) and v > 0 for v in (beta_ra, target_weight, safety_tolerance)):
            raise ValueError("Schedule thresholds and weight must be finite and positive")
        self.beta_ra = beta_ra
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.target_weight = target_weight
        self.safety_tolerance = safety_tolerance
        self.weight = 0.0
        self.started = False

    def to_dict(self):
        return dict(beta_ra=self.beta_ra, warmup_epochs=self.warmup_epochs,
                    ramp_epochs=self.ramp_epochs, target_weight=self.target_weight,
                    safety_tolerance=self.safety_tolerance)

    def __call__(self, epoch, bounds):
        if epoch < self.warmup_epochs:
            return 0.0
        with torch.no_grad():
            violations = [
                torch.relu(self.beta_ra * (1 + UNSAFE_TRAINING_MARGIN_RATIO) - bounds['unsafe'][0]),
                torch.relu(bounds['init'][1] - 1.0),
                torch.relu(-bounds['goal'][0]), torch.relu(-bounds['outside'][0]),
            ]
            ready = all(v.numel() > 0 and torch.isfinite(v).all() and
                        float(v.max()) <= self.safety_tolerance for v in violations)
        if ready:
            if not self.started:
                print(f"[Safety curriculum] Value constraints nearly satisfied at epoch {epoch}; starting generator ramp")
                self.started = True
            self.weight = min(self.target_weight, self.weight + self.target_weight / self.ramp_epochs)
        else:
            self.weight *= 0.5
            if self.weight < 1e-8:
                self.weight = 0.0
        return self.weight
