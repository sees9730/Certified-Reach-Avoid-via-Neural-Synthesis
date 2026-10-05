"""Independent audit of density/mass generator bounds against actual V autograd.

These numerical tests supplement the real-arithmetic argument; tolerances do
not certify floating-point rounding error. No training checkpoints are changed.
"""
from copy import deepcopy
from itertools import product
import json

import pytest
import torch

from examples.xv15_uncertain.model import load_config
from examples.xv15_uncertain.neural_certified_nominal_drift.main import (
    configure_reproducibility, make_hyperparameters,
)
from examples.xv15_uncertain.neural_certified_uncertain_param.main import build_problem
from src.crown_bounds import SymbolicCROWNCache_Phi


def reference_terms(value, controller, aero, diffusion, states):
    """Differentiate the actual V.forward in float64, outside the GV graph."""
    value, controller, aero, diffusion = [deepcopy(model).double() for model in
                                          (value, controller, aero, diffusion)]
    x = states.double().detach().requires_grad_(True)
    grad = torch.autograd.grad(value(x).sum(), x, create_graph=True)[0]
    diagonal = torch.stack([torch.autograd.grad(grad[:, i].sum(), x, retain_graph=True)[0][:, i]
                            for i in range(3)], dim=1)
    with torch.no_grad():
        u = controller(x)
        ito = 0.5 * (diffusion(x).square() * diagonal).sum(dim=1)
    return aero, x.detach(), u.detach(), grad.detach(), ito.detach()


def physical_generators(terms, density_interval, mass_interval, fractions):
    aero, x, u, grad, ito = terms
    for a, b in product(fractions, repeat=2):
        rho = density_interval[0] + a * (density_interval[1] - density_interval[0])
        mass = mass_interval[0] + b * (mass_interval[1] - mass_interval[0])
        with torch.no_grad():
            # Use the independently parameterized physical model, not support().
            drift = aero(x, u, density=rho, mass=mass)
            yield (drift * grad).sum(dim=1) + ito


def audit_case(tmp_path, case):
    config = load_config()
    seed = {"current": 0, "one_percent": 1, "asymmetric": 7}[case]
    configure_reproducibility(seed)
    if case == "one_percent":
        config["uncertainty"]["mass_kg"] = [5841., 5959.]
    elif case == "asymmetric":
        config["uncertainty"] = dict(air_density_kg_m3=[1.1, 1.4], mass_kg=[5600., 6200.])
    density, mass = config["uncertainty"]["air_density_kg_m3"], config["uncertainty"]["mass_kg"]
    params = make_hyperparameters(config, seed, tmp_path)
    value, generator, controller, dynamics, _, arrays = build_problem(config, params)
    full = torch.tensor(arrays["full_range"])
    span = full[:, 1] - full[:, 0]
    widths = span * 10 ** (-3 + 2 * torch.rand(21, 1))
    lower = full[:, 0] + torch.rand(21, 3) * (span - widths)
    upper = lower + widths
    bits = torch.tensor(list(product((0., 1.), repeat=3)))
    boundary_lower = full[:, 0] + bits * (span - .01 * span)
    boundary_upper = boundary_lower + .01 * span
    special = torch.tensor([arrays[name].tolist() for name in ("full_range", "init_range", "goal_range")])
    lower = torch.cat([lower, boundary_lower, special[:, :, 0]])
    upper = torch.cat([upper, boundary_upper, special[:, :, 1]])
    cache = SymbolicCROWNCache_Phi(generator, len(lower), input_dim=3)
    with torch.no_grad():
        bounds = cache.compute_bounds(lower, upper).double()
    assert torch.isfinite(bounds).all()
    # Exact state-box corners plus random interior states in float64.
    fractions = torch.cat([bits.double(), torch.rand(8, 3, dtype=torch.float64)])
    states = lower.double()[None] + fractions[:, None] * (upper.double() - lower.double())[None]
    flat = states.reshape(-1, 3)
    terms = reference_terms(value, controller, dynamics.f.aero, dynamics.g, flat)
    reference = torch.stack(list(physical_generators(terms, density, mass, (0., .25, .5, .75, 1.))))
    bound_at_state = bounds.repeat(len(fractions))
    excess = reference - bound_at_state
    tolerance = 2e-5 + 2e-5 * reference.abs()
    assert (excess <= tolerance).all(), f"Physical generator exceeds IBP bound: {float(excess.max())}"
    # At a point, the robust generator must match the four physical corners.
    corner_max = torch.stack(list(physical_generators(terms, density, mass, (0., 1.)))).amax(dim=0)
    with torch.no_grad():
        robust = generator(flat.float()).flatten().double()
    torch.testing.assert_close(robust, corner_max, atol=2e-5, rtol=2e-4)
    # Deliberately measure, rather than hide, rounding on degenerate state cells.
    mid = (lower + upper) / 2
    point_terms = reference_terms(value, controller, dynamics.f.aero, dynamics.g, mid)
    point_reference = torch.stack(list(physical_generators(point_terms, density, mass, (0., 1.)))).amax(dim=0)
    with torch.no_grad():
        point_bounds = cache.compute_bounds(mid, mid).double()
    point_excess = point_reference - point_bounds
    assert (point_excess <= 2e-5 + 2e-5 * point_reference.abs()).all()
    return dict(case=case, seed=seed, density=density, mass=mass, cells=len(lower),
                physical_evaluations=reference.numel(), violations_beyond_tolerance=int((excess > tolerance).sum()),
                minimum_sampled_cell_slack=float((-excess).min()),
                maximum_pointwise_generator_error=float((robust-corner_max).abs().max()),
                point_cells_below_float64_reference=int((point_excess > 0).sum()),
                maximum_point_cell_underestimate=float(point_excess.clamp_min(0).max()))


@pytest.mark.parametrize("case", ["current", "one_percent", "asymmetric"])
def test_joint_cell_bounds_against_independent_float64_generator(tmp_path, case):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = audit_case(tmp_path, case)
        print(json.dumps(report, sort_keys=True))
    finally:
        torch.set_num_threads(old_threads)
