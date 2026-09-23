"""Fine-tune a SAT XV-15 density/mass certificate under a control-effort budget.

Coordinates are z = [v, gamma, beta, E], with E(0) = 0 and
    dE/dt = (T/(m0*g))**2 + (alpha/alpha_max)**2 + (delta/delta_max)**2,
    GV = sup_{rho,m} G_{rho,m} V + dE/dt * dV/dE.
The controller still sees only [v, gamma, beta]. The energy rate is the
normalized squared control of XV15EqMLPControl.raw_control, which is exactly
run_mc.py's normalized_effort integrand; it is not physical energy. The rate
does not depend on density or mass, so adding it after the robust support is
still the exact worst case over the uncertainty rectangle.

Because the rate is nonnegative the energy state is nondecreasing, and it has
finite variation, so the generator carries no diffusion or mixed term in E.

Follows inv_pend_adversarial/neural_certified/main_constraint_energy.py.
With the effort ceiling

    E_bar = energy_max - energy_margin,    0 < energy_margin < energy_max,

the certificate domain is X x [0, energy_max]. The unsafe set is
    (X_u x [0, energy_max]) union (X x [E_bar, energy_max]).
The goal is X_g x [0, E_goal], where
    E_goal = E_bar - 0.01 * min(energy_margin, E_bar)
leaves the same positive separation gap used by the pendulum script.
There is no effort-related unsafe condition at E = 0. The conditions are

    V(x, e)     >= 0      on X x [0, energy_max],
    V(x, 0)     <= alpha  on X_0,
    V(x, e)     >= beta   on the physical unsafe sets and upper effort band,
    sup_{rho,m} G^{pi,E}_{rho,m} V(x, e) <= -eta
                          on {V < beta} minus X_g x [0, E_goal],

giving a success probability at least 1 - alpha/beta for the event
tau_goal < tau_unsafe ^ tau_budget. Success therefore requires reaching the
goal strictly before the accumulated effort reaches E_bar. alpha is fixed at
1 by the shared init loss (src/training_utils.compute_loss_init_bounds), and
eta is 1e-4, the GENERATOR_MARGIN that src/training_utils applies in both the
generator loss and its sat check. Between E_goal and E_bar, physical goal
cells join the outside and generator covers. Inside the upper unsafe band,
goal + outside spatial cells enforce V >= beta over all of X; the band is
excluded from the generator cover. Closed slabs share their boundaries,
so generator checks conservatively include the endpoints of the gap.
The guarantee applies up to reaching the goal, and only after all
certificate checks pass; it is not an all-path hard cap.

Start from seed0/outputs/eval_bundle.pth by default. Lift its value network
exactly to V(x,E), freeze the controller for sample pretraining, then
fine-tune both networks using the shared bound trainer. --freeze-controller
instead searches for an energy certificate for the unchanged robust
controller. Initial cells reuse the checkpoint's final spatial partition
exactly; --energy-cells controls uniform energy slabs over [0, energy_max],
with extra edges at E_goal and E_bar. V uses energy_max for its energy
input scale and zero for its energy input offset, with zero energy weights.
Initial-set cells remain at E=0. Outputs use a fresh energy/ run folder;
the baseline checkpoint is never replaced.
The XV-15 CLI defaults are retained: energy_max=200, energy-cells=2, and
pretrain-epochs=0; pass --pretrain-epochs to enable V-only sample pretraining.

Example (from the repository root):
    .venv/bin/python examples/xv15_uncertain/neural_certified_uncertain_param/main_constraint_energy.py --energy-max 20 --energy-margin 1
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from examples.xv15_uncertain.model import (
    DiagonalDiffusion, XV15Aero, XV15EqMLPControl, load_region_arrays,
)
from examples.xv15_uncertain.neural_certified_nominal_drift.main import configure_reproducibility
from examples.xv15_uncertain.uncertain_density import load_density_interval
from examples.xv15_uncertain.uncertain_parameters import DensityMassIntervalDrift, load_mass_interval
from src.dynamics import Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV, compute_GV_autograd
from src.pretrainer import pretrain_network_samples
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging, save_eval_bundle
from src.trainer import train_network_bounds
from src.training_utils import GENERATOR_MARGIN, evaluate_constraints, print_constraint_summary
from src.visualization import create_summary_plots

GROUPS = ("init", "goal", "unsafe", "outside", "generator")
SAT_KEYS = tuple(f"{name}_satisfied" for name in GROUPS)
DEFAULT_BASELINE = HERE / "seed0" / "outputs" / "eval_bundle.pth"
ENERGY_DEFINITION = "integral((T/(m0*g))**2 + (alpha/alpha_max)**2 + (delta/delta_max)**2 dt)"
# alpha in the corollary. Fixed at 1 by compute_loss_init_bounds, which
# penalizes V_upper > 1.0 and sets its sat flag from the same threshold.
ALPHA = 1.0
# eta in the corollary: the strict margin that sup_{rho,m} G V must clear.
# Must stay equal to src.training_utils.GENERATOR_MARGIN, which is what
# compute_loss_generator_bounds and generator_bound_masks actually apply.
ETA = GENERATOR_MARGIN


def energy_levels(energy_max, energy_margin):
    """Return float32 (domain maximum, unsafe-band start, goal ceiling)."""
    if not (math.isfinite(energy_max) and math.isfinite(energy_margin)
            and 0 < energy_margin < energy_max):
        raise ValueError("Require finite 0 < energy_margin < energy_max")
    ceiling = np.float32(energy_max - energy_margin)
    maximum = np.float32(energy_max)
    if not (np.isfinite(maximum) and 0 < ceiling < maximum):
        raise ValueError("Energy bounds must remain distinct and positive in float32")
    # Match the pendulum's gap; one ULP can collapse under float32 midpoints.
    goal_max = np.float32(ceiling - 0.01 * min(energy_margin, float(ceiling)))
    if not (0 < goal_max < ceiling):
        raise ValueError("Energy goal and unsafe band must be separated in float32")
    return maximum, ceiling, goal_max


def energy_budget(energy_max, energy_margin):
    """Return the effort ceiling (unsafe-band start), not the domain maximum."""
    return energy_levels(energy_max, energy_margin)[1]


class EnergyHyperparameters(Hyperparameters):
    """Preserve the augmentation flag in the shared trainer's saved bundles."""

    include_energy = True
    energy_cells = 2
    energy_max = 200.0
    energy_margin = 10.0
    baseline_checkpoint = ""

    @classmethod
    def from_dict(cls, data):
        result = super().from_dict(data)
        for name in ("energy_max", "energy_margin", "energy_cells", "baseline_checkpoint"):
            if name in data:
                setattr(result, name, data[name])
        return result

    def to_dict(self):
        maximum, ceiling, goal_max = energy_levels(self.energy_max, self.energy_margin)
        result = super().to_dict()
        result.update(include_energy=True, include_time=False,
                      energy_max=self.energy_max, energy_margin=self.energy_margin,
                      energy_budget=float(ceiling), energy_ceiling=float(ceiling),
                      goal_energy_max=float(goal_max),
                      energy_domain=[0.0, float(maximum)],
                      unsafe_energy_band=[float(ceiling), float(maximum)],
                      alpha=ALPHA, eta=ETA, energy_cells=self.energy_cells,
                      baseline_checkpoint=self.baseline_checkpoint,
                      energy_definition=ENERGY_DEFINITION,
                      initial_discretization="baseline_final_cells_extruded_in_energy",
                      budget_condition="V(x,E)>=beta on X x [energy_max-energy_margin, energy_max]",
                      coordinate_order=["v", "gamma", "beta", "E"])
        return result


REFINEMENT_FLAGS = ("outside_merge_margin", "generator_merge_margin", "n_to_refine",
                    "max_generator_cells", "merge_interval")


def refinement_overrides(args):
    """Only the refinement flags actually passed; the rest keep the baseline."""
    return {name: getattr(args, name) for name in REFINEMENT_FLAGS
            if getattr(args, name) is not None}


def apply_refinement_overrides(params, args):
    """Retune refinement/merging without touching the shared src/ defaults.

    Sign conventions differ between the two groups that merge at all.
    v_outside merges a cell when V_lower > merge_relax_margin, so a LARGER
    margin merges less. gv_generator merges when phi_upper <= min(margin,
    -GENERATOR_MARGIN), so a MORE NEGATIVE margin merges less; the baseline
    -1000 is below every observed Phi and therefore never merges a pair.
    """
    if args.outside_merge_margin is not None:
        params.refinement.v_outside.merge_relax_margin = args.outside_merge_margin
    if args.generator_merge_margin is not None:
        params.refinement.gv_generator.merge_relax_margin = args.generator_merge_margin
    if args.n_to_refine is not None:
        params.refinement.gv_generator.N_to_refine = args.n_to_refine
    if args.max_generator_cells is not None:
        params.refinement.gv_generator.max_cells = args.max_generator_cells
    if args.merge_interval is not None:
        for name in ("v_goal", "v_init", "v_unsafe", "v_outside", "gv_generator"):
            getattr(params.refinement, name).merge_interval = args.merge_interval
    return params


def load_sat_baseline(checkpoint):
    """Use the saved problem, never the potentially edited current config.json."""
    checkpoint = Path(checkpoint).resolve()
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    results = bundle.get("final_results") or {}
    if not all(bool(results.get(key, False)) for key in SAT_KEYS):
        raise ValueError("Baseline must be an eval_bundle.pth with SAT final_results for every constraint")
    for key in ("V_state_dict", "control_state_dict", "region_cells", "hyperparameters", "regions"):
        if not bundle.get(key):
            raise ValueError(f"Baseline is missing {key}")
    metadata = json.loads((checkpoint.parent / "run_config.json").read_text())
    config = deepcopy(metadata["example"])
    hp = bundle["hyperparameters"]
    if hp["network"]["n_inputs"] != 3 or hp.get("include_time", False) or hp.get("include_energy", False):
        raise ValueError("Expected a purely spatial XV-15 checkpoint, not an augmented one")
    if not math.isfinite(float(hp["constraints"]["beta_ra"])) or hp["constraints"]["beta_ra"] <= 1:
        raise ValueError("Reach-avoid under an effort budget requires beta_ra > 1")
    arrays = load_region_arrays(config)
    for name in ("init", "goal", "full"):
        saved = np.asarray(bundle["regions"][name], dtype=np.float32)
        if not np.array_equal(saved, arrays[f"{name}_range"]):
            raise ValueError(f"Saved {name} region disagrees with run_config.json")
    for key, interval in (("density", load_density_interval(config)), ("mass", load_mass_interval(config))):
        saved = (bundle.get("GV_state_dict") or {}).get(f"f.{key}_interval")
        if saved is None or not torch.equal(saved.cpu(), torch.tensor(interval, dtype=saved.dtype)):
            raise ValueError(f"Saved robust {key} interval disagrees with run_config.json")
    return bundle, config


def build_energy_regions(config, energy_max, energy_margin):
    """Lift physical boxes and add the pendulum-style upper unsafe band."""
    maximum, ceiling, goal_max = energy_levels(energy_max, energy_margin)
    arrays = load_region_arrays(config)

    def lift(box, lower=0.0, upper=maximum):
        return np.vstack((box, np.array([[lower, upper]], dtype=np.float32)))

    unsafe_boxes = [lift(box) for box in arrays["unsafe_ranges"]]
    # The whole upper band is unsafe at every physical state, goal included.
    unsafe_boxes.append(lift(arrays["full_range"], lower=ceiling))
    regions = Regions(init=Region(lift(arrays["init_range"], upper=0.0)),
                      goal=Region(lift(arrays["goal_range"], upper=goal_max)),
                      unsafe=Region.union(*(Region(box) for box in unsafe_boxes)),
                      full=Region(lift(arrays["full_range"])))
    return regions, np.stack(unsafe_boxes)


def energy_cell_edges(regions, energy_cells):
    """Uniform energy grid with exact goal-ceiling and unsafe-band edges."""
    if not isinstance(energy_cells, int) or isinstance(energy_cells, bool) or energy_cells < 1:
        raise ValueError("energy-cells must be a positive integer")
    edges = np.unique(np.concatenate((
        np.linspace(0.0, regions.full.upper[-1], energy_cells + 1, dtype=np.float32),
        [regions.goal.upper[-1], regions.unsafe.components[-1].lower[-1]],
    ))).astype(np.float32)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Energy slab edges must remain distinct in float32")
    return edges


def augment_baseline_cells(baseline_cells, regions, energy_cells=4):
    """Extrude final 3D cells without changing any physical-state endpoints.

    Follow the pendulum's group transfers: goal cells become outside above
    the goal ceiling and join generator only in the gap below the unsafe
    band. Goal + outside cells tile X and enforce the upper unsafe band.
    """
    source = {}
    for name in GROUPS:
        if not isinstance(baseline_cells, dict) or not baseline_cells.get(name):
            raise ValueError(f"Missing nonempty baseline cells for {name}")
        source[name] = []
        for lower, upper in baseline_cells[name]:
            lower, upper = [torch.as_tensor(v).detach().cpu().float().clone() for v in (lower, upper)]
            if (lower.shape != (3,) or upper.shape != (3,)
                    or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
                    or not (lower <= upper).all()):
                raise ValueError(f"Invalid 3D baseline cell in {name}")
            source[name].append((lower, upper))

    def extrude(name, e_lower, e_upper):
        return [(torch.cat((lower, lower.new_tensor([e_lower]))),
                 torch.cat((upper, upper.new_tensor([e_upper]))))
                for lower, upper in source[name]]

    goal_max = regions.goal.upper[-1]
    ceiling = regions.unsafe.components[-1].lower[-1]
    result = {name: [] for name in GROUPS}
    result["init"] = extrude("init", 0.0, 0.0)
    edges = energy_cell_edges(regions, energy_cells)
    for e_lower, e_upper in zip(edges[:-1], edges[1:]):
        result["outside"].extend(extrude("outside", e_lower, e_upper))
        result["unsafe"].extend(extrude("unsafe", e_lower, e_upper))
        if e_upper <= goal_max:
            result["goal"].extend(extrude("goal", e_lower, e_upper))
        else:
            result["outside"].extend(extrude("goal", e_lower, e_upper))
        if e_upper <= ceiling:
            result["generator"].extend(extrude("generator", e_lower, e_upper))
            if e_lower >= goal_max:
                result["generator"].extend(extrude("goal", e_lower, e_upper))
        else:
            # Baseline goal + outside cover X, including physical unsafe sets.
            result["unsafe"].extend(extrude("goal", e_lower, e_upper))
            result["unsafe"].extend(extrude("outside", e_lower, e_upper))
    return result


def lift_baseline_value_state(baseline_state, energy_max):
    """Preserve V(x) at every E by appending a zero column to the first layer."""
    state = {key: value.detach().clone() for key, value in baseline_state.items()}
    weight = state["layer1.weight"]
    if weight.shape[1] != 3 or state["input_scale"].numel() != 3:
        raise ValueError("Expected three spatial value-network inputs")
    state["layer1.weight"] = torch.cat((weight, weight.new_zeros(weight.shape[0], 1)), dim=1)
    state["input_scale"] = torch.cat((state["input_scale"], weight.new_tensor([float(energy_max)])))
    state["input_offset"] = torch.cat((state["input_offset"], weight.new_zeros(1)))
    return state


def make_energy_parameters(bundle, checkpoint, args):
    params = EnergyHyperparameters.from_dict(bundle["hyperparameters"])
    params.energy_max, params.energy_margin = args.energy_max, args.energy_margin
    params.energy_cells = args.energy_cells
    params.baseline_checkpoint = str(Path(checkpoint).resolve())
    maximum, _, _ = energy_levels(params.energy_max, params.energy_margin)
    lifted = lift_baseline_value_state(bundle["V_state_dict"], maximum)
    params.include_time, params.include_energy = False, True
    params.compute_V = params.compute_GV = True
    params.network.n_inputs = 4
    params.network.input_scale = lifted["input_scale"].tolist()
    params.network.scale_factor = float(lifted["scale_factor"])
    params.training.device = args.device
    params.training.num_epochs = args.epochs
    params.training.random_seed = args.seed
    params.training.pretrain_epochs = args.pretrain_epochs
    params.training.pretrain_n_samples = args.samples
    params.training.enable_pretraining = args.pretrain_epochs > 0
    params.training.curriculum_mode = "none"
    params.training.generator_start_epoch = 0
    if params.training.generator_weight <= 0:
        raise ValueError("The baseline must enable generator training")
    # De-emphasize the energy axis; this entry point extrudes the saved
    # partition instead of calling discretize_regions, the only consumer.
    params.discretization.axis_weights = [1.0, 1.0, 1.0, 0.2]
    apply_refinement_overrides(params, args)
    if args.no_plots:
        params.logging.visualize_interval = 0
    return params


def build_networks(config, params, bundle, freeze_controller=False):
    """Load V and u; let GV strip E before calling the physical dynamics/u."""
    maximum, _, _ = energy_levels(params.energy_max, params.energy_margin)
    lifted = lift_baseline_value_state(bundle["V_state_dict"], maximum)
    control_state = bundle["control_state_dict"]
    controller = XV15EqMLPControl(config, control_state["x_eq"], control_state["u_eq"], control_state["input_scale"])
    controller.load_state_dict(control_state, strict=True)
    controller.requires_grad_(not freeze_controller)
    controller.to(params.training.device)
    aero = XV15Aero(config)
    drift = DensityMassIntervalDrift(aero, controller, load_density_interval(config), load_mass_interval(config)).to(params.training.device)
    diffusion = DiagonalDiffusion(config).to(params.training.device)
    dynamics = Dynamics.dynamics(f=drift, g=diffusion, state_dim=3)
    value = create_V(params.network, input_offset=lifted["input_offset"].tolist(), output_offset=float(lifted["output_offset"]))
    value.load_state_dict(lifted, strict=True)
    value.to(params.training.device)
    # create_GV reads dE/dt from drift.controller.raw_control, so the rate is
    # the same normalized effort that run_mc.py integrates.
    generator = create_GV(V_net=value, dynamics=dynamics, network_config=params.network,
                          input_offset=lifted["input_offset"].tolist(),
                          include_time=False, include_energy=True, verify=False).to(params.training.device)
    return value, generator, controller


def pretrain_value_network(value, generator, controller, regions, unsafe_boxes, params, outputs):
    """Optimize only V, retaining the generator loss with a fixed controller."""
    flags = [p.requires_grad for p in controller.parameters()]
    controller.requires_grad_(False)
    try:
        print("Pretraining V only; controller fixed (including in the generator loss).")
        pretrain_network_samples(
            model=value, GV_net=generator, control_net=None, params=params,
            x_goal_range=regions.goal.bounds, x_unsafe_range=unsafe_boxes,
            x_init_range=regions.init.bounds, x_range=regions.full.bounds,
            num_epochs=params.training.pretrain_epochs, lr=params.training.pretrain_lr,
            device=params.training.device, n_each=params.training.pretrain_n_samples,
            # Every unsafe box receives samples, including the energy band.
            unsafe_sample_fraction=1.0 / len(unsafe_boxes), save_v_path=outputs / "V_pretrained.pth")
    finally:
        for parameter, flag in zip(controller.parameters(), flags):
            parameter.requires_grad_(flag)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--energy-max", type=float, default=500.0,
                        help="Upper end of the accumulated-effort domain (default: %(default)s)")
    parser.add_argument("--energy-margin", type=float, default=None,
                        help="Unsafe band width; default 5%% of energy-max")
    parser.add_argument("--energy-cells", type=int, default=1,
                        help="Uniform slabs over [0, energy-max]; goal/unsafe boundaries add edges. Spatial cells are inherited.")
    parser.add_argument("--outside-merge-margin", type=float, default=50.0,
                        help="v_outside merge_relax_margin; merges cells with V_lower > this, so larger merges less (baseline: 20)")
    parser.add_argument("--generator-merge-margin", type=float, default=None,
                        help="gv_generator merge_relax_margin; merges cells with Phi_upper <= this, so more negative merges less (baseline: -1000, which never merges)")
    parser.add_argument("--n-to-refine", type=int, default=None,
                        help="Generator cells split per refinement event; each yields refine_factor**4 = 16 subcells (baseline: 100)")
    parser.add_argument("--max-generator-cells", type=int, default=None,
                        help="Cap on generator cells; refinement stops above it (baseline: 100000)")
    parser.add_argument("--merge-interval", type=int, default=None,
                        help="Epochs between merge passes for every group; keep it out of phase with refine_interval (baseline: 501 vs 500)")
    parser.add_argument("--freeze-controller", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Fine-tuning RNG seed; default source remains the seed0 SAT bundle")
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--pretrain-epochs", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, help="New or empty directory; default: baseline seed0/energy/<timestamp>")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip both periodic training-progress plots and final summary plots")
    args = parser.parse_args(argv)
    if args.energy_margin is None:
        args.energy_margin = 0.05 * args.energy_max
    try:
        energy_levels(args.energy_max, args.energy_margin)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs < 1 or args.pretrain_epochs < 0 or args.samples < 1 or args.energy_cells < 1 or args.threads < 1:
        parser.error("epochs/samples/energy-cells/threads must be positive; pretrain-epochs must be nonnegative")
    for name in ("outside_merge_margin", "generator_merge_margin"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    for name in ("n_to_refine", "max_generator_cells", "merge_interval"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be between 0 and 2**32-1")
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_reproducibility(args.seed)
    torch.set_num_threads(args.threads)
    bundle, config = load_sat_baseline(args.baseline_checkpoint)
    params = make_energy_parameters(bundle, args.baseline_checkpoint, args)
    regions, unsafe_boxes = build_energy_regions(config, params.energy_max, params.energy_margin)
    if args.pretrain_epochs and args.samples < len(unsafe_boxes):
        raise ValueError(f"samples must be at least {len(unsafe_boxes)} so every unsafe box is sampled")
    cells = augment_baseline_cells(bundle["region_cells"], regions, args.energy_cells)
    value, generator, controller = build_networks(config, params, bundle, args.freeze_controller)
    # Domain-valid consistency check (generic create_GV checks include v=0).
    physical = torch.tensor(load_region_arrays(config)["init_range"], device=args.device)
    x = physical[:, 0] + torch.rand(16, 3, device=args.device) * (physical[:, 1] - physical[:, 0])
    maximum, ceiling, goal_max = energy_levels(params.energy_max, params.energy_margin)
    xe = torch.cat([x, torch.rand(16, 1, device=args.device) * float(maximum)], dim=1)
    torch.testing.assert_close(generator(xe), compute_GV_autograd(generator, xe), atol=1e-4, rtol=1e-4)
    run_dir = args.output_dir or HERE / "seed0" / "energy" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {run_dir}")
    outputs, results_dir, progress = (run_dir / name for name in ("outputs", "results", "training_progress"))
    for directory in (outputs, results_dir, progress):
        directory.mkdir(parents=True, exist_ok=True)
    params.training.resume_checkpoint_path = str(outputs / "resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(outputs)
    params.training.progress_output_dir = str(progress)
    edges = energy_cell_edges(regions, args.energy_cells)
    beta_ra = params.constraints.beta_ra
    metadata = dict(mode="effort_constrained_uncertain_parameters", example=config,
                    hyperparameters=params.to_dict(),
                    baseline_checkpoint=str(args.baseline_checkpoint.resolve()),
                    baseline_sha256=hashlib.sha256(args.baseline_checkpoint.read_bytes()).hexdigest(),
                    baseline_sat_results=bundle["final_results"],
                    energy_max=params.energy_max, energy_margin=params.energy_margin,
                    energy_budget=float(ceiling), energy_ceiling=float(ceiling),
                    goal_energy_max=float(goal_max),
                    energy_domain=[0.0, float(maximum)],
                    unsafe_energy_band=[float(ceiling), float(maximum)],
                    alpha=ALPHA, beta=beta_ra, eta=ETA,
                    probability_lower_bound=1 - ALPHA / beta_ra,
                    energy_cell_edges=edges.tolist(),
                    freeze_controller=args.freeze_controller,
                    coordinate_order=["v", "gamma", "beta", "E"],
                    energy_definition=ENERGY_DEFINITION,
                    initial_discretization="baseline_final_cells_extruded_in_energy",
                    budget_condition="V(x,E)>=beta on X x [energy_max-energy_margin, energy_max]",
                    refinement_overrides=refinement_overrides(args),
                    baseline_cell_counts={k: len(v) for k, v in bundle["region_cells"].items()},
                    initial_cell_counts={k: len(v) for k, v in cells.items()})
    (outputs / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    enable_terminal_logging(outputs / "terminal_log.txt")
    print(f"Effort-constrained uncertain XV-15: V(v,gamma,beta,E), u(v,gamma,beta)")
    print(f"SAT baseline: {args.baseline_checkpoint.resolve()}")
    print(f"Density: {load_density_interval(config)}; mass: {load_mass_interval(config)}")
    print(f"Energy: {ENERGY_DEFINITION}")
    print(f"Normalized effort ceiling: {float(ceiling):g}; "
          f"domain E in [0, {float(maximum):g}], starting at E = 0")
    print(f"Upper unsafe band: [{float(ceiling):g}, {float(maximum):g}] at every physical state")
    print(f"Goal energy upper bound (with separation gap): {float(goal_max):g}")
    print(f"Energy slab edges: {metadata['energy_cell_edges']}")
    pretraining = f"certificate-only pretraining for {args.pretrain_epochs} epochs" if args.pretrain_epochs else "no pretraining"
    print(f"Controller: {'frozen' if args.freeze_controller else 'fine-tuned'}, {pretraining}")
    print("Progress and final plots: off" if args.no_plots
          else f"Progress plots: every {params.logging.visualize_interval} epochs")
    overrides = metadata["refinement_overrides"]
    print(f"Refinement overrides: {overrides}" if overrides
          else "Refinement: baseline settings inherited from the checkpoint")
    for name in GROUPS:
        print(f"{name}: {len(bundle['region_cells'][name])} baseline spatial cells -> {len(cells[name])} augmented cells")
    print(f"Outputs: {run_dir.resolve()}")
    print("The lifted baseline is not yet an energy certificate; the augmented conditions require new SAT.")
    torch.save(cells, outputs / "initial_region_cells.pth")
    if args.pretrain_epochs:
        pretrain_value_network(value, generator, controller, regions, unsafe_boxes, params, outputs)
    started = time.time()
    history, final_beta_s, refinements = train_network_bounds(
        V_net=value, GV_net=generator, control_net=None if args.freeze_controller else controller,
        region_cells=cells, regions=regions, params=params, start_time=started,
        create_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=.95))
    results = evaluate_constraints(value, generator, cells, beta_ra=beta_ra, device=args.device)
    print_constraint_summary(results)
    passed = all(bool(results[key]) for key in SAT_KEYS)
    if passed:
        print(f"Effort-constrained SAT: robust reach-avoid probability >= {1 - ALPHA / beta_ra:g} "
              f"with normalized effort <= {float(goal_max):g} < {float(ceiling):g} at the hitting time.")
    else:
        print("Effort-constrained conditions remain unsatisfied; no energy guarantee established.")
    # Always save final weights and verification results, including epoch-limit
    # exits. Store the controller even when it was frozen during training.
    save_eval_bundle(outputs, V_net=value, GV_net=generator, control_net=controller, params=params,
                     regions=regions, region_cells=cells, final_beta_s=final_beta_s,
                     loss_history=history, refinement_epochs=refinements, results=results)
    torch.save(value.state_dict(), outputs / "V_final.pth")
    torch.save(controller.state_dict(), outputs / "controller_final.pth")
    (outputs / "energy_result.json").write_text(json.dumps(dict(
        energy_max=params.energy_max, energy_margin=params.energy_margin,
        energy_budget=float(ceiling), energy_ceiling=float(ceiling),
        goal_energy_max=float(goal_max), energy_domain=[0.0, float(maximum)],
        unsafe_energy_band=[float(ceiling), float(maximum)],
        alpha=ALPHA, beta=beta_ra, eta=ETA, satisfied=passed,
        probability_lower_bound=1 - ALPHA / beta_ra if passed else None,
        results=results), indent=2) + "\n")
    if not args.no_plots:
        create_summary_plots(value, generator, regions, cells, beta_ra,
                             final_beta_s, history, refinements, results, str(results_dir))
    return results


if __name__ == "__main__":
    main()
