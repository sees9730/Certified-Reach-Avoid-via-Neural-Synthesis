"""Fine-tune neural_certified's robust controller with an energy certificate.

Coordinates are z = [theta, omega, E], with E(0) = 0 and
    dE/dt = raw_control(theta, omega)**2,
    GV = G_nominal V + DRIFT_MAG * abs(dV/domega)
         + raw_control(theta, omega)**2 * dV/dE.
The policy still has exactly two inputs. Energy measures squared normalized
policy output, as in WrapperConterlNN.raw_control; squared physical torque
integral is M_torque**2 times this quantity (36 with the baseline parameters).
This is exactly run_mc.py's energy measure: energy_acc += u_raw**2 * dt,
integrated until the physical goal is first reached for successful rollouts.

Following Corollary 2 of Neural_Certificate_TAC.pdf, the unsafe set includes
the closed band [energy_max - energy_margin, energy_max] at every physical
state. Thus the requested effort ceiling is energy_max - energy_margin.
The reach-avoid/energy probability bound applies up to reaching the goal,
and only after all certificate checks pass; it is not an all-path hard cap.

By default, load neural_certified/seed<seed>/outputs/eval_bundle.pth, lift V(x)
exactly to V(x,E), pretrain only V with the controller fixed, then fine-tune
both networks during bound-based training. --freeze-controller instead
searches for an energy certificate for the unchanged robust controller.
Initial cells reuse the checkpoint's final spatial discretization exactly.
--energy-cells controls uniform energy slabs; goal/unsafe energy boundaries
are inserted as additional slab edges. Initial-set cells remain at E=0.
Outputs go into a fresh directory; the baseline checkpoint is never replaced.

Example (from the repository root):
    .venv/bin/python examples/inv_pend_adversarial/neural_certified/main_constraint_energy.py --energy-max 2.0 --energy-margin 0.03

Compare a saved energy run using the same MC metrics and disturbance regimes:
    .venv/bin/python examples/inv_pend_adversarial/run_mc.py --energy-controller-dir /path/to/energy/run
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from examples.inv_pend_adversarial.neural_certified.main import (
    CONTROLLER_HIDDEN_DIM, DRIFT_MAG, DYNAMICS, REGIONS_CFG, TRAIN_SEED,
    configure_reproducibility, pretrain_network_samples,
)
from src.control_network import InvertControlNN, WrapperConterlNN
from src.dynamics import Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging, save_eval_bundle
from src.set_values import AdditiveBoxSetDrift, ClosedLoopSetValuedDrift
from src.trainer import train_network_bounds
from src.training_utils import evaluate_constraints, print_constraint_summary
from src.visualization import create_summary_plots


class EnergyHyperparameters(Hyperparameters):
    """Preserve the augmentation flag in the shared trainer's saved bundles."""

    include_energy = True
    energy_cells = 4

    def to_dict(self):
        result = super().to_dict()
        result.update(include_energy=True, energy_max=self.energy_max,
                      energy_margin=self.energy_margin,
                      energy_definition="integral(raw_control(x)**2 dt)",
                      drift_mag=DRIFT_MAG,
                      initial_discretization="baseline_final_cells_extruded_in_energy",
                      energy_cells=self.energy_cells,
                      coordinate_order=["theta", "omega", "E"])
        return result


def build_energy_regions(energy_max, energy_margin):
    """Lift physical boxes and include a positive-width energy unsafe band."""
    if not (math.isfinite(energy_max) and math.isfinite(energy_margin)
            and 0 < energy_margin < energy_max):
        raise ValueError("Require finite 0 < energy_margin < energy_max")
    ceiling = np.float32(energy_max - energy_margin)
    maximum = np.float32(energy_max)
    if not (np.isfinite(maximum) and 0 < ceiling < maximum):
        raise ValueError("Energy bounds must remain distinct and positive in float32")

    def lift(box, lower=0.0, upper=maximum):
        return np.vstack((box, np.array([[lower, upper]], dtype=np.float32)))

    init = Region(lift(REGIONS_CFG["init_range"], upper=0.0))
    # Leave a small positive gap so the closed goal lies below the unsafe band.
    # A one-ULP gap is insufficient: the shared partitioner's float32 midpoint
    # can round onto an endpoint and omit that generator strip.
    goal_energy_max = np.float32(ceiling - 0.01 * min(energy_margin, float(ceiling)))
    if not (0 < goal_energy_max < ceiling):
        raise ValueError("Energy goal and unsafe band must be separated in float32")
    goal = Region(lift(REGIONS_CFG["goal_range"],
                       upper=goal_energy_max))
    unsafe_boxes = [lift(REGIONS_CFG[name]) for name in (
        "unsafe_down1", "unsafe_down2", "unsafe_lb", "unsafe_rb",
        "unsafe_tb", "unsafe_bb",
    )]
    unsafe_boxes.append(lift(REGIONS_CFG["full_range"], lower=ceiling))
    regions = Regions(init=init, goal=goal,
                      unsafe=Region.union(*(Region(box) for box in unsafe_boxes)),
                      full=Region(lift(REGIONS_CFG["full_range"])))
    return regions, np.stack(unsafe_boxes)


def energy_cell_edges(regions, energy_cells):
    """Uniform energy grid with exact edges at both new region boundaries."""
    if not isinstance(energy_cells, int) or isinstance(energy_cells, bool) or energy_cells < 1:
        raise ValueError("energy_cells must be a positive integer")
    return np.unique(np.concatenate((
        np.linspace(0.0, regions.full.upper[-1], energy_cells + 1, dtype=np.float32),
        [regions.goal.upper[-1], regions.unsafe.components[-1].lower[-1]],
    ))).astype(np.float32)


def augment_baseline_cells(baseline_cells, regions, energy_cells=4):
    """Extrude final 2D cells without changing any physical-state endpoints.

    Below the goal energy cutoff, retain each baseline constraint group.
    Above that cutoff, the old goal cells also become outside/ generator
    cells. In the unsafe energy band, goal + outside cells cover the full
    spatial domain and provide the new unsafe cells. These transfers only
    change region membership and the energy interval, never spatial bounds.
    """
    groups = ("init", "goal", "unsafe", "outside", "generator")
    if not isinstance(baseline_cells, dict) or any(
        name not in baseline_cells or len(baseline_cells[name]) == 0 for name in groups
    ):
        raise ValueError("Baseline checkpoint must contain nonempty final region_cells for all constraint groups")
    source = {}
    for name in groups:
        cells = []
        for lower, upper in baseline_cells[name]:
            lower = torch.as_tensor(lower).detach().cpu()
            upper = torch.as_tensor(upper).detach().cpu()
            if (lower.shape != (2,) or upper.shape != (2,)
                    or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
                    or not (lower <= upper).all()):
                raise ValueError(f"Expected valid 2D baseline cells in {name}")
            cells.append((lower, upper))
        source[name] = cells

    def extrude(name, e_lower, e_upper):
        return [(torch.cat((lower, lower.new_tensor([e_lower]))),
                 torch.cat((upper, upper.new_tensor([e_upper]))))
                for lower, upper in source[name]]

    result = {name: [] for name in groups}
    result["init"] = extrude("init", 0.0, 0.0)
    goal_max = regions.goal.upper[-1]
    ceiling = regions.unsafe.components[-1].lower[-1]
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
    if weight.shape[1] != 2 or state["input_scale"].numel() != 2:
        raise ValueError("Expected a two-state V checkpoint from neural_certified/main.py")
    state["layer1.weight"] = torch.cat((weight, weight.new_zeros(weight.shape[0], 1)), dim=1)
    state["input_scale"] = torch.cat((state["input_scale"], weight.new_tensor([energy_max])))
    state["input_offset"] = torch.cat((state["input_offset"], weight.new_zeros(1)))
    return state


def build_networks(params, bundle, freeze_controller=False):
    """Load V and u; let GV strip E before calling the physical dynamics/u."""
    controller = WrapperConterlNN(InvertControlNN(hidden_dim=CONTROLLER_HIDDEN_DIM))
    controller.load_state_dict(bundle["control_state_dict"], strict=True)
    controller.requires_grad_(not freeze_controller)
    controller.to(params.training.device)

    def f_ol(x):
        theta, omega = x[:, 0], x[:, 1]
        acceleration = (DYNAMICS["g"] / DYNAMICS["L"]) * torch.sin(theta)
        acceleration = acceleration - DYNAMICS["b"] / (DYNAMICS["m"] * DYNAMICS["L"]**2) * omega
        return torch.stack((omega, acceleration), dim=1)

    # Match neural_certified/main.py: physical disturbance acts only on omega.
    # The support function adds DRIFT_MAG * |dV/domega| to the generator.
    drift = ClosedLoopSetValuedDrift(
        AdditiveBoxSetDrift(f_ol, torch.tensor([0.0, DRIFT_MAG])), controller=controller,
    ).to(params.training.device)
    dynamics = Dynamics.dynamics(f=drift, g=torch.tensor([0.0, DYNAMICS["sigma"]]), state_dim=2)
    lifted = lift_baseline_value_state(bundle["V_state_dict"], params.energy_max)
    V_net = create_V(params.network, input_offset=lifted["input_offset"].tolist(),
                     output_offset=float(lifted["output_offset"]))
    V_net.load_state_dict(lifted, strict=True)
    V_net.to(params.training.device)
    GV_net = create_GV(V_net=V_net, dynamics=dynamics, network_config=params.network,
                       input_offset=lifted["input_offset"].tolist(),
                       include_time=False, include_energy=True).to(params.training.device)
    return V_net, GV_net, controller


def pretrain_value_network(V_net, GV_net, controller, regions, unsafe_boxes, params, outputs):
    """Optimize only V, retaining the generator loss with a fixed controller."""
    control_params = list(controller.parameters())
    trainable_flags = [parameter.requires_grad for parameter in control_params]
    controller.requires_grad_(False)
    try:
        print("Pretraining V only; controller fixed (including in the generator loss).")
        pretrain_network_samples(
            model=V_net, x_goal_range=regions.goal.bounds, x_unsafe_range=unsafe_boxes,
            x_init_range=regions.init.bounds, x_range=regions.full.bounds,
            params=params, GV_net=GV_net, num_epochs=params.training.pretrain_epochs,
            lr=params.training.pretrain_lr, device=params.training.device, control_net=None,
            n_each=params.training.pretrain_n_samples, save_v_path=outputs / "V_pretrained.pth",
        )
    finally:
        for parameter, trainable in zip(control_params, trainable_flags):
            parameter.requires_grad_(trainable)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-checkpoint", type=Path,
                        help="Default: neural_certified/seed<seed>/outputs/eval_bundle.pth")
    parser.add_argument("--energy-max", type=float, default=2.0)
    parser.add_argument("--energy-margin", type=float, default=None,
                        help="Unsafe band width; default 5%% of energy-max")
    parser.add_argument("--freeze-controller", action="store_true")
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--pretrain-epochs", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--energy-cells", type=int, default=4,
                        help="Uniform energy slabs over [0, energy-max]; goal/unsafe boundaries add edges. Spatial cells are inherited.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path,
                        help="New or empty run directory; default timestamped energy run")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.baseline_checkpoint is None:
        args.baseline_checkpoint = HERE / f"seed{args.seed}" / "outputs" / "eval_bundle.pth"
    if args.energy_margin is None:
        args.energy_margin = 0.05 * args.energy_max
    if not (math.isfinite(args.energy_max) and math.isfinite(args.energy_margin)
            and 0 < args.energy_margin < args.energy_max):
        parser.error("require finite 0 < energy-margin < energy-max")
    if args.epochs < 1 or args.pretrain_epochs < 0 or args.samples < 6:
        parser.error("require epochs >= 1, pretrain-epochs >= 0, samples >= 6")
    if args.energy_cells < 1:
        parser.error("energy-cells must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_reproducibility(args.seed)
    if not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint missing: {args.baseline_checkpoint}. Run neural_certified/main.py first or use --baseline-checkpoint.")
    bundle = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    params = EnergyHyperparameters.from_dict(bundle["hyperparameters"])
    params.energy_max, params.energy_margin = args.energy_max, args.energy_margin
    params.energy_cells = args.energy_cells
    params.include_time = False
    params.compute_V = params.compute_GV = True
    params.network.n_inputs = 3
    params.network.input_scale = [*bundle["V_state_dict"]["input_scale"].tolist(), args.energy_max]
    params.training.device = args.device
    params.training.num_epochs = args.epochs
    params.training.random_seed = args.seed
    params.training.pretrain_epochs = args.pretrain_epochs
    params.training.pretrain_n_samples = args.samples
    params.training.enable_pretraining = args.pretrain_epochs > 0
    params.training.curriculum_mode = "none"
    params.discretization.axis_weights = [1.0, 1.0, 0.2]
    if args.no_plots:
        params.logging.visualize_interval = 0

    regions, unsafe_boxes = build_energy_regions(args.energy_max, args.energy_margin)
    region_cells = augment_baseline_cells(bundle.get("region_cells"), regions, args.energy_cells)
    V_net, GV_net, controller = build_networks(params, bundle, args.freeze_controller)
    run_dir = args.output_dir or HERE / f"seed{args.seed}" / "energy" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {run_dir}")
    outputs, results_dir, progress = (run_dir / name for name in ("outputs", "results", "training_progress"))
    for directory in (outputs, results_dir, progress):
        directory.mkdir(parents=True, exist_ok=True)
    params.training.resume_checkpoint_path = str(outputs / "resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(outputs)
    params.training.progress_output_dir = str(progress)
    enable_terminal_logging(outputs / "terminal_log.txt")
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(energy_ceiling=float(unsafe_boxes[-1, -1, 0]),
                    goal_energy_max=float(regions.goal.upper[-1]),
                    initial_discretization="baseline_final_cells_extruded_in_energy",
                    energy_cell_edges=energy_cell_edges(regions, args.energy_cells).tolist(),
                    baseline_cell_counts={name: len(cells) for name, cells in bundle["region_cells"].items()},
                    initial_cell_counts={name: len(cells) for name, cells in region_cells.items()},
                    energy_definition="integral(raw_control(x)**2 dt)",
                    coordinate_order=["theta", "omega", "E"], drift_mag=DRIFT_MAG)
    (outputs / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Energy-constrained robust pendulum: V(theta, omega, E), u(theta, omega)")
    print(f"Loaded neural_certified networks: {args.baseline_checkpoint}")
    print(f"Angular-acceleration disturbance bound: {DRIFT_MAG:g}")
    print(f"Controller: {'frozen' if args.freeze_controller else 'fine-tuned'}")
    print(f"Normalized effort ceiling: {metadata['energy_ceiling']:g}; domain E <= {args.energy_max:g}")
    print(f"Goal energy upper bound (with separation gap): {metadata['goal_energy_max']:g}")
    print(f"Run directory: {run_dir}")
    print(f"Energy slab edges: {metadata['energy_cell_edges']}")
    for name, cells in region_cells.items():
        print(f"{name}: {len(bundle['region_cells'][name])} baseline spatial cells -> {len(cells)} augmented cells")
    torch.save(region_cells, outputs / "initial_region_cells.pth")
    train_control = None if args.freeze_controller else controller
    if args.pretrain_epochs:
        pretrain_value_network(V_net, GV_net, controller, regions, unsafe_boxes, params, outputs)
    started = time.time()
    loss_history, final_beta_s, refinement_epochs = train_network_bounds(
        V_net=V_net, GV_net=GV_net, region_cells=region_cells, regions=regions,
        params=params, control_net=train_control, start_time=started,
        create_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95),
    )
    results = evaluate_constraints(V_net, GV_net, region_cells,
                                   beta_ra=params.constraints.beta_ra, device=args.device)
    print_constraint_summary(results)
    passed = all(results[f"{name}_satisfied"] for name in ("goal", "unsafe", "init", "outside", "generator"))
    print("All certificate checks passed." if passed else "Certificate checks remain unsatisfied; no energy guarantee established.")
    # Always save final weights and verification results, including epoch-limit
    # exits. Store the controller even when it was frozen during training.
    save_eval_bundle(outputs, V_net=V_net, GV_net=GV_net, control_net=controller,
                     params=params, regions=regions, region_cells=region_cells,
                     final_beta_s=final_beta_s, loss_history=loss_history,
                     refinement_epochs=refinement_epochs, results=results)
    torch.save(V_net.state_dict(), outputs / "V_final.pth")
    torch.save(controller.state_dict(), outputs / "controller_final.pth")
    if not args.no_plots:
        create_summary_plots(V_net=V_net, GV_net=GV_net, regions=regions,
                             region_cells=region_cells, beta_ra=params.constraints.beta_ra,
                             beta_s=final_beta_s, loss_history=loss_history,
                             refinement_epochs=refinement_epochs, results=results,
                             output_dir=str(results_dir))
    return results


if __name__ == "__main__":
    main()
