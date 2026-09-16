"""Nominal 4D double integrator: neural certificate and controller synthesis.

Run from the repository root:
  .venv/bin/python examples/asteroid_landing_uncertain/neural_certified_nominal_drift/main.py --seed 0

Follows examples/xv15_uncertain/neural_certified_nominal_drift: sample
pretraining, adaptive interval-bound training, final evaluation and plots.
Parameters are fixed; Brownian diffusion remains.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

TRAIN_SEED = 0
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from examples.double_integrator_uncertain.model import (
    DiagonalDiffusion, LearnablePDControl, create_control, NominalClosedLoopDrift,
    domain_box, load_config, load_region_arrays,
)
# from examples.double_integrator_uncertain.neural_certified_nominal_drift.cell_grid import (
#     discretize_unsafe_boxes,
# )
from examples.double_integrator_uncertain.neural_certified_nominal_drift.unsat_diagnostics import (
    compute_diagnostic_bounds, save_unsat_diagnostics,
)
from src.crown_bounds import validate_bound_method
from src.discretization import discretize_regions
from src.dynamics import Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV, verify_GV
from src.pretrainer import pretrain_network_samples
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging, save_eval_bundle
from src.trainer import train_network_bounds
from src.training_utils import evaluate_constraints, print_constraint_summary
from src.utils import cleanup_and_setup_directories
from src.visualization import create_summary_plots


def configure_reproducibility(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def make_hyperparameters(config, seed, seed_dir, n_unsafe_boxes):
    settings = config["learning"]
    integer_keys = ("pretrain_epochs", "pretrain_n_samples", "epochs", "region_budget",
                    "generator_budget", "refine_interval", "refine_n_cells",
                    "max_value_cells", "max_generator_cells")
    for name in integer_keys:
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"learning.{name} must be a positive integer")
    hidden = settings["certificate_hidden_dims"]
    if len(hidden) != 2 or any(not isinstance(n, int) or n < 1 for n in hidden):
        raise ValueError("Specify two positive certificate hidden-layer widths")
    for name in ("pretrain_lr", "lr", "certificate_scale_factor", "generator_weight"):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"learning.{name} must be positive and finite")
    if not np.isfinite(settings["pretrain_l2"]) or settings["pretrain_l2"] < 0:
        raise ValueError("learning.pretrain_l2 must be finite and nonnegative")
    # Give every unsafe-box component at least one cell in the shared budget allocator.
    if min(settings["region_budget"], settings["generator_budget"]) < 10 * n_unsafe_boxes:
        raise ValueError(f"Both discretization budgets must be at least {10 * n_unsafe_boxes} "
                          f"to cover all {n_unsafe_boxes} boundary strips")
    if settings["pretrain_n_samples"] < n_unsafe_boxes:
        raise ValueError(f"Use at least {n_unsafe_boxes} samples to cover all boundary strips")
    if (settings["max_value_cells"] < settings["region_budget"]
            or settings["max_generator_cells"] < settings["generator_budget"]):
        raise ValueError("Refinement cell limits must not be smaller than the initial budgets")
    # Absent keys keep the historical interval-bound behaviour.
    bound_method = validate_bound_method(settings.get("bound_method", "IBP"))
    generator_bound_method = validate_bound_method(settings.get("generator_bound_method", "IBP"))
    # Step decay: multiply the rate by lr_decay_gamma every lr_decay_step
    # epochs. Absent keys keep the historical StepLR(2000, 0.95) shape.
    lr_decay_gamma = settings.get("lr_decay_gamma", 0.95)
    lr_decay_step = settings.get("lr_decay_step", 2000)
    if type(lr_decay_step) is not int or lr_decay_step < 1:
        raise ValueError("learning.lr_decay_step must be a positive integer")
    if not (np.isfinite(lr_decay_gamma) and 0 < lr_decay_gamma <= 1.0):
        raise ValueError("learning.lr_decay_gamma must be finite and in (0, 1]")
    # Learning-rate floor: hold lr_floor from lr_floor_epoch on, so that the
    # decay does not shrink late refinements' updates towards zero. An absent
    # lr_floor_epoch disables the floor and restores the pure decay.
    lr_floor_epoch = settings.get("lr_floor_epoch", 0)
    lr_floor = settings.get("lr_floor", 0.0)
    if type(lr_floor_epoch) is not int or lr_floor_epoch < 0:
        raise ValueError("learning.lr_floor_epoch must be a nonnegative integer")
    if lr_floor_epoch and not (np.isfinite(lr_floor) and 0 < lr_floor <= settings["lr"]):
        raise ValueError("learning.lr_floor must be positive, finite and at most learning.lr")

    params = Hyperparameters.default()
    params.include_time = False
    params.include_energy = False
    params.network.n_inputs = 4
    params.network.n_hidden_1, params.network.n_hidden_2 = hidden
    full = domain_box(config)
    params.network.input_scale = ((full[:, 1] - full[:, 0]) / 2.0).tolist()
    params.network.scale_factor = float(settings["certificate_scale_factor"])

    params.training.learning_rate = float(settings["lr"])
    params.training.num_epochs = settings["epochs"]
    params.training.generator_weight = float(settings["generator_weight"])
    params.training.generator_start_epoch = 0
    params.training.random_seed = seed
    params.training.enable_pretraining = True
    params.training.pretrain_epochs = settings["pretrain_epochs"]
    params.training.pretrain_lr = float(settings["pretrain_lr"])
    params.training.pretrain_n_samples = settings["pretrain_n_samples"]
    params.training.bound_method = bound_method
    params.training.generator_bound_method = generator_bound_method
    params.training.lr_decay_gamma = float(lr_decay_gamma)
    params.training.lr_decay_step = lr_decay_step
    params.training.lr_floor = float(lr_floor)
    params.training.lr_floor_epoch = lr_floor_epoch
    params.training.curriculum_mode = "none"
    params.training.resume_checkpoint_path = str(seed_dir / "outputs" / "resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(seed_dir / "outputs")
    params.training.progress_output_dir = str(seed_dir / "training_progress")

    params.discretization.axis_weights = [1.0] * 4
    params.discretization.max_region_budget = settings["region_budget"]
    params.discretization.max_generator_budget = settings["generator_budget"]
    params.constraints.beta_ra = float(config["beta_ra"])
    params.compute_V = True
    params.compute_GV = True

    # Same refinement/merging schedule as the XV-15 nominal-drift baseline.
    # Each 4D refinement creates 16 children, so N_to_refine comes from the
    # configured budget rather than XV-15's fixed 100.
    for name in ("v_goal", "v_init", "v_unsafe", "v_outside", "gv_generator"):
        cfg = getattr(params.refinement, name)
        cfg.enable_refinement = True
        cfg.refine_interval = settings["refine_interval"]
        cfg.late_epoch_threshold = 3500 if name == "gv_generator" else 2500
        cfg.refine_interval_late = 250 if name == "v_outside" else settings["refine_interval"]
        cfg.refine_factor = 2
        cfg.max_cells = settings["max_generator_cells"] if name == "gv_generator" else settings["max_value_cells"]
        cfg.N_to_refine = settings["refine_n_cells"]
        cfg.enable_merging = name in ("v_outside", "gv_generator")
        cfg.merge_interval = settings["refine_interval"] + 1
        cfg.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 20.0
    params.refinement.gv_generator.merge_relax_margin = -1000.0
    params.refinement.refine_interval_after_first_sat = settings["refine_interval"]
    params.logging.visualize_interval = 2000
    return params


def build_problem(config, params):
    arrays = load_region_arrays(config)
    regions = Regions(
        init=Region(arrays["init_range"]), goal=Region(arrays["goal_range"]),
        unsafe=Region.union(*[Region(box) for box in arrays["unsafe_ranges"]]),
        full=Region(arrays["full_range"]),
    )
    # Both controller forms use the configured goal center as their reference.
    input_offset = arrays["goal_range"].mean(axis=1).tolist()
    controller = create_control(config, input_offset, params.network.input_scale).to(params.training.device)
    drift = NominalClosedLoopDrift(controller).to(params.training.device)
    diffusion = DiagonalDiffusion(config).to(params.training.device)
    dynamics = Dynamics.dynamics(f=drift, g=diffusion, state_dim=4)
    # Sigmoid architecture, anchored at V(goal center) = 0.1. This does not
    # constrain the controller's command there.
    value = create_V(params.network, input_offset=input_offset, output_offset=np.float32(0.1))
    value = value.to(params.training.device)
    generator = create_GV(
        V_net=value, dynamics=dynamics, network_config=params.network,
        input_offset=input_offset, include_time=False, include_energy=False,
        # Check the configured domain explicitly below, including the origin.
        verify=False,
    ).to(params.training.device)
    return value, generator, controller, dynamics, regions, arrays


def check_generator(value, generator, dynamics, arrays):
    """At startup, compare the analytic generator with autograd in the domain."""
    device = next(value.parameters()).device
    full = torch.as_tensor(arrays["full_range"], device=device)
    rng = torch.Generator(device=device).manual_seed(0)
    points = full[:, 0] + torch.rand(32, 4, generator=rng, device=device) * (full[:, 1] - full[:, 0])
    corners = torch.cartesian_prod(*[full[i] for i in range(4)])
    points = torch.cat([points, corners, generator.f.controller.input_offset[None]])
    if not verify_GV(generator, dynamics=dynamics, x=points, tol=1e-4):
        raise RuntimeError("Double-integrator generator failed its autograd consistency check")


def evaluate_and_visualize(value, generator, controller, regions, cells, params,
                           output_dir, results_dir, loss_history, refinement_epochs,
                           final_beta_s, no_plots=False):
    print("\n" + "=" * 20 + "\nFinal Evaluation\n" + "=" * 20)
    if isinstance(controller, LearnablePDControl):
        print("Final learned controller gains:", controller.gains())
        with (output_dir / 'controller_gains.json').open('w', encoding='utf-8') as stream:
            json.dump(controller.gains(), stream, indent=2)
    # Always recompute the final cells, including children created by the
    # last refinement round. Bound batches limit memory.
    bounds, phi_upper = compute_diagnostic_bounds(value, generator, cells, params)
    results = evaluate_constraints(
        value, generator, cells, beta_ra=params.constraints.beta_ra,
        device=params.training.device,
        precomputed_region_bounds=bounds, precomputed_phi_uppers=phi_upper,
        precomputed_v_gen_lowers=bounds["generator"][0],
    )
    print_constraint_summary(results)
    # Save even when the epoch budget ends without SAT, and before plotting.
    save_eval_bundle(
        output_dir, V_net=value, GV_net=generator, control_net=controller,
        params=params, regions=regions, region_cells=cells,
        final_beta_s=final_beta_s, loss_history=loss_history,
        refinement_epochs=refinement_epochs, results=results,
    )
    save_unsat_diagnostics(cells, regions, params, bounds, phi_upper, results_dir / "unsat_cells")
    if not no_plots:
        create_summary_plots(
            V_net=value, GV_net=generator, regions=regions, region_cells=cells,
            beta_s=final_beta_s, beta_ra=params.constraints.beta_ra,
            loss_history=loss_history, refinement_epochs=refinement_epochs,
            results=results, output_dir=str(results_dir),
        )
    return results


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--config", type=Path, default=EXAMPLE_ROOT / "config.json")
    parser.add_argument("--epochs", type=positive_int, default=None, help="Override bound-training epoch budget")
    parser.add_argument("--pretrain-epochs", type=positive_int, default=None, help="Override sample-pretraining epochs")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable routine plots; UNSAT cell diagnostics are still saved")
    parser.add_argument("--run-tag", default="double_integrator", help="Suffix for a separate seed output folder")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32-1")
    if args.run_tag and not all(c.isalnum() or c in "_-" for c in args.run_tag):
        parser.error("--run-tag may contain only letters, numbers, underscores and hyphens")
    configure_reproducibility(args.seed)
    config = load_config(args.config)
    n_unsafe_boxes = len(load_region_arrays(config)["unsafe_ranges"])
    suffix = f"_{args.run_tag}" if args.run_tag else ""
    seed_dir = HERE / f"seed{args.seed}{suffix}"
    params = make_hyperparameters(config, args.seed, seed_dir, n_unsafe_boxes)
    params.training.device = args.device
    if args.epochs is not None:
        params.training.num_epochs = args.epochs
    if args.pretrain_epochs is not None:
        params.training.pretrain_epochs = args.pretrain_epochs
    if args.no_plots:
        params.logging.visualize_interval = 0

    output_dir, results_dir, progress_dir = [seed_dir / name for name in ("outputs", "results", "training_progress")]
    cleanup_and_setup_directories([output_dir, results_dir, progress_dir])
    enable_terminal_logging(output_dir / "terminal_log.txt", append=False)
    print("=" * 20 + "\nDouble Integrator: Nominal 4D SDE\n" + "=" * 20)
    print(f"Random seed: {args.seed}; device: {args.device}")
    print("Fixed nondimensional parameters; acceleration Brownian noise; no time/energy augmentation.")
    print(f"Configuration: {args.config.resolve()}\nOutputs: {output_dir}")
    print("State: [px, py, vx, vy]; nominal drift: [vx, vy, ux, uy].")
    print("All coordinates, controls, and time are dimensionless; no unit conversion.")
    print(json.dumps(params.to_dict(), indent=2))
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(mode="nominal_drift", example=config, hyperparameters=params.to_dict()), stream, indent=2)

    value, generator, controller, dynamics, regions, arrays = build_problem(config, params)
    print("Controller type:", config['control'].get('type', 'neural'))
    if isinstance(controller, LearnablePDControl):
        print("Initial controller gains:", controller.gains())
    print("Goal center / controller input_offset [px, py, vx, vy]:",
          controller.input_offset.detach().cpu().tolist())
    print("Controller output at input_offset:",
          controller(controller.input_offset[None]).detach().cpu().tolist())
    print("Nondimensional diffusion:", dynamics.g.sigma.detach().cpu().tolist())
    for name in ("full_range", "init_range", "goal_range"):
        print(f"{name} [px, py, vx, vy]:", arrays[name].tolist())
    print(f"Avoid set: {len(arrays['unsafe_ranges'])} domain-boundary strips")
    check_generator(value, generator, dynamics, arrays)
    cells = discretize_regions(regions, params.discretization, use_radial_generator=False)
    # cells["unsafe"] = discretize_unsafe_boxes(
    #     arrays["unsafe_ranges"], params.network.input_scale,
    #     params.discretization.max_region_budget,
    # )

    started = time.time()
    pretrain_network_samples(
        model=value, GV_net=generator, control_net=controller,
        x_goal_range=arrays["goal_range"], x_unsafe_range=arrays["unsafe_ranges"],
        x_init_range=arrays["init_range"], x_range=arrays["full_range"],
        params=params, num_epochs=params.training.pretrain_epochs,
        lr=params.training.pretrain_lr, device=params.training.device,
        n_each=params.training.pretrain_n_samples, lambda_w=float(config["learning"]["pretrain_l2"]),
        unsafe_sample_fraction=1.0 / len(arrays["unsafe_ranges"]),
        save_v_path=output_dir / "V_pretrained.pth",
        save_control_path=output_dir / "controller_pretrained.pth",
    )
    print(f"Pre-training time: {time.time() - started:.1f}s")
    if isinstance(controller, LearnablePDControl):
        print("Controller gains after pretraining:", controller.gains())
    started = time.time()

    def create_scheduler(optimizer):
        # An unfloored decay makes the total remaining progress, which is
        # proportional to the sum of the learning rates, converge: two thirds
        # of it was already spent by epoch 40000 of the seed-0 run. Holding the
        # rate from learning.lr_floor_epoch keeps that sum growing, so cells
        # created by late refinements can still be trained.
        floor_epoch = params.training.lr_floor_epoch
        floor_factor = params.training.lr_floor / params.training.learning_rate
        gamma, step = params.training.lr_decay_gamma, params.training.lr_decay_step

        def factor(epoch):
            if floor_epoch and epoch >= floor_epoch:
                return floor_factor
            return gamma ** (epoch // step)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    loss_history, final_beta_s, refinement_epochs = train_network_bounds(
        V_net=value, GV_net=generator, region_cells=cells, regions=regions,
        params=params, control_net=controller, create_scheduler=create_scheduler,
        start_time=started,
    )
    print(f"Bound-training time: {time.time() - started:.1f}s")
    evaluate_and_visualize(
        value, generator, controller, regions, cells, params, output_dir, results_dir,
        loss_history, refinement_epochs, final_beta_s, no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
