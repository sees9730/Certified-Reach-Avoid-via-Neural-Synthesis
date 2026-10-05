"""Joint controller/certificate synthesis for the four-state planar Kepler SDE.

XV-15 layout: config/model → sample pretraining → adaptive bound training →
final evaluation bundle. The 4D value/generator networks and rectangular
discretization use the same shared framework as 4D_gbm_veri.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from examples.planar_kepler.model import (
    AccelerationDiffusion, KeplerEqMLPControl, NominalClosedLoopDrift,
    PlanarKepler, find_goal_equilibrium, load_config, load_region_arrays, validate_config,
)
from examples.planar_kepler.neural_certified_nominal_drift.training_schedule import (
    SafetyFirstGeneratorSchedule, initialize_boundary_certificate,
)
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
from src.visualization import create_summary_plots

SAT_KEYS = ("goal_satisfied", "unsafe_satisfied", "init_satisfied",
            "outside_satisfied", "generator_satisfied")


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


def make_hyperparameters(config, seed, seed_dir,
                         outside_merge_margin=20.0, generator_merge_margin=-1000.0,
                         max_cells=200000):
    params = Hyperparameters.default()
    params.include_time = False
    params.include_energy = False
    params.network.n_inputs = 4
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    arrays = load_region_arrays(config)
    params.network.input_scale = ((arrays["full_range"][:, 1] - arrays["full_range"][:, 0]) / 2).tolist()
    params.network.scale_factor = 20.0
    params.training.learning_rate = 0.001
    params.training.num_epochs = 30000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0
    params.training.loss_reduction = 'mean'
    params.training.loss_weights = dict(goal=1.0, unsafe=5.0, init=1.0, outside=1.0)
    params.training.max_grad_norm = 1.0
    params.training.random_seed = seed
    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 15000
    params.training.pretrain_lr = 0.001
    params.training.pretrain_n_samples = 2048
    params.training.curriculum_mode = "none"
    params.training.resume_checkpoint_path = str(seed_dir / "outputs/resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(seed_dir / "outputs")
    params.training.progress_output_dir = str(seed_dir / "training_progress")
    params.discretization.axis_weights = [1.0] * 4
    params.discretization.max_region_budget = 256
    params.discretization.max_generator_budget = 512
    params.constraints.beta_ra = float(config["beta_ra"])
    params.compute_V = True
    params.compute_GV = True
    # Each refinement splits a 4D cell into 2^4 children.
    for name in ("v_goal", "v_init", "v_unsafe", "v_outside", "gv_generator"):
        cfg = getattr(params.refinement, name)
        cfg.enable_refinement = True
        cfg.refine_interval = 500
        cfg.late_epoch_threshold = 2500
        cfg.refine_interval_late = 100 if name == "gv_generator" else 250
        cfg.refine_factor = 2
        cfg.max_cells = max_cells
        cfg.N_to_refine = 25
        cfg.enable_merging = name in ("v_outside", "gv_generator")
        cfg.merge_interval = 501
        cfg.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = outside_merge_margin
    params.refinement.gv_generator.merge_relax_margin = generator_merge_margin
    params.refinement.refine_interval_after_first_sat = 500
    return params


def build_problem(config, params):
    validate_config(config)
    arrays = load_region_arrays(config)
    regions = Regions(
        init=Region(arrays["init_range"]), goal=Region(arrays["goal_range"]),
        unsafe=Region.union(*[Region(box) for box in arrays["unsafe_ranges"]]),
        full=Region(arrays["full_range"]),
    )
    x_eq, u_eq = find_goal_equilibrium(config)
    controller = KeplerEqMLPControl(config, x_eq, u_eq, params.network.input_scale).to(params.training.device)
    drift = NominalClosedLoopDrift(PlanarKepler(config), controller).to(params.training.device)
    diffusion = AccelerationDiffusion(config).to(params.training.device)
    dynamics = Dynamics.dynamics(f=drift, g=diffusion, state_dim=4)
    offset = x_eq.tolist()
    value = create_V(params.network, input_offset=offset, output_offset=np.float32(0.1)).to(params.training.device)
    initialize_boundary_certificate(value, arrays, config['regions']['unsafe_boundary_width'],
                                    params.constraints.beta_ra)
    generator = create_GV(
        V_net=value, dynamics=dynamics, network_config=params.network,
        input_offset=offset, include_time=False, include_energy=False, verify=False,
    ).to(params.training.device)
    return value, generator, controller, dynamics, regions, arrays


def check_generator(value, generator, dynamics, arrays):
    device = next(value.parameters()).device
    full = torch.as_tensor(arrays["full_range"], device=device)
    rng = torch.Generator(device=device).manual_seed(0)
    points = full[:, 0] + torch.rand(32, 4, generator=rng, device=device) * (full[:, 1] - full[:, 0])
    corners = torch.cartesian_prod(*[full[i] for i in range(4)])
    points = torch.cat([points, corners, generator.f.controller.x_eq[None]])
    if not verify_GV(generator, dynamics=dynamics, x=points, tol=1e-3):
        raise RuntimeError("Planar Kepler generator failed its autograd consistency check")


def evaluate_and_visualize(problem, cells, params, output_dir, results_dir,
                           loss_history, refinement_epochs, final_beta_s, no_plots=False):
    value, generator, controller, _, regions, _ = problem
    results = evaluate_constraints(value, generator, cells,
                                   beta_ra=params.constraints.beta_ra, device=params.training.device)
    print_constraint_summary(results)
    save_eval_bundle(output_dir, V_net=value, GV_net=generator, control_net=controller,
                     params=params, regions=regions, region_cells=cells, final_beta_s=final_beta_s,
                     loss_history=loss_history, refinement_epochs=refinement_epochs, results=results)
    satisfied = all(bool(results[key]) for key in SAT_KEYS)
    summary = dict(all_constraints_satisfied=satisfied, beta_ra=params.constraints.beta_ra,
                   state_order=["r", "theta", "r_dot", "theta_dot"],
                   interpretation="Final bound checks for the configured reach-avoid problem. "
                   "A saved checkpoint alone does not establish certification.",
                   constraints={key: bool(results[key]) for key in SAT_KEYS})
    (output_dir / "certification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not no_plots:
        create_summary_plots(V_net=value, GV_net=generator, regions=regions, region_cells=cells,
                             beta_s=final_beta_s, beta_ra=params.constraints.beta_ra,
                             loss_history=loss_history, refinement_epochs=refinement_epochs,
                             results=results, output_dir=str(results_dir))
    return results


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--config", type=Path, default=EXAMPLE_ROOT / "config.json")
    parser.add_argument("--epochs", type=positive_int, default=None)
    parser.add_argument("--pretrain-epochs", type=nonnegative_int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=positive_int, default=1, help="Torch CPU threads (default: 1)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the seed run directory")
    parser.add_argument("--evaluate-only", action="store_true", help="Reload the run's final bundle and evaluate")
    parser.add_argument("--generator-warmup", type=nonnegative_int, default=1000,
                        help="Minimum value-only bound epochs (default: 1000)")
    parser.add_argument("--generator-ramp", type=positive_int, default=3000,
                        help="Safe epochs needed to reach full generator weight (default: 3000)")
    parser.add_argument("--outside-merge-margin", type=float, default=20.0,
                        help="Minimum outside V lower bound for merging (default: 20.0)")
    parser.add_argument("--generator-merge-margin", type=float, default=-1000.0,
                        help="Maximum generator upper bound for merging; capped at -1e-4 (default: -1000.0)")
    parser.add_argument("--max-cells", type=positive_int, default=200000,
                        help="Refinement cell-count threshold per region, including generator (default: 200000)")
    args = parser.parse_args(argv)
    if args.seed >= 2 ** 32:
        parser.error("--seed must be less than 2**32")
    torch.set_num_threads(args.threads)
    configure_reproducibility(args.seed)
    seed_dir = args.output_dir or HERE / f"seed{args.seed}"
    output_dir, results_dir, progress_dir = [seed_dir / name for name in ("outputs", "results", "training_progress")]
    bundle = None
    if args.evaluate_only:
        config = json.loads((output_dir / "run_config.json").read_text())["example"]
        bundle = torch.load(output_dir / "eval_bundle.pth", map_location="cpu", weights_only=False)
        params = Hyperparameters.from_dict(bundle["hyperparameters"])
    else:
        if (output_dir / "eval_bundle.pth").exists():
            parser.error("This run already has a final bundle; choose another --output-dir or --evaluate-only")
        config = load_config(args.config)
        params = make_hyperparameters(config, args.seed, seed_dir,
                                      outside_merge_margin=args.outside_merge_margin,
                                      generator_merge_margin=args.generator_merge_margin,
                                      max_cells=args.max_cells)
    params.training.device = args.device
    schedule = SafetyFirstGeneratorSchedule(params.constraints.beta_ra,
                                            warmup_epochs=args.generator_warmup, ramp_epochs=args.generator_ramp)
    if args.epochs is not None:
        params.training.num_epochs = args.epochs
    if args.pretrain_epochs is not None:
        params.training.pretrain_epochs = args.pretrain_epochs
    if args.no_plots:
        params.logging.visualize_interval = 0
    for directory in (output_dir, results_dir, progress_dir):
        directory.mkdir(parents=True, exist_ok=True)
    enable_terminal_logging(output_dir / ("evaluation_log.txt" if bundle else "terminal_log.txt"))
    print("Planar Kepler nominal-drift controller synthesis; state = [r, theta, r_dot, theta_dot]")
    print(f"Seed: {args.seed}; device: {args.device}; run directory: {seed_dir.resolve()}")
    print("Controls: radial/tangential acceleration. Goal: fixed position at zero velocity.")
    if bundle is None:
        (output_dir / "run_config.json").write_text(json.dumps(
            dict(mode="nominal_drift", example=config, hyperparameters=params.to_dict(),
                 generator_curriculum=schedule.to_dict()), indent=2) + "\n")
        print("Bound losses are per-cell means; unsafe weight=5; gradient norm capped at 1.")
        print(f"Generator warmup: {args.generator_warmup} epochs; safety-gated ramp: {args.generator_ramp} epochs.")
        print(f"Merging margins: outside={params.refinement.v_outside.merge_relax_margin}; "
              f"generator={params.refinement.gv_generator.merge_relax_margin}.")
        print(f"Refinement max_cells per region: {args.max_cells}.")
    problem = build_problem(config, params)
    value, generator, controller, dynamics, regions, arrays = problem
    if bundle is not None:
        value.load_state_dict(bundle["V_state_dict"])
        controller.load_state_dict(bundle["control_state_dict"])
    print("Goal equilibrium:", controller.x_eq.detach().cpu().tolist())
    print("Equilibrium control:", controller.u_eq.detach().cpu().tolist())
    check_generator(value, generator, dynamics, arrays)
    if bundle is not None:
        cells = bundle["region_cells"]
        history, beta_s, refinements = bundle["loss_history"], bundle["final_beta_s"], bundle["refinement_epochs"]
    else:
        cells = discretize_regions(regions, params.discretization, use_radial_generator=False)
        started = time.time()
        if params.training.pretrain_epochs > 0:
            print("Certificate-only sample pretraining; controller updates begin with the bound-generator ramp.")
            pretrain_network_samples(
                model=value, GV_net=None, control_net=controller,
                x_goal_range=arrays["goal_range"], x_unsafe_range=arrays["unsafe_ranges"],
                x_init_range=arrays["init_range"], x_range=arrays["full_range"], params=params,
                num_epochs=params.training.pretrain_epochs, lr=params.training.pretrain_lr,
                device=params.training.device, n_each=params.training.pretrain_n_samples, lambda_w=1e-5,
                unsafe_sample_fraction=1 / len(arrays["unsafe_ranges"]),
                save_v_path=output_dir / "V_pretrained.pth",
                save_control_path=output_dir / "controller_pretrained.pth")
        else:
            torch.save(value.state_dict(), output_dir / "V_pretrained.pth")
            torch.save(controller.state_dict(), output_dir / "controller_pretrained.pth")
        print(f"Sample pretraining: {time.time() - started:.1f}s")
        started = time.time()
        def create_scheduler(optimizer):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
        history, beta_s, refinements = train_network_bounds(
            V_net=value, GV_net=generator, region_cells=cells, regions=regions, params=params,
            control_net=controller, create_scheduler=create_scheduler, start_time=started,
            generator_weight_schedule=schedule)
        print(f"Bound training: {time.time() - started:.1f}s")
    evaluate_and_visualize(problem, cells, params, output_dir, results_dir,
                           history, refinements, beta_s, no_plots=args.no_plots)


if __name__ == "__main__":
    main()
