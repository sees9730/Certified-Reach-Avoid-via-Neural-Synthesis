"""XV-15 nominal-drift baseline: neural certificate and controller synthesis.

Follows examples/inv_pend_adversarial/neural_certified_nominal_drift:
sample pretraining from scratch, adaptive bound training, evaluation and plots.
The SDE retains its Brownian diffusion; physical parameters are nominal.
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

from examples.xv15_uncertain.model import (
    DEG, DiagonalDiffusion, NominalClosedLoopDrift, XV15Aero,
    XV15EqMLPControl, find_goal_equilibrium, load_config, load_region_arrays,
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
from src.utils import cleanup_and_setup_directories
from src.visualization import create_summary_plots


def configure_reproducibility(seed: int) -> None:
    """Use deterministic Torch operations on a fixed hardware/software stack."""
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


def make_hyperparameters(config, seed, seed_dir):
    params = Hyperparameters.default()
    params.include_time = False
    params.include_energy = False
    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [100.0, 20.0 * DEG, 90.0 * DEG]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 30000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0
    params.training.random_seed = seed
    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 15000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1200
    params.training.curriculum_mode = "none"
    params.training.resume_checkpoint_path = str(seed_dir / "outputs" / "resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(seed_dir / "outputs")
    params.training.progress_output_dir = str(seed_dir / "training_progress")

    params.discretization.axis_weights = [1.0, 1.0, 1.0]
    params.discretization.max_region_budget = 1000
    params.discretization.max_generator_budget = 1000
    params.constraints.beta_ra = float(config["beta_ra"])
    params.compute_V = True
    params.compute_GV = True

    # Same refinement/merging schedule as the nominal pendulum baseline.
    for name in ("v_goal", "v_init", "v_unsafe", "v_outside", "gv_generator"):
        cfg = getattr(params.refinement, name)
        cfg.enable_refinement = True
        cfg.refine_interval = 500
        cfg.late_epoch_threshold = 3500 if name == "gv_generator" else 2500
        cfg.refine_interval_late = 100 if name == "v_outside" else 500
        cfg.refine_factor = 2
        cfg.max_cells = 100000 if name in ("v_goal", "gv_generator") else 50000
        cfg.N_to_refine = 100
        cfg.enable_merging = name in ("v_outside", "gv_generator")
        cfg.merge_interval = 501
        cfg.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 20.0
    params.refinement.gv_generator.merge_relax_margin = -1000.0
    params.refinement.refine_interval_after_first_sat = 500
    return params


def build_problem(config, params):
    arrays = load_region_arrays(config)
    regions = Regions(
        init=Region(arrays["init_range"]), goal=Region(arrays["goal_range"]),
        unsafe=Region.union(*[Region(box) for box in arrays["unsafe_ranges"]]),
        full=Region(arrays["full_range"]),
    )
    aero = XV15Aero(config)
    x_eq, u_eq = find_goal_equilibrium(config, aero)
    controller = XV15EqMLPControl(config, x_eq, u_eq, params.network.input_scale)
    drift = NominalClosedLoopDrift(aero, controller).to(params.training.device)
    diffusion = DiagonalDiffusion(config).to(params.training.device)
    dynamics = Dynamics.dynamics(f=drift, g=diffusion, state_dim=3)
    input_offset = x_eq.tolist()
    value = create_V(params.network, input_offset=input_offset, output_offset=np.float32(0.1))
    value = value.to(params.training.device)
    generator = create_GV(
        V_net=value, dynamics=dynamics, network_config=params.network,
        input_offset=input_offset, include_time=False, include_energy=False,
        # The generic check samples negative velocities; use in-domain points below.
        verify=False,
    ).to(params.training.device)
    return value, generator, controller, dynamics, regions, arrays


def check_generator(value, generator, dynamics, arrays):
    """Compare analytic and autograd generators at valid aircraft states."""
    device = next(value.parameters()).device
    full = torch.as_tensor(arrays["full_range"], device=device)
    rng = torch.Generator(device=device).manual_seed(0)
    points = full[:, 0] + torch.rand(32, 3, generator=rng, device=device) * (full[:, 1] - full[:, 0])
    corners = torch.cartesian_prod(*[full[i] for i in range(3)])
    points = torch.cat([points, corners, generator.f.controller.x_eq[None]])
    if not verify_GV(generator, dynamics=dynamics, x=points, tol=1e-3):
        raise RuntimeError("XV-15 generator failed its autograd consistency check")


def evaluate_and_visualize(value, generator, controller, regions, cells, params,
                           output_dir, results_dir, loss_history, refinement_epochs,
                           final_beta_s, no_plots=False):
    print("\n" + "=" * 20 + "\nFinal Evaluation\n" + "=" * 20)
    results = evaluate_constraints(
        value, generator, cells, beta_ra=params.constraints.beta_ra,
        device=params.training.device,
    )
    print_constraint_summary(results)
    # Save even when the epoch budget ends without SAT, and before plotting.
    save_eval_bundle(
        output_dir, V_net=value, GV_net=generator, control_net=controller,
        params=params, regions=regions, region_cells=cells,
        final_beta_s=final_beta_s, loss_history=loss_history,
        refinement_epochs=refinement_epochs, results=results,
    )
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
    parser.add_argument("--epochs", type=positive_int, default=None, help="Bound-training epoch budget (default: 30000)")
    parser.add_argument("--pretrain-epochs", type=positive_int, default=None, help="Sample pretraining epochs (default: 15000)")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--no-plots", action="store_true", help="Disable progress and final plots")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32-1")
    configure_reproducibility(args.seed)

    seed_dir = HERE / f"seed{args.seed}"
    output_dir, results_dir, progress_dir = [seed_dir / name for name in ("outputs", "results", "training_progress")]
    cleanup_and_setup_directories([output_dir, results_dir, progress_dir])
    enable_terminal_logging(output_dir / "terminal_log.txt", append=False)
    print("=" * 20 + "\nXV-15 Nominal-Drift Baseline\n" + "=" * 20)
    print(f"Random seed: {args.seed}; device: {args.device}")
    print("Nominal physical parameters; Brownian diffusion enabled; no time/energy augmentation or curriculum.")
    print(f"Configuration: {args.config.resolve()}")
    print(f"Outputs: {output_dir}")

    config = load_config(args.config)
    params = make_hyperparameters(config, args.seed, seed_dir)
    params.training.device = args.device
    if args.epochs is not None:
        params.training.num_epochs = args.epochs
    if args.pretrain_epochs is not None:
        params.training.pretrain_epochs = args.pretrain_epochs
    if args.no_plots:
        params.logging.visualize_interval = 0
    print(json.dumps(params.to_dict(), indent=2))
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(mode="nominal_drift", example=config, hyperparameters=params.to_dict()), stream, indent=2)

    value, generator, controller, dynamics, regions, arrays = build_problem(config, params)
    print("x_eq [m/s, deg, deg]:", (controller.x_eq.detach().cpu() / torch.tensor([1.0, DEG, DEG])).tolist())
    print("u_eq [N, deg, deg/s]:", (controller.u_eq.detach().cpu() / torch.tensor([1.0, DEG, DEG])).tolist())
    print("Nominal drift at trim:", dynamics.f(controller.x_eq[None]).detach().cpu().tolist())
    print("Diffusion [m/s, rad, rad] / sqrt(s):", dynamics.g.sigma.detach().cpu().tolist())
    check_generator(value, generator, dynamics, arrays)
    cells = discretize_regions(regions, params.discretization, use_radial_generator=False)

    started = time.time()
    pretrain_network_samples(
        model=value, GV_net=generator, control_net=controller,
        x_goal_range=arrays["goal_range"], x_unsafe_range=arrays["unsafe_ranges"],
        x_init_range=arrays["init_range"], x_range=arrays["full_range"],
        params=params, num_epochs=params.training.pretrain_epochs,
        lr=params.training.pretrain_lr, device=params.training.device,
        n_each=params.training.pretrain_n_samples, lambda_w=1e-3,
        unsafe_sample_fraction=1.0 / len(arrays["unsafe_ranges"]),
        save_v_path=output_dir / "V_pretrained.pth",
        save_control_path=output_dir / "controller_pretrained.pth",
    )
    print(f"Pre-training time: {time.time() - started:.1f}s")
    started = time.time()

    def create_scheduler(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

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
