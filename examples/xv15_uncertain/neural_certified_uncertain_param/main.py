"""Neural certificate/controller synthesis for XV-15 with uncertain density.

Use the nominal XV-15 settings and the support-function robust generator
interface from inv_pend_adversarial/neural_certified. Both sample pretraining
and bound training enforce the worst-case generator over the density interval.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

TRAIN_SEED = 0
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from examples.xv15_uncertain.model import DEG, load_config
from examples.xv15_uncertain.neural_certified_nominal_drift.main import (
    build_problem as build_nominal_problem,
    check_generator, configure_reproducibility, evaluate_and_visualize,
    make_hyperparameters, positive_int,
)
from examples.xv15_uncertain.uncertain_density import DensityIntervalDrift, load_density_interval
from src.discretization import discretize_regions
from src.dynamics import Dynamics
from src.phi_module import create_GV
from src.pretrainer import pretrain_network_samples
from src.save_load_utils import enable_terminal_logging
from src.trainer import train_network_bounds
from src.utils import cleanup_and_setup_directories


def build_problem(config, params):
    density_interval = load_density_interval(config)
    value, _, controller, nominal, regions, arrays = build_nominal_problem(config, params)
    drift = DensityIntervalDrift(nominal.f.aero, controller, density_interval).to(params.training.device)
    dynamics = Dynamics.dynamics(f=drift, g=nominal.g, state_dim=3)
    # create_GV detects drift.support(x, grad V), just as for the robust
    # pendulum. The diffusion contribution remains the full nominal Itô term.
    generator = create_GV(
        V_net=value, dynamics=dynamics, network_config=params.network,
        input_offset=value.input_offset, include_time=False, include_energy=False,
        verify=False,
    ).to(params.training.device)
    return value, generator, controller, dynamics, regions, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--config", type=Path, default=EXAMPLE_ROOT / "config.json")
    parser.add_argument("--density-min", type=float, default=None, help="Air-density lower endpoint (kg/m^3)")
    parser.add_argument("--density-max", type=float, default=None, help="Air-density upper endpoint (kg/m^3)")
    parser.add_argument("--epochs", type=positive_int, default=None, help="Bound-training epoch budget (default: nominal settings)")
    parser.add_argument("--pretrain-epochs", type=positive_int, default=None, help="Sample pretraining epochs (default: nominal settings)")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--no-plots", action="store_true", help="Disable progress and final plots")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32-1")
    config = load_config(args.config)
    interval = list(load_density_interval(config))
    if args.density_min is not None:
        interval[0] = args.density_min
    if args.density_max is not None:
        interval[1] = args.density_max
    config["uncertainty"]["air_density_kg_m3"] = interval
    try:
        lower, upper = load_density_interval(config)
    except ValueError as exc:
        parser.error(str(exc))

    configure_reproducibility(args.seed)
    seed_dir = HERE / f"seed{args.seed}"
    output_dir, results_dir, progress_dir = [seed_dir / name for name in ("outputs", "results", "training_progress")]
    cleanup_and_setup_directories([output_dir, results_dir, progress_dir])
    enable_terminal_logging(output_dir / "terminal_log.txt", append=False)
    print("=" * 20 + "\nXV-15 Uncertain-Air-Density Synthesis\n" + "=" * 20)
    print(f"Random seed: {args.seed}; device: {args.device}")
    print(f"Air density: [{lower}, {upper}] kg/m^3; nominal: {config['dynamics']['density']}")
    print("Robust generator: max over one shared density parameter, with Brownian diffusion enabled.")
    print("Certificate/controller inputs: [v, gamma, beta]; density is unobserved.")
    print("Training from scratch; no time/energy augmentation or curriculum.")
    print(f"Configuration: {args.config.resolve()}")
    print(f"Outputs: {output_dir}")

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
        json.dump(dict(mode="uncertain_air_density", example=config,
                       hyperparameters=params.to_dict()), stream, indent=2)

    value, generator, controller, dynamics, regions, arrays = build_problem(config, params)
    angle_scale = torch.tensor([1.0, DEG, DEG])
    print("Nominal x_eq [m/s, deg, deg]:", (controller.x_eq.detach().cpu() / angle_scale).tolist())
    print("Nominal u_eq [N, deg, deg/s]:", (controller.u_eq.detach().cpu() / angle_scale).tolist())
    print("The trim anchor is nominal; the robust generator checks every density outside the goal.")
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
    print(f"Robust pre-training time: {time.time() - started:.1f}s")
    started = time.time()

    def create_scheduler(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

    loss_history, final_beta_s, refinement_epochs = train_network_bounds(
        V_net=value, GV_net=generator, region_cells=cells, regions=regions,
        params=params, control_net=controller, create_scheduler=create_scheduler,
        start_time=started,
    )
    print(f"Robust bound-training time: {time.time() - started:.1f}s")
    # All generator evaluations and plot values use the robust generator.
    # The final bundle is saved even when the epoch budget ends without SAT.
    evaluate_and_visualize(
        value, generator, controller, regions, cells, params, output_dir, results_dir,
        loss_history, refinement_epochs, final_beta_s, no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
