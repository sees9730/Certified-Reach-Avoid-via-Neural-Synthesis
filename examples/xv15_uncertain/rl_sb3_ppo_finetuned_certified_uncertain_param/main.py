"""Certify and fine-tune a pretrained XV-15 RL controller under density/mass uncertainty.

Loads rl_sb3_ppo/seed0/outputs/rl_controller.pth (the distilled physical
XV15EqMLPControl, not the SB3 teacher zip). A fresh certificate V is trained
against the fixed RL controller during sample pretraining. Bound training
then jointly trains V and, by default, only the controller's last layer.

The problem builder, robust generator, hyperparameters, discretization,
adaptive bound trainer, and final evaluation are reused from
neural_certified_uncertain_param/main.py. beta_ra is fixed at 5.0; there is
no time or energy augmentation. Brownian diffusion remains enabled.

Run from the repository root:
    ./.venv/bin/python -u examples/xv15_uncertain/rl_sb3_ppo_finetuned_certified_uncertain_param/main.py

Optional: --controller-training all (both layers) or frozen (V only),
--rl-seed N, --controller-checkpoint PATH, --seed N, --pretrain-epochs N,
--epochs N, --learning-rate RATE, --pretrain-lr RATE, --device cuda, --no-plots.
Use --pretrain-epochs 0 to skip sample pretraining.

Each run recreates this experiment's seed<seed>/{outputs,results,training_progress}
directories, following the certified baseline. The source RL checkpoint is
read only. Outputs include run_config.json, controller_initial.pth,
V_pretrained.pth, controller_pretrained.pth, eval_bundle.pth, terminal_log.txt,
and certification_summary.json. A final bundle is saved even without SAT;
certification_summary.json reports whether all final bound checks passed.
The resulting certificate applies to the final fine-tuned controller.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from examples.xv15_uncertain.model import DEG, load_config
from examples.xv15_uncertain.neural_certified_uncertain_param.main import (
    build_problem as build_uncertain_problem,
    check_generator, configure_reproducibility, evaluate_and_visualize,
    make_hyperparameters, positive_int,
)
from examples.xv15_uncertain.uncertain_density import load_density_interval
from examples.xv15_uncertain.uncertain_parameters import load_mass_interval
from src.discretization import discretize_regions
from src.pretrainer import pretrain_network_samples
from src.save_load_utils import enable_terminal_logging
from src.trainer import train_network_bounds
from src.utils import cleanup_and_setup_directories

TRAIN_SEED = 0
BETA_RA = 5.0
SAT_KEYS = ("goal_satisfied", "unsafe_satisfied", "init_satisfied",
            "outside_satisfied", "generator_satisfied")


def controller_state_cpu(controller):
    return {name: tensor.detach().cpu().clone()
            for name, tensor in controller.state_dict().items()}


def set_controller_training(controller, mode):
    """Freeze all buffers/hidden weights; optionally unfreeze selected weights."""
    if mode not in ("last-layer", "all", "frozen"):
        raise ValueError(f"Unknown controller training mode: {mode}")
    controller.requires_grad_(False)
    for parameter in controller.parameters():
        parameter.grad = None
    if mode == "last-layer":
        controller.fc2.requires_grad_(True)
    elif mode == "all":
        controller.requires_grad_(True)
    return [name for name, parameter in controller.named_parameters() if parameter.requires_grad]


def load_rl_weights(controller, checkpoint):
    """Strictly load the distilled controller, retaining its trim and scaling.

    Compare fixed buffers with the chosen certification problem before loading:
    incompatible limits, normalization or trim must not silently change it.
    Accept the PPO export payload or a raw XV15EqMLPControl state_dict.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".zip":
        raise ValueError("Use the distilled rl_controller.pth, not the SB3 teacher zip")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected an XV15EqMLPControl checkpoint dictionary")
    state = payload.get("control_state_dict", payload)
    expected = controller.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint is not a complete XV15EqMLPControl state_dict")
    learned = set(dict(controller.named_parameters()))
    for name, target in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape:
            raise ValueError(f"Controller shape mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite controller checkpoint tensor: {name}")
        if name not in learned and not torch.allclose(
                tensor.to(dtype=target.dtype), target.detach().cpu(), rtol=1e-6, atol=1e-7):
            raise ValueError(f"Checkpoint {name} is incompatible with the certification config")
    controller.load_state_dict(state, strict=True)
    return controller_state_cpu(controller)


def build_problem(config, params, checkpoint, controller_training="last-layer"):
    if float(config["beta_ra"]) != BETA_RA or float(params.constraints.beta_ra) != BETA_RA:
        raise ValueError("This experiment requires beta_ra = 5.0")
    if params.include_time or params.include_energy:
        raise ValueError("This experiment uses only the three physical state inputs")
    problem = build_uncertain_problem(config, params)
    _, generator, controller, dynamics, _, _ = problem
    load_rl_weights(controller, checkpoint)
    set_controller_training(controller, controller_training)
    # Loading in place updates the controller used by BOTH dynamics and GV.
    if dynamics.f.controller is not controller or generator.f.controller is not controller:
        raise RuntimeError("The robust generator must share the loaded RL controller")
    return problem


def assert_fixed_state(controller, initial, trainable_names=()):
    """Ensure frozen weights and all physical buffers are preserved exactly."""
    for name, tensor in controller.state_dict().items():
        if name not in trainable_names and not torch.equal(tensor.detach().cpu(), initial[name]):
            raise RuntimeError(f"Frozen controller tensor changed: {name}")


def make_bound_scheduler(controller, trainable_names):
    """Restrict optimizer membership before the shared trainer's first update.

    auto_LiRPA's BoundParams.forward may re-enable requires_grad on frozen
    parameters. Excluding them from Adam is therefore necessary even after
    setting requires_grad=False, and remains effective across cache rebuilds.
    The shared trainer calls this factory after creating its Adam optimizer.
    """
    selected = set(trainable_names)

    def create_scheduler(optimizer):
        excluded = {id(parameter) for name, parameter in controller.named_parameters()
                    if name not in selected}
        for group in optimizer.param_groups:
            group["params"] = [parameter for parameter in group["params"]
                               if id(parameter) not in excluded]
        # Clear any gradients produced during setup. Later bound evaluations
        # can still compute unused gradients, but Adam cannot update these weights.
        controller.zero_grad(set_to_none=True)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

    return create_scheduler


def pretrain_certificate(value, generator, controller, arrays, params, output_dir):
    """Warm-start V against the unchanged RL policy, with the robust GV loss."""
    flags = [parameter.requires_grad for parameter in controller.parameters()]
    initial = controller_state_cpu(controller)
    controller.requires_grad_(False)
    try:
        if params.training.pretrain_epochs > 0:
            pretrain_network_samples(
                model=value, GV_net=generator, control_net=None,
                x_goal_range=arrays["goal_range"], x_unsafe_range=arrays["unsafe_ranges"],
                x_init_range=arrays["init_range"], x_range=arrays["full_range"],
                params=params, num_epochs=params.training.pretrain_epochs,
                lr=params.training.pretrain_lr, device=params.training.device,
                n_each=params.training.pretrain_n_samples, lambda_w=1e-3,
                unsafe_sample_fraction=1.0 / len(arrays["unsafe_ranges"]),
                save_v_path=output_dir / "V_pretrained.pth",
            )
        else:
            print("Sample pretraining skipped.")
        assert_fixed_state(controller, initial)
        torch.save(controller_state_cpu(controller), output_dir / "controller_pretrained.pth")
    finally:
        for parameter, flag in zip(controller.parameters(), flags):
            parameter.requires_grad_(flag)
            parameter.grad = None


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=nonnegative_int, default=TRAIN_SEED, help="Certificate initialization seed")
    parser.add_argument("--rl-seed", type=nonnegative_int, default=0, help="Source PPO seed (default: 0)")
    parser.add_argument("--controller-checkpoint", type=Path, default=None,
                        help="Default: rl_sb3_ppo/seed<rl-seed>/outputs/rl_controller.pth")
    parser.add_argument("--config", type=Path, default=EXAMPLE_ROOT / "config.json")
    parser.add_argument("--controller-training", choices=("last-layer", "all", "frozen"), default="last-layer",
                        help="Controller weights allowed to change in bound training (default: last-layer)")
    parser.add_argument("--epochs", type=positive_int, default=None, help="Bound-training epochs (default: 30000)")
    parser.add_argument("--pretrain-epochs", type=nonnegative_int, default=None,
                        help="V-only sample pretraining epochs (default: 15000; 0 skips)")
    parser.add_argument("--learning-rate", type=positive_float, default=None,
                        help="Joint bound-training learning rate (default: 0.005)")
    parser.add_argument("--pretrain-lr", type=positive_float, default=None,
                        help="V-only sample pretraining learning rate (default: 0.01)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.seed >= 2 ** 32 or args.rl_seed >= 2 ** 32:
        parser.error("seeds must be less than 2**32")
    checkpoint = args.controller_checkpoint or EXAMPLE_ROOT / "rl_sb3_ppo" / f"seed{args.rl_seed}" / "outputs" / "rl_controller.pth"
    args.controller_checkpoint = checkpoint.expanduser().resolve()
    if not args.controller_checkpoint.is_file():
        parser.error(f"RL controller checkpoint not found: {args.controller_checkpoint}")
    # This experiment clears its own output directories before a fresh run.
    # Never allow that operation to remove the supplied source checkpoint.
    if args.controller_checkpoint.is_relative_to((HERE / f"seed{args.seed}").resolve()):
        parser.error("The source controller must be outside this run's output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if float(config["beta_ra"]) != BETA_RA:
        raise ValueError("This experiment requires config beta_ra = 5.0")
    density_interval = load_density_interval(config)
    mass_interval = load_mass_interval(config)
    configure_reproducibility(args.seed)
    seed_dir = HERE / f"seed{args.seed}"
    output_dir, results_dir, progress_dir = [seed_dir / name for name in ("outputs", "results", "training_progress")]
    params = make_hyperparameters(config, args.seed, seed_dir)
    params.training.device = args.device
    for argument, field in (("epochs", "num_epochs"), ("pretrain_epochs", "pretrain_epochs"),
                            ("learning_rate", "learning_rate"), ("pretrain_lr", "pretrain_lr")):
        if getattr(args, argument) is not None:
            setattr(params.training, field, getattr(args, argument))
    params.training.enable_pretraining = params.training.pretrain_epochs > 0
    if args.no_plots:
        params.logging.visualize_interval = 0

    # Validate/load the checkpoint before clearing any previous certification run.
    value, generator, controller, dynamics, regions, arrays = build_problem(
        config, params, args.controller_checkpoint, args.controller_training,
    )
    initial = controller_state_cpu(controller)
    trainable_names = [name for name, parameter in controller.named_parameters() if parameter.requires_grad]
    source = dict(checkpoint=str(args.controller_checkpoint),
                  sha256=hashlib.sha256(args.controller_checkpoint.read_bytes()).hexdigest())
    cleanup_and_setup_directories([output_dir, results_dir, progress_dir])
    stdout, stderr = sys.stdout, sys.stderr
    log_handle = enable_terminal_logging(output_dir / "terminal_log.txt", append=False)
    try:
        print("XV-15: certify and fine-tune the pretrained RL controller")
        print(f"Source: {args.controller_checkpoint}")
        print(f"Seed: {args.seed}; device: {args.device}; beta_ra: {BETA_RA}")
        print(f"Density interval: {density_interval}; mass interval: {mass_interval}")
        print("Robust density/mass generator with Brownian diffusion; physical state only.")
        print("Sample pretraining: V only, with the loaded RL controller fixed.")
        print(f"Bound training: V plus controller weights {trainable_names or '(none)'}")
        print(f"Outputs: {output_dir}")
        print(json.dumps(params.to_dict(), indent=2))
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
            json.dump(dict(mode="rl_finetuned_uncertain_air_density_and_mass", example=config,
                           hyperparameters=params.to_dict(), source_controller=source,
                           controller_training=args.controller_training,
                           trainable_controller_parameters=trainable_names,
                           pretrain_controller=False), stream, indent=2)
        torch.save(initial, output_dir / "controller_initial.pth")
        angle_scale = torch.tensor([1.0, DEG, DEG])
        print("Nominal x_eq [m/s, deg, deg]:", (controller.x_eq.detach().cpu() / angle_scale).tolist())
        print("Nominal u_eq [N, deg, deg/s]:", (controller.u_eq.detach().cpu() / angle_scale).tolist())
        check_generator(value, generator, dynamics, arrays)
        cells = discretize_regions(regions, params.discretization, use_radial_generator=False)

        started = time.time()
        pretrain_certificate(value, generator, controller, arrays, params, output_dir)
        pretrain_seconds = time.time() - started
        print(f"V-only robust pretraining time: {pretrain_seconds:.1f}s")
        assert_fixed_state(controller, initial)
        started = time.time()

        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=value, GV_net=generator, region_cells=cells, regions=regions,
            params=params, control_net=controller,
            create_scheduler=make_bound_scheduler(controller, trainable_names),
            start_time=started,
        )
        bound_seconds = time.time() - started
        print(f"Robust bound-training time: {bound_seconds:.1f}s")
        assert_fixed_state(controller, initial, trainable_names)
        # Recompute all bounds on the FINAL controller/certificate pair. The
        # shared helper saves an eval_bundle whether or not all checks pass.
        results = evaluate_and_visualize(
            value, generator, controller, regions, cells, params, output_dir, results_dir,
            loss_history, refinement_epochs, final_beta_s, no_plots=args.no_plots,
        )
        set_controller_training(controller, args.controller_training)
        assert_fixed_state(controller, initial, trainable_names)
        certified = all(bool(results[name]) for name in SAT_KEYS)
        changes = {name: float((tensor.detach().cpu() - initial[name]).norm())
                   for name, tensor in controller.named_parameters()}
        with (output_dir / "certification_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(dict(certified=certified, beta_ra=BETA_RA, source_controller=source,
                           controller_training=args.controller_training,
                           trainable_controller_parameters=trainable_names,
                           controller_parameter_change_l2=changes,
                           pretrain_time_sec=pretrain_seconds, bound_training_time_sec=bound_seconds,
                           final_results=results, checkpoint="eval_bundle.pth"), stream, indent=2)
        if certified:
            print("SAT: all final robust reach-avoid bound checks passed for the saved controller.")
        else:
            print("NOT CERTIFIED: final robust bound checks did not all pass; the final bundle was saved.")
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        log_handle.close()


if __name__ == "__main__":
    main()
