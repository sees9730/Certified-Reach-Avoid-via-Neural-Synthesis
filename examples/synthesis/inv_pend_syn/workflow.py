"""Workflow helpers for 2D inverted-pendulum synthesis."""

import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.control_network import InvertControlNN, WrapperConterlNN
from src.discretization import discretize_regions
from src.dynamics import ClosedLoopDrift, Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.pretrainer import pretrain_network_samples
from src.regions import Region, Regions
from src.save_load_utils import (
    enable_terminal_logging,
    load_eval_bundle,
    log_loaded_training_epochs,
    save_eval_bundle,
)
from src.trainer import train_network_bounds
from src.training_utils import evaluate_constraints, print_constraint_summary
from src.utils import cleanup_and_setup_directories
from src.visualization import create_summary_plots


def set_global_reproducibility(seed: int) -> None:
    """Configure deterministic behavior for initialization, pretraining, and bound training."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_controller(device: str) -> torch.nn.Module:
    """Create state-feedback controller for synthesis on state [x1, x2]."""
    return WrapperConterlNN(InvertControlNN(hidden_dim=64)).to(device)


def make_invpend_dynamics(control_net: torch.nn.Module) -> Dynamics:
    """Build closed-loop inverted-pendulum dynamics over state [x1, x2]."""

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        g_grav = 9.81
        length = 0.5
        damping = 0.1
        mass = 0.15
        x1, x2 = x[:, 0], x[:, 1]
        f1 = x2
        f2 = (g_grav / length) * torch.sin(x1) - (damping / (mass * length ** 2)) * x2
        return torch.stack([f1, f2], dim=1)

    g_coeffs = torch.tensor([0.0, 2.0], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        base = g_coeffs.to(device=x.device, dtype=x.dtype)
        if x.dim() == 1:
            return base
        if x.dim() == 2:
            return base.unsqueeze(0).expand(x.shape[0], -1)
        raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")

    return Dynamics.dynamics(f=ClosedLoopDrift(f_ol, control_net), g=g)


def configure_default_params() -> Hyperparameters:
    """Build default hyperparameters with two-stage training and simple refinement."""
    params = Hyperparameters.default()

    pi = np.pi
    params.network.n_inputs = 2
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [2 * pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    # Two-stage setup: GV active from start, no generator-threshold curriculum.
    params.training.generator_after_v_sat = False
    params.training.use_generator_threshold_curriculum = False

    params.logging.visualize_interval = 0

    params.discretization.n_goal = 20
    params.discretization.n_outside_goal = 10
    params.discretization.n_generator = 2
    params.discretization.n_unsafe = 20
    params.discretization.n_init = 20

    params.constraints.beta_ra = 20.0
    # Stage-1 with beta_ra_k=1.0, then increase to final beta_ra.
    params.constraints.use_beta_ra_curriculum = True
    params.constraints.beta_ra_init = 1.0
    params.constraints.beta_ra_step = 0.5
    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 100
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1000

    # Last-layer LP solving (direct feasibility on active constraints):
    # - use_last_layer_lp: enable LP updates for V's final layer (w, c).
    # - last_layer_lp_interval: run LP every N epochs (1 = every epoch).
    # - LP includes only currently active constraints (stage-aware), and if feasible
    #   applies the last-layer solution directly (no gradient step that epoch).
    # - Generator LP constraint uses current generator threshold (typically 0.0).
    # - last_layer_lp_max_cells: skip LP if total selected cells exceed this cap (runtime guard).
    # - last_layer_lp_timeout_sec: per-LP solver time limit (seconds).
    # - last_layer_lp_verbose: print LP/SAT solver diagnostics.
    params.training.use_last_layer_lp = True
    params.training.last_layer_lp_interval = 500
    params.training.last_layer_lp_max_cells = 9999999
    params.training.last_layer_lp_timeout_sec = 30.0
    params.training.last_layer_lp_verbose = False
    params.training.verify_lp_apply_sat = False

    params.refinement.v_goal.refine_factor = 2
    params.refinement.v_goal.max_cells = 100000
    params.refinement.v_goal.enable_merging = False

    params.refinement.v_unsafe.refine_factor = 2
    params.refinement.v_unsafe.max_cells = 100000
    params.refinement.v_unsafe.enable_merging = False

    params.refinement.v_init.refine_factor = 2
    params.refinement.v_init.max_cells = 100000
    params.refinement.v_init.enable_merging = False

    params.refinement.v_outside.refine_factor = 2
    params.refinement.v_outside.max_cells = 100000
    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 501
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 20.0
    params.refinement.gv_generator.refine_factor = 2
    params.refinement.gv_generator.max_cells = 50000
    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -500.0

    params.refinement.use_simple_refinement = True
    params.refinement.simple_refine_every = 200
    params.refinement.simple_loss_improve_tol = 0.01
    params.refinement.simple_refine_budget = 100
    params.refinement.simple_regions_per_trigger = 5
    params.refinement.stage2_force_refine_interval = 1000

    return params


def build_regions() -> Tuple[Regions, np.ndarray, np.ndarray, np.ndarray]:
    """Create init/goal/unsafe/full regions and return box arrays for pretraining."""
    pi = np.pi
    init_range = np.array([[(3 / 4) * pi, (5 / 4) * pi], [-1.0, 1.0]], dtype=np.float32)
    goal_range = np.array([[-0.4 * pi, 0.4 * pi], [-4.0, 4.0]], dtype=np.float32)
    full_range = np.array([[-2 * pi, 2 * pi], [-20.0, 20.0]], dtype=np.float32)

    unsafe_down1 = Region(np.array([[-2 * pi, -1.5 * pi], [-20.0, -10.0]], dtype=np.float32))
    unsafe_down2 = Region(np.array([[1.5 * pi, 2 * pi], [10.0, 20.0]], dtype=np.float32))
    unsafe_lb = Region(np.array([[-2 * pi, -2 * pi + 0.5], [-20.0, 20.0]], dtype=np.float32))
    unsafe_rb = Region(np.array([[2 * pi - 0.5, 2 * pi], [-20.0, 20.0]], dtype=np.float32))
    unsafe_tb = Region(np.array([[-2 * pi, 2 * pi], [19.5, 20.0]], dtype=np.float32))
    unsafe_bb = Region(np.array([[-2 * pi, 2 * pi], [-20.0, -19.5]], dtype=np.float32))

    regions = Regions(
        init=Region(init_range),
        goal=Region(goal_range),
        unsafe=Region.union(unsafe_down1, unsafe_down2, unsafe_tb, unsafe_bb, unsafe_lb, unsafe_rb),
        full=Region(full_range),
    )
    return regions, init_range, goal_range, full_range


def create_gv_net(V_net: torch.nn.Module, dynamics: Dynamics, params: Hyperparameters) -> torch.nn.Module:
    """Create generator network for synthesis."""
    return create_GV(V_net=V_net, dynamics=dynamics, network_config=params.network, verify=True)


def create_v_net_for_training(params: Hyperparameters, goal_range: np.ndarray, use_v_offset: bool = True) -> torch.nn.Module:
    """Create V network for training path."""
    if not use_v_offset:
        return create_V(params.network)
    input_offset = np.array([
        0.5 * (goal_range[0, 0] + goal_range[0, 1]),
        0.5 * (goal_range[1, 0] + goal_range[1, 1]),
    ], dtype=np.float32)
    output_offset = np.array([0.1], dtype=np.float32)
    return create_V(params.network, input_offset=input_offset, output_offset=output_offset)


def restore_v_net_from_state(params: Hyperparameters, v_state_dict: dict, device: str) -> torch.nn.Module:
    """Restore V network (offset/non-offset) from state dict."""
    has_offset_buffers = "input_offset" in v_state_dict and "output_offset" in v_state_dict
    if has_offset_buffers:
        input_offset = v_state_dict["input_offset"].detach().cpu().numpy()
        output_offset = v_state_dict["output_offset"].detach().cpu().numpy()
        v_net = create_V(params.network, input_offset=input_offset, output_offset=output_offset).to(device)
    else:
        v_net = create_V(params.network).to(device)
    v_net.load_state_dict(v_state_dict)
    return v_net


def final_evaluation(
    V_net: torch.nn.Module,
    GV_net: torch.nn.Module,
    region_cells: dict,
    params: Hyperparameters,
    device: str,
    title: str,
    cached_results: Optional[dict] = None,
) -> dict:
    """Evaluate constraints and print a summary."""
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)
    if cached_results is not None:
        results = cached_results
    else:
        results = evaluate_constraints(
            V_net,
            GV_net,
            region_cells,
            beta_ra=params.constraints.beta_ra,
            device=device,
        )
    print_constraint_summary(results)
    return results


def visualize_results(
    V_net: torch.nn.Module,
    GV_net: torch.nn.Module,
    regions: Regions,
    region_cells: dict,
    params: Hyperparameters,
    loss_history: list,
    refinement_epochs: dict,
    results: dict,
    final_beta_s: Optional[float],
    loaded: bool = False,
) -> None:
    """Render plots for final/loaded results."""
    print("\n" + "=" * 20)
    print("Visualizations (loaded)" if loaded else "Visualizations")
    print("=" * 20)
    if loaded:
        log_loaded_training_epochs(loss_history)
    beta_s_plot = final_beta_s if final_beta_s is not None else float(results.get("V_outside_min", 0.0))
    create_summary_plots(
        V_net=V_net,
        GV_net=GV_net,
        regions=regions,
        region_cells=region_cells,
        beta_s=beta_s_plot,
        beta_ra=params.constraints.beta_ra,
        loss_history=loss_history,
        refinement_epochs=refinement_epochs,
        results=results,
        output_dir="results",
    )


def run_training_path(
    output_dir: Path,
    params: Hyperparameters,
    regions: Regions,
    init_range: np.ndarray,
    goal_range: np.ndarray,
    full_range: np.ndarray,
    V_net: torch.nn.Module,
    GV_net: torch.nn.Module,
    control_net: torch.nn.Module,
    device: str,
) -> float:
    """Run pretraining + bound-training + save bundle."""
    cleanup_and_setup_directories(["results", "training_progress"])
    enable_terminal_logging(output_dir / "terminal_log.txt")

    training_start_time = time.time()

    if params.training.enable_pretraining:
        unsafe_pretrain_range = np.stack([comp.bounds for comp in regions.unsafe.components], axis=0)
        pretrain_network_samples(
            model=V_net,
            x_goal_range=goal_range,
            x_unsafe_range=unsafe_pretrain_range,
            x_init_range=init_range,
            x_range=full_range,
            params=params,
            GV_net=GV_net,
            num_epochs=params.training.pretrain_epochs,
            lr=params.training.pretrain_lr,
            device=params.training.device,
            control_net=control_net,
            n_each=params.training.pretrain_n_samples,
            lambda_w=1e-3,
            unsafe_sample_fraction=1.0 / 6.0,
            save_v_path=output_dir / "V_pretrained.pth",
            save_control_path=output_dir / "controller_pretrained.pth",
        )

    def create_scheduler(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

    region_cells = discretize_regions(regions, params.discretization, use_radial_generator=False)
    loss_history, final_beta_s, refinement_epochs = train_network_bounds(
        V_net=V_net,
        GV_net=GV_net,
        region_cells=region_cells,
        regions=regions,
        params=params,
        control_net=control_net,
        create_scheduler=create_scheduler,
        start_time=training_start_time,
    )

    total_training_time = time.time() - training_start_time

    results = final_evaluation(
        V_net=V_net,
        GV_net=GV_net,
        region_cells=region_cells,
        params=params,
        device=device,
        title="Final Evaluation",
    )
    beta_s_plot = final_beta_s if final_beta_s is not None else float(results.get("V_outside_min", 0.0))
    visualize_results(
        V_net=V_net,
        GV_net=GV_net,
        regions=regions,
        region_cells=region_cells,
        params=params,
        loss_history=loss_history,
        refinement_epochs=refinement_epochs,
        results=results,
        final_beta_s=beta_s_plot,
        loaded=False,
    )

    print("\n" + "=" * 20)
    print("Saving Bundle")
    print("=" * 20)
    save_eval_bundle(
        output_dir,
        V_net=V_net,
        GV_net=GV_net,
        control_net=control_net,
        params=params,
        regions=regions,
        region_cells=region_cells,
        final_beta_s=beta_s_plot,
        loss_history=loss_history,
        refinement_epochs=refinement_epochs,
        results=results,
    )

    return total_training_time


def run_load_path(output_dir: Path, bundle_path: Path, regions: Regions) -> None:
    """Load bundle, evaluate, and visualize."""
    print("\n" + "=" * 20)
    print("Loading Bundle")
    print("=" * 20)
    bundle = load_eval_bundle(bundle_path, map_location="cpu")

    params = Hyperparameters.from_dict(bundle["hyperparameters"])
    region_cells = bundle["region_cells"]
    device = params.training.device

    control_net = make_controller(device)
    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    V_net = restore_v_net_from_state(params, bundle["V_state_dict"], device)

    dynamics = make_invpend_dynamics(control_net)
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        verify=True,
    ).to(device)
    if bundle["GV_state_dict"] is not None:
        GV_net.load_state_dict(bundle["GV_state_dict"])

    region_cells = {k: [(lo.to(device), hi.to(device)) for (lo, hi) in v] for k, v in region_cells.items()}
    final_beta_s = bundle["final_beta_s"]
    loss_history = bundle["loss_history"]
    refinement_epochs = bundle["refinement_epochs"]

    results = bundle.get("final_results", None)
    if results is None:
        print("No saved results found in bundle, recomputing evaluation...")
        results = final_evaluation(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            params=params,
            device=device,
            title="Final Evaluation (loaded)",
        )
    else:
        results = final_evaluation(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            params=params,
            device=device,
            title="Final Evaluation (loaded)",
            cached_results=results,
        )

    beta_s_plot = final_beta_s if final_beta_s is not None else float(results.get("V_outside_min", 0.0))
    visualize_results(
        V_net=V_net,
        GV_net=GV_net,
        regions=regions,
        region_cells=region_cells,
        params=params,
        loss_history=loss_history,
        refinement_epochs=refinement_epochs,
        results=results,
        final_beta_s=beta_s_plot,
        loaded=True,
    )
