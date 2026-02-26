"""Workflow helpers for temporal inverted-pendulum synthesis."""

import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.control_network import InvertControlNN, WrapperConterlNN
from src.discretization import clip_cell_against_exclusions, discretize_region, discretize_regions
from src.dynamics import ClosedLoopDrift, Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.pretrainer import pretrain_network_samples
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging, load_eval_bundle, log_loaded_training_epochs, save_eval_bundle
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


def make_invpend_dynamics(control_net: torch.nn.Module) -> Dynamics:
    """Build closed-loop inverted-pendulum dynamics over spatial state [x1, x2]."""

    def f_ol_spatial(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        g = 9.81
        L = 0.5
        b = 0.1
        m = 0.15
        x1, x2 = x[:, 0], x[:, 1]
        f1 = x2
        f2 = (g / L) * torch.sin(x1) - (b / (m * L ** 2)) * x2
        return torch.stack([f1, f2], dim=1)

    g_coeffs = torch.tensor([0.0, 0.2], dtype=torch.float32)

    def g_spatial(x: torch.Tensor) -> torch.Tensor:
        base = g_coeffs.to(device=x.device, dtype=x.dtype)
        if x.dim() == 1:
            return base
        if x.dim() == 2:
            return base.unsqueeze(0).expand(x.shape[0], -1)
        raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")

    f_cl_module = ClosedLoopDrift(f_ol_spatial, control_net)
    return Dynamics.dynamics(f=f_cl_module, g=g_spatial)


def make_controller(device: str) -> torch.nn.Module:
    """Create state-feedback controller for synthesis on spatial state [x1, x2]."""
    return WrapperConterlNN(InvertControlNN(hidden_dim=64)).to(device)


def configure_default_params(time_horizon: float) -> Hyperparameters:
    """Build the default experiment hyperparameters."""
    params = Hyperparameters.default()

    pi = np.pi
    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [time_horizon, 2 * pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0
    params.training.generator_after_v_sat = True
    params.training.generator_ramp_epochs = 0
    params.training.use_generator_threshold_curriculum = True
    params.training.generator_threshold_final = 0.0
    params.training.generator_threshold_step = 1.0
    params.training.generator_threshold_step_fine = 0.1
    params.training.generator_threshold_switch = 1.0

    params.logging.visualize_interval = 0

    params.discretization.n_goal = [20, 20, 20]
    params.discretization.n_outside_goal = [10, 10, 10]
    params.discretization.n_generator = [2, 2, 2]
    params.discretization.n_unsafe = [20, 20, 20]
    params.discretization.n_init = [1, 100, 100]

    params.constraints.beta_ra = 20.0
    params.constraints.use_beta_ra_curriculum = True
    params.constraints.beta_ra_init = 2.0
    params.constraints.beta_ra_step = 0.5
    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 1000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1000

    params.refinement.v_goal.enable_refinement = True
    params.refinement.v_goal.refine_factor = [2, 2, 2]
    params.refinement.v_goal.max_cells = 100000
    params.refinement.v_goal.enable_merging = False

    params.refinement.v_unsafe.enable_refinement = True
    params.refinement.v_unsafe.refine_factor = [2, 2, 2]
    params.refinement.v_unsafe.max_cells = 100000
    params.refinement.v_unsafe.enable_merging = False

    params.refinement.v_init.enable_refinement = True
    params.refinement.v_init.refine_factor = [1, 2, 2]
    params.refinement.v_init.max_cells = 100000
    params.refinement.v_init.enable_merging = False

    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_factor = [2, 2, 2]
    params.refinement.v_outside.max_cells = 100000
    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 501
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 20.0

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_factor = [2, 2, 2]
    params.refinement.gv_generator.max_cells = 100000
    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -500.0

    params.refinement.use_simple_refinement = True
    params.refinement.simple_refine_every = 200
    params.refinement.simple_loss_improve_tol = 0.01
    params.refinement.simple_refine_budget = 100
    params.refinement.simple_regions_per_trigger = 5
    params.refinement.stage3_force_refine_interval = 1000

    return params


def build_regions(time_horizon: float, params: Hyperparameters) -> Tuple[Regions, np.ndarray, np.ndarray, np.ndarray]:
    """Create init/goal/unsafe/full regions and set terminal-unsafe cell count."""
    pi = np.pi
    init_range = np.array([[0.0, 0.0], [(3 / 4) * pi, (5 / 4) * pi], [-1.0, 1.0]], dtype=np.float32)
    goal_range = np.array([[0.0, time_horizon], [-0.4 * pi, 0.4 * pi], [-4.0, 4.0]], dtype=np.float32)
    full_range = np.array([[0.0, time_horizon], [-2 * pi, 2 * pi], [-20.0, 20.0]], dtype=np.float32)

    unsafe_down1 = np.array([[0.0, time_horizon], [-2 * pi, -1.5 * pi], [-20.0, -10.0]], dtype=np.float32)
    unsafe_down2 = np.array([[0.0, time_horizon], [1.5 * pi, 2 * pi], [10.0, 20.0]], dtype=np.float32)
    unsafe_lb = np.array([[0.0, time_horizon], [-2 * pi, -2 * pi + 0.5], [-20.0, 20.0]], dtype=np.float32)
    unsafe_rb = np.array([[0.0, time_horizon], [2 * pi - 0.5, 2 * pi], [-20.0, 20.0]], dtype=np.float32)
    unsafe_tb = np.array([[0.0, time_horizon], [-2 * pi, 2 * pi], [19.5, 20.0]], dtype=np.float32)
    unsafe_bb = np.array([[0.0, time_horizon], [-2 * pi, 2 * pi], [-20.0, -19.5]], dtype=np.float32)
    unsafe_terminal_full = np.array([[time_horizon, time_horizon], [-2 * pi, 2 * pi], [-20.0, 20.0]], dtype=np.float32)
    goal_terminal = np.array([[time_horizon, time_horizon], [-0.4 * pi, 0.4 * pi], [-4.0, 4.0]], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe_tube_components = [
        Region(unsafe_down1),
        Region(unsafe_down2),
        Region(unsafe_tb),
        Region(unsafe_bb),
        Region(unsafe_lb),
        Region(unsafe_rb),
    ]

    terminal_exclusions = [
        np.array([[time_horizon, time_horizon], [b[1, 0], b[1, 1]], [b[2, 0], b[2, 1]]], dtype=np.float32)
        for b in (unsafe_down1, unsafe_down2, unsafe_tb, unsafe_bb, unsafe_lb, unsafe_rb)
    ]
    terminal_exclusions.append(goal_terminal)
    terminal_miss_boxes = clip_cell_against_exclusions(
        unsafe_terminal_full[:, 0],
        unsafe_terminal_full[:, 1],
        terminal_exclusions,
    )
    terminal_miss_components = [Region(np.stack([lo, hi], axis=1)) for (lo, hi) in terminal_miss_boxes]
    if not terminal_miss_components:
        raise ValueError("Terminal miss set is empty; adjust goal/unsafe/full definitions.")

    unsafe = Region.union(*(unsafe_tube_components + terminal_miss_components))
    full = Region(full_range)
    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)
    params.constraints.unsafe_terminal_cell_count = len(
        discretize_region(Region.union(*terminal_miss_components), params.discretization.n_unsafe)
    )

    return regions, init_range, goal_range, full_range


def create_v_net_for_training(params: Hyperparameters, goal_range: np.ndarray, use_v_offset: bool = True) -> torch.nn.Module:
    """Create V network for training path."""
    if not use_v_offset:
        return create_V(params.network)
    input_offset = np.array([
        0.5 * (goal_range[0, 0] + goal_range[0, 1]),
        0.5 * (goal_range[1, 0] + goal_range[1, 1]),
        0.5 * (goal_range[2, 0] + goal_range[2, 1]),
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


def create_gv_net(
    V_net: torch.nn.Module,
    dynamics: Dynamics,
    params: Hyperparameters,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """Create GV/GV_offset network."""
    gv_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        verify=True,
        include_time_derivative=True,
        time_index=0,
    )
    return gv_net if device is None else gv_net.to(device)


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
    print("Creating Visualizations (loaded)" if loaded else "Creating Visualizations")
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
            control_net=control_net,
            num_epochs=params.training.pretrain_epochs,
            lr=params.training.pretrain_lr,
            device=params.training.device,
            lambda_w=1e-3,
            n_each=params.training.pretrain_n_samples,
            save_v_path=output_dir / "V_pretrained.pth",
            save_control_path=output_dir / "controller_pretrained.pth",
        )
    else:
        V_net.load_state_dict(torch.load(output_dir / "V_pretrained.pth", map_location="cpu"))
        control_net.load_state_dict(torch.load(output_dir / "controller_pretrained.pth", map_location="cpu"))

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
    print("Saving Evaluation Bundle")
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


def run_load_path(output_dir: Path, bundle_path: Path) -> None:
    """Load bundle, evaluate, and visualize."""
    print("\n" + "=" * 20)
    print("Loading Saved Bundle")
    print("=" * 20)
    bundle = load_eval_bundle(bundle_path, map_location="cpu")

    params = Hyperparameters.from_dict(bundle["hyperparameters"])
    regions = Regions.from_dict(bundle["regions"])
    region_cells = bundle["region_cells"]

    device = params.training.device
    control_net = make_controller(device)
    if bundle["control_state_dict"] is not None:
        control_net.load_state_dict(bundle["control_state_dict"])

    V_net = restore_v_net_from_state(params, bundle["V_state_dict"], device)

    dynamics = make_invpend_dynamics(control_net)
    GV_net = create_gv_net(V_net, dynamics, params, device=device)
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
