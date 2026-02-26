"""Workflow helpers for temporal 2D GBM verification."""

import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.discretization import discretize_region, discretize_regions
from src.dynamics import Dynamics
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


def make_gbm_dynamics() -> Dynamics:
    """Build closed-loop GBM dynamics over spatial state [x1, x2]."""

    def f_spatial(x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = -1.5 * x1 + 1.0 * x2
        f2 = -1.0 * x1 - 1.5 * x2
        return torch.stack([f1, f2], dim=1)

    g_coeffs = torch.tensor([0.2, 0.2], dtype=torch.float32)

    def g_spatial(x: torch.Tensor) -> torch.Tensor:
        return g_coeffs.to(device=x.device, dtype=x.dtype) * x

    return Dynamics.dynamics(f=f_spatial, g=g_spatial)


def configure_default_params(time_horizon: float) -> Hyperparameters:
    """Build default hyperparameters using the same curriculum/refinement structure as inv_pend_syn_unsafe."""
    params = Hyperparameters.default()

    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [time_horizon, 100.0, 100.0]
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

    params.discretization.n_goal = [10, 10, 10]
    params.discretization.n_outside_goal = [4, 4, 4]
    params.discretization.n_generator = [1, 1, 1]
    params.discretization.n_unsafe = [10, 9, 9]
    params.discretization.n_init = [10, 11, 11]

    params.constraints.beta_ra = 20.0
    params.constraints.use_beta_ra_curriculum = True
    params.constraints.beta_ra_init = 2.0
    params.constraints.beta_ra_step = 0.5
    params.constraints.use_terminal_beta_curriculum = False
    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 1000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1000

    params.refinement.v_goal.enable_refinement = True
    params.refinement.v_goal.refine_factor = [2, 2, 2]
    params.refinement.v_goal.max_cells = 50000
    params.refinement.v_goal.enable_merging = False

    params.refinement.v_unsafe.enable_refinement = True
    params.refinement.v_unsafe.refine_factor = [2, 2, 2]
    params.refinement.v_unsafe.max_cells = 50000
    params.refinement.v_unsafe.enable_merging = False

    params.refinement.v_init.enable_refinement = True
    params.refinement.v_init.refine_factor = [1, 2, 2]
    params.refinement.v_init.max_cells = 50000
    params.refinement.v_init.enable_merging = False

    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_factor = [2, 2, 2]
    params.refinement.v_outside.max_cells = 30000
    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 502
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 20.0

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_factor = [2, 2, 2]
    params.refinement.gv_generator.max_cells = 30000
    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 502
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
    init_range = np.array([[0.0, 0.0], [45.0, 55.0], [-55.0, -45.0]], dtype=np.float32)
    goal_range = np.array([[0.0, time_horizon], [-25.0, 25.0], [-25.0, 25.0]], dtype=np.float32)
    full_range = np.array([[0.0, time_horizon], [-100.0, 100.0], [-100.0, 100.0]], dtype=np.float32)
    unsafe_tube_range = np.array([[0.0, time_horizon], [-100.0, -80.0], [-100.0, 100.0]], dtype=np.float32)
    unsafe_terminal_full_range = np.array([[time_horizon, time_horizon], [-100.0, 100.0], [-100.0, 100.0]], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    full = Region(full_range)
    unsafe_tube = Region(unsafe_tube_range)
    unsafe_terminal_full = Region(unsafe_terminal_full_range)
    unsafe = Region.union(unsafe_tube, unsafe_terminal_full)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)
    params.constraints.unsafe_terminal_cell_count = len(
        discretize_region(unsafe_terminal_full, params.discretization.n_unsafe)
    )
    return regions, init_range, goal_range, full_range


def create_v_net_for_training(params: Hyperparameters) -> torch.nn.Module:
    """Create V network for training path."""
    return create_V(params.network)


def restore_v_net_from_state(params: Hyperparameters, v_state_dict: dict, device: str) -> torch.nn.Module:
    """Restore V network from state dict."""
    v_net = create_V(params.network).to(device)
    v_net.load_state_dict(v_state_dict)
    return v_net


def create_gv_net(
    V_net: torch.nn.Module,
    dynamics: Dynamics,
    params: Hyperparameters,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """Create GV network."""
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
            control_net=None,
            num_epochs=params.training.pretrain_epochs,
            lr=params.training.pretrain_lr,
            device=params.training.device,
            lambda_w=1e-3,
            n_each=params.training.pretrain_n_samples,
            save_v_path=output_dir / "V_pretrained.pth",
            save_control_path=None,
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
        control_net=None,
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
        control_net=None,
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
    V_net = restore_v_net_from_state(params, bundle["V_state_dict"], device)

    dynamics = make_gbm_dynamics()
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
