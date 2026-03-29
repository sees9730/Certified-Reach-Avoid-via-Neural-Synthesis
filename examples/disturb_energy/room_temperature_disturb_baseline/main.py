"""
2D Room Temperature Baseline

Single baseline pipeline only:
1) pre-train value/controller from scratch,
2) run bound-based training,
3) evaluate and generate plots.

No energy dimension, no finite-time dimension, no curriculum, no resume/warm-start.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

ROOM_TEMP_DIR = ROOT / "examples" / "disturb_energy" / "room_temperature_disturb"
sys.path.insert(0, str(ROOM_TEMP_DIR))

from src.control_network import RoomTempControlNN, RoomTempControlWrapper
from src.discretization import discretize_regions
from src.dynamics import Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging
from src.set_values import ClosedLoopSetValuedDrift
from src.trainer import train_network_bounds
from src.utils import cleanup_and_setup_directories
from test import (
    B_INPUT,
    KELVIN_OFFSET,
    RADIATION_COEFF,
    X0_INIT,
    XG_GOAL,
    X_DOMAIN,
    XS_SAFE,
    U_MAX,
    g_torch, f_ol_set,
)

# Reuse shared implementations from room_temperature_disturb/main.py
ROOM_TEMP_MAIN = ROOM_TEMP_DIR / "main.py"
_spec = importlib.util.spec_from_file_location("room_temp_disturb_main_shared", ROOM_TEMP_MAIN)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load shared module: {ROOM_TEMP_MAIN}")
_room_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_room_main)
pretrain_network_samples = _room_main.pretrain_network_samples
evaluate_and_visualize = _room_main.evaluate_and_visualize

torch.manual_seed(0)


def main():
    print("=" * 20)
    print("2D Room Temperature Baseline")
    print("=" * 20)
    print("Baseline mode: no energy dimension, no time dimension, no curriculum.")

    active_output_dir = OUTPUT_DIR
    active_results_dir = HERE / "results"
    active_progress_dir = HERE / "training_progress"

    # === Hyperparameters ===
    params = Hyperparameters.default()
    params.include_time = False
    params.include_energy = False

    state_dim = 2
    params.network.n_inputs = state_dim
    params.network.n_hidden_1 = 32
    params.network.n_hidden_2 = 32

    x_min, x_max = X_DOMAIN
    x_center = (x_min + x_max) / 2.0
    x_scale = (x_max - x_min) / 2.0
    params.network.input_scale = [x_scale, x_scale]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.001
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0

    # Discretization in [x1, x2]
    params.discretization.axis_weights = [1.0, 1.0]
    params.discretization.max_region_budget = 10000
    params.discretization.max_generator_budget = 1000

    params.constraints.beta_ra = 10.0

    params.compute_V = True
    params.compute_GV = True

    # Always pre-train from scratch.
    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 2000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1200
    params.training.curriculum_mode = "none"
    params.training.resume_checkpoint_path = str(OUTPUT_DIR / "resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(active_output_dir)
    params.training.progress_output_dir = str(active_progress_dir)

    print(f"[Dynamics] Radiation term enabled: -{RADIATION_COEFF:.3e} * (x + {KELVIN_OFFSET})^4")

    # Refinement settings
    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_interval = 500
    params.refinement.v_outside.late_epoch_threshold = 2500
    params.refinement.v_outside.refine_interval_late = 100
    params.refinement.v_outside.refine_factor = 2
    params.refinement.v_outside.max_cells = 100000
    params.refinement.v_outside.N_to_refine = 100
    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 501
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 40.0

    params.refinement.v_unsafe.enable_refinement = True
    params.refinement.v_unsafe.refine_interval = 500
    params.refinement.v_unsafe.late_epoch_threshold = 2500
    params.refinement.v_unsafe.refine_interval_late = 500
    params.refinement.v_unsafe.refine_factor = 2
    params.refinement.v_unsafe.max_cells = 50000
    params.refinement.v_unsafe.N_to_refine = 100
    params.refinement.v_unsafe.enable_merging = False

    params.refinement.v_init.enable_refinement = True
    params.refinement.v_init.refine_interval = 500
    params.refinement.v_init.late_epoch_threshold = 2500
    params.refinement.v_init.refine_interval_late = 500
    params.refinement.v_init.refine_factor = 2
    params.refinement.v_init.max_cells = 50000
    params.refinement.v_init.N_to_refine = 100
    params.refinement.v_init.enable_merging = False

    params.refinement.v_goal.enable_refinement = True
    params.refinement.v_goal.refine_interval = 500
    params.refinement.v_goal.late_epoch_threshold = 2500
    params.refinement.v_goal.refine_interval_late = 500
    params.refinement.v_goal.refine_factor = 2
    params.refinement.v_goal.max_cells = 100000
    params.refinement.v_goal.N_to_refine = 100
    params.refinement.v_goal.enable_merging = False

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_interval = 500
    params.refinement.gv_generator.late_epoch_threshold = 3500
    params.refinement.gv_generator.refine_interval_late = 500
    params.refinement.gv_generator.refine_factor = 2
    params.refinement.gv_generator.max_cells = 100000
    params.refinement.gv_generator.N_to_refine = 100
    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -500.0
    params.refinement.refine_interval_after_first_sat = 100

    # === Dynamics ===
    rl_policy_net = RoomTempControlNN(input_dim=state_dim, u_max=U_MAX)
    u_nn = RoomTempControlWrapper(rl_policy_net, B_INPUT)

    f_cl_module = ClosedLoopSetValuedDrift(
        f_ol_set.to(params.training.device), controller=u_nn
    ).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g_torch, state_dim=state_dim)

    # === Regions ===
    x0_lo, x0_hi = X0_INIT
    xg_lo, xg_hi = XG_GOAL
    xs_lo, xs_hi = XS_SAFE

    init_range = np.array([[x0_lo, x0_hi], [x0_lo, x0_hi]], dtype=np.float32)
    goal_range = np.array([[xg_lo, xg_hi], [xg_lo, xg_hi]], dtype=np.float32)
    full_range = np.array([[x_min, x_max], [x_min, x_max]], dtype=np.float32)

    unsafe_x1_lo = np.array([[x_min, xs_lo], [x_min, x_max]], dtype=np.float32)
    unsafe_x1_hi = np.array([[xs_hi, x_max], [x_min, x_max]], dtype=np.float32)
    unsafe_x2_lo = np.array([[xs_lo, xs_hi], [x_min, xs_lo]], dtype=np.float32)
    unsafe_x2_hi = np.array([[xs_lo, xs_hi], [xs_hi, x_max]], dtype=np.float32)
    unsafe_range = np.vstack((unsafe_x1_lo, unsafe_x1_hi, unsafe_x2_lo, unsafe_x2_hi))

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe = Region.union(
        Region(unsafe_x1_lo),
        Region(unsafe_x1_hi),
        Region(unsafe_x2_lo),
        Region(unsafe_x2_hi),
    )
    full = Region(full_range)
    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # === Networks ===
    input_offset = [x_center, x_center]
    output_offset = np.float32(0.1)
    V_net = create_V(params.network, input_offset=input_offset, output_offset=output_offset)
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        input_offset=input_offset,
        include_time=False,
        include_energy=False,
    )

    cleanup_and_setup_directories([active_results_dir, active_progress_dir])
    enable_terminal_logging(active_output_dir / "terminal_log.txt", append=False)

    region_cells = discretize_regions(regions, params.discretization, use_radial_generator=False)

    pretrain_start_time = time.time()
    pretrain_network_samples(
        model=V_net,
        x_goal_range=goal_range,
        x_unsafe_range=unsafe_range,
        x_init_range=init_range,
        x_range=full_range,
        params=params,
        GV_net=GV_net,
        num_epochs=params.training.pretrain_epochs,
        lr=params.training.pretrain_lr,
        lambda_w=1e-3,
        device=params.training.device,
        control_net=u_nn,
        n_each=params.training.pretrain_n_samples,
        save_v_path=active_output_dir / "V_pretrained.pth",
        save_control_path=active_output_dir / "controller_pretrained.pth",
    )
    pretrain_time = time.time() - pretrain_start_time
    print(f"Pre-training time: {pretrain_time:.1f}s")

    training_start_time = time.time()

    def create_scheduler(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

    loss_history, final_beta_s, refinement_epochs = train_network_bounds(
        V_net=V_net,
        GV_net=GV_net,
        region_cells=region_cells,
        regions=regions,
        params=params,
        control_net=u_nn,
        create_scheduler=create_scheduler,
        start_time=training_start_time,
    )

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    evaluate_and_visualize(
        V_net=V_net,
        GV_net=GV_net,
        regions=regions,
        region_cells=region_cells,
        params=params,
        active_results_dir=active_results_dir,
        loss_history=loss_history,
        refinement_epochs=refinement_epochs,
        final_beta_s=final_beta_s,
        results=None,
        loaded_mode=False,
    )

    return total_training_time


if __name__ == "__main__":
    main()
