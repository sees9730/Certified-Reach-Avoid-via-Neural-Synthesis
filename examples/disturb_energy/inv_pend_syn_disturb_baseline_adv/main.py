"""
2D Inverted Pendulum Baseline

Single baseline pipeline only:
1) pre-train value/controller from scratch,
2) run bound-based training,
3) evaluate and generate plots.

No energy dimension, no finite-time dimension, no curriculum, no resume/warm-start.
"""
import sys
import time
from math import pi
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN
from src.discretization import discretize_regions
from src.dynamics import Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging
from src.set_values import AdditiveBoxSetDrift, ClosedLoopSetValuedDrift
from src.trainer import train_network_bounds
from src.training_utils import evaluate_constraints, print_constraint_summary
from src.utils import cleanup_and_setup_directories
from src.visualization import create_summary_plots

torch.manual_seed(0)


def pretrain_network_samples(
    model,
    x_goal_range,
    x_unsafe_range,
    x_init_range,
    x_range,
    params,
    GV_net=None,
    num_epochs=1000,
    lr=1e-4,
    device='cpu',
    control_net=None,
    n_each: int = 400,
    lambda_w=1e-3,
    save_v_path=None,
    save_control_path=None,
):
    """Pre-train V and GV networks using sampled points."""
    print("\n" + "=" * 20)
    print("Pre-training using samples")
    print("=" * 20)

    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    if GV_net is not None:
        print("GV pre-training enabled")
    else:
        print("GV pre-training disabled")

    best_loss = float('inf')
    best_model_state = None
    best_control_state = None

    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    init_t = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
    low = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low

    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        return torch.rand(N, D, device=device) * (box[:, 1] - box[:, 0]) + box[:, 0]

    def _in_box(x_batch: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return ((x_batch >= box[:, 0]) & (x_batch <= box[:, 1])).all(dim=1)

    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
        t = torch.as_tensor(x_unsafe, dtype=torch.float32, device=device)
        if t.dim() == 2:
            if t.shape == (D, 2):
                return t.unsqueeze(0)
            if t.shape[1] == 2 and (t.shape[0] % D == 0):
                return t.view(int(t.shape[0] // D), D, 2)
            raise ValueError(f"x_unsafe_range 2D must be (D,2) or (K*D,2); got {tuple(t.shape)}")
        if t.dim() == 3:
            if t.shape[1:] != (D, 2):
                raise ValueError(f"x_unsafe_range 3D must be (K,D,2) with D={D}; got {tuple(t.shape)}")
            return t
        raise ValueError(f"x_unsafe_range must be (D,2), (K,D,2), or (K*D,2); got {tuple(t.shape)}")

    unsafe_boxes = _unsafe_to_boxes(x_unsafe_range)
    K_unsafe = int(unsafe_boxes.shape[0])

    def _in_unsafe_union(x_batch: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(x_batch.shape[0], dtype=torch.bool, device=device)
        for k in range(K_unsafe):
            mask |= _in_box(x_batch, unsafe_boxes[k])
        return mask

    def _sample_in_unsafe_union(N: int) -> torch.Tensor:
        if K_unsafe == 1:
            return _sample_in_box(unsafe_boxes[0], N)
        xs = [_sample_in_box(unsafe_boxes[k], N) for k in range(K_unsafe)]
        x = torch.cat(xs, dim=0)
        return x[torch.randperm(x.shape[0], device=device)]

    def _l2_weight_penalty(net: torch.nn.Module) -> torch.Tensor:
        reg = 0.0
        for name, p in net.named_parameters():
            if p.requires_grad and p.dim() > 1 and not name.endswith("bias"):
                reg = reg + (p ** 2).sum()
        return reg

    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        x_full = torch.rand(n_each, D, device=device) * span + low
        v_loss_full = F.relu(-model(x_full).squeeze(-1)).sum()

        x_init = _sample_in_box(init_t, n_each)
        v_loss_init = F.relu(model(x_init).squeeze(-1) - 1.0).sum()

        x_unsafe = _sample_in_unsafe_union(int(n_each / 6))
        v_loss_unsafe = F.relu(params.constraints.beta_ra - model(x_unsafe).squeeze(-1)).sum()

        x_others_list = []
        need = n_each
        for _ in range(20):
            if need <= 0:
                break
            x_cand = torch.rand(max(need * 4, 32), D, device=device) * span + low
            keep = ~(_in_box(x_cand, goal_t) | _in_unsafe_union(x_cand))
            x_keep = x_cand[keep]
            if x_keep.shape[0] > 0:
                take = min(need, x_keep.shape[0])
                x_others_list.append(x_keep[:take])
                need -= take

        x_others = torch.cat(x_others_list, dim=0) if x_others_list else x_full.detach()

        loss_gv = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_gv = x_others.detach().clone().requires_grad_(True)
            loss_gv = F.relu(GV_net(x_gv).squeeze(-1)).sum()

        total_loss = v_loss_full + v_loss_init + v_loss_unsafe + loss_gv
        total_loss = total_loss + lambda_w * _l2_weight_penalty(model)

        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        if epoch % 100 == 0:
            if GV_net is not None:
                print(f"Epoch {epoch} | V_loss={v_loss_full.item() + v_loss_init.item() + v_loss_unsafe.item():8.4f} | GV_loss={loss_gv.item():8.4f}")
            else:
                print(f"Epoch {epoch} | V_loss={v_loss_full.item() + v_loss_init.item() + v_loss_unsafe.item():8.4f}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\nBest loss: {best_loss:.6f}")
        if save_v_path is not None:
            torch.save(best_model_state, save_v_path)
            print(f"Saved pretrained V_net to: {save_v_path}")
        if control_net is not None and best_control_state is not None and save_control_path is not None:
            torch.save(best_control_state, save_control_path)
            print(f"Saved pretrained Controller_net to: {save_control_path}")

    networks_trained = ["V"] + (["GV"] if GV_net is not None else []) + (["Controller"] if control_net is not None else [])
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")


def evaluate_and_visualize(
    V_net,
    GV_net,
    regions,
    region_cells,
    params,
    active_results_dir,
    loss_history,
    refinement_epochs,
    final_beta_s=None,
    results=None,
    loaded_mode: bool = False,
):
    print("\n" + "=" * 20)
    print("Final Evaluation" + (" (loaded)" if loaded_mode else ""))
    print("=" * 20)

    if results is None:
        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_ra=params.constraints.beta_ra,
            device=params.training.device,
        )
    print_constraint_summary(results)

    print("\n" + "=" * 20)
    print("Visualizations")
    print("=" * 20)
    create_summary_plots(
        V_net=V_net,
        GV_net=GV_net,
        regions=regions,
        region_cells=region_cells,
        beta_s=final_beta_s,
        beta_ra=params.constraints.beta_ra,
        loss_history=loss_history,
        refinement_epochs=refinement_epochs,
        results=results,
        output_dir=str(active_results_dir),
    )


def main():
    print("=" * 20)
    print("2D Inverted Pendulum Baseline")
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
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [2 * pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0

    params.discretization.axis_weights = [1.0, 1.0]
    params.discretization.max_region_budget = 1000
    params.discretization.max_generator_budget = 1000

    params.constraints.beta_ra = 3.0

    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 5000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1200
    params.training.curriculum_mode = "none"
    params.training.resume_checkpoint_path = str(OUTPUT_DIR / "resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(active_output_dir)
    params.training.progress_output_dir = str(active_progress_dir)

    # Refinement settings
    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_interval = 500
    params.refinement.v_outside.late_epoch_threshold = 2500
    params.refinement.v_outside.refine_interval_late = 100
    params.refinement.v_outside.refine_factor = 2
    params.refinement.v_outside.max_cells = 50000
    params.refinement.v_outside.N_to_refine = 100
    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 501
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 20.0

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
    params.refinement.gv_generator.merge_relax_margin = -1000.0
    params.refinement.refine_interval_after_first_sat = 500

    # === Dynamics ===
    rl_policy_net = InvertControlNN()
    u_nn = WrapperConterlNN(rl_policy_net)

    def f_ol(x: torch.Tensor, _u: torch.Tensor = None) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x2
        f2 = (9.81 / 0.5) * torch.sin(x1) - (0.1 / (0.15 * 0.5 ** 2)) * x2
        return torch.stack([f1, f2], dim=1)

    _g_coeffs = torch.tensor([0.0, 0.2], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        base = _g_coeffs.to(device=x.device, dtype=x.dtype)
        if x.dim() == 1:
            return base
        elif x.dim() == 2:
            return base.unsqueeze(0).expand(x.shape[0], -1)
        else:
            raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")

    drift_unc = torch.tensor([2.0, 2.0], dtype=torch.float32)
    f_ol_set = AdditiveBoxSetDrift(f_ol, drift_unc)
    f_cl_module = ClosedLoopSetValuedDrift(
        f_ol_set.to(params.training.device), controller=u_nn
    ).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=state_dim)

    # === Regions ===
    init_range  = np.array([[(3/4)*pi, (5/4)*pi], [-1.0,  1.0]],  dtype=np.float32)
    goal_range  = np.array([[-0.4*pi,  0.4*pi],   [-4.0,  4.0]],  dtype=np.float32)
    full_range  = np.array([[-2*pi,    2*pi],      [-20.0, 20.0]], dtype=np.float32)

    unsafe_down1 = np.array([[-2*pi,       -2*pi+0.5*pi], [-20.0, -10.0]], dtype=np.float32)
    unsafe_down2 = np.array([[ 2*pi-0.5*pi, 2*pi],        [ 10.0,  20.0]], dtype=np.float32)
    unsafe_lb    = np.array([[-2*pi,        -2*pi+0.5],   [-20.0,  20.0]], dtype=np.float32)
    unsafe_rb    = np.array([[ 2*pi-0.5,    2*pi],        [-20.0,  20.0]], dtype=np.float32)
    unsafe_tb    = np.array([[-2*pi,        2*pi],        [ 19.5,  20.0]], dtype=np.float32)
    unsafe_bb    = np.array([[-2*pi,        2*pi],        [-20.0, -19.5]], dtype=np.float32)

    init   = Region(init_range)
    goal   = Region(goal_range)
    unsafe = Region.union(
        Region(unsafe_down1), Region(unsafe_down2),
        Region(unsafe_lb),    Region(unsafe_rb),
        Region(unsafe_tb),    Region(unsafe_bb),
    )
    full = Region(full_range)
    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    unsafe_range = np.vstack((unsafe_down1, unsafe_down2, unsafe_tb, unsafe_bb, unsafe_lb, unsafe_rb))

    # === Networks ===
    input_offset = [0.0, 0.0]
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
