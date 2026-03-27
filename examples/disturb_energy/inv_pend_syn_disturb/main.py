"""
2D Inverted Pendulum Control Synthesis

How To Run
==========
Assume you are in this folder:
`examples/disturb_temporal/inv_pend_syn_disturb/`

Pipeline
--------
1) Nominal training to first SAT:
- `python main.py --train 1`
- saves checkpoint to `outputs/resume_checkpoint.pth`

2) From nominal SAT checkpoint, choose one curriculum mode:
- Beta curriculum (`beta_ra` increases):
  `python main.py --train 1 --curriculum_mode beta --resume_checkpoint outputs/resume_checkpoint.pth`
- Energy curriculum (`energy_max` decreases in-loop):
  `python main.py --train 1 --curriculum_mode energy --resume_checkpoint outputs/resume_checkpoint.pth --energy_max_step 0.1 --energy_max_min 0.2`

Optional stop/resume:
- Beta: during training, type `stop` in terminal.
  Resume with:
  `python main.py --train 1 --curriculum_mode beta --resume_checkpoint outputs/beta_latest_resume_checkpoint.pth`
- Energy: during training, type `stop` in terminal.
  Resume with:
  `python main.py --train 1 --curriculum_mode energy --resume_checkpoint outputs/energy_latest_resume_checkpoint.pth --energy_max_step 0.1 --energy_max_min 0.2`

Visualization:
- Nominal:
  `python main.py --train 0`
- Beta:
  `python main.py --train 0 --curriculum_mode beta`
- Energy:
  `python main.py --train 0 --curriculum_mode energy`

Outputs (default paths):
- Nominal mode:
  - `outputs/eval_bundle.pth`, `outputs/resume_checkpoint.pth`, `outputs/terminal_log.txt`
  - `results/`, `training_progress/`
- Beta mode:
  - run artifacts in `outputs/beta_runs/...` (single folder):
    - `outputs/beta_runs/eval_bundle.pth`
    - `outputs/beta_runs/terminal_log.txt` (appended across resumes)
    - `outputs/beta_runs/results/`
    - `outputs/beta_runs/training_progress/`
  - rolling checkpoint `outputs/beta_latest_resume_checkpoint.pth`
- Energy mode:
  - run artifacts in `outputs/energy_runs/...` (single folder across stages):
    - `outputs/energy_runs/eval_bundle.pth`
    - `outputs/energy_runs/terminal_log.txt` (appended across stages)
    - `outputs/energy_runs/results/`
    - `outputs/energy_runs/training_progress/`
  - rolling checkpoint `outputs/energy_latest_resume_checkpoint.pth`
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN
from src.discretization import discretize_regions, debug_print_region_bounds
from src.dynamics import Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging, load_eval_bundle, log_loaded_training_epochs
from src.trainer import train_network_bounds
from src.training_utils import evaluate_constraints, print_constraint_summary
from src.utils import cleanup_and_setup_directories
from src.visualization import create_summary_plots
from src.set_values import AdditiveBoxSetDrift, ClosedLoopSetValuedDrift

torch.manual_seed(0)


def debug_print_all_region_cells(region_cells: dict, label: str = "Region cells") -> None:
    print(f"{label}:")
    for name in ("init", "goal", "unsafe", "outside", "generator"):
        cells = region_cells.get(name, None)
        if cells is None:
            continue
        debug_print_region_bounds(cells, label=f"{name} cells")


def sync_regions_from_region_cells(regions: Regions, region_cells: dict, params) -> Regions:
    """
    Rebuild region metadata from loaded cells and clip first-dimension bounds to current horizon.
    """
    def _bounds(cells):
        if not cells:
            return None
        lowers = torch.stack([lo.detach().cpu() for (lo, _) in cells], dim=0)
        uppers = torch.stack([hi.detach().cpu() for (_, hi) in cells], dim=0)
        lo = torch.min(lowers, dim=0).values.numpy()
        hi = torch.max(uppers, dim=0).values.numpy()
        return np.stack([lo, hi], axis=1).astype(np.float32)

    init_b = _bounds(region_cells.get("init", []))
    goal_b = _bounds(region_cells.get("goal", []))
    unsafe_b = _bounds(region_cells.get("unsafe", []))
    full_cells = (
        list(region_cells.get("init", []))
        + list(region_cells.get("goal", []))
        + list(region_cells.get("unsafe", []))
        + list(region_cells.get("outside", []))
    )
    full_b = _bounds(full_cells)

    if init_b is not None:
        regions.init = Region(init_b)
    if goal_b is not None:
        regions.goal = Region(goal_b)
    if unsafe_b is not None:
        regions.unsafe = Region(unsafe_b)
    if full_b is not None:
        regions.full = Region(full_b)

    if bool(getattr(params, "include_energy", False)):
        scales = getattr(params.network, "input_scale", None)
        if isinstance(scales, (list, tuple)) and len(scales) >= 1:
            emax = float(scales[0])
            for r in (regions.init, regions.goal, regions.unsafe, regions.full):
                r.bounds[0, 0] = min(float(r.bounds[0, 0]), 0.0)
                r.bounds[0, 1] = min(float(r.bounds[0, 1]), emax)

    return regions


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
    lambda_w = 1e-3,
    save_v_path=None,
    save_control_path=None,
):
    """
    Pre-train V and GV networks using sampled points.
    """
    print("\n" + "="*20)
    print("Pre-training using samples")
    print("="*20)

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

    # Convert ranges to torch once: each is (D,2)
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    init_t = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
    low  = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low

    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        b_low = box[:, 0]
        b_high = box[:, 1]
        return torch.rand(N, D, device=device) * (b_high - b_low) + b_low

    def _in_box(x_batch: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return ((x_batch >= box[:, 0]) & (x_batch <= box[:, 1])).all(dim=1)

    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
        t = torch.as_tensor(x_unsafe, dtype=torch.float32, device=device)

        if t.dim() == 2:
            if t.shape == (D, 2):
                return t.unsqueeze(0)  # (1,D,2)
            if t.shape[1] == 2 and (t.shape[0] % D == 0):
                K = int(t.shape[0] // D)
                return t.view(K, D, 2)  # (K,D,2)  <-- handles vstack case
            raise ValueError(f"x_unsafe_range 2D must be (D,2) or (K*D,2); got {tuple(t.shape)}")

        if t.dim() == 3:
            if t.shape[1:] != (D, 2):
                raise ValueError(f"x_unsafe_range 3D must be (K,D,2) with D={D}; got {tuple(t.shape)}")
            return t

        raise ValueError(f"x_unsafe_range must be (D,2), (K,D,2), or (K*D,2); got {tuple(t.shape)}")

    unsafe_boxes = _unsafe_to_boxes(x_unsafe_range)  # (K,D,2)
    K_unsafe = int(unsafe_boxes.shape[0])

    def _in_unsafe_union(x_batch: torch.Tensor) -> torch.Tensor:
        """mask True if x is inside ANY unsafe box."""
        mask = torch.zeros(x_batch.shape[0], dtype=torch.bool, device=device)
        for k in range(K_unsafe):
            mask |= _in_box(x_batch, unsafe_boxes[k])
        return mask

    def _sample_in_unsafe_union(N: int) -> torch.Tensor:
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")

        if K_unsafe == 1:
            return _sample_in_box(unsafe_boxes[0], N)

        xs = []
        for k in range(K_unsafe):
            xs.append(_sample_in_box(unsafe_boxes[k], N))  # (N, D) per box

        x = torch.cat(xs, dim=0)  # (K_unsafe * N, D)
        x = x[torch.randperm(x.shape[0], device=device)]  # shuffle
        return x
    
    def _l2_weight_penalty(model: torch.nn.Module, exclude_bias: bool = True) -> torch.Tensor:
        reg = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_bias and (p.dim() == 1 or name.endswith("bias")):
                continue
            reg = reg + (p ** 2).sum()
        return reg

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # full-range samples -> enforce v(t, x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # init-range samples -> enforce v(t=0, x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # unsafe-range samples -> enforce v(t, x) >= beta_ra
        x_unsafe = _sample_in_unsafe_union(int(n_each /6))
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.beta_ra - v_unsafe).sum()

        # samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= 0.0
        x_others_list = []
        need = n_each
        max_tries = 20
        tries = 0
        while need > 0 and tries < max_tries:
            tries += 1
            x_cand = torch.rand(max(need * 4, 32), D, device=device) * span + low
            cand_in_goal = _in_box(x_cand, goal_t)
            cand_in_unsafe = _in_unsafe_union(x_cand)
            keep = ~(cand_in_goal | cand_in_unsafe)
            x_keep = x_cand[keep]
            if x_keep.shape[0] > 0:
                take = min(need, x_keep.shape[0])
                x_others_list.append(x_keep[:take])
                need -= take

        if len(x_others_list) == 0:
            x_others = x_full.detach()
        else:
            x_others = torch.cat(x_others_list, dim=0)

        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
        )

        loss_gv = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_gv = x_others.detach().clone().requires_grad_(True)
            gv_output = GV_net(x_gv).squeeze(-1)
            loss_gv = F.relu(gv_output).sum()

        total_loss = loss_v + loss_gv

        # L2 weight penalty
        reg_w = _l2_weight_penalty(model, exclude_bias=True)
        total_loss = total_loss + lambda_w * reg_w

        # Track best
        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        if epoch % 100 == 0:
            if GV_net is not None:
                print(f"Epoch {epoch} | V_loss={loss_v.item():8.4f} | GV_loss={loss_gv.item():8.4f}")
            else:
                print(f"Epoch {epoch} | V_loss={loss_v.item():8.4f}")

        # Optimize/update networks
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\nBest loss: {best_loss:.6f}")

        # Save best pretrained weights (state_dict)
        if save_v_path is not None:
            torch.save(best_model_state, save_v_path)
            print(f"Saved pretrained V_net to: {save_v_path}")
        if (control_net is not None) and (best_control_state is not None) and (save_control_path is not None):
            torch.save(best_control_state, save_control_path)
            print(f"Saved pretrained Controller_net to: {save_control_path}")

    networks_trained = ["V"]
    if GV_net is not None:
        networks_trained.append("GV")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")


def resolve_mode_paths(mode: str):
    if mode == "none":
        return OUTPUT_DIR, Path("results"), Path("training_progress")
    if mode == "energy":
        out_dir = OUTPUT_DIR / "energy_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, out_dir / "results", out_dir / "training_progress"
    if mode == "beta":
        out_dir = OUTPUT_DIR / "beta_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, out_dir / "results", out_dir / "training_progress"
    raise ValueError(f"Unsupported curriculum mode: {mode}")


def apply_input_scale_to_models(params, V_net, GV_net):
    with torch.no_grad():
        v_scale = torch.as_tensor(
            params.network.input_scale,
            dtype=V_net.input_scale.dtype,
            device=V_net.input_scale.device,
        )
        if V_net.input_scale.numel() == v_scale.numel():
            V_net.input_scale.copy_(v_scale)
        if GV_net.input_scale.numel() == v_scale.numel():
            GV_net.input_scale.copy_(v_scale.to(device=GV_net.input_scale.device, dtype=GV_net.input_scale.dtype))
            GV_net.input_scale_sq.copy_(GV_net.input_scale * GV_net.input_scale)
        if hasattr(GV_net, "_cached_scale_D"):
            GV_net._cached_scale_D = None
        if hasattr(GV_net, "_cached_inv_scale"):
            GV_net._cached_inv_scale = None
        if hasattr(GV_net, "_cached_inv_scale_sq"):
            GV_net._cached_inv_scale_sq = None


def load_model_states_from_checkpoint(ckpt_path: Path, V_net, GV_net, u_nn):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    V_net.load_state_dict(ckpt["V_state_dict"], strict=True)
    if ckpt.get("GV_state_dict") is not None:
        GV_net.load_state_dict(ckpt["GV_state_dict"], strict=False)
    if ckpt.get("control_state_dict") is not None:
        u_nn.load_state_dict(ckpt["control_state_dict"], strict=True)
    return ckpt


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
    if loaded_mode:
        print("\n" + "=" * 20)
        print("Final Evaluation (loaded)")
        print("=" * 20)
    else:
        print("\n" + "=" * 20)
        print("Final Evaluation")
        print("=" * 20)

    latest_sat_beta_ra = getattr(params.training, "latest_sat_beta_ra", None)
    if latest_sat_beta_ra is not None:
        params.constraints.beta_ra = float(latest_sat_beta_ra)
        tag = "[LoadMode]" if loaded_mode else "[FinalEval]"
        print(f"{tag} Using latest SAT beta_ra={params.constraints.beta_ra:.4f}")

    if results is None:
        results = evaluate_constraints(
            V_net,
            GV_net,
            region_cells,
            beta_ra=params.constraints.beta_ra,
            device=params.training.device,
        )
    print_constraint_summary(results)

    print("\n" + "=" * 20)
    print("Visualizations (loaded)" if loaded_mode else "Visualizations")
    print("=" * 20)
    if loaded_mode:
        log_loaded_training_epochs(loss_history)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to resume checkpoint (.pth) saved at first SAT")
    parser.add_argument("--load_model_only_checkpoint", type=str, default="",
                        help="Path to checkpoint (.pth) to load V/GV/controller only (fresh optimizer/LR)")
    parser.add_argument("--curriculum_mode", type=str, default=None, choices=["beta", "energy"],
                        help="Curriculum mode: beta (increase beta_ra) or energy (decrease energy_max across stages)")
    parser.add_argument("--energy_max", type=float, default=None,
                        help="Override energy upper bound for this run")
    parser.add_argument("--energy_max_step", type=float, default=0.1,
                        help="Energy curriculum decrement step used inside trainer loop")
    parser.add_argument("--energy_max_min", type=float, default=None,
                        help="Minimum energy_max target for energy curriculum")
    args = parser.parse_args()

    print("="*20)
    print("2D Inverted Pendulum Synthesis")
    print("="*20)

    training_time_result = None

    mode = (args.curriculum_mode if args.curriculum_mode is not None else "none")
    active_output_dir, active_results_dir, active_progress_dir = resolve_mode_paths(mode)

    # === Hyperparameters ===
    params = Hyperparameters.default()

    # Toggle energy-dependent certificate: False -> V(x), True -> V(E, x)
    params.include_time = False
    params.include_energy = True
    state_dim = 2

    energy_max = float(args.energy_max) if args.energy_max is not None else 1.0
    energy_range = np.array([[0.0, energy_max]], dtype=np.float32)
    params.network.n_inputs = state_dim + (1 if params.include_energy else 0)
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    pi = np.pi
    if params.include_energy:
        params.network.input_scale = [energy_max, 2*pi, 20.0]
    else:
        params.network.input_scale = [2*pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0

    # Weights-based discretization in [t, x1, x2].
    # Example: (1,1,1) -> approximately uniform splits across dimensions.
    params.discretization.axis_weights = [0.2, 1.0, 1.0]
    params.discretization.max_region_budget = 15000      # shared for init/goal/unsafe/outside
    params.discretization.max_generator_budget = 1000    # separate generator budget

    params.constraints.beta_ra = 5.0 # max beta_ra = 20.0 by default
    params.constraints.beta_increment = 0.5

    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 5000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1200
    params.training.curriculum_mode = mode
    params.training.time_user_stop = False
    params.training.energy_max_step = float(args.energy_max_step)
    params.training.energy_max_min = (float(args.energy_max_min) if args.energy_max_min is not None else None)
    if mode == "none":
        params.training.resume_checkpoint_path = str(OUTPUT_DIR / "resume_checkpoint.pth")
    else:
        # Keep a mode-specific rolling checkpoint path so multi-stage retraining can chain.
        params.training.resume_checkpoint_path = str(OUTPUT_DIR / f"{mode}_latest_resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(active_output_dir)
    params.training.progress_output_dir = str(active_progress_dir)

    if params.include_energy:
        print(f"[Energy] Current energy_max = {energy_max}")

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
    params.refinement.gv_generator.merge_relax_margin = -500.0
    params.refinement.refine_interval_after_first_sat = 500 # increase the refinement frequecny for all region after first SAT

    # === Dynamics ===
    rl_policy_net = InvertControlNN()
    u_nn = WrapperConterlNN(rl_policy_net)

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        g = 9.81
        L = 0.5
        b = 0.1
        m = 0.15
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x2
        f2 = (g/L)*torch.sin(x1) - (b/(m*L**2))*x2
        return torch.stack([f1, f2], dim=1)

    g_coeffs = torch.tensor([0.0, 0.2], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        base = g_coeffs.to(device=x.device, dtype=x.dtype)
        if x.dim() == 1:
            return base
        elif x.dim() == 2:
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)
        else:
            raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")

    # Additive disturbance on open-loop drift: f_ol(x) + w, w in [-d, d].
    drift_unc = torch.tensor([1.0, 1.0], dtype=torch.float32)  # per-dimension disturbance box
    f_ol_set = AdditiveBoxSetDrift(f_ol, drift_unc).to(params.training.device)
    f_cl_module = ClosedLoopSetValuedDrift(f_ol_set, u_nn).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=state_dim)

    # === Regions ===
    init_range = np.array([[(3/4)*pi, (5/4)*pi], [-1.0, 1.0]], dtype=np.float32)
    goal_range = np.array([[-0.4*pi, 0.4*pi], [-4.0, 4.0]], dtype=np.float32)
    unsafe_down1 = np.array([[-2*pi, -2*pi+0.5*pi], [-20.0, -10.0]], dtype=np.float32)
    unsafe_down2 = np.array([[2*pi-0.5*pi, 2*pi], [10.0, 20.0]], dtype=np.float32)
    unsafe_lb = np.array([[-2*pi, -2*pi+0.5], [-20.0, 20.0]], dtype=np.float32)
    unsafe_rb = np.array([[2*pi-0.5, 2*pi], [-20.0, 20.0]], dtype=np.float32)
    unsafe_tb = np.array([[-2*pi, 2*pi], [20.0-0.5, 20.0]], dtype=np.float32)
    unsafe_bb = np.array([[-2*pi, 2*pi], [-20.0, -20.0+0.5]], dtype=np.float32)
    full_range = np.array([[-2*pi, 2*pi], [-20.0, 20.0]], dtype=np.float32)

    def _with_energy(rng: np.ndarray) -> np.ndarray:
        return np.vstack((energy_range, rng)) if params.include_energy else rng

    init_range = _with_energy(init_range)
    if params.include_energy:
        init_range[0, :] = 0.0

    goal_range = _with_energy(goal_range)
    unsafe_down1 = _with_energy(unsafe_down1)
    unsafe_down2 = _with_energy(unsafe_down2)
    unsafe_lb = _with_energy(unsafe_lb)
    unsafe_rb = _with_energy(unsafe_rb)
    unsafe_tb = _with_energy(unsafe_tb)
    unsafe_bb = _with_energy(unsafe_bb)
    full_range = _with_energy(full_range)
    unsafe_range = np.vstack((unsafe_down1, unsafe_down2, unsafe_tb, unsafe_bb, unsafe_lb, unsafe_rb))
    terminal_unsafe_range = np.array([[energy_max, energy_max], [-2*pi, 2*pi], [-20.0, 20.0]], dtype=np.float32)
    unsafe_range = np.vstack((unsafe_range, terminal_unsafe_range))

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe_down1 = Region(unsafe_down1)
    unsafe_down2 = Region(unsafe_down2)
    unsafe_tb = Region(unsafe_tb)
    unsafe_bb = Region(unsafe_bb)
    unsafe_lb = Region(unsafe_lb)
    unsafe_rb = Region(unsafe_rb)
    unsafe_components = [unsafe_down1, unsafe_down2, unsafe_tb, unsafe_bb, unsafe_lb, unsafe_rb]
    if terminal_unsafe_range.shape[0] > 0:
        D_tot = int(full_range.shape[0])
        K_term = int(terminal_unsafe_range.shape[0] // D_tot)
        for i in range(K_term):
            box_i = terminal_unsafe_range[i * D_tot:(i + 1) * D_tot, :]
            unsafe_components.append(Region(box_i))
        print(f"[Energy] Added {K_term} terminal-energy unsafe boxes at E={energy_max}")
    unsafe = Region.union(*unsafe_components)
    full = Region(full_range)
    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # === Networks ===
    input_offset = [0.04, 0.0, 0.0]
    output_offset = np.float32(0.1)
    V_net = create_V(params.network,
        input_offset=input_offset, output_offset=output_offset
    )
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        input_offset=input_offset,
        include_time=False,
        include_energy=params.include_energy
    )

    bundle_path = active_output_dir / "eval_bundle.pth"

    if args.train == 1:
        cleanup_and_setup_directories([active_results_dir, active_progress_dir])
        enable_terminal_logging(active_output_dir / "terminal_log.txt", append=False)
        # === Discretization ===
        region_cells = discretize_regions(
            regions,
            params.discretization,
            use_radial_generator=False
        )
        # debug_print_all_region_cells(region_cells, label="Initial discretized region cells")

        training_start_time = time.time()
        resumed_from_checkpoint = False
        model_loaded_only = False

        if args.load_model_only_checkpoint:
            ckpt_path = Path(args.load_model_only_checkpoint)
            _ = load_model_states_from_checkpoint(ckpt_path, V_net, GV_net, u_nn)
            model_loaded_only = True
            print("\n" + "="*20)
            print("Loaded model-only checkpoint")
            print("="*20)
            print(f"Path: {ckpt_path}")
            print("Optimizer/scheduler state NOT loaded (fresh optimizer and LR).")

        if args.resume_checkpoint:
            ckpt_path = Path(args.resume_checkpoint)
            ckpt = load_model_states_from_checkpoint(ckpt_path, V_net, GV_net, u_nn)

            if ckpt.get("region_cells") is not None:
                region_cells = {
                    k: [(lo.to(params.training.device), hi.to(params.training.device)) for (lo, hi) in v]
                    for k, v in ckpt["region_cells"].items()
                }
                debug_print_all_region_cells(region_cells, label="Resumed region cells")

            params.training.resume_optimizer_state_dict = ckpt.get("optimizer_state_dict", None)
            params.training.resume_scheduler_state_dict = ckpt.get("scheduler_state_dict", None)

            ckpt_hparams = ckpt.get("hyperparameters", None)
            if isinstance(ckpt_hparams, dict):
                ckpt_network = ckpt_hparams.get("network", {})
                if "input_scale" in ckpt_network:
                    params.network.input_scale = list(ckpt_network["input_scale"])
                    apply_input_scale_to_models(params, V_net, GV_net)
                ckpt_constraints = ckpt_hparams.get("constraints", {})
                if "beta_ra" in ckpt_constraints:
                    params.constraints.beta_ra = float(ckpt_constraints["beta_ra"])

            # Keep region metadata aligned with resumed cells/horizon.
            regions = sync_regions_from_region_cells(regions, region_cells, params)

            resumed_from_checkpoint = True
            print("\n" + "="*20)
            print("Resumed from checkpoint")
            print("="*20)
            print(f"Path: {ckpt_path}")
            print(f"Loaded beta_ra: {params.constraints.beta_ra}")
            if getattr(params, "include_energy", False) and len(params.network.input_scale) > 0:
                print(f"Loaded energy_max from checkpoint: {float(params.network.input_scale[0])}")

        if (not resumed_from_checkpoint) and (not model_loaded_only) and params.training.enable_pretraining:
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
                save_control_path=active_output_dir / "controller_pretrained.pth"
            )
        elif (not resumed_from_checkpoint) and (not model_loaded_only):
            print("\n" + "="*20)
            print("Pre-training disabled; loading local pretrained checkpoints")
            print("="*20)
            v_pretrained = active_output_dir / "V_pretrained.pth"
            c_pretrained = active_output_dir / "controller_pretrained.pth"
            if not v_pretrained.exists():
                v_pretrained = OUTPUT_DIR / "V_pretrained.pth"
            if not c_pretrained.exists():
                c_pretrained = OUTPUT_DIR / "controller_pretrained.pth"
            V_net.load_state_dict(torch.load(v_pretrained, map_location="cpu"))
            u_nn.load_state_dict(torch.load(c_pretrained, map_location="cpu"))

        def create_scheduler(optimizer):
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=2000,
                gamma=0.95
            )

        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            control_net=u_nn,
            create_scheduler=create_scheduler,
            start_time=training_start_time
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
        training_time_result = total_training_time

    else:
        print("\n" + "="*20)
        print("Loading Bundle")
        print("="*20)
        print(bundle_path)

        bundle = load_eval_bundle(bundle_path, map_location="cpu")

        # In load mode, fully restore hyperparameters (including network config) from bundle.
        params = Hyperparameters.from_dict(bundle["hyperparameters"])
        print(
            f"[LoadMode] Loaded network config: "
            f"n_inputs={params.network.n_inputs}, "
            f"hidden=({params.network.n_hidden_1}, {params.network.n_hidden_2}), "
            f"input_scale={params.network.input_scale}"
        )
        latest_sat_beta_ra = bundle.get("latest_sat_beta_ra", None)
        if latest_sat_beta_ra is not None:
            params.constraints.beta_ra = float(latest_sat_beta_ra)
            print(f"[LoadMode] Using saved latest SAT beta_ra={params.constraints.beta_ra:.4f}")
        
        # In load mode, always use regions saved in eval bundle.
        regions = Regions.from_dict(bundle["regions"])
        region_cells = bundle["region_cells"]

        V_net.load_state_dict(bundle["V_state_dict"], strict=True)
        u_nn.load_state_dict(bundle["control_state_dict"], strict=True)
        GV_net.load_state_dict(bundle["GV_state_dict"], strict=False)
        apply_input_scale_to_models(params, V_net, GV_net)

        region_cells = {
            k: [(lo.to(params.training.device), hi.to(params.training.device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }
        loss_history = bundle["loss_history"]
        refinement_epochs = bundle["refinement_epochs"]

        results = bundle.get("final_results", None)
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
        evaluate_and_visualize(
            V_net=V_net,
            GV_net=GV_net,
            regions=regions,
            region_cells=region_cells,
            params=params,
            active_results_dir=active_results_dir,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            final_beta_s=None,
            results=results,
            loaded_mode=True,
        )

    return training_time_result


if __name__ == '__main__':
    main()
