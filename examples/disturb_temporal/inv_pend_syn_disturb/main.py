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
- Time curriculum (`time_horizon` decreases in-loop):
  `python main.py --train 1 --curriculum_mode time --resume_checkpoint outputs/resume_checkpoint.pth --time_horizon_step 0.5 --time_horizon_min 1.0`

Optional stop/resume:
- Beta: during training, type `stop` in terminal.
  Resume with:
  `python main.py --train 1 --curriculum_mode beta --resume_checkpoint outputs/beta_latest_resume_checkpoint.pth`
- Time: during training, type `stop` in terminal.
  Resume with:
  `python main.py --train 1 --curriculum_mode time --resume_checkpoint outputs/time_latest_resume_checkpoint.pth --time_horizon_step 0.5 --time_horizon_min 1.0`

Visualization:
- Nominal:
  `python main.py --train 0`
- Beta:
  `python main.py --train 0 --curriculum_mode beta`
- Time:
  `python main.py --train 0 --curriculum_mode time`

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
- Time mode:
  - run artifacts in `outputs/time_runs/...` (single folder across stages):
    - `outputs/time_runs/eval_bundle.pth`
    - `outputs/time_runs/terminal_log.txt` (appended across stages)
    - `outputs/time_runs/results/`
    - `outputs/time_runs/training_progress/`
  - rolling checkpoint `outputs/time_latest_resume_checkpoint.pth`
"""
import argparse
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
WARMSTART_DIR = HERE / "outputs_warm"
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


def warm_start_from_spatial_bundle(
    V_net,
    control_net,
    include_time: bool,
    bundle_path: Path,
):
    """
    Warm-start temporal model from a previously trained spatial bound-training bundle.
    - V_net: copy all matching tensors; if include_time and input dim increased by 1,
      map old layer1 weights to spatial columns (new[:, 1:] <- old[:, :]).
    - control_net: load directly (same architecture).
    """
    if not bundle_path.exists():
        print(f"[WarmStart] Bundle not found: {bundle_path}")
        return False

    bundle = torch.load(bundle_path, map_location="cpu")
    v_old = bundle.get("V_state_dict", None)
    u_old = bundle.get("control_state_dict", None)
    if v_old is None:
        print(f"[WarmStart] No V_state_dict in bundle: {bundle_path}")
        return False

    v_new = V_net.state_dict()
    copied = 0
    for k, v in v_old.items():
        if k not in v_new:
            continue
        if k == "layer1.weight":
            tgt = v_new[k]
            if tgt.shape == v.shape:
                v_new[k] = v
                copied += 1
            elif include_time and tgt.shape[0] == v.shape[0] and tgt.shape[1] == v.shape[1] + 1:
                # Keep time column as initialized; copy spatial columns.
                tgt[:, 1:] = v
                v_new[k] = tgt
                copied += 1
            continue

        if v_new[k].shape == v.shape:
            v_new[k] = v
            copied += 1

    V_net.load_state_dict(v_new, strict=False)
    print(f"[WarmStart] Loaded V with {copied} matching tensors from: {bundle_path}")

    if (u_old is not None) and (control_net is not None):
        control_net.load_state_dict(u_old, strict=True)
        print(f"[WarmStart] Loaded controller from: {bundle_path}")
    else:
        print("[WarmStart] Controller state missing; skipped controller load.")

    return True


def add_terminal_ftr_unsafe_boxes(
    *,
    include_time: bool,
    time_horizon: float,
    full_range_spatial: np.ndarray,
    goal_range_spatial: np.ndarray,
    unsafe_ranges_spatial,
) -> np.ndarray:
    """
    Build terminal-time unsafe set for finite-time reach-avoid:
        {(t, x): t = T, x in X \\ (Goal U Unsafe)}

    Returns stacked boxes in shape (K*(D+1), 2) if include_time=True,
    otherwise empty array with shape (0, 2).
    """
    if not include_time:
        return np.zeros((0, 2), dtype=np.float32)

    def _as_box_list(arr: np.ndarray, D: int):
        t = np.asarray(arr, dtype=np.float32)
        if t.ndim == 2 and t.shape == (D, 2):
            return [t.copy()]
        if t.ndim == 2 and t.shape[1] == 2 and (t.shape[0] % D == 0):
            k = t.shape[0] // D
            return [t[i * D:(i + 1) * D, :].copy() for i in range(k)]
        if t.ndim == 3 and t.shape[1:] == (D, 2):
            return [t[i].copy() for i in range(t.shape[0])]
        raise ValueError(f"Expected box array as (D,2), (K*D,2), or (K,D,2); got {t.shape}")

    def _subtract_one_box(base_box: np.ndarray, cut_box: np.ndarray):
        """
        Exact subtraction of one axis-aligned box:
            base_box \\ cut_box = union of axis-aligned boxes.
        """
        D = base_box.shape[0]
        inter_lo = np.maximum(base_box[:, 0], cut_box[:, 0])
        inter_hi = np.minimum(base_box[:, 1], cut_box[:, 1])
        if np.any(inter_lo >= inter_hi):
            return [base_box]

        segments = []
        for d in range(D):
            segs_d = []
            a0, a1 = base_box[d, 0], base_box[d, 1]
            i0, i1 = inter_lo[d], inter_hi[d]
            if a0 < i0:
                segs_d.append((a0, i0, 0))
            segs_d.append((i0, i1, 1))
            if i1 < a1:
                segs_d.append((i1, a1, 2))
            segments.append(segs_d)

        out = []
        def _dfs(dim, idx_codes, cur):
            if dim == D:
                if all(c == 1 for c in idx_codes):
                    return
                out.append(cur.copy())
                return
            for lo, hi, code in segments[dim]:
                if hi <= lo:
                    continue
                cur[dim, 0] = lo
                cur[dim, 1] = hi
                _dfs(dim + 1, idx_codes + [code], cur)

        _dfs(0, [], np.zeros_like(base_box))
        return out

    D = int(full_range_spatial.shape[0])
    full_box = np.asarray(full_range_spatial, dtype=np.float32).copy()
    goal_boxes = _as_box_list(goal_range_spatial, D)
    unsafe_boxes = []
    for u in unsafe_ranges_spatial:
        unsafe_boxes.extend(_as_box_list(u, D))
    obstacles = goal_boxes + unsafe_boxes

    remain = [full_box]
    for obs in obstacles:
        nxt = []
        for b in remain:
            nxt.extend(_subtract_one_box(b, obs))
        remain = nxt
        if not remain:
            break

    if not remain:
        return np.zeros((0, 2), dtype=np.float32)

    t_box = np.array([[float(time_horizon), float(time_horizon)]], dtype=np.float32)
    terminal_boxes = [np.vstack((t_box, b)) for b in remain]
    return np.vstack(terminal_boxes).astype(np.float32)


def debug_print_regions_ranges(regions: Regions, label: str = "Regions") -> None:
    """Print min/max bounds for each region in a Regions object."""
    print(f"{label} bounds:")
    for name in ("init", "goal", "unsafe", "full"):
        reg = getattr(regions, name, None)
        if reg is None:
            continue
        b = np.asarray(reg.bounds, dtype=np.float32)
        print(f"  {name}:")
        for d in range(b.shape[0]):
            print(f"    dim {d}: [{b[d, 0]:.6g}, {b[d, 1]:.6g}]")
        if getattr(reg, "is_union", False) and getattr(reg, "components", None):
            print(f"    union components: {len(reg.components)}")


def sync_regions_from_region_cells(regions: Regions, region_cells: dict, params) -> Regions:
    """
    Rebuild region metadata from loaded cells and clip time bounds to current horizon.
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

    if bool(getattr(params, "include_time", False)):
        scales = getattr(params.network, "input_scale", None)
        if isinstance(scales, (list, tuple)) and len(scales) >= 1:
            h = float(scales[0])
            regions.init.bounds[0, 0] = 0.0
            regions.init.bounds[0, 1] = 0.0
            for r in (regions.goal, regions.unsafe, regions.full):
                r.bounds[0, 0] = min(float(r.bounds[0, 0]), 0.0)
                r.bounds[0, 1] = min(float(r.bounds[0, 1]), h)

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
    lambda_w = 0.1,
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


def main(benchmark_mode=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    parser.add_argument("--benchmark", type=int, default=0, choices=[0, 1],
                        help="1: run benchmark (5 runs), 0: normal run")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to resume checkpoint (.pth) saved at first SAT")
    parser.add_argument("--curriculum_mode", type=str, default=None, choices=["beta", "time"],
                        help="Curriculum mode: beta (increase beta_ra) or time (decrease time_horizon across stages)")
    parser.add_argument("--time_horizon", type=float, default=None,
                        help="Override time horizon for this run")
    parser.add_argument("--time_horizon_step", type=float, default=1.0,
                        help="Time curriculum decrement step used inside trainer loop")
    parser.add_argument("--time_horizon_min", type=float, default=None,
                        help="Minimum time horizon target (used by time curriculum visualization gating)")
    args = parser.parse_args()

    if benchmark_mode:
        args.benchmark = 1

    print("="*20)
    print("2D Inverted Pendulum Synthesis")
    print("="*20)

    training_time_result = None

    mode = (args.curriculum_mode if args.curriculum_mode is not None else "none")
    if mode == "none":
        active_output_dir = OUTPUT_DIR
        active_results_dir = Path("results")
        active_progress_dir = Path("training_progress")
    elif mode == "time":
        # Keep all time-curriculum stages in one folder (latest artifacts overwrite).
        active_output_dir = OUTPUT_DIR / "time_runs"
        active_results_dir = active_output_dir / "results"
        active_progress_dir = active_output_dir / "training_progress"
        active_output_dir.mkdir(parents=True, exist_ok=True)
    elif mode == "beta":
        # Keep all beta-curriculum runs in one folder (latest artifacts overwrite).
        active_output_dir = OUTPUT_DIR / "beta_runs"
        active_results_dir = active_output_dir / "results"
        active_progress_dir = active_output_dir / "training_progress"
        active_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Unsupported curriculum mode: {mode}")

    # === Hyperparameters ===
    params = Hyperparameters.default()

    # Toggle time-dependent certificate: False -> V(x), True -> V(t, x)
    params.include_time = True
    state_dim = 2

    # In load/visualization mode, infer horizon from the selected bundle unless explicitly overridden.
    # Skip this for time mode: we load everything directly from eval_bundle in the load branch.
    if (args.train == 0) and (args.time_horizon is None) and (mode != "time"):
        infer_bundle_path = active_output_dir / "eval_bundle.pth"
        if infer_bundle_path.exists():
            try:
                infer_bundle = torch.load(infer_bundle_path, map_location="cpu")
                infer_hparams = infer_bundle.get("hyperparameters", {})
                infer_scale = infer_hparams.get("network", {}).get("input_scale", None)
                if isinstance(infer_scale, (list, tuple)) and len(infer_scale) >= 1:
                    args.time_horizon = float(infer_scale[0])
                    print(f"[LoadMode] Inferred time_horizon={args.time_horizon} from {infer_bundle_path}")
            except Exception as e:
                print(f"[LoadMode] Could not infer time_horizon from {infer_bundle_path}: {e}")

    time_horizon = float(args.time_horizon) if args.time_horizon is not None else 10.0
    time_range = np.array([[0.0, time_horizon]], dtype=np.float32)
    params.network.n_inputs = state_dim + (1 if params.include_time else 0)
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    pi = np.pi
    if params.include_time:
        params.network.input_scale = [time_horizon, 2*pi, 20.0]
    else:
        params.network.input_scale = [2*pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0

    # Weights-based discretization in [t, x1, x2].
    # Example: (1,1,1) -> approximately uniform splits across dimensions.
    params.discretization.axis_weights = [0.5, 1.0, 1.0]
    params.discretization.max_region_budget = 15000      # shared for init/goal/unsafe/outside
    params.discretization.max_generator_budget = 1000    # separate generator budget

    params.constraints.beta_ra = 5.0 # max beta_ra = 20.0 by default
    params.constraints.beta_increment = 0.5

    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = False
    params.training.pretrain_epochs = 5000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1200
    params.training.curriculum_mode = mode
    params.training.time_user_stop = False
    params.training.time_horizon_step = float(args.time_horizon_step)
    params.training.time_horizon_min = (float(args.time_horizon_min) if args.time_horizon_min is not None else None)
    if mode == "none":
        params.training.resume_checkpoint_path = str(OUTPUT_DIR / "resume_checkpoint.pth")
    else:
        # Keep a mode-specific rolling checkpoint path so multi-stage retraining can chain.
        params.training.resume_checkpoint_path = str(OUTPUT_DIR / f"{mode}_latest_resume_checkpoint.pth")
    params.training.eval_bundle_output_dir = str(active_output_dir)
    params.training.progress_output_dir = str(active_progress_dir)

    if mode == "time":
        print(f"[TimeCurriculum] Current time_horizon = {time_horizon}")

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
    params.refinement.v_goal.max_cells = 50000
    params.refinement.v_goal.N_to_refine = 100
    params.refinement.v_goal.enable_merging = False

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_interval = 500
    params.refinement.gv_generator.late_epoch_threshold = 3500
    params.refinement.gv_generator.refine_interval_late = 500
    params.refinement.gv_generator.refine_factor = 2
    params.refinement.gv_generator.max_cells = 50000
    params.refinement.gv_generator.N_to_refine = 100
    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -500.0
    params.refinement.refine_interval_after_first_sat = 100 # increase the refinement frequecny for all region after first SAT

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
    goal_range_spatial = goal_range.copy()
    full_range_spatial = full_range.copy()
    unsafe_ranges_spatial = [
        unsafe_down1.copy(),
        unsafe_down2.copy(),
        unsafe_tb.copy(),
        unsafe_bb.copy(),
        unsafe_lb.copy(),
        unsafe_rb.copy(),
    ]

    def _with_time(rng: np.ndarray) -> np.ndarray:
        return np.vstack((time_range, rng)) if params.include_time else rng

    init_range = _with_time(init_range)
    init_range[0, :] = 0.0 # assume initial time (the first input) is 0.0

    goal_range = _with_time(goal_range)
    unsafe_down1 = _with_time(unsafe_down1)
    unsafe_down2 = _with_time(unsafe_down2)
    unsafe_lb = _with_time(unsafe_lb)
    unsafe_rb = _with_time(unsafe_rb)
    unsafe_tb = _with_time(unsafe_tb)
    unsafe_bb = _with_time(unsafe_bb)
    full_range = _with_time(full_range)
    unsafe_range = np.vstack((unsafe_down1, unsafe_down2, unsafe_tb, unsafe_bb, unsafe_lb, unsafe_rb))
    terminal_unsafe_range = add_terminal_ftr_unsafe_boxes(
        include_time=params.include_time,
        time_horizon=time_horizon,
        full_range_spatial=full_range_spatial,
        goal_range_spatial=goal_range_spatial,
        unsafe_ranges_spatial=unsafe_ranges_spatial,
    )
    if terminal_unsafe_range.shape[0] > 0:
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
        print(f"[FTRA] Added {K_term} terminal-time unsafe boxes at t={time_horizon}")

    unsafe = Region.union(*unsafe_components)
    full = Region(full_range)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # === Networks ===
    V_net = create_V(params.network)
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        include_time=params.include_time
    )

    # === Discretization ===
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=False
    )

    bundle_path = active_output_dir / "eval_bundle.pth"

    if args.train == 1:
        cleanup_and_setup_directories([active_results_dir, active_progress_dir])
        # Always start a fresh terminal log per run (nominal/beta/time).
        enable_terminal_logging(active_output_dir / "terminal_log.txt", append=False)

        training_start_time = time.time()
        resumed_from_checkpoint = False

        if args.resume_checkpoint:
            ckpt_path = Path(args.resume_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            V_net.load_state_dict(ckpt["V_state_dict"], strict=True)
            if ckpt.get("GV_state_dict") is not None:
                GV_net.load_state_dict(ckpt["GV_state_dict"], strict=False)
            if ckpt.get("control_state_dict") is not None:
                u_nn.load_state_dict(ckpt["control_state_dict"], strict=True)

            if ckpt.get("region_cells") is not None:
                region_cells = {
                    k: [(lo.to(params.training.device), hi.to(params.training.device)) for (lo, hi) in v]
                    for k, v in ckpt["region_cells"].items()
                }

            params.training.resume_optimizer_state_dict = ckpt.get("optimizer_state_dict", None)
            params.training.resume_scheduler_state_dict = ckpt.get("scheduler_state_dict", None)

            ckpt_hparams = ckpt.get("hyperparameters", None)
            if isinstance(ckpt_hparams, dict):
                ckpt_network = ckpt_hparams.get("network", {})
                if "input_scale" in ckpt_network:
                    params.network.input_scale = list(ckpt_network["input_scale"])
                    # Keep loaded models consistent with resumed horizon/normalization.
                    with torch.no_grad():
                        v_scale = torch.as_tensor(params.network.input_scale, dtype=V_net.input_scale.dtype, device=V_net.input_scale.device)
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
            if params.include_time and len(params.network.input_scale) > 0:
                print(f"Loaded time_horizon from checkpoint: {float(params.network.input_scale[0])}")

        if (not resumed_from_checkpoint) and params.training.enable_pretraining:
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
                device=params.training.device,
                control_net=u_nn,
                n_each=params.training.pretrain_n_samples,
                save_v_path=active_output_dir / "V_pretrained.pth",
                save_control_path=active_output_dir / "controller_pretrained.pth"
            )
        elif not resumed_from_checkpoint:
            print("\n" + "="*20)
            print("Warm-start from previous bound model without time-dependency")
            print("="*20)
            spatial_bundle_path = WARMSTART_DIR / "eval_bundle.pth"
            warm_start_ok = warm_start_from_spatial_bundle(
                V_net=V_net,
                control_net=u_nn,
                include_time=params.include_time,
                bundle_path=spatial_bundle_path,
            )

            if not warm_start_ok:
                print("[WarmStart] Falling back to local pretrained checkpoints.")
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

        user_stop_requested = bool(getattr(params.training, "user_stop_requested", False))
        if mode in {"beta", "time"} and user_stop_requested:
            print("\n" + "="*20)
            print("Loading Latest SAT Bundle After User Stop")
            print("="*20)
            sat_bundle = load_eval_bundle(bundle_path, map_location="cpu")

            params = Hyperparameters.from_dict(sat_bundle["hyperparameters"])
            latest_sat_beta_ra = sat_bundle.get("latest_sat_beta_ra", None)
            if latest_sat_beta_ra is not None:
                params.constraints.beta_ra = float(latest_sat_beta_ra)

            regions = Regions.from_dict(sat_bundle["regions"])
            region_cells = sat_bundle["region_cells"]

            V_net = create_V(params.network).to(params.training.device)
            V_net.load_state_dict(sat_bundle["V_state_dict"])

            rl_policy_net = InvertControlNN()
            u_nn = WrapperConterlNN(rl_policy_net).to(params.training.device)
            if sat_bundle["control_state_dict"] is not None:
                u_nn.load_state_dict(sat_bundle["control_state_dict"])

            f_cl_module = ClosedLoopSetValuedDrift(f_ol_set, u_nn).to(params.training.device)
            dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=state_dim)
            GV_net = create_GV(
                V_net=V_net,
                dynamics=dynamics,
                network_config=params.network,
                include_time=params.include_time
            ).to(params.training.device)
            if sat_bundle.get("GV_state_dict", None) is not None:
                GV_net.load_state_dict(sat_bundle["GV_state_dict"], strict=False)

            region_cells = {
                k: [(lo.to(params.training.device), hi.to(params.training.device)) for (lo, hi) in v]
                for k, v in region_cells.items()
            }
            loss_history = sat_bundle.get("loss_history", loss_history)
            refinement_epochs = sat_bundle.get("refinement_epochs", refinement_epochs)
            if sat_bundle.get("final_beta_s", None) is not None:
                final_beta_s = sat_bundle["final_beta_s"]

        print("\n" + "="*20)
        print("Final Evaluation")
        print("="*20)

        latest_sat_beta_ra = getattr(params.training, "latest_sat_beta_ra", None)
        if latest_sat_beta_ra is not None:
            params.constraints.beta_ra = float(latest_sat_beta_ra)
            print(f"[FinalEval] Using latest SAT beta_ra={params.constraints.beta_ra:.4f}")

        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_ra=params.constraints.beta_ra,
            device=params.training.device
        )
        print_constraint_summary(results)

        print("\n" + "="*20)
        print("Visualizations")
        print("="*20)
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
            output_dir=str(active_results_dir)
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
        debug_print_regions_ranges(regions, label="Loaded regions")
        region_cells = bundle["region_cells"]
        debug_print_region_bounds(region_cells["goal"], label="goal cells")

        V_net = create_V(params.network).to(params.training.device)
        V_net.load_state_dict(bundle["V_state_dict"])

        rl_policy_net = InvertControlNN()
        u_nn = WrapperConterlNN(rl_policy_net).to(params.training.device)
        u_nn.load_state_dict(bundle["control_state_dict"])

        # Rebuild additive-disturbance dynamics in loaded mode.
        f_cl_module = ClosedLoopSetValuedDrift(f_ol_set, u_nn).to(params.training.device)
        dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=state_dim)
        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network,
            include_time=params.include_time
        )

        region_cells = {
            k: [(lo.to(params.training.device), hi.to(params.training.device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }
        loss_history = bundle["loss_history"]
        refinement_epochs = bundle["refinement_epochs"]

        results = bundle.get("final_results", None)
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_ra=params.constraints.beta_ra,
                device=params.training.device
            )

        print("\n" + "="*20)
        print("Final Evaluation (loaded)")
        print("="*20)
        print_constraint_summary(results)

        print("\n" + "="*20)
        print("Visualizations (loaded)")
        print("="*20)
        log_loaded_training_epochs(loss_history)

        create_summary_plots(
            V_net=V_net,
            GV_net=GV_net,
            regions=regions,
            region_cells=region_cells,
            beta_ra=params.constraints.beta_ra,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
            output_dir=str(active_results_dir)
        )

    return training_time_result


if __name__ == '__main__':
    import sys
    is_benchmark = '--benchmark=1' in sys.argv or '--benchmark' in sys.argv and '1' in sys.argv

    if is_benchmark:
        n_runs = 2
        times = []
        cells = []

        print("="*20)
        print(f"Benchmark: {n_runs} runs")
        print("="*20)

        for i in range(n_runs):
            print(f"\n*** Run {i+1}/{n_runs} ***")
            training_time = main(benchmark_mode=True)
            if training_time is not None:
                times.append(training_time)

            bundle_path = OUTPUT_DIR / "eval_bundle.pth"
            if bundle_path.exists():
                import torch
                bundle = torch.load(bundle_path, map_location='cpu')
                cell_counts = {
                    'init': len(bundle['region_cells']['init']),
                    'goal': len(bundle['region_cells']['goal']),
                    'unsafe': len(bundle['region_cells']['unsafe']),
                    'outside': len(bundle['region_cells']['outside']),
                    'generator': len(bundle['region_cells']['generator']),
                }
                cell_counts['total_v'] = cell_counts['init'] + cell_counts['goal'] + cell_counts['unsafe'] + cell_counts['outside']
                cell_counts['total'] = cell_counts['total_v'] + cell_counts['generator']
                cells.append(cell_counts)

        print("\n" + "="*20)
        print("Benchmark Results")
        print("="*20)
        avg_time = stats.mean(times)
        std_time = stats.stdev(times)
        print(f"Training time (pretrain+train): {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"  Individual times: {[f'{t:.2f}s' for t in times]}")

        if cells:
            categories = ['init', 'goal', 'unsafe', 'outside', 'generator', 'total_v', 'total']
            print("\nCell counts:")
            for cat in categories:
                values = [c[cat] for c in cells]
                avg = stats.mean(values)
                std = stats.stdev(values)
                print(f"  {cat:12s}: {avg:.0f} ± {std:.0f}")
    else:
        main()
