"""
2D Inverted Pendulum Control Synthesis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import argparse

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.hyperparameters import Hyperparameters
from src.dynamics import Dynamics, ClosedLoopDrift
from src.regions import Regions, Region
from src.network import create_V
from src.control_network import WrapperConterlNN, InvertControlNN
from src.phi_module import create_GV
from src.discretization import discretize_regions
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.training_utils import (
    compute_total_loss_bounds,
    compute_loss_goal_bounds,
    evaluate_constraints,
    print_loss_summary,
    merge_passing_neighbor_cells,
    print_constraint_summary,
    print_cell_counts,
    refine_failing_cells
)
from src.save_load_utils import save_eval_bundle, load_eval_bundle, log_loaded_training_epochs, enable_terminal_logging
from src.utils import cleanup_and_setup_directories, print_training_config
from src.visualization import (
    visualize_training_progress,
    create_summary_plots
)

# Set random seed immediately after imports (matching testing_simple3.py)
torch.manual_seed(0)


class LearnableBetaS(nn.Module):
    """
    Learnable beta_s parameter constrained to (0, 1) using sigmoid.
    """
    def __init__(self, initial_value: float = 0.6):
        super().__init__()
        # Use logit to initialize so that sigmoid(logit) = initial_value
        initial_logit = math.log(initial_value / (1.0 - initial_value))
        self.beta_s_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def forward(self):
        """Return beta_s constrained to (0, 1) via sigmoid."""
        return torch.sigmoid(self.beta_s_logit)

    @property
    def value(self):
        """Get the current value of beta_s."""
        return torch.sigmoid(self.beta_s_logit)
    
# def pretrain_network_samples(model, x_goal_range, x_unsafe_range, x_init_range, x_range,
#                                  params, GV_net=None, num_epochs=1000, lr=0.01, device='cpu',
#                                  control_net=None):
#     """Pre-train V network using MSE to auto-derived target values (minimal hyperparameters).

#     Creates tight value clusters for CROWN-friendly geometry.

#     V targets (derived from beta_s and beta_ra only):
#     - goal: 0.0
#     - outside: beta_s
#     - init: (beta_s + 1.0) / 2
#     - unsafe: beta_ra

#     Φ target (derived from beta_s):
#     - generator: -beta_s
#     """
#     print("\n" + "="*80)
#     print("PRE-TRAINING: MSE to Auto-Derived Targets (Minimal Hyperparameters)")
#     print("="*80)

#     opt_params = list(model.parameters())
#     if control_net is not None:
#         opt_params += list(control_net.parameters())
#     optimizer = torch.optim.Adam(opt_params, lr=lr)

#     beta_s = params.constraints.beta_s
#     beta_ra = params.constraints.beta_ra

#     print(f"  V targets: goal=0.0, outside={beta_s:.3f}, init={(beta_s + 1.0)/2:.3f}, unsafe={beta_ra:.3f}")
#     print(f"  Φ target: generator={-beta_s:.3f}")

#     best_loss = float('inf')
#     best_model_state = None
#     best_control_state = None

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         # Stratified sampling: sample each region separately for balanced coverage
#         n_per_region = 100

#         # Goal samples
#         x_goal = torch.rand(n_per_region, 2, device=device)
#         x_goal[:, 0] = x_goal[:, 0] * (x_goal_range[0, 1] - x_goal_range[0, 0]) + x_goal_range[0, 0]
#         x_goal[:, 1] = x_goal[:, 1] * (x_goal_range[1, 1] - x_goal_range[1, 0]) + x_goal_range[1, 0]
#         target_goal = torch.zeros(n_per_region, device=device)

#         # Init samples
#         x_init = torch.rand(n_per_region, 2, device=device)
#         x_init[:, 0] = x_init[:, 0] * (x_init_range[0, 1] - x_init_range[0, 0]) + x_init_range[0, 0]
#         x_init[:, 1] = x_init[:, 1] * (x_init_range[1, 1] - x_init_range[1, 0]) + x_init_range[1, 0]
#         target_init = torch.full((n_per_region,), (0.5 + 1.0) / 2, device=device)

#         # Unsafe samples (handle union of regions - sample each separately)
#         if x_unsafe_range.shape[0] == 4:
#             n_per_unsafe = n_per_region
#             x_unsafe1 = torch.rand(n_per_unsafe, 2, device=device)
#             x_unsafe1[:, 0] = x_unsafe1[:, 0] * (x_unsafe_range[0, 1] - x_unsafe_range[0, 0]) + x_unsafe_range[0, 0]
#             x_unsafe1[:, 1] = x_unsafe1[:, 1] * (x_unsafe_range[1, 1] - x_unsafe_range[1, 0]) + x_unsafe_range[1, 0]
#             x_unsafe2 = torch.rand(n_per_unsafe, 2, device=device)
#             x_unsafe2[:, 0] = x_unsafe2[:, 0] * (x_unsafe_range[2, 1] - x_unsafe_range[2, 0]) + x_unsafe_range[2, 0]
#             x_unsafe2[:, 1] = x_unsafe2[:, 1] * (x_unsafe_range[3, 1] - x_unsafe_range[3, 0]) + x_unsafe_range[3, 0]
#             x_unsafe = torch.cat([x_unsafe1, x_unsafe2], dim=0)
#         else:
#             x_unsafe = torch.rand(n_per_region, 2, device=device)
#             x_unsafe[:, 0] = x_unsafe[:, 0] * (x_unsafe_range[0, 1] - x_unsafe_range[0, 0]) + x_unsafe_range[0, 0]
#             x_unsafe[:, 1] = x_unsafe[:, 1] * (x_unsafe_range[1, 1] - x_unsafe_range[1, 0]) + x_unsafe_range[1, 0]
#         target_unsafe = torch.full((len(x_unsafe),), beta_ra, device=device)

#         # Outside/generator samples (larger sample since it's the biggest region)
#         x_outside = torch.rand(n_per_region, 2, device=device)
#         x_outside[:, 0] = x_outside[:, 0] * (x_range[0, 1] - x_range[0, 0]) + x_range[0, 0]
#         x_outside[:, 1] = x_outside[:, 1] * (x_range[1, 1] - x_range[1, 0]) + x_range[1, 0]
#         target_outside = torch.full((len(x_outside),), 0.5, device=device)

#         # Concatenate all samples
#         x = torch.cat([x_goal, x_init, x_unsafe, x_outside], dim=0)
#         target_v = torch.cat([target_goal, target_init, target_unsafe, target_outside], dim=0)
#         is_generator = torch.cat([
#             torch.zeros(len(x_goal), dtype=torch.bool, device=device),
#             torch.zeros(len(x_init), dtype=torch.bool, device=device),
#             torch.zeros(len(x_unsafe), dtype=torch.bool, device=device),
#             torch.ones(len(x_outside), dtype=torch.bool, device=device)
#         ], dim=0)

#         # V network loss
#         v_output = model(x).squeeze()
#         loss_v = F.mse_loss(v_output, target_v)

#         # GV (Φ) network loss
#         if GV_net is not None:
#             loss_phi = torch.tensor(0.0, device=device)
#             x_gen = x[is_generator]
#             if len(x_gen) > 0:
#                 phi_output = GV_net(x_gen).squeeze()
#                 target_phi = torch.full_like(phi_output, -beta_ra)
#                 loss_phi = F.mse_loss(phi_output, target_phi)

#         # # Combined loss (equal weighting)
#         total_loss = loss_v + loss_phi
#         # total_loss = loss_v

#         total_loss.backward()
#         optimizer.step()

#         # Track best
#         if total_loss.item() < best_loss:
#             best_loss = total_loss.item()
#             best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             if control_net is not None:
#                 best_control_state = {k: v.cpu().clone() for k, v in control_net.state_dict().items()}

#         if epoch % 100 == 0:
#             gv_info = f", Φ_loss={loss_phi.item():.6f}" if GV_net is not None else ""
#             print(f"  Epoch [{epoch}/{num_epochs}]: "
#                   f"V_loss={loss_v.item():.6f}{gv_info}, "
#                   f"Total={total_loss.item():.6f}")

#     # Restore best model
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)
#         if control_net is not None and best_control_state is not None:
#             control_net.load_state_dict(best_control_state)
#         print(f"\n  Best loss: {best_loss:.6f}")

#     print("="*80)
#     print("Pre-training complete (MSE-based, minimal hyperparameters).")
#     print("="*80 + "\n")

def pretrain_network_samples(
    model,
    x_goal_range,
    x_unsafe_range,
    x_init_range,
    x_range,
    params,
    GV_net=None,
    num_epochs=1000,
    lr=0.01,
    device='cpu',
    control_net=None,
    n_each: int = 50,   # samples per region per epoch
):
    """
    Pre-train V network using sampled points to match constraint structure.

    Supports x_unsafe_range as:
      1) (D,2) single box
      2) (K,D,2) union of K boxes
      3) (2*D,2) produced by np.vstack((box1, box2, ...))  <-- your case

    Losses:
      - full-range: enforce v(x) >= 0
      - init-range: enforce v(x) <= 1
      - unsafe-range: enforce v(x) >= pretrain_unsafe_target
      - goal-range: enforce v(x) <= pretrain_goal_target
      - others (outside goal ∪ unsafe): enforce v(x) >= pretrain_goal_target
      - optional phi loss on x_others: enforce phi(x) <= 0
    """
    print("\n" + "="*80)
    print("PRE-TRAINING: Constraint-Structured Initialization (Sample-Based)")
    print("="*80)

    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    if GV_net is not None:
        print("  GV (Φ) pre-training ENABLED (using provided GV_net)")
    else:
        print("  GV (Φ) pre-training DISABLED (no GV_net provided)")

    best_loss = float('inf')
    best_model_state = None
    best_control_state = None

    # Convert ranges to torch once: each is (D,2)
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t    = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    init_t    = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
    low  = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low

    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        """box: (D,2) -> samples: (N,D) uniform in box."""
        b_low = box[:, 0]
        b_high = box[:, 1]
        return torch.rand(N, D, device=device) * (b_high - b_low) + b_low

    def _in_box(x_batch: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """x_batch: (N,D), box: (D,2) -> mask: (N,)"""
        return ((x_batch >= box[:, 0]) & (x_batch <= box[:, 1])).all(dim=1)

    # -----------------------------
    # Unsafe region: allow union of boxes
    # -----------------------------
    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
        """
        Returns unsafe_boxes as torch.Tensor of shape (K,D,2).
        Accepts:
          - (D,2)
          - (K,D,2)
          - (K*D,2) from np.vstack((box1, box2, ...))
        """
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
        """
        Sample N points from union of K unsafe boxes using torch only.
        Picks a box index ~ Categorical(probs) where probs ∝ volume, then samples uniformly in that box.
        """
        if K_unsafe == 1:
            return _sample_in_box(unsafe_boxes[0], N)

        # probs ∝ volume
        widths = (unsafe_boxes[:, :, 1] - unsafe_boxes[:, :, 0]).clamp_min(0.0)  # (K,D)
        vols = widths.prod(dim=1)                                                # (K,)
        probs = vols / vols.sum().clamp_min(1e-12)                               # safe normalize

        # choose box for each sample (N,)
        choices = torch.multinomial(probs, num_samples=N, replacement=True)      # (N,)

        # gather bounds for chosen boxes: (N,D)
        b_low  = unsafe_boxes[choices, :, 0]
        b_high = unsafe_boxes[choices, :, 1]

        # uniform in chosen box
        u = torch.rand(N, D, device=device)
        return u * (b_high - b_low) + b_low

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # 1) full-range samples -> enforce v(x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # 2) init-range samples -> enforce v(x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # 3) unsafe-range samples -> enforce v(x) >= pretrain_unsafe_target
        x_unsafe = _sample_in_unsafe_union(n_each)
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.pretrain_unsafe_target - v_unsafe).sum()

        # 4) goal-range samples -> enforce v(x) <= pretrain_goal_target
        x_goal = _sample_in_box(goal_t, n_each)
        v_goal = model(x_goal).squeeze(-1)
        v_loss_inside_goal = F.relu(v_goal - params.constraints.pretrain_goal_target).sum()

        # 5) samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= pretrain_goal_target
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

        v_others = model(x_others).squeeze(-1)
        v_loss_others = F.relu(params.constraints.pretrain_goal_target - v_others).sum()

        # Total V loss
        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
            + v_loss_inside_goal
            + v_loss_others
        )

        # Phi loss on SAME x_others -> enforce phi(x) <= 0
        loss_phi = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_phi = x_others.detach().clone().requires_grad_(True)
            phi_output = GV_net(x_phi).squeeze(-1)
            loss_phi = F.relu(phi_output).sum()

        total_loss = loss_v + loss_phi

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track best
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        # Logging
        if epoch % 100 == 0:
            if GV_net is not None:
                print(
                    f"  Epoch [{epoch}/{num_epochs}]: "
                    f"V={loss_v.item():.6f} "
                    f"(full={v_loss_full.item():.3f}, init={v_loss_init.item():.3f}, "
                    f"unsafe={v_loss_unsafe.item():.3f}, goal={v_loss_inside_goal.item():.3f}, "
                    f"others={v_loss_others.item():.3f}), "
                    f"Φ={loss_phi.item():.6f}, Total={total_loss.item():.6f}"
                )
            else:
                print(
                    f"  Epoch [{epoch}/{num_epochs}]: "
                    f"V={loss_v.item():.6f} "
                    f"(full={v_loss_full.item():.3f}, init={v_loss_init.item():.3f}, "
                    f"unsafe={v_loss_unsafe.item():.3f}, goal={v_loss_inside_goal.item():.3f}, "
                    f"others={v_loss_others.item():.3f})"
                )

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\n  Best loss: {best_loss:.6f}")

    print("="*80)
    networks_trained = ["V"]
    if GV_net is not None:
        networks_trained.append("GV (Φ)")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")
    print("="*80 + "\n")


def train_network_bounds(
    V_net,
    GV_net,
    region_cells: dict,
    regions: Regions,
    params: Hyperparameters,
    device: str = 'cpu',
    visualize_interval: int = 5000,
    control_net: nn.Module = None
):
    """
    Train the value network using CROWN bounds.

    Args:
        V_net: Value network
        GV_net: Generator network
        region_cells: Dictionary of discretized cells
        regions: Regions object
        params: Hyperparameters
        device: Device for training
        visualize_interval: Interval for visualization (0 to disable)
        control_net: if this is provided, then we do [control synthesis]
    """
    print("\n" + "="*80)
    print("BOUND-BASED TRAINING (using CROWN)")
    print("="*80)

    # Move models to device
    V_net = V_net.to(device)
    # GV_net = GV_net.to(device)

    # Collect ALL cells in the same order as original (init, goal, unsafe, outside, generator)
    print("\nCollecting all cells for V network...")
    all_cells_V = []
    cell_counts_V = {}
    region_order_V = ['init', 'goal', 'unsafe', 'outside']  # Match original order

    for name in region_order_V:
        cells = region_cells[name]
        all_cells_V.extend(cells)
        cell_counts_V[name] = len(cells)
        print(f"  {name}: {len(cells)} cells")

    total_cells_V = len(all_cells_V)
    print(f"  Total V cells: {total_cells_V}")

    # Prepare ALL input bounds at once (matching original)
    print("\nPreparing concatenated input bounds for V network...")
    if total_cells_V > 0:
        input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device)
    else:
        input_lowers_all = torch.empty(0, 2, device=device)
        input_uppers_all = torch.empty(0, 2, device=device)

    # Create ONE big CROWN cache for ALL V cells (matching original!)
    print(f"\nInitializing CROWN cache for ALL {total_cells_V} V cells...")
    crown_cache_all = SymbolicCROWNCache(
        model=V_net,
        num_cells=total_cells_V,
        input_dim=2,
        device=device
    )

    # Create CROWN cache for generator (Phi) - separate cache
    crown_cache_phi = None
    input_lowers_gen = None
    input_uppers_gen = None
    if len(region_cells['generator']) > 0 and params.training.generator_weight > 0:
        print(f"\nCreating CROWN cache for 'generator' (Phi)...")
        print(f"  generator: {len(region_cells['generator'])} cells")
        crown_cache_phi = SymbolicCROWNCache_Phi(
            phi_module=GV_net,
            num_cells=len(region_cells['generator']),
            input_dim=2,
            device=device
        )
        # Prepare generator input bounds
        input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device)

    # Check if beta_s should be learnable
    learnable_beta_s = None
    beta_s_value = None

    if params.training.learnable_beta_s or params.constraints.beta_s is None:
        # Create learnable beta_s
        initial_beta_s = params.constraints.beta_s if params.constraints.beta_s is not None else 0.6
        learnable_beta_s = LearnableBetaS(initial_value=initial_beta_s).to(device)
        print(f"\nUsing LEARNABLE beta_s (initialized to {initial_beta_s})")

        # # Optimizer includes both V_net and learnable beta_s
        # optimizer = torch.optim.Adam(
        #     list(V_net.parameters()) + list(learnable_beta_s.parameters()),
        #     lr=params.training.learning_rate
        # )
        opt_params = list(V_net.parameters()) + list(learnable_beta_s.parameters())
    else:
        # Use constant beta_s
        beta_s_value = params.constraints.beta_s
        print(f"\nUsing CONSTANT beta_s = {beta_s_value}")

        # # Optimizer only for V_net
        # optimizer = torch.optim.Adam(
        #     V_net.parameters(),
        #     lr=params.training.learning_rate
        # )
        opt_params = list(V_net.parameters())

    # [control synthesis]
    if control_net is not None:
        opt_params += list(control_net.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)

    # Scheduler (matching testing_simple3.py)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # minimize the loss
        factor=0.5,           # reduce LR by half when plateau detected
        patience=300,         # wait 100 epochs of no improvement before reducing
        verbose=True,         # print when LR changes
        min_lr=1e-6,          # minimum learning rate
        threshold=1e-3        # minimum change to qualify as improvement
    )
    final_beta_s = None

    # Training loop
    loss_history = []
    refinement_epochs = {'outside': [], 'generator': []}
    start_time = time.time()

    # Constraint switching state
    locked_to_all_sum = False
    current_sum_constraint = None
    constraint_largest_counts = {}
    prev_total_loss = None  # Track previous epoch's total loss for switching logic

    for epoch in range(params.training.num_epochs):
        V_net.train()

        optimizer.zero_grad()

        if params.compute_V:
            # Compute bounds for ALL V cells at once (matching original!)
            if total_cells_V > 0:
                v_lowers_all, v_uppers_all = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all = torch.tensor([], device=device)
                v_uppers_all = torch.tensor([], device=device)

            # Split bounds by region (matching original's split_bounds_by_region)
            bounds = {}
            cell_idx = 0
            for name in region_order_V:
                num_cells = cell_counts_V[name]
                if num_cells > 0:
                    bounds[name] = (
                        v_lowers_all[cell_idx:cell_idx + num_cells],
                        v_uppers_all[cell_idx:cell_idx + num_cells]
                    )
                    cell_idx += num_cells
                else:
                    bounds[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        if params.compute_GV: 
            # Compute generator bounds if enabled
            needs_cache_rebuild = False
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None):
                phi_lowers, phi_uppers = crown_cache_phi.compute_bounds(input_lowers_gen, input_uppers_gen)
                current_gen_weight = params.training.generator_weight

                # Track failing cells for adaptive refinement
                phi_upper_failing_mask = phi_uppers > 0.0
                num_total_failing = phi_upper_failing_mask.sum().item()
                
            else:
                phi_lowers = torch.tensor([], device=device)
                phi_uppers = torch.tensor([], device=device)
                current_gen_weight = 0.0
                num_total_failing = 0

        # Get current beta_s value (learnable or constant)
        if learnable_beta_s is not None:
            current_beta_s = learnable_beta_s.value
        else:
            current_beta_s = beta_s_value

        # Compute total loss from bounds
        loss_kwargs = {
            'model': V_net,
            'goal_region': regions.goal,
            'beta_s': current_beta_s,
            'beta_ra': params.constraints.beta_ra,
            'device': device,
            'compute_V': params.compute_V,
            'compute_GV': params.compute_GV,
            'epoch': epoch,
            'locked_to_all_sum': locked_to_all_sum,
            'current_sum_constraint': current_sum_constraint,
            'constraint_largest_counts': constraint_largest_counts,
            'prev_total_loss': prev_total_loss
        }

        # Add V bounds if computing V
        if params.compute_V:
            loss_kwargs.update({
                'V_goal_lower': bounds['goal'][0],
                'V_goal_upper': bounds['goal'][1],
                'V_unsafe_lower': bounds['unsafe'][0],
                'V_unsafe_upper': bounds['unsafe'][1],
                'V_init_lower': bounds['init'][0],
                'V_init_upper': bounds['init'][1],
                'V_outside_lower': bounds['outside'][0],
                'V_outside_upper': bounds['outside'][1]
            })

        # Add GV bounds if computing GV
        if params.compute_GV:
            loss_kwargs.update({
                'Phi_lower': phi_lowers,
                'Phi_upper': phi_uppers,
                'generator_weight': current_gen_weight
            })
        
        prev_locked = locked_to_all_sum
        prev_sum_constraint = current_sum_constraint
        total_loss, loss_dict, locked_to_all_sum, current_sum_constraint, constraint_largest_counts = compute_total_loss_bounds(**loss_kwargs)

        # Update prev_total_loss for next iteration
        prev_total_loss = total_loss.item()

        # Reset optimizer if we just switched focus (loss scale changes dramatically)
        should_skip_step = False
        if locked_to_all_sum and not prev_locked:
            print(f"  Resetting optimizer and scheduler state after switching to all sums")
            # Recreate optimizer with fresh state
            current_lr = optimizer.param_groups[0]['lr']
            optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)
            # Recreate scheduler with new optimizer
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=300,
                verbose=True,
                min_lr=1e-6,
                threshold=1e-3
            )
            should_skip_step = True  # Skip this step since we're resetting
        elif current_sum_constraint != prev_sum_constraint and prev_sum_constraint is not None:
            print(f"  Resetting optimizer and scheduler state after switching focus")
            # Recreate optimizer with fresh state when switching sum focus
            current_lr = optimizer.param_groups[0]['lr']
            optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)
            # Recreate scheduler with new optimizer
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=300,
                verbose=True,
                min_lr=1e-6,
                threshold=1e-3
            )
            should_skip_step = True  # Skip this step since we're resetting

        # Backward pass
        if not should_skip_step:
            total_loss.backward()

        # Recompute bounds after optimizer step for verification
        V_net.eval()
        with torch.no_grad():
            if total_cells_V > 0:
                v_lowers_all_updated, v_uppers_all_updated = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all_updated = torch.tensor([], device=device)
                v_uppers_all_updated = torch.tensor([], device=device)

            # Split updated bounds by region
            bounds_updated = {}
            cell_idx = 0
            for name in region_order_V:
                num_cells = cell_counts_V[name]
                if num_cells > 0:
                    bounds_updated[name] = (
                        v_lowers_all_updated[cell_idx:cell_idx + num_cells],
                        v_uppers_all_updated[cell_idx:cell_idx + num_cells]
                    )
                    cell_idx += num_cells
                else:
                    bounds_updated[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        # Update scheduler after bounds recomputation
        scheduler.step(total_loss.item())

        # Get current beta_s value for constraint checks
        beta_s_check = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s

        # Adaptive refinement based on highest loss region
        needs_cache_rebuild = False
        if params.compute_V:
            # Compute per-region losses
            region_losses = {}
            if len(bounds_updated['outside'][0]) > 0:
                region_losses['outside'] = F.relu(beta_s_check - bounds_updated['outside'][0]).sum().item()
            if len(bounds_updated['unsafe'][0]) > 0:
                region_losses['unsafe'] = F.relu(params.constraints.beta_ra - bounds_updated['unsafe'][0]).sum().item()
            if len(bounds_updated['init'][0]) > 0:
                region_losses['init'] = F.relu(bounds_updated['init'][1] - 1.0).sum().item()

            # Find region with highest loss
            if region_losses:
                highest_loss_region = max(region_losses, key=region_losses.get)
                highest_loss_value = region_losses[highest_loss_region]

                # Only refine if loss is significant
                if highest_loss_value > 0:
                    # Set up region-specific failing masks
                    if highest_loss_region == 'outside':
                        failing_mask = bounds_updated['outside'][0] < beta_s_check
                    elif highest_loss_region == 'unsafe':
                        failing_mask = bounds_updated['unsafe'][0] < params.constraints.beta_ra
                    elif highest_loss_region == 'init':
                        failing_mask = bounds_updated['init'][1] > 1.0

                    num_failing = failing_mask.sum().item()

                    if num_failing > 0 and len(failing_mask) == len(region_cells[highest_loss_region]):
                        REFINE_INTERVAL_V = 500
                        REFINE_FACTOR = 2
                        MAX_CELLS = 10000

                        if epoch > 2500:
                            REFINE_INTERVAL_V = 100

                        if ((epoch + 1) % REFINE_INTERVAL_V == 0 and
                            len(region_cells[highest_loss_region]) < MAX_CELLS):
                            new_cells, num_refined = refine_failing_cells(
                                region_cells[highest_loss_region],
                                failing_mask,
                                REFINE_FACTOR
                            )
                            region_cells[highest_loss_region] = new_cells
                            print(f"[Refine-{highest_loss_region.capitalize()}] Epoch {epoch+1}: Loss={highest_loss_value:.3f}, {num_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                            needs_cache_rebuild = True
                            if highest_loss_region not in refinement_epochs:
                                refinement_epochs[highest_loss_region] = []
                            refinement_epochs[highest_loss_region].append(epoch + 1)
                    
                    

        # Adaptive refinement for generator cells (before optimizer step)
        if params.compute_GV:
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):

                # Refinement parameters (matching testing_simple3.py)
                REFINE_INTERVAL = 500
                REFINE_FACTOR = 2  # Split into 2x2 subcells
                MAX_CELLS = 10000  # Don't refine if we already have too many cells

                # Adjust interval for later epochs
                if epoch > 2500:
                    REFINE_INTERVAL = 250

                # Check if it's time to refine
                if ((epoch + 1) % REFINE_INTERVAL == 0 and
                    len(region_cells['generator']) < MAX_CELLS):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['generator'],
                        phi_upper_failing_mask,
                        REFINE_FACTOR
                    )
                    region_cells['generator'] = new_cells
                    print(f"[Refine-Generator] Epoch {epoch+1}: {num_total_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['generator'].append(epoch + 1)

        # Logging
        if epoch % 10 == 0 or epoch == params.training.num_epochs - 1 or epoch == 0:
            # Add beta_s to loss dict for logging
            if learnable_beta_s is not None:
                beta_s_log = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s
                print(f"Epoch [{epoch}/{params.training.num_epochs}]: Loss={total_loss.item():.4f}, β_s={beta_s_log:.4f}")
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV)
            if control_net is not None:
                print("control_net parameters:")
                for name, param in control_net.named_parameters():
                    if param.requires_grad:
                        print(f"  {name} =\n{param.data}")
            loss_dict['epoch'] = epoch
            if learnable_beta_s is not None:
                loss_dict['beta_s'] = beta_s_log
            loss_history.append(loss_dict.copy())

        # Early stopping check
        with torch.no_grad():
            # Check V constraints
            all_satisfied = True
            if params.compute_V:
                show = (epoch % 10 == 0)
                # _, goal_satisfied = compute_loss_goal_bounds(V_net, regions.goal, bounds_updated["goal"][0], bounds_updated["goal"][1], beta_s_check, 
                #                                              bounds_updated['outside'][0], device=device, show=show, check=True, n_samples=10000)
                unsafe_satisfied = (bounds_updated['unsafe'][0].min() >= params.constraints.beta_ra)
                init_satisfied = (bounds_updated['init'][0].min() >= beta_s_check and bounds_updated['init'][1].max() <= 1.0)
                # outside_satisfied = (bounds_updated['outside'][0].min() >= beta_s_check)
                outside_satisfied = (bounds_updated['outside'][0].min() >= 0.0)
                # all_satisfied = all_satisfied and goal_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied
                all_satisfied = all_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied

            # Check GV constraints
            if params.compute_GV:
                generator_satisfied = (phi_uppers.max() <= 0.0)
                all_satisfied = all_satisfied and generator_satisfied

            # Early stop if all active constraints are satisfied
            if all_satisfied:
                print("\n" + "="*80)
                print("ALL CONSTRAINTS SATISFIED - EARLY STOPPING!")
                print("="*80)
                print(f"Training converged at epoch {epoch}")

                # Print relevant losses
                loss_parts = []
                if params.compute_V:
                    # loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
                    loss_parts.append(f"Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
                if params.compute_GV:
                    loss_parts.append(f"Gen={loss_dict['generator']:.4f}")
                print(f"Final losses: {', '.join(loss_parts)}")

                if control_net is not None:
                    print("control_net parameters:")
                    # for name, param in control_net.named_parameters():
                    #     if param.requires_grad:
                    #         print(f"  {name} =\n{param.data}")

                final_beta_s = bounds_updated['outside'][0].min()
                print("final beta_s: {:.4f}".format(final_beta_s))

                # Print final cell counts
                print("")
                print_cell_counts(region_cells)

                break

        # Detailed evaluation and visualization
        if (epoch % 1000 == 0) or epoch == params.training.num_epochs - 1:
            print(f"\nEpoch {epoch} - Detailed Evaluation:")
            # For evaluation, we can just create temporary caches (not in the hot path)
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                crown_cache_all=crown_cache_all,
                crown_cache_phi=crown_cache_phi,
                input_bounds_all=(input_lowers_all, input_uppers_all),
                cell_counts_V=cell_counts_V,
                input_bounds_gen=(input_lowers_gen, input_uppers_gen) if input_lowers_gen is not None else None,
                beta_s=current_beta_s,
                beta_ra=params.constraints.beta_ra,
                device=device,
                n_samples=1000
            )
            print_constraint_summary(results, prefix="  ")

            # Print bound statistics
            if params.compute_V:
                if len(bounds['goal'][0]) > 0:
                    print(f"  Goal bounds: V ∈ [{bounds['goal'][0].min().item():.3f}, {bounds['goal'][1].max().item():.3f}]")
                if len(bounds['unsafe'][0]) > 0:
                    print(f"  Unsafe bounds: V ∈ [{bounds['unsafe'][0].min().item():.3f}, {bounds['unsafe'][1].max().item():.3f}]")
            if params.compute_GV:
                if len(phi_uppers) > 0:
                    print(f"  Generator bounds: Φ ∈ [{phi_lowers.min().item():.3f}, {phi_uppers.max().item():.3f}]")
                    failing_cells = (phi_uppers > 0).sum().item()
                    print(f"  Generator failing cells: {failing_cells}/{len(phi_uppers)} ({100*failing_cells/len(phi_uppers):.1f}%)")

        # Visualize progress
        if visualize_interval > 0 and epoch % visualize_interval == 0:
            print(f"  Creating visualization for epoch {epoch}...")
            visualize_training_progress(
                V_net, GV_net, regions, region_cells,
                epoch=epoch,
                output_dir="training_progress",
            )
        
        if not should_skip_step:
            optimizer.step()

        # Rebuild CROWN caches after optimizer step if needed (after adaptive refinement)
        if params.compute_GV:
            if needs_cache_rebuild:
                # with torch.no_grad():
                print(f"  Rebuilding CROWN caches with new generator cells...")

                # Rebuild V cache with all cells (in order: init, goal, unsafe, outside)
                # Note: generator cells are NOT included in V cache
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])

                total_cells_V = len(all_cells_V)
                input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device)

                # Rebuild V CROWN cache
                print(f"    Rebuilding V cache with {total_cells_V} cells...")
                crown_cache_all = SymbolicCROWNCache(
                    model=V_net,
                    num_cells=total_cells_V,
                    input_dim=2,
                    device=device
                )

                # Rebuild Phi CROWN cache (only for generator region)
                num_generator_cells = len(region_cells['generator'])
                print(f"    Rebuilding Phi cache with {num_generator_cells} cells...")
                crown_cache_phi = SymbolicCROWNCache_Phi(
                    phi_module=GV_net,
                    num_cells=num_generator_cells,
                    input_dim=2,
                    device=device
                )

                # Rebuild generator input bounds
                input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device)

                # Update cell counts after refinement
                for name in region_order_V:
                    cell_counts_V[name] = len(region_cells[name])
                
                # Rebuild concatenated input bounds for V
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])
                    input_lowers_all, input_uppers_all =  prepare_cell_bounds(all_cells_V, device)

                print(f"  Caches rebuilt successfully!")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Return final beta_s value along with loss history
    # final_beta_s = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s
    # if learnable_beta_s is not None:
    #     print(f"\nFinal learned β_s = {final_beta_s:.4f}")

    return loss_history, final_beta_s, refinement_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    args = parser.parse_args()

    """Main training function."""
    print("="*80)
    print("MODULAR RL VERIFICATION - BOUND-BASED TRAINING")
    print("="*80)

    # ========================================================================
    # 1. HYPERPARAMETERS
    # ========================================================================
    print("\n" + "="*80)
    print("HYPERPARAMETERS")
    print("="*80)

    params = Hyperparameters.default()

    # Customize configuration
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 16
    pi = np.pi
    params.network.input_scale = [2*pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000  # Adjust as needed
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 1.0  # Enable generator constraint
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 1
    params.discretization.n_outside_goal = 1
    params.discretization.n_generator = 1  # Will be overridden by radial discretization
    params.discretization.n_unsafe = 1
    params.discretization.n_init = 1

    # Set beta_s to a value (constant), or set to None to make it learnable
    # If learnable_beta_s is True, this value will be used as initialization
    params.training.learnable_beta_s = False  # Set to True to make beta_s learnable
    params.constraints.beta_s = 0.1
    params.constraints.beta_ra = 20.0

    # Control what to compute during training
    params.compute_V = True
    params.compute_GV = True

    # ========================================================================
    # 2. SYSTEM DYNAMICS
    # ========================================================================
    print("\n" + "="*80)
    print("SYSTEM DYNAMICS")
    print("="*80)

    # pre-trained in RL deterministic env
    # rl_policy_net = TanhPolicy(2, 1, 64, device=device)
    # rl_policy_net.load_state_dict(torch.load(
    #     OUTPUT_DIR / "pendulum_policy.pt",
    #     map_location=device,
    #     weights_only=True
    # ))
    # rl_policy_net.requires_grad_(False)
    # create a u_nn wrapper to wrap around the rl_policy_net and retrun [rl_policy_net(x), 0] vector
    # u_nn = WrapperConterlNN(rl_policy_net)

    rl_policy_net = InvertControlNN()
    u_nn = WrapperConterlNN(rl_policy_net) # wrapper around InvertControlNN to create u_nn for ClosedLoopDrift

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        g = 9.81
        L = 0.5
        b = 0.1
        m = 0.15
        # Batch
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x2
        f2 = (g/L)*torch.sin(x1) - (b/(m*L**2))*x2
        return torch.stack([f1, f2], dim=1)


    # Create diffusion coefficients as a persistent tensor to avoid TracerWarnings
    g_coeffs = torch.tensor([0.0, 2.0], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term g(x) that is constant: [0.0, 0.2] for every x.

        If x has shape (2,), returns (2,).
        If x has shape (N, 2), returns (N, 2) with each row [0.0, 0.2].
        """
        base = g_coeffs.to(device=x.device, dtype=x.dtype)  # shape (2,)

        if x.dim() == 1:
            # x is shape (2,)
            return base
        elif x.dim() == 2:
            # x is shape (N, 2)
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)  # (N, 2)
        else:
            raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")


    # Create closed-loop drift for control synthesis
    f_cl_module = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)

    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    # ========================================================================
    # 3. SPATIAL REGIONS
    # ========================================================================
    print("\n" + "="*80)
    print("SPATIAL REGIONS")
    print("="*80)

    init_range = np.array([[(3/4)*pi, (5/4)*pi], [-1.0, 1.0]], dtype=np.float32)
    goal_range = np.array([[-0.5*pi, 0.5*pi], [-4.0, 4.0]], dtype=np.float32)

    # Create unsafe region as union of two rectangles (matching paper exactly)
    unsafe_down1 = np.array([[-2*pi, -(3/2)*pi], [-20.0, -10.0]], dtype=np.float32)
    unsafe_down2 = np.array([[(3/2)*pi, 2*pi], [10.0, 20.0]], dtype=np.float32)
    unsafe_range = np.vstack((unsafe_down1, unsafe_down2))
    full_range = np.array([[-2*pi, 2*pi], [-20.0, 20.0]], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe_down1 = Region(unsafe_down1)
    unsafe_down2 = Region(unsafe_down2)
    unsafe = Region.union(unsafe_down1, unsafe_down2)
    full = Region(full_range)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # ========================================================================
    # 4. CREATE NETWORKS
    # ========================================================================
    print("\n" + "="*80)
    print("NETWORKS")
    print("="*80)

    V_net = create_V(params.network)
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        training_config=params.training
    )

    # ========================================================================
    # 5. DISCRETIZE REGIONS
    # ========================================================================
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=True  # Use radial + clipping for generator
    )

    # ========================================================================
    # 6.0 Setup saving
    # ========================================================================
    device = params.training.device
    delta = 0.1
    params.constraints.pretrain_goal_target = params.constraints.beta_s - delta
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = params.constraints.beta_s + delta
    params.constraints.pretrain_phi_target = 1.0
    # Path to bundle we will save/load
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        # ========================================================================
        # 6.1. PRE-TRAINING (Optional)
        # ========================================================================
        cleanup_and_setup_directories(["results", "training_progress"]) # CLEANUP: Remove old results and training progress
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")
        print_training_config(params)
        print(f"\n{dynamics}")
        print(f"V network: {V_net}")
        print(f"\nUsing device: {device}")
        
        ENABLE_PRETRAINING = True  # Set to True to enable
        PRETRAIN_EPOCHS = 10000
        PRETRAIN_LR = 1e-3

        if ENABLE_PRETRAINING:
            pretrain_network_samples(
                model=V_net,
                x_goal_range=goal_range,
                x_unsafe_range=unsafe_range,
                x_init_range=init_range,
                x_range=full_range,
                params=params,
                GV_net=GV_net,  # Pass GV_net if you want GV loss, None otherwise
                num_epochs=PRETRAIN_EPOCHS,
                lr=PRETRAIN_LR,
                device=params.training.device,
                control_net=u_nn
            )

            print(f"Pretraining completed!\n")

        # ========================================================================
        # 6.2 TRAIN WITH BOUNDS
        # ========================================================================

        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            visualize_interval=500,
            control_net=u_nn,   # [control synthesis]
        )

        # ========================================================================
        # 7. FINAL EVALUATION
        # ========================================================================
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80)

        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_s=final_beta_s,
            beta_ra=params.constraints.beta_ra,
            device=device,
            n_samples=5000
        )
        print_constraint_summary(results)

        # ========================================================================
        # 8. FINAL VISUALIZATIONS
        # ========================================================================
        print("\n" + "="*80)
        print("CREATING FINAL VISUALIZATIONS")
        print("="*80)

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
            output_dir="results"
        )

        # ====================================================================
        # 9. SAVE BUNDLE (for future eval/plots)
        # ====================================================================
        print("\n" + "="*80)
        print("SAVING EVAL BUNDLE")
        print("="*80)
        save_eval_bundle(
            OUTPUT_DIR,
            V_net=V_net,
            GV_net=GV_net,
            control_net=u_nn,
            params=params,
            regions=regions,
            region_cells=region_cells,
            final_beta_s=final_beta_s,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
        )

    else:
        # ====================================================================
        # LOAD + EVAL + PLOT
        # ====================================================================
        print("\n" + "="*80)
        print("LOADING SAVED BUNDLE (skip training)")
        print("="*80)

        bundle = load_eval_bundle(bundle_path, map_location="cpu")

        # Rebuild params/regions from dicts
        params = Hyperparameters.from_dict(bundle["hyperparameters"])

        # Rebuild discretization cells
        region_cells = bundle["region_cells"]  # already list of (cpu tensors)

        # Rebuild networks (same as before) and load weights
        V_net = create_V(params.network).to(device)
        V_net.load_state_dict(bundle["V_state_dict"])

        # Recreate dynamics + GV_net (same construction as training path)
        # NOTE: uses your existing dynamics creation earlier in main()
        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network,
            training_config=params.training
        ).to(device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"])

        # Control net (only needed if your plots depend on it)
        rl_policy_net = InvertControlNN(input_dim=2, hidden_dim=8, output_dim=1)
        u_nn =  WrapperConterlNN(rl_policy_net).to(device)
        if bundle["control_state_dict"] is not None:
            u_nn.load_state_dict(bundle["control_state_dict"])

        # Move cells to device for evaluation/plots
        region_cells = {
            k: [(lo.to(device), hi.to(device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }

        final_beta_s = bundle["final_beta_s"]
        loss_history = bundle["loss_history"]
        refinement_epochs = bundle["refinement_epochs"]

        # You can reuse saved results, or recompute
        results = bundle.get("final_results", None)
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_s=final_beta_s,
                beta_ra=params.constraints.beta_ra,
                device=device,
                n_samples=5000
            )

        print("\n" + "="*80)
        print("FINAL EVALUATION (LOADED)")
        print("="*80)
        print_constraint_summary(results)

        print("\n" + "="*80)
        print("CREATING FINAL VISUALIZATIONS (LOADED)")
        print("="*80)
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
            output_dir="results"
        )


if __name__ == '__main__':
    main()