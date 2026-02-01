"""
3D Geometric Brownian Motion Control Synthesis
NOTE: the training progress visualization is turned off because it has not been changed to allow generic x dimension
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]   # repo_root
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.hyperparameters import Hyperparameters
from src.dynamics import Dynamics, ClosedLoopDrift
from src.regions import Regions, Region
from src.network import create_V
from src.control_network import LorentzLinearControlNN
from src.phi_module import create_GV
from src.discretization import discretize_regions
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.training_utils import (
    compute_total_loss_bounds,
    compute_loss_goal_bounds,
    evaluate_constraints,
    print_loss_summary,
    print_constraint_summary,
    refine_failing_cells,
    merge_passing_neighbor_cells,
)
from src.save_load_utils import save_eval_bundle, load_eval_bundle, log_loaded_training_epochs, enable_terminal_logging
from src.utils import cleanup_and_setup_directories
from src.visualization import (
    # visualize_training_progress,
    create_summary_plots
)

# Set random seed immediately after imports (matching testing_simple3.py)
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
    lr=0.01,
    device='cpu',
    control_net=None,
    n_each: int = 400,   # samples per region per epoch
    lambda_w = 1.0,
    save_v_path=None,            # NEW
    save_control_path=None,      # NEW
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
        Sample N points from EACH of the K unsafe boxes (torch only).

        Returns:
        x: (K_unsafe * N, D)

        Notes:
        - If K_unsafe == 1, this is just N samples from that box.
        - Shuffles so the batch is not grouped by box.
        """
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

        # 4) goal-range samples -> enforce v(x) >= 0.0
        x_goal = _sample_in_box(goal_t, n_each)
        v_goal = model(x_goal).squeeze(-1)
        v_loss_inside_goal = F.relu(0.0 - v_goal).sum()

        # 5) samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= 0.0
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
        v_loss_others = F.relu(0.0 - v_others).sum()

        # Total V loss (v_loss_inside_goal and v_loss_others are not used anymore)
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

        # Add regularization for V network
        reg_w = _l2_weight_penalty(model, exclude_bias=True)
        # print(reg_w)
        total_loss = total_loss + lambda_w * reg_w

        # Track best
        if total_loss.item() <= best_loss:
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
                    f"Φ={loss_phi.item():.6f}, Total={total_loss.item():.5e}"
                )
            else:
                print(
                    f"  Epoch [{epoch}/{num_epochs}]: "
                    f"V={loss_v.item():.6f} "
                    f"(full={v_loss_full.item():.3f}, init={v_loss_init.item():.3f}, "
                    f"unsafe={v_loss_unsafe.item():.3f}, goal={v_loss_inside_goal.item():.3f}, "
                    f"others={v_loss_others.item():.3f})"
                )

        # Optimize/update networks
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\n  Best loss: {best_loss:.6f}")

        # NEW: save best pretrained weights (state_dict)
        if save_v_path is not None:
            torch.save(best_model_state, save_v_path)
            print(f"  Saved pretrained V_net to: {save_v_path}")
        if (control_net is not None) and (best_control_state is not None) and (save_control_path is not None):
            torch.save(best_control_state, save_control_path)
            print(f"  Saved pretrained Controller_net to: {save_control_path}")

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

    # Collect ALL cells in the same order as original (init, goal, unsafe, outside, generator)
    print("\nCollecting all cells for V network...")
    all_cells_V = []
    cell_counts_V = {}
    region_order_V = ['init', 'goal', 'unsafe', 'outside'] 

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
        input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)
    else:
        input_lowers_all = torch.empty(0, params.network.n_inputs, device=device)
        input_uppers_all = torch.empty(0, params.network.n_inputs, device=device)

    # Create ONE big CROWN cache for ALL V cells (matching original!)
    print(f"\nInitializing CROWN cache for ALL {total_cells_V} V cells...")
    crown_cache_all = SymbolicCROWNCache(
        model=V_net,
        num_cells=total_cells_V,
        input_dim=params.network.n_inputs,
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
            input_dim=params.network.n_inputs,
            device=device
        )
        # Prepare generator input bounds
        input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

    # Check if beta_s should be learnable
    learnable_beta_s = None
    beta_s_value = params.constraints.beta_s
    print(f"\nUsing CONSTANT beta_s = {beta_s_value}")
    opt_params = list(V_net.parameters())

    # [control synthesis]
    if control_net is not None:
        opt_params += list(control_net.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)

    # Scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,   # every 2000 epochs
        gamma=0.95         # multiply lr by 0.5
    )

    # Training loop
    loss_history = []
    refinement_epochs = {'outside': [], 'generator': []}
    start_time = time.time()
    final_beta_s = params.constraints.beta_s 
    loss_kwargs = {
        'beta_ra': params.constraints.beta_ra,
        'device': params.training.device,
        'compute_V': params.compute_V,
        'compute_GV': params.compute_GV,
    }

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
                phi_uppers = crown_cache_phi.compute_bounds(input_lowers_gen, input_uppers_gen)
                current_gen_weight = params.training.generator_weight

                # Track failing cells for adaptive refinement
                phi_upper_failing_mask = phi_uppers > 0.0
                num_total_failing = phi_upper_failing_mask.sum().item()
                phi_upper_failing_mask_relax = phi_uppers > -1000.0
                
            else:
                phi_uppers = torch.tensor([], device=device)
                current_gen_weight = 0.0
                num_total_failing = 0

        # Get current beta_s value (learnable or constant)
        if learnable_beta_s is not None:
            current_beta_s = learnable_beta_s.value
        else:
            current_beta_s = beta_s_value

        # Update loss kwargs with current bounds (reuse pre-allocated dict)
        if params.compute_V:
            loss_kwargs['beta_ra'] = params.constraints.beta_ra
            loss_kwargs['V_goal_lower'] = bounds['goal'][0]
            loss_kwargs['V_unsafe_lower'] = bounds['unsafe'][0]
            loss_kwargs['V_init_upper'] = bounds['init'][1]
            loss_kwargs['V_outside_lower'] = bounds['outside'][0]

        if params.compute_GV:
            loss_kwargs['Phi_upper'] = phi_uppers
            loss_kwargs['generator_weight'] = current_gen_weight

        total_loss, loss_dict, _ = compute_total_loss_bounds(**loss_kwargs)

        # Backward pass
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

        # Get current beta_s value for constraint checks
        beta_s_check = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s

        # Adaptive refinement for V outside region cells
        needs_cache_rebuild = False
        if params.compute_V and len(bounds_updated['outside'][0]) > 0:
            # Track failing cells in outside region
            outside_failing_mask = bounds_updated['outside'][0] < beta_s_check
            num_outside_failing = outside_failing_mask.sum().item()
            outside_failing_mask_relax = bounds_updated['outside'][0] < (beta_s_check + 0.9)

            # Ensure bounds match current cell count
            if len(outside_failing_mask) == len(region_cells['outside']) and num_outside_failing > 0:
                REFINE_INTERVAL = 250
                REFINE_FACTOR = 2
                MAX_CELLS = 60000

                if epoch > 2500:
                    REFINE_INTERVAL = 50

                if ((epoch + 1) % REFINE_INTERVAL == 0 and
                    len(region_cells['outside']) < MAX_CELLS):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['outside'],
                        outside_failing_mask,
                        REFINE_FACTOR
                    )
                    region_cells['outside'] = new_cells
                    print(f"[Refine-Outside] Epoch {epoch+1}: {num_outside_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['outside'].append(epoch + 1)

            if((epoch + 1) % 501 == 0):
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['outside'],
                    outside_failing_mask_relax,
                    max_passes=8,
                    max_merges=None,   # cap work; set None for full greedy
                    seed=0,
                    eps=1e-6,
                )
                region_cells['outside'] = merged_cells
                print(f"[Merge-Outside] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Adaptive refinement for generator cells (before optimizer step)
        if params.compute_GV:
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):

                # Refinement parameters (matching testing_simple3.py)
                REFINE_INTERVAL = 500  # Refine every 100k epochs
                REFINE_FACTOR = 2  # Split into 2x2 subcells
                MAX_CELLS = 60000  # Don't refine if we already have too many cells

                # Adjust interval for later epochs
                if epoch > 2500:
                    REFINE_INTERVAL = 200

                # Check if it's time to refine
                if ((epoch + 1) % REFINE_INTERVAL == 0 and
                    len(region_cells['generator']) < MAX_CELLS):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['generator'],
                        phi_upper_failing_mask,
                        REFINE_FACTOR,
                        scores=phi_uppers,
                    )
                    region_cells['generator'] = new_cells
                    print(f"[Refine-Generator] Epoch {epoch+1}: {num_total_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['generator'].append(epoch + 1)
            
            if((epoch + 1) % 501 == 0):
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['generator'],
                    phi_upper_failing_mask_relax,
                    max_passes=8,
                    max_merges=None,   # cap work; set None for full greedy
                    seed=0,
                    eps=1e-6,
                )
                region_cells['generator'] = merged_cells
                print(f"[Merge-Generator] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Logging
        if epoch % 10 == 0 or epoch == params.training.num_epochs - 1 or epoch == 0:
            # Add beta_s to loss dict for logging
            if learnable_beta_s is not None:
                beta_s_log = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s
                print(f"Epoch [{epoch}/{params.training.num_epochs}]: Loss={total_loss.item():.4f}, β_s={beta_s_log:.4f}")
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV)
            # if control_net is not None:
            #     for name, param in control_net.named_parameters():
            #         if param.requires_grad:
            #             print(f" [Controller Params] {name} = {param.data}")
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
                _, goal_satisfied = compute_loss_goal_bounds(bounds_updated["goal"][0])
                unsafe_satisfied = (bounds_updated['unsafe'][0].min() >= params.constraints.beta_ra)
                init_satisfied = (bounds_updated['init'][1].max() <= 1.0)
                outside_satisfied = (bounds_updated['outside'][0].min() >= 0.0)
                all_satisfied = all_satisfied and goal_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied

            # Check GV constraints
            if params.compute_GV:
                generator_satisfied = (phi_uppers.max() < 0.0)
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
                    loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
                if params.compute_GV:
                    loss_parts.append(f"Gen={loss_dict['generator']:.4f}")
                print(f"Final losses: {', '.join(loss_parts)}")

                if control_net is not None:
                    for name, param in control_net.named_parameters():
                        if param.requires_grad:
                            print(f" [Controller Params] {name} = {param.data}")

                final_beta_s = bounds_updated['outside'][0].min()
                print("final beta_s: {:.4f}".format(final_beta_s))

                break

        # Detailed evaluation and visualization
        if (epoch % 500 == 0) or epoch == params.training.num_epochs - 1:
            print(f"\nEpoch {epoch} - Detailed Evaluation:")
            # For evaluation, we can just create temporary caches (not in the hot path)
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                crown_cache_all=crown_cache_all,
                crown_cache_phi=crown_cache_phi,
                input_bounds_all=(input_lowers_all, input_uppers_all),
                cell_counts_V=cell_counts_V,
                input_bounds_gen=(input_lowers_gen, input_uppers_gen) if input_lowers_gen is not None else None,
                beta_ra=params.constraints.beta_ra,
                device=device,
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
                    print(f"  Generator bounds: Φ ∈ [{phi_uppers.min().item():.6e}, {phi_uppers.max().item():.6e}]")
                    failing_cells = (phi_uppers > 0).sum().item()
                    print(f"  Generator failing cells: {failing_cells}/{len(phi_uppers)} ({100*failing_cells/len(phi_uppers):.1f}%)")
            print()

            # Visualize progress
            if visualize_interval > 0 and epoch % visualize_interval == 0:
                print(f"  Creating visualization for epoch {epoch}...")
                # visualize_training_progress(
                #     V_net, GV_net, regions, region_cells,
                #     epoch=epoch,
                #     output_dir="training_progress"
                # )
        
        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Rebuild CROWN caches after optimizer step if needed (after adaptive refinement)
        # if params.compute_GV:
        if True:
            if needs_cache_rebuild:
                # with torch.no_grad():
                print(f"  Rebuilding CROWN caches with new generator cells...")

                # Rebuild V cache with all cells (in order: init, goal, unsafe, outside)
                # Note: generator cells are NOT included in V cache
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])

                total_cells_V = len(all_cells_V)
                input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

                # Rebuild V CROWN cache
                print(f"    Rebuilding V cache with {total_cells_V} cells...")
                crown_cache_all = SymbolicCROWNCache(
                    model=V_net,
                    num_cells=total_cells_V,
                    input_dim=params.network.n_inputs,
                    device=device
                )

                # Rebuild Phi CROWN cache (only for generator region)
                num_generator_cells = len(region_cells['generator'])
                print(f"    Rebuilding Phi cache with {num_generator_cells} cells...")
                crown_cache_phi = SymbolicCROWNCache_Phi(
                    phi_module=GV_net,
                    num_cells=num_generator_cells,
                    input_dim=params.network.n_inputs,
                    device=device
                )

                # Rebuild generator input bounds
                input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

                # Update cell counts after refinement
                for name in region_order_V:
                    cell_counts_V[name] = len(region_cells[name])
                
                # Rebuild concatenated input bounds for V
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])
                    input_lowers_all, input_uppers_all =  prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

                print(f"  Caches rebuilt successfully!")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    return loss_history, final_beta_s, refinement_epochs


def animate_closed_loop_3d_state_control(
    *,
    f_cl_module,               # closed-loop drift module: x_dot = f_cl_module(x)
    g_fn=None,                 # optional diffusion: g(x) (diag (3,) or matrix (3,3))
    init_range: np.ndarray,    # (3,2)
    goal_range: np.ndarray,    # (3,2)
    full_range: np.ndarray,    # (3,2)
    unsafe_boxes: np.ndarray,  # (K*3,2) or (K,3,2) or (3,2)
    device: str = "cpu",
    dt: float = 0.02,
    T: float = 12.0,
    seed: int = 0,

    # --- optional extras (kept separate from the "must keep" argument block above) ---
    state_labels=("x1", "x2", "x3"),     # axis labels for the 3D plot + time-series
    control_labels=("u1", "u2", "u3"),   # y-labels on control plots (expects 3 controls)
    controller_label: str | None = None,
    save_path: str | None = None,       # e.g. "outputs/anim.mp4" (ffmpeg needed)
    show: bool = True,
    frame_skip: int = 20,
    interval_ms: int = 30,
):
    """
    General 3D closed-loop rollout animation.

    Layout (3 columns):
      LEFT:   3D state trajectory (x1,x2,x3) with init/goal boxes + unsafe union boxes (+ full range)
      MIDDLE: 3 rows state vs time (x1,x2,x3) with shaded init/goal/unsafe bands (only when tighter than full_range)
      RIGHT:  3 rows control vs time (u1,u2,u3)

    Assumptions:
      - state dimension is 3 (for 3D visualization + ranges)
      - control dimension is 3 for the right column plots; if your controller returns != 3,
        you can adapt the U handling below (or pass a wrapper controller).

    Stochastic rollout (if g_fn is provided):
      Euler–Maruyama: x_{k+1} = x_k + f(x_k) dt + G(x_k) dW
      - if g_fn returns (3,), treats it as diagonal coefficients
      - if g_fn returns (3,3), uses matrix diffusion
    """

    rng = np.random.default_rng(seed)

    init_range = np.asarray(init_range, dtype=np.float32)
    goal_range = np.asarray(goal_range, dtype=np.float32)
    full_range = np.asarray(full_range, dtype=np.float32)

    def _as_union_boxes(x_unsafe: np.ndarray) -> np.ndarray:
        x = np.asarray(x_unsafe, dtype=np.float32)
        if x.ndim == 2:
            if x.shape == (3, 2):
                return x[None, ...]
            if x.shape[1] == 2 and (x.shape[0] % 3 == 0):
                K = x.shape[0] // 3
                return x.reshape(K, 3, 2)
            raise ValueError(f"unsafe_boxes 2D must be (3,2) or (K*3,2), got {x.shape}")
        if x.ndim == 3:
            if x.shape[1:] != (3, 2):
                raise ValueError(f"unsafe_boxes 3D must be (K,3,2), got {x.shape}")
            return x
        raise ValueError(f"unsafe_boxes must be (3,2), (K*3,2), or (K,3,2), got {x.shape}")

    unsafeK = _as_union_boxes(unsafe_boxes)  # (K,3,2)

    # -----------------------------------------
    # Rollout
    # -----------------------------------------
    x0 = np.array([
        rng.uniform(init_range[0, 0], init_range[0, 1]),
        rng.uniform(init_range[1, 0], init_range[1, 1]),
        rng.uniform(init_range[2, 0], init_range[2, 1]),
    ], dtype=np.float32)

    N = int(T / dt) + 1
    t = np.linspace(0.0, T, N, dtype=np.float32)

    X = np.zeros((N, 3), dtype=np.float32)
    U = np.zeros((N, 3), dtype=np.float32)

    X[0] = x0

    def _get_control(xk_t: torch.Tensor) -> np.ndarray:
        # Preferred: f_cl_module.controller(x)
        if hasattr(f_cl_module, "controller") and callable(getattr(f_cl_module, "controller")):
            uk_t = f_cl_module.controller(xk_t)
            uk = uk_t.detach().cpu().numpy().reshape(-1)
            return uk.astype(np.float32)
        # Fallbacks
        for name in ("u", "policy", "actor"):
            if hasattr(f_cl_module, name) and callable(getattr(f_cl_module, name)):
                uk_t = getattr(f_cl_module, name)(xk_t)
                uk = uk_t.detach().cpu().numpy().reshape(-1)
                return uk.astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _eval_drift(xk_t: torch.Tensor) -> np.ndarray:
        xdot_t = f_cl_module(xk_t)
        xdot = xdot_t.detach().cpu().numpy().reshape(-1)
        if xdot.size != 3:
            raise ValueError(f"f_cl_module(x) must return 3D drift, got shape {xdot_t.shape} -> size {xdot.size}")
        return xdot.astype(np.float32)

    def _eval_diffusion(xk_t: torch.Tensor):
        gk_t = g_fn(xk_t)
        gk = gk_t.detach().cpu().numpy()
        gk = np.asarray(gk, dtype=np.float32).reshape(-1)
        # allow (3,) diag or (9,) flattened 3x3
        if gk.size == 3:
            return ("diag", gk)
        if gk.size == 9:
            return ("mat", gk.reshape(3, 3))
        # also allow raw (3,3) in numpy shape form
        gk2 = np.asarray(gk_t.detach().cpu().numpy(), dtype=np.float32)
        if gk2.shape == (3, 3):
            return ("mat", gk2)
        raise ValueError(f"g_fn(x) must return (3,) or (3,3); got {gk_t.shape}")

    f_cl_module.eval()
    if g_fn is not None and hasattr(g_fn, "eval"):
        g_fn.eval()

    with torch.no_grad():
        for k in range(N - 1):
            xk_t = torch.tensor(X[k:k + 1], dtype=torch.float32, device=device)  # (1,3)

            uk = _get_control(xk_t)
            xdot = _eval_drift(xk_t)

            if g_fn is None:
                xnext = X[k] + dt * xdot
            else:
                kind, G = _eval_diffusion(xk_t)
                dW = (np.sqrt(dt) * rng.standard_normal(3)).astype(np.float32)  # (3,)
                if kind == "diag":
                    xnext = X[k] + dt * xdot + G * dW
                else:
                    xnext = X[k] + dt * xdot + (G @ dW)

            X[k + 1] = xnext
            # keep only first 3 controls for plotting (or pad if fewer)
            if uk.size >= 3:
                U[k] = uk[:3]
            else:
                tmp = np.zeros(3, dtype=np.float32)
                tmp[:uk.size] = uk
                U[k] = tmp

        U[-1] = U[-2]

    # -----------------------------------------
    # Helpers for shaded bands + limits
    # -----------------------------------------
    def _is_tighter(rng_1d: np.ndarray, full_1d: np.ndarray, eps: float = 1e-9) -> bool:
        return (rng_1d[0] > full_1d[0] + eps) or (rng_1d[1] < full_1d[1] - eps)

    def _add_band(ax, lo, hi, color, label=None, alpha=0.12):
        ax.axhspan(lo, hi, color=color, alpha=alpha, label=label, zorder=0)

    def _add_state_bands(ax, dim: int, ax_label: str):
        if _is_tighter(init_range[dim], full_range[dim]):
            _add_band(ax, init_range[dim, 0], init_range[dim, 1], color="green", label=f"init ({ax_label})")
        if _is_tighter(goal_range[dim], full_range[dim]):
            _add_band(ax, goal_range[dim, 0], goal_range[dim, 1], color="blue", label=f"goal ({ax_label})")
        any_unsafe = False
        for k in range(unsafeK.shape[0]):
            lo, hi = unsafeK[k, dim, 0], unsafeK[k, dim, 1]
            if _is_tighter(unsafeK[k, dim], full_range[dim]):
                _add_band(ax, lo, hi, color="red", label=("unsafe" if not any_unsafe else None), alpha=0.10)
                any_unsafe = True

    def _collect_band_extents_for_dim(d: int):
        lows, highs = [], []
        if _is_tighter(init_range[d], full_range[d]):
            lows.append(float(init_range[d, 0])); highs.append(float(init_range[d, 1]))
        if _is_tighter(goal_range[d], full_range[d]):
            lows.append(float(goal_range[d, 0])); highs.append(float(goal_range[d, 1]))
        for k in range(unsafeK.shape[0]):
            if _is_tighter(unsafeK[k, d], full_range[d]):
                lows.append(float(unsafeK[k, d, 0])); highs.append(float(unsafeK[k, d, 1]))
        if len(lows) == 0:
            return None
        return (min(lows), max(highs))

    def _set_ylim_with_bands(ax, y_data, band_extents):
        y = np.asarray(y_data, dtype=np.float32)
        y0, y1 = float(np.min(y)), float(np.max(y))
        if band_extents is not None:
            b0, b1 = band_extents
            y0 = min(y0, b0)
            y1 = max(y1, b1)
        if np.isclose(y0, y1):
            y0 -= 1.0
            y1 += 1.0
        pad = 0.10 * (y1 - y0)
        ax.set_ylim(y0 - pad, y1 + pad)

    def _set_ylim(ax, y):
        y = np.asarray(y, dtype=np.float32)
        y0, y1 = float(np.min(y)), float(np.max(y))
        if np.isclose(y0, y1):
            y0 -= 1.0
            y1 += 1.0
        pad = 0.10 * (y1 - y0)
        ax.set_ylim(y0 - pad, y1 + pad)

    # -----------------------------------------
    # 3D box drawing
    # -----------------------------------------
    def _draw_box3d(ax3d, box3x2, *, lw=1.5, color="k", alpha=1.0):
        x0, x1 = float(box3x2[0, 0]), float(box3x2[0, 1])
        y0, y1 = float(box3x2[1, 0]), float(box3x2[1, 1])
        z0, z1 = float(box3x2[2, 0]), float(box3x2[2, 1])

        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float32)

        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]
        for (i, j) in edges:
            ax3d.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                lw=lw, color=color, alpha=alpha
            )

    # -----------------------------------------
    # Figure layout: 6 rows x 3 cols
    # Left column is one big 3D plot
    # -----------------------------------------
    fig = plt.figure(figsize=(18, 9))
    title = "Closed-loop rollout: 3D state + time series"
    if controller_label:
        title += f"  |  Controller: {controller_label}"
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92)

    gs = fig.add_gridspec(
        6, 3,
        width_ratios=[1.55, 1.0, 1.0],
        height_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.30,
        hspace=0.55,
    )

    ax_3d  = fig.add_subplot(gs[0:6, 0], projection="3d")

    ax_s1 = fig.add_subplot(gs[0:2, 1])
    ax_s2 = fig.add_subplot(gs[2:4, 1], sharex=ax_s1)
    ax_s3 = fig.add_subplot(gs[4:6, 1], sharex=ax_s1)

    ax_u1 = fig.add_subplot(gs[0:2, 2])
    ax_u2 = fig.add_subplot(gs[2:4, 2], sharex=ax_u1)
    ax_u3 = fig.add_subplot(gs[4:6, 2], sharex=ax_u1)

    # -----------------------------------------
    # LEFT: 3D trajectory + boxes
    # -----------------------------------------
    ax_3d.set_title("3D state trajectory + init/goal/unsafe boxes")
    ax_3d.set_xlabel(state_labels[0])
    ax_3d.set_ylabel(state_labels[1])
    ax_3d.set_zlabel(state_labels[2])

    _draw_box3d(ax_3d, full_range, lw=1.0, color="0.5", alpha=0.35)
    _draw_box3d(ax_3d, init_range, lw=2.0, color="green", alpha=0.9)
    _draw_box3d(ax_3d, goal_range, lw=2.0, color="blue", alpha=0.9)
    for k in range(unsafeK.shape[0]):
        _draw_box3d(ax_3d, unsafeK[k], lw=1.8, color="red", alpha=0.75)

    line3d, = ax_3d.plot([], [], [], lw=2)
    pt3d,   = ax_3d.plot([], [], [], marker="o")
    txt3d = ax_3d.text2D(0.02, 0.98, "", transform=ax_3d.transAxes, va="top")

    ax_3d.set_xlim(float(full_range[0, 0]), float(full_range[0, 1]))
    ax_3d.set_ylim(float(full_range[1, 0]), float(full_range[1, 1]))
    ax_3d.set_zlim(float(full_range[2, 0]), float(full_range[2, 1]))
    ax_3d.view_init(elev=22, azim=-55)

    # -----------------------------------------
    # MIDDLE: states vs time (with bands)
    # -----------------------------------------
    ax_s1.set_title("States vs time")
    ax_s1.set_ylabel(state_labels[0])
    ax_s2.set_ylabel(state_labels[1])
    ax_s3.set_ylabel(state_labels[2])
    ax_s3.set_xlabel("t [s]")
    for ax in (ax_s1, ax_s2, ax_s3):
        ax.grid(True, alpha=0.3)

    _add_state_bands(ax_s1, 0, state_labels[0])
    _add_state_bands(ax_s2, 1, state_labels[1])
    _add_state_bands(ax_s3, 2, state_labels[2])

    _set_ylim_with_bands(ax_s1, X[:, 0], _collect_band_extents_for_dim(0))
    _set_ylim_with_bands(ax_s2, X[:, 1], _collect_band_extents_for_dim(1))
    _set_ylim_with_bands(ax_s3, X[:, 2], _collect_band_extents_for_dim(2))
    ax_s1.set_xlim(0, float(T))

    handles, labels = ax_s1.get_legend_handles_labels()
    if len(handles) > 0:
        ax_s1.legend(loc="upper right", framealpha=0.85)

    # -----------------------------------------
    # RIGHT: controls vs time
    # -----------------------------------------
    ax_u1.set_title("Controls vs time")
    ax_u1.set_ylabel(control_labels[0])
    ax_u2.set_ylabel(control_labels[1])
    ax_u3.set_ylabel(control_labels[2])
    ax_u3.set_xlabel("t [s]")
    for ax in (ax_u1, ax_u2, ax_u3):
        ax.grid(True, alpha=0.3)

    _set_ylim(ax_u1, U[:, 0])
    _set_ylim(ax_u2, U[:, 1])
    _set_ylim(ax_u3, U[:, 2])
    ax_u1.set_xlim(0, float(T))

    # -----------------------------------------
    # Time-series lines
    # -----------------------------------------
    ls1, = ax_s1.plot([], [], lw=2)
    ls2, = ax_s2.plot([], [], lw=2)
    ls3, = ax_s3.plot([], [], lw=2)

    lu1, = ax_u1.plot([], [], lw=2)
    lu2, = ax_u2.plot([], [], lw=2)
    lu3, = ax_u3.plot([], [], lw=2)

    # -----------------------------------------
    # Animation init/update
    # -----------------------------------------
    def init_anim():
        line3d.set_data([], [])
        line3d.set_3d_properties([])
        pt3d.set_data([], [])
        pt3d.set_3d_properties([])
        txt3d.set_text("")

        ls1.set_data([], [])
        ls2.set_data([], [])
        ls3.set_data([], [])
        lu1.set_data([], [])
        lu2.set_data([], [])
        lu3.set_data([], [])

        return (line3d, pt3d, txt3d, ls1, ls2, ls3, lu1, lu2, lu3)

    def update(i: int):
        i = int(i)
        xs = X[:i + 1]
        us = U[:i + 1]

        # LEFT 3D
        line3d.set_data(xs[:, 0], xs[:, 1])
        line3d.set_3d_properties(xs[:, 2])
        pt3d.set_data([xs[-1, 0]], [xs[-1, 1]])
        pt3d.set_3d_properties([xs[-1, 2]])

        # MIDDLE states
        ls1.set_data(t[:i + 1], xs[:, 0])
        ls2.set_data(t[:i + 1], xs[:, 1])
        ls3.set_data(t[:i + 1], xs[:, 2])

        # RIGHT controls
        lu1.set_data(t[:i + 1], us[:, 0])
        lu2.set_data(t[:i + 1], us[:, 1])
        lu3.set_data(t[:i + 1], us[:, 2])

        txt3d.set_text(
            f"t={t[i]:.2f}s\n"
            f"{state_labels[0]}={xs[-1,0]:.3g}, {state_labels[1]}={xs[-1,1]:.3g}, {state_labels[2]}={xs[-1,2]:.3g}\n"
            f"{control_labels[0]}={us[-1,0]:.3g}, {control_labels[1]}={us[-1,1]:.3g}, {control_labels[2]}={us[-1,2]:.3g}"
        )

        return (line3d, pt3d, txt3d, ls1, ls2, ls3, lu1, lu2, lu3)

    frames = range(0, N, max(1, int(frame_skip)))
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init_anim,
        interval=int(interval_ms),
        blit=False,
    )

    if save_path is not None:
        ani.save(save_path, dpi=150)

    if show:
        plt.show()

    return {"t": t, "X": X, "U": U, "x0": x0}


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
    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [6.0, 6.0, 6.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000  # Adjust as needed
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 1.0  # Enable generator constraint
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 13
    params.discretization.n_outside_goal = 5
    params.discretization.n_generator = 1  # Will be overridden by radial discretization
    params.discretization.n_unsafe = 12
    params.discretization.n_init = 20

    # Set beta_s to a value (constant), or set to None to make it learnable
    # If learnable_beta_s is True, this value will be used as initialization
    params.training.learnable_beta_s = False  # Set to True to make beta_s learnable
    params.constraints.beta_s = 0.00
    params.constraints.beta_ra = 5.0

    # Control what to compute during training
    params.compute_V = True
    params.compute_GV = True

    # ========================================================================
    # 2. SYSTEM DYNAMICS
    # ========================================================================
    print("\n" + "="*80)
    print("SYSTEM DYNAMICS")
    print("="*80)

    u_nn = LorentzLinearControlNN()

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        # Batch
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        # f1 = -33.71*x1 - 8.49*x2
        f1 = -10.0*x1 + 10.0*x2
        f2 = -x1*x3 + 28.0*x1 - x2
        f3 =  x1*x2 - 8/3 *x3
        return torch.stack([f1, f2, f3], dim=1)


    # Create diffusion coefficients as a persistent tensor to avoid TracerWarnings
    g_coeffs = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term g(x) that is constant: [1.0, 1.0, 1.0] for every x.

        If x has shape (D,), returns (D,).
        If x has shape (N, D), returns (N, D) with each row [1.0, 1.0, 1.0].
        """
        base = g_coeffs.to(device=x.device, dtype=x.dtype)

        if x.dim() == 1:
            # x is shape (D,)
            return base
        elif x.dim() == 2:
            # x is shape (N, D)
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)  # (N, D)
        else:
            raise ValueError(f"g(x) expects x of shape (D,) or (N, D), got {tuple(x.shape)}")

    # Create closed-loop drift for control synthesis
    f_cl_module = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)

    dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=3)

    # ========================================================================
    # 3. SPATIAL REGIONS
    # ========================================================================
    print("\n" + "="*80)
    print("SPATIAL REGIONS")
    print("="*80)

    init_range = np.array([
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    ], dtype=np.float32)

    goal_range = np.array([
        [-0.3, 0.3],
        [-0.3, 0.3],
        [-0.3, 0.3],
    ], dtype=np.float32)

    unsafe_tb = np.array([
        [-6.0, 6.0], 
        [-6.0, 6.0],
        [-6.0, -6.0+0.5]
    ], dtype=np.float32)
    unsafe_db = np.array([
        [-6.0, 6.0], 
        [-6.0, 6.0],
        [6.0-0.5, 6.0]
    ], dtype=np.float32)
    unsafe_fb = np.array([
        [-6.0, 6.0], 
        [-6.0, -6.0+0.5],
        [-6.0, 6.0]
    ], dtype=np.float32)
    unsafe_bb = np.array([
        [-6.0, 6.0], 
        [6.0-0.5, 6.0],
        [-6.0, 6.0]
    ], dtype=np.float32)
    unsafe_lb = np.array([
        [-6.0, -6.0+0.5], 
        [-6.0, 6.0],
        [-6.0, 6.0]
    ], dtype=np.float32)
    unsafe_rb = np.array([
        [6.0-0.5, 6.0], 
        [-6.0, 6.0],
        [-6.0, 6.0]
    ], dtype=np.float32)
    unsafe_range = np.vstack((unsafe_tb, unsafe_db, 
                              unsafe_fb, unsafe_bb,
                              unsafe_lb, unsafe_rb))

    full_range = np.array([
        [-6.0, 6.0],
        [-6.0, 6.0],
        [-6.0, 6.0],
    ], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe_tb = Region(unsafe_tb)
    unsafe_db = Region(unsafe_db)
    unsafe_fb = Region(unsafe_fb)
    unsafe_bb = Region(unsafe_bb)
    unsafe_lb = Region(unsafe_lb)
    unsafe_rb = Region(unsafe_rb)
    unsafe = Region.union(unsafe_tb, unsafe_db, 
                          unsafe_fb, unsafe_bb,
                          unsafe_lb, unsafe_rb)
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
        verify=False
    )

    # ========================================================================
    # 5. DISCRETIZE REGIONS
    # ========================================================================
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=True,  # Use radial + clipping for generator
    )

    # ========================================================================
    # 6.0 Setup saving
    # ========================================================================
    device = params.training.device
    params.constraints.pretrain_goal_target = params.constraints.beta_s
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = 1.0
    params.constraints.pretrain_phi_target = 0.0
    # Path to bundle we will save/load
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        # ========================================================================
        # 6.1. PRE-TRAINING (Optional)
        # ========================================================================
        cleanup_and_setup_directories(["results", "training_progress"]) # CLEANUP: Remove old results and training progress
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")
        print(f"\n{dynamics}")
        print(f"V network: {V_net}")
        print(f"\nUsing device: {device}")

        ENABLE_PRETRAINING = True  # Set to True to enable
        PRETRAIN_EPOCHS = 15000
        PRETRAIN_LR = 0.01

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
                control_net=u_nn,
                n_each=1000,
                lambda_w=1e-3,
                save_v_path= OUTPUT_DIR / "V_pretrained.pth",
                save_control_path= OUTPUT_DIR / "controller_pretrained.pth"
            )
            print(f"Pretraining completed!\n")
        V_net.load_state_dict(torch.load(OUTPUT_DIR / "V_pretrained.pth", map_location=device))
        u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))

        # ========================================================================
        # 6. TRAIN WITH BOUNDS
        # ========================================================================
        device = params.training.device
        print(f"\nUsing device: {device}")

        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            visualize_interval=1000,
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
            beta_ra=params.constraints.beta_ra,
            device=params.training.device
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
            verify=False,
        ).to(device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"])

        # Control net
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
        results = None
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_ra=params.constraints.beta_ra,
                device=device,
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
