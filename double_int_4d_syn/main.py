"""
4D Double Integrator Control Synthesis
Two particles that need to swap positions while avoiding collision

State: x = [p1, v1, p2, v2]
- p1, p2: positions of particle 1 and 2
- v1, v2: velocities of particle 1 and 2

Goal: Swap positions (p1 → top-right, p2 → bottom-left) with low velocity
Unsafe: Particles too close (collision)
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
from src.control_network import LinearControlNN, NonlinearControlNN
from src.phi_module import create_GV
from src.discretization import discretize_regions
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.training_utils import (
    compute_total_loss_bounds,
    evaluate_constraints,
    print_loss_summary,
    print_constraint_summary,
    refine_failing_cells,
    merge_passing_neighbor_cells,
)
from src.save_load_utils import save_eval_bundle, load_eval_bundle, log_loaded_training_epochs, enable_terminal_logging
from src.utils import cleanup_and_setup_directories, print_training_config
from src.visualization import (
    visualize_training_progress,
    create_summary_plots
)

# Set random seed
torch.manual_seed(0)


class DoubleIntegratorControlNN(nn.Module):
    """
    Control network for 4D double integrator.
    Maps state [p1, v1, p2, v2] to control [a1, a2] (accelerations).
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 16, bias=True)
        self.fc3 = nn.Linear(16, 2, bias=False)  # output: [a1, a2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 4) state [p1, v1, p2, v2]
        returns: (N, 2) control [a1, a2]
        """
        h1 = F.tanh(self.fc1(x))
        h2 = F.tanh(self.fc2(h1))
        u = 20.0 * F.tanh(self.fc3(h2))  # bound control to [-2, 2]
        return u


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
    n_each: int = 400,
):
    """
    Pre-train V network using sampled points to match constraint structure.
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

    # Convert ranges to torch
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t    = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    unsafe_t  = torch.as_tensor(x_unsafe_range, dtype=torch.float32, device=device)
    init_t    = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = x_range_t.shape[0]
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
        x_unsafe = _sample_in_box(unsafe_t, n_each)
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.pretrain_unsafe_target - v_unsafe).sum()

        # 4) samples outside (goal ∪ unsafe) -> enforce v(x) >= pretrain_goal_target
        x_others_list = []
        need = n_each
        max_tries = 20
        tries = 0
        while need > 0 and tries < max_tries:
            tries += 1
            x_cand = torch.rand(max(need * 4, 32), D, device=device) * span + low
            cand_in_goal = _in_box(x_cand, goal_t)
            cand_in_unsafe = _in_box(x_cand, unsafe_t)
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

        # 5) goal-range samples -> enforce v(x) <= pretrain_goal_target
        x_goal = _sample_in_box(goal_t, n_each)
        v_goal = model(x_goal).squeeze(-1)
        v_loss_inside_goal = F.relu(v_goal - params.constraints.pretrain_goal_target).sum()

        # Total V loss
        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
            + v_loss_inside_goal
            + v_loss_others
        )

        # Phi loss on x_others -> enforce phi(x) <= 0
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
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.cpu().clone() for k, v in control_net.state_dict().items()}

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
    control_net: nn.Module = None,
    n_samples_per_region: int = 100,
    sample_weight: float = 0.1
):
    """Train the value network using CROWN bounds + hybrid sampling."""
    print("\n" + "="*80)
    print("HYBRID TRAINING (CROWN bounds + sampling)")
    print("="*80)
    print(f"Sample weight: {sample_weight}, samples per region: {n_samples_per_region}")

    V_net = V_net.to(device)

    # Collect ALL cells for V network
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

    # Prepare input bounds
    print("\nPreparing concatenated input bounds for V network...")
    if total_cells_V > 0:
        input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)
    else:
        input_lowers_all = torch.empty(0, params.network.n_inputs, device=device)
        input_uppers_all = torch.empty(0, params.network.n_inputs, device=device)

    # Sampling helper
    def sample_from_cells(cells, n_samples):
        """Sample points uniformly from cells."""
        if len(cells) == 0 or n_samples == 0:
            return torch.empty(0, params.network.n_inputs, device=device)
        samples = []
        for lo, hi in cells:
            n = max(1, n_samples // len(cells))
            s = torch.rand(n, params.network.n_inputs, device=device) * (hi - lo) + lo
            samples.append(s)
        return torch.cat(samples, dim=0) if samples else torch.empty(0, params.network.n_inputs, device=device)

    # Generate fixed sample points at start
    print("\nGenerating fixed sample points...")
    fixed_samples = {}
    if n_samples_per_region > 0:
        for region_name in ['init', 'unsafe', 'outside', 'generator']:
            fixed_samples[region_name] = sample_from_cells(region_cells[region_name], n_samples_per_region)
            print(f"  {region_name}: {fixed_samples[region_name].shape[0]} samples")

    # Create CROWN cache for V
    print(f"\nInitializing CROWN cache for ALL {total_cells_V} V cells...")
    crown_cache_all = SymbolicCROWNCache(
        model=V_net,
        num_cells=total_cells_V,
        input_dim=params.network.n_inputs,
        device=device
    )

    # Create CROWN cache for generator (Phi)
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
        input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

    # Setup optimizer
    beta_s_value = params.constraints.beta_s
    print(f"\nUsing CONSTANT beta_s = {beta_s_value}")

    opt_params = list(V_net.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=300,
        verbose=True,
        min_lr=1e-6,
        threshold=1e-3
    )

    # Training loop
    loss_history = []
    refinement_epochs = {'outside': [], 'generator': []}
    start_time = time.time()
    final_beta_s = params.constraints.beta_s

    # Constraint switching state
    locked_to_all_sum = False
    current_sum_constraint = None
    constraint_largest_counts = {}
    prev_total_loss = None

    for epoch in range(params.training.num_epochs):
        V_net.train()

        optimizer.zero_grad()
        if params.compute_V:
            # Compute bounds for ALL V cells
            if total_cells_V > 0:
                v_lowers_all, v_uppers_all = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all = torch.tensor([], device=device)
                v_uppers_all = torch.tensor([], device=device)

            # Split bounds by region
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

                phi_upper_failing_mask = phi_uppers > 0.0
                num_total_failing = phi_upper_failing_mask.sum().item()
                phi_upper_failing_mask_relax = phi_uppers > -100.0
            else:
                phi_lowers = torch.tensor([], device=device)
                phi_uppers = torch.tensor([], device=device)
                current_gen_weight = 0.0
                num_total_failing = 0

        current_beta_s = beta_s_value

        # Compute total loss
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

        if params.compute_GV:
            loss_kwargs.update({
                'Phi_lower': phi_lowers,
                'Phi_upper': phi_uppers,
                'generator_weight': current_gen_weight
            })

        prev_locked = locked_to_all_sum
        prev_sum_constraint = current_sum_constraint
        total_loss, loss_dict, locked_to_all_sum, current_sum_constraint, constraint_largest_counts = compute_total_loss_bounds(**loss_kwargs)

        # Hybrid sampling: add direct sample-based losses using fixed samples
        sample_loss = torch.tensor(0.0, device=device)
        if n_samples_per_region > 0:
            if fixed_samples['init'].numel() > 0:
                sample_loss += F.relu(V_net(fixed_samples['init']).squeeze(-1) - 1.0).mean()
            if fixed_samples['unsafe'].numel() > 0:
                sample_loss += F.relu(params.constraints.beta_ra - V_net(fixed_samples['unsafe']).squeeze(-1)).mean()
            if fixed_samples['outside'].numel() > 0:
                sample_loss += F.relu(current_beta_s - V_net(fixed_samples['outside']).squeeze(-1)).mean()
            if fixed_samples['generator'].numel() > 0 and epoch >= params.training.generator_start_epoch:
                x_gen_s = fixed_samples['generator'].clone().requires_grad_(True)
                sample_loss += F.relu(GV_net(x_gen_s).squeeze(-1)).mean()

            total_loss = total_loss + sample_weight * sample_loss

        prev_total_loss = total_loss.item()

        # Reset optimizer if switching
        should_skip_step = False
        if locked_to_all_sum and not prev_locked:
            print(f"  Resetting optimizer and scheduler state after switching to all sums")
            print(f"  constraint_largest_counts: {constraint_largest_counts}")
            optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=300,
                verbose=True, min_lr=1e-6, threshold=1e-3
            )
            should_skip_step = True
        elif current_sum_constraint != prev_sum_constraint and prev_sum_constraint is not None:
            print(f"  Resetting optimizer and scheduler state after switching focus")
            print(f"  constraint_largest_counts: {constraint_largest_counts}")
            optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=300,
                verbose=True, min_lr=1e-6, threshold=1e-3
            )
            should_skip_step = True

        # Backward pass
        if not should_skip_step:
            total_loss.backward()

        # Recompute bounds after optimizer step
        V_net.eval()
        with torch.no_grad():
            if total_cells_V > 0:
                v_lowers_all_updated, v_uppers_all_updated = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all_updated = torch.tensor([], device=device)
                v_uppers_all_updated = torch.tensor([], device=device)

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

        scheduler.step(total_loss.item())

        beta_s_check = current_beta_s if isinstance(current_beta_s, float) else current_beta_s.item()

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
            if len(bounds_updated['goal'][0]) > 0:
                region_losses['goal'] = F.relu(bounds_updated['goal'][1] - 1.0).sum().item()

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
                    elif highest_loss_region == 'goal':
                        failing_mask = bounds_updated['goal'][1] > 1.0

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


            # if((epoch + 1) % 502 == 0):
            #     merged_cells, num_merges = merge_passing_neighbor_cells(
            #         region_cells['outside'],
            #         outside_failing_mask_relax,
            #         max_passes=8,
            #         max_merges=None,
            #         seed=0,
            #         eps=1e-6,
            #     )
            #     region_cells['outside'] = merged_cells
            #     print(f"[Merge-Outside] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
            #     needs_cache_rebuild = True

        # Adaptive refinement for generator
        if params.compute_GV and epoch >= params.training.generator_start_epoch:
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):

                REFINE_INTERVAL = 500
                REFINE_FACTOR = 2
                MAX_CELLS = 30000

                if epoch > 4000:
                    REFINE_INTERVAL = 250

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

            if((epoch + 1) % 502 == 0):
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['generator'],
                    phi_upper_failing_mask_relax,
                    max_passes=8,
                    max_merges=None,
                    seed=0,
                    eps=1e-6,
                )
                region_cells['generator'] = merged_cells
                print(f"[Merge-Generator] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Logging
        if epoch % 10 == 0 or epoch == params.training.num_epochs - 1 or epoch == 0:
            if n_samples_per_region > 0 and sample_loss.item() > 0:
                loss_dict['sample'] = sample_loss.item()
            if locked_to_all_sum:
                current_sum_constraint = 'all'
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV, focus=current_sum_constraint)
            loss_dict['epoch'] = epoch
            loss_history.append(loss_dict.copy())

        # Early stopping check
        with torch.no_grad():
            all_satisfied = True
            if params.compute_V:
                unsafe_satisfied = (bounds_updated['unsafe'][0].min() >= params.constraints.beta_ra)
                goal_satisfied = (bounds_updated['goal'][1].max() <= 1.0)
                init_satisfied = (bounds_updated['init'][1].max() <= 1.0)
                outside_satisfied = (bounds_updated['outside'][0].min() >= 0.0)
                all_satisfied = all_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied and goal_satisfied

            if params.compute_GV and epoch >= params.training.generator_start_epoch:
                if len(phi_uppers) > 0:
                    generator_satisfied = (phi_uppers.max() <= 0.0)
                    all_satisfied = all_satisfied and generator_satisfied
                else:
                    all_satisfied = False

            can_early_exit = all_satisfied
            if params.compute_GV and epoch < params.training.generator_start_epoch + 1000:
                can_early_exit = False

            if can_early_exit:
                print("\n" + "="*80)
                print("ALL CONSTRAINTS SATISFIED - EARLY STOPPING!")
                print("="*80)
                print(f"Training converged at epoch {epoch}")

                loss_parts = []
                if params.compute_V:
                    loss_parts.append(f"Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
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

        # Detailed evaluation
        if (epoch % 1000 == 0) or epoch == params.training.num_epochs - 1:
            print(f"\nEpoch {epoch} - Detailed Evaluation:")
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
            print()

        # Optimizer step
        if not should_skip_step:
            optimizer.step()

        # Rebuild CROWN caches if needed
        if needs_cache_rebuild:
            print(f"  Rebuilding CROWN caches with new generator cells...")

            all_cells_V = []
            for name in region_order_V:
                all_cells_V.extend(region_cells[name])

            total_cells_V = len(all_cells_V)
            input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

            print(f"    Rebuilding V cache with {total_cells_V} cells...")
            crown_cache_all = SymbolicCROWNCache(
                model=V_net,
                num_cells=total_cells_V,
                input_dim=params.network.n_inputs,
                device=device
            )

            num_generator_cells = len(region_cells['generator'])
            print(f"    Rebuilding Phi cache with {num_generator_cells} cells...")
            crown_cache_phi = SymbolicCROWNCache_Phi(
                phi_module=GV_net,
                num_cells=num_generator_cells,
                input_dim=params.network.n_inputs,
                device=device
            )

            input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

            for name in region_order_V:
                cell_counts_V[name] = len(region_cells[name])

            all_cells_V = []
            for name in region_order_V:
                all_cells_V.extend(region_cells[name])
            input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

            # Regenerate fixed samples with new cells
            if n_samples_per_region > 0:
                print(f"    Regenerating fixed samples...")
                for region_name in ['init', 'unsafe', 'outside', 'generator']:
                    fixed_samples[region_name] = sample_from_cells(region_cells[region_name], n_samples_per_region)

            print(f"  Caches rebuilt successfully!")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return loss_history, final_beta_s, refinement_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    args = parser.parse_args()

    print("="*80)
    print("4D DOUBLE INTEGRATOR CONTROL SYNTHESIS")
    print("="*80)

    # ========================================================================
    # 1. HYPERPARAMETERS
    # ========================================================================
    print("\n" + "="*80)
    print("HYPERPARAMETERS")
    print("="*80)

    params = Hyperparameters.default()

    # Network configuration
    params.network.n_inputs = 4  # [p1, v1, p2, v2]
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 16
    params.network.input_scale = [6.0, 3.0, 6.0, 3.0]  # scale positions more than velocities
    params.network.scale_factor = 20.0

    # Training configuration
    params.training.learning_rate = 0.005
    params.training.num_epochs = 50000
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0

    # Discretization
    params.discretization.n_goal = 1
    params.discretization.n_outside_goal = 1
    params.discretization.n_generator = 1
    params.discretization.n_unsafe = 1
    params.discretization.n_init = 1

    # Constraints
    params.training.learnable_beta_s = False
    params.constraints.beta_s = 0.0
    params.constraints.beta_ra = 20.0  # Much lower than Lorenz (easier problem)

    # What to compute
    params.compute_V = True
    params.compute_GV = True

    # ========================================================================
    # 2. SYSTEM DYNAMICS
    # ========================================================================
    print("\n" + "="*80)
    print("SYSTEM DYNAMICS")
    print("="*80)

    # Control network: 4 inputs → 2 outputs (accelerations)
    u_nn = DoubleIntegratorControlNN(hidden_dim=128)

    def f_ol(x: torch.Tensor) -> torch.Tensor:
        """
        Open-loop drift for double integrator.
        x: (N, 4) = [p1, v1, p2, v2]
        returns: (N, 4) drift
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        p1 = x[:, 0]
        v1 = x[:, 1]
        p2 = x[:, 2]
        v2 = x[:, 3]

        # dp1/dt = v1
        # dv1/dt = 0 (coasting)
        # dp2/dt = v2
        # dv2/dt = -0.5*v2 (light damping)
        return torch.stack([
            v1,
            torch.zeros_like(v1),
            v2,
            -0.5 * v2
        ], dim=1)

    # Constant diagonal diffusion
    g_coeffs = torch.tensor([0.0, 0.15, 0.0, 0.15], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion (diagonal, constant).
        Noise only on velocities.
        """
        base = g_coeffs.to(device=x.device, dtype=x.dtype)

        if x.dim() == 1:
            return base
        elif x.dim() == 2:
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)
        else:
            raise ValueError(f"g(x) expects x of shape (D,) or (N, D), got {tuple(x.shape)}")

    # Closed-loop drift: f_cl(x) = f_ol(x) + [0, u1, 0, u2]
    # where u = u_nn(x) outputs [u1, u2]
    class ClosedLoopDriftDouble(nn.Module):
        def __init__(self, f_ol_fn, controller):
            super().__init__()
            self.f_ol_fn = f_ol_fn
            self.controller = controller

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (N, 4)
            u: (N, 2) = [a1, a2]
            Returns: (N, 4) drift
            """
            f_open = self.f_ol_fn(x)  # (N, 4)
            u = self.controller(x)     # (N, 2)

            # CROWN-friendly: use torch.cat instead of indexing
            # Split f_open into components
            # f_open = [v1, 0, v2, -0.5*v2]
            # control adds [0, a1, 0, a2]
            # result = [v1, a1, v2, -0.5*v2 + a2]

            return torch.cat([
                f_open[:, 0:1],           # v1
                f_open[:, 1:2] + u[:, 0:1],  # 0 + a1 = a1
                f_open[:, 2:3],           # v2
                f_open[:, 3:4] + u[:, 1:2],  # -0.5*v2 + a2
            ], dim=1)

    f_cl_module = ClosedLoopDriftDouble(f_ol, u_nn).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    # ========================================================================
    # 3. SPATIAL REGIONS
    # ========================================================================
    print("\n" + "="*80)
    print("SPATIAL REGIONS")
    print("="*80)

    # Init: particle 1 at bottom-left, particle 2 at top-right
    init_range = np.array([
        [-2.0, -1.0],   # p1
        [-0.2,  0.2],   # v1
        [ 1.0,  2.0],   # p2
        [-0.2,  0.2],   # v2
    ], dtype=np.float32)

    goal_range = np.array([
        [ 0.0,  2.0],   # p1 at top-right
        [-1.3,  0.3],   # v1 small
        [-2.0, 0.0],   # p2 at bottom-left
        [-1.3,  0.3],   # v2 small
    ], dtype=np.float32)

    unsafe_range = np.array([
        [ 2.5,  3.0],   # p1 far right
        [0.5,  1.5],   # any v1
        [ 2.5,  3.0],   # p2 also far right  
        [0.5,  1.5],   # any v2
    ], dtype=np.float32)

    full_range = np.array([
        [-3.0,  3.0],   # p1
        [-1.5,  1.5],   # v1
        [-3.0,  3.0],   # p2
        [-1.5,  1.5],   # v2
    ], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe = Region(unsafe_range)
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
        use_radial_generator=False
    )

    # ========================================================================
    # 6. TRAINING OR LOADING
    # ========================================================================
    device = params.training.device
    delta = 0.5
    params.constraints.pretrain_goal_target = params.constraints.beta_s - delta
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = params.constraints.beta_s + delta
    params.constraints.pretrain_phi_target = 0.0

    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")
        print_training_config(params)
        print(f"\n{dynamics}")
        print(f"V network: {V_net}")
        print(f"\nUsing device: {device}")

        # Pre-training
        ENABLE_PRETRAINING = True
        PRETRAIN_EPOCHS = 5000
        PRETRAIN_LR = 1e-2

        if ENABLE_PRETRAINING:
            pretrain_network_samples(
                model=V_net,
                x_goal_range=goal_range,
                x_unsafe_range=unsafe_range,
                x_init_range=init_range,
                x_range=full_range,
                params=params,
                GV_net=GV_net,
                num_epochs=PRETRAIN_EPOCHS,
                lr=PRETRAIN_LR,
                device=params.training.device,
                n_each=100,
                control_net=u_nn
            )
            print(f"Pretraining completed!\n")

        # Main training
        print(f"\nUsing device: {device}")
        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            visualize_interval=0,  # Disable visualization for now
            control_net=u_nn,
            n_samples_per_region=200,  # Hybrid: sample 200 pts per region
            sample_weight=1.0,  # Weight for sample-based loss
        )

        # Final evaluation
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

        # Save bundle
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
        # Load and evaluate
        print("\n" + "="*80)
        print("LOADING SAVED BUNDLE (skip training)")
        print("="*80)

        bundle = load_eval_bundle(bundle_path, map_location="cpu")
        params = Hyperparameters.from_dict(bundle["hyperparameters"])
        region_cells = bundle["region_cells"]

        V_net = create_V(params.network).to(device)
        V_net.load_state_dict(bundle["V_state_dict"])

        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network,
            training_config=params.training
        ).to(device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"])

        u_nn = DoubleIntegratorControlNN()
        if bundle["control_state_dict"] is not None:
            u_nn.load_state_dict(bundle["control_state_dict"])

        region_cells = {
            k: [(lo.to(device), hi.to(device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }

        final_beta_s = bundle["final_beta_s"]
        loss_history = bundle["loss_history"]
        refinement_epochs = bundle["refinement_epochs"]

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


if __name__ == '__main__':
    main()
