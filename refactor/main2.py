"""
Bound-based training script using CROWN for NN verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
OUTPUT_DIR = ROOT / "refactor"/ "outputs"

# Set random seed immediately after imports (matching testing_simple3.py)
torch.manual_seed(0)

from hyperparameters import Hyperparameters
from dynamics import Dynamics, ClosedLoopDrift
from regions import Regions, Region
from network import create_V
from control_network import LinearControlNN # [control synthesis]
from phi_module import create_GV
from discretization import discretize_regions
from crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from training_utils import (
    compute_total_loss_bounds,
    compute_loss_goal_bounds,
    evaluate_constraints,
    print_loss_summary,
    print_constraint_summary,
    refine_failing_cells,
    clear_goal_samples_cache
)
from utils import cleanup_and_setup_directories, print_training_config
from visualization import (
    visualize_training_progress,
    create_summary_plots
)
from controls import u_control


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
    

def pretrain_network_samples(model, goal_region, unsafe_region, init_region, full_region,
                             params, GV_net=None, num_epochs=1000, lr=0.01, device='cpu', control_net=None):
    """Pre-train V network using sampled points to match constraint structure.

    Args:
        model: V network to train
        goal_region: Goal Region object
        unsafe_region: Unsafe Region object (can be union)
        init_region: Init Region object
        full_region: Full state space Region object
        params: Hyperparameters object (for target values)
        GV_net: Optional GV network for additional loss (if None, only V is trained)
        control_net: Optional control network
    """
    print("\n" + "="*80)
    print("PRE-TRAINING: Structure-Aware Initialization (Sample-Based)")
    print("="*80)

    # Include control network parameters if provided
    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    # Use provided GV module if available
    if GV_net is not None:
        print(f"  GV (Φ) pre-training ENABLED (using provided GV_net)")
    else:
        print(f"  GV (Φ) pre-training DISABLED (no GV_net provided)")

    # Track best model
    best_loss = float('inf')
    best_model_state = None
    best_control_state = None

    # Get full range bounds for sampling
    full_bounds = full_region.bounds

    for epoch in range(num_epochs):
        # Sample points uniformly from state space
        x = torch.rand(1000, 2, device=device)
        x[:, 0] = x[:, 0] * (full_bounds[0, 1] - full_bounds[0, 0]) + full_bounds[0, 0]
        x[:, 1] = x[:, 1] * (full_bounds[1, 1] - full_bounds[1, 0]) + full_bounds[1, 0]

        # Assign target V values based on region
        target_v = torch.zeros(1000, device=device)
        for i in range(1000):
            point = x[i].cpu().numpy()

            # Use Region.contains() which handles unions properly
            in_goal = goal_region.contains(point)
            in_unsafe = unsafe_region.contains(point)
            in_init = init_region.contains(point)

            if in_goal:
                target_v[i] = params.constraints.pretrain_goal_target + 0.001 * torch.rand(1, device=device).item()
            elif in_unsafe:
                target_v[i] = params.constraints.pretrain_unsafe_target + 3.0 * torch.rand(1, device=device).item()
            elif in_init:
                target_v[i] = params.constraints.pretrain_init_target + 0.05 * torch.rand(1, device=device).item()
            else:
                target_v[i] = params.constraints.pretrain_goal_target + 0.3 * torch.rand(1, device=device).item()

        # V network loss
        v_output = model(x).squeeze()
        loss_v = F.mse_loss(v_output, target_v)

        # GV (Φ) network loss
        loss_phi = torch.tensor(0.0, device=device)
        if GV_net is not None:
            # Sample from generator region: X \ (Goal ∪ Unsafe)
            x_gen = torch.rand(100, 2, device=device)
            x_gen[:, 0] = x_gen[:, 0] * (full_bounds[0, 1] - full_bounds[0, 0]) + full_bounds[0, 0]
            x_gen[:, 1] = x_gen[:, 1] * (full_bounds[1, 1] - full_bounds[1, 0]) + full_bounds[1, 0]

            # Filter to generator region only using Region.contains()
            in_generator = []
            for i in range(len(x_gen)):
                point = x_gen[i].cpu().numpy()
                in_goal = goal_region.contains(point)
                in_unsafe = unsafe_region.contains(point)
                in_generator.append(not (in_goal or in_unsafe))

            in_generator_mask = torch.tensor(in_generator, device=device)
            x_gen_filtered = x_gen[in_generator_mask]

            if len(x_gen_filtered) > 0:
                phi_output = GV_net(x_gen_filtered).squeeze()
                loss_phi = torch.nn.functional.relu(phi_output + params.constraints.pretrain_phi_target).sum()

        # Combined loss and optimization step
        total_loss = loss_v + 1.0 * loss_phi

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.cpu().clone() for k, v in control_net.state_dict().items()}

        if epoch % 100 == 0:
            if GV_net is not None:
                print(f"  Epoch [{epoch}/{num_epochs}]: V_loss={loss_v.item():.6f}, Φ_loss={loss_phi.item():.6f}, Total={total_loss.item():.6f}")
            else:
                print(f"  Epoch [{epoch}/{num_epochs}]: V_loss={loss_v.item():.6f}")

            if control_net is not None and epoch % 500 == 0:
                print("  Control Network Parameters:")
                for name, param in control_net.named_parameters():
                    if param.requires_grad:
                        param_str = str(param.data.numpy()).replace('\n', '\n    ')
                        print(f"    {name}:\n    {param_str}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\n  Best loss: {best_loss:.6f}")

    print("="*80)
    networks_trained = []
    if GV_net is not None:
        networks_trained.append("V and GV (Φ)")
    else:
        networks_trained.append("V")
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

    # Training loop
    loss_history = []
    refinement_epochs = {'outside': [], 'generator': []}
    start_time = time.time()

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
            'epoch': epoch
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

        total_loss, loss_dict = compute_total_loss_bounds(**loss_kwargs)

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

        # Update scheduler after bounds recomputation
        scheduler.step(total_loss.item())

        # Get current beta_s value for constraint checks
        beta_s_check = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s

        # Adaptive refinement for V outside region cells
        needs_cache_rebuild = False
        if params.compute_V and len(bounds_updated['outside'][0]) > 0:
            # Track failing cells in outside region
            outside_failing_mask = bounds_updated['outside'][0] < beta_s_check
            num_outside_failing = outside_failing_mask.sum().item()

            # Ensure bounds match current cell count
            if len(outside_failing_mask) == len(region_cells['outside']) and num_outside_failing > 0:
                REFINE_INTERVAL = 500
                REFINE_FACTOR = 2
                MAX_CELLS = 2000

                if epoch > 2500:
                    REFINE_INTERVAL = 100

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

        # Adaptive refinement for generator cells (before optimizer step)
        if params.compute_GV:
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):

                # Refinement parameters (matching testing_simple3.py)
                REFINE_INTERVAL = 500  # Refine every 100k epochs
                REFINE_FACTOR = 2  # Split into 2x2 subcells
                MAX_CELLS = 2000  # Don't refine if we already have too many cells

                # Adjust interval for later epochs
                if epoch > 2500:
                    REFINE_INTERVAL = 100

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
                for name, param in control_net.named_parameters():
                    if param.requires_grad:
                        print(f" [Controller Params] {name} = {param.data}")
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
                _, goal_satisfied = compute_loss_goal_bounds(V_net, regions.goal, bounds_updated["goal"][0], bounds_updated["goal"][1], beta_s_check, device=device, show=show, check=True, n_samples=10000)
                unsafe_satisfied = (bounds_updated['unsafe'][0].min() >= params.constraints.beta_ra)
                init_satisfied = (bounds_updated['init'][0].min() >= beta_s_check and bounds_updated['init'][1].max() <= 1.0)
                outside_satisfied = (bounds_updated['outside'][0].min() >= beta_s_check)
                all_satisfied = all_satisfied and goal_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied

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
                    loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
                if params.compute_GV:
                    loss_parts.append(f"Gen={loss_dict['generator']:.4f}")
                print(f"Final losses: {', '.join(loss_parts)}")

                if control_net is not None:
                    for name, param in control_net.named_parameters():
                        if param.requires_grad:
                            print(f" [Controller Params] {name} = {param.data}")
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
            print()

            # Visualize progress
            if visualize_interval > 0 and epoch % visualize_interval == 0:
                print(f"  Creating visualization for epoch {epoch}...")
                visualize_training_progress(
                    V_net, GV_net, regions, region_cells,
                    epoch=epoch,
                    output_dir="training_progress"
                )
        
        # Optimizer step
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
    final_beta_s = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s
    if learnable_beta_s is not None:
        print(f"\nFinal learned β_s = {final_beta_s:.4f}")

    return loss_history, final_beta_s, refinement_epochs

class TanhPolicy(nn.Sequential):
    """
    A policy with three layers and tanh activations.
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 1,
        n_hidden: int = 64,
        device: torch.device | str = "cpu"
    ):
        super().__init__(
            nn.Linear(n_in, n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out, dtype=torch.float32, device=device),
        )

def main():
    """Main training function."""
    print("="*80)
    print("MODULAR RL VERIFICATION - BOUND-BASED TRAINING")
    print("="*80)

    # 

    # ========================================================================
    # 1. HYPERPARAMETERS
    # ========================================================================
    print("\n" + "="*80)
    print("HYPERPARAMETERS")
    print("="*80)

    params = Hyperparameters.default()

    # Customize configuration
    params.network.n_hidden_1 = 256
    params.network.n_hidden_2 = 32
    params.network.input_scale = [2*np.pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000  # Adjust as needed
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 1.0  # Enable generator constraint
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 4
    params.discretization.n_outside_goal = 5
    params.discretization.n_generator = 1  # Will be overridden by radial discretization
    params.discretization.n_unsafe = 3
    params.discretization.n_init = 3

    # Set beta_s to a value (constant), or set to None to make it learnable
    # If learnable_beta_s is True, this value will be used as initialization
    params.training.learnable_beta_s = False  # Set to True to make beta_s learnable
    params.constraints.beta_s = 0.6
    params.constraints.beta_ra = 20.0

    # Control what to compute during training
    params.compute_V = True
    params.compute_GV = True

    # params.training.random_seed = 0

    # ========================================================================
    # CLEANUP: Remove old results and training progress
    # ========================================================================
    cleanup_and_setup_directories(["results", "training_progress"])
    print_training_config(params)

    # ========================================================================
    # 2. SYSTEM DYNAMICS
    # ========================================================================
    print("\n" + "="*80)
    print("SYSTEM DYNAMICS")
    print("="*80)

    # Option 2: Neural network control (implements same K @ x)
    # u_nn = LinearControlNN(prior_knowledge=False)
    device_str = "cpu"
    rl_policy_net = TanhPolicy(2, 1, 64, device=device_str)
    rl_policy_net.load_state_dict(torch.load("rl_agent/pendulum_policy.pt",
                                             map_location=device_str,
                                             weights_only=True))
    rl_policy_net.requires_grad_(False)
    rl_policy_net.eval()
    u_nn = rl_policy_net

    # Pendulum parameters (from paper)
    grav = 9.81      # gravity (m/s²)
    L = 0.5       # pendulum length (m)
    m = 0.15      # ball mass (kg)
    b = 0.1       # friction coefficient
    M = 6.0       # maximum torque
    sigma = 2.0   # diffusion coefficient

    def f_ol(x: torch.Tensor) -> torch.Tensor:
        """
        Open-loop drift for pendulum (without control).

        Args:
            x: State (x₁=angle, x₂=angular velocity)
               Shape: (batch_size, 2) or (2,)

        Returns:
            f(x) = [x₂, (g/L)sin(x₁) - (b/(m·L²))·x₂]
        """
        # Batch
        x1 = x[:, 0]  # angle
        x2 = x[:, 1]  # angular velocity
        f1 = x2
        f2 = (grav/L) * torch.sin(x1) - (b/(m * L**2)) * x2
        return torch.stack([f1, f2], dim=1)

    # Wrap controller to apply actuation matrix B = [0, M/(mL²)]
    class ActuatedController(nn.Module):
        """Applies actuation matrix to controller output."""
        def __init__(self, controller, actuation_gain):
            super().__init__()
            self.controller = controller
            self.actuation_gain = actuation_gain

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply B @ u(x) where B = [0, M/(mL²)]."""
            u = self.controller(x)  # Shape: (batch, 1) or (batch,)
            if u.dim() == 1:
                u = u.unsqueeze(-1)  # (batch, 1)

            # Create [0, actuation_gain * u(x)]
            batch_size = x.shape[0]
            zeros = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            actuated_u = torch.cat([zeros, self.actuation_gain * u], dim=1)  # (batch, 2)
            return actuated_u

    # Create actuated controller
    actuation_gain = M / (m * L**2)
    u_actuated = ActuatedController(u_nn, actuation_gain)

    # Create diffusion: [0, σ]
    g_coeffs = torch.tensor([0.0, sigma], dtype=torch.float32)
    def g(x: torch.Tensor) -> torch.Tensor:
        """Diffusion: [0, σ]"""
        return g_coeffs.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1)

    # Create closed-loop drift: f_cl(x) = f_ol(x) + B @ u(x)
    f_cl_module = ClosedLoopDrift(f_ol, u_actuated).to(params.training.device)

    # For pretraining compatibility
    # u_fn = u_control(u_nn)
    # def F_CL(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
    #     """Closed-loop drift: f_cl(x) = f(x) + u(x)"""
    #     return f_ol(x) + u_fn(x)

    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    print(f"\n{dynamics}")

    # ========================================================================
    # 3. SPATIAL REGIONS
    # ========================================================================
    print("\n" + "="*80)
    print("SPATIAL REGIONS")
    print("="*80)

    init_range = np.array([[3*np.pi/4, 5*np.pi/4], [-1.0, 1.0]], dtype=np.float32)
    goal_range = np.array([[-np.pi/2, np.pi/2], [-4.0, 4.0]], dtype=np.float32)
    full_range = np.array([[-2*np.pi, 2*np.pi], [-20.0, 20.0]], dtype=np.float32)

    # Create unsafe region as union of two rectangles (matching paper exactly)
    unsafe_left_range = np.array([[-2*np.pi, -3*np.pi/2], [-20.0, -10.0]], dtype=np.float32)
    unsafe_right_range = np.array([[3*np.pi/2, 2*np.pi], [10.0, 20.0]], dtype=np.float32)

    # Create regions with union unsafe region
    init = Region(init_range)
    goal = Region(goal_range)
    unsafe_left = Region(unsafe_left_range)
    unsafe_right = Region(unsafe_right_range)
    unsafe = Region.union(unsafe_left, unsafe_right)
    full = Region(full_range)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # ========================================================================
    # 4. CREATE NETWORKS
    # ========================================================================
    print("\n" + "="*80)
    print("NETWORKS")
    print("="*80)

    V_net = create_V(params.network)
    print(f"V network: {V_net}")

    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        training_config=params.training
    )
    print(f"GV network created")

    # ========================================================================
    # 4.5. PRE-TRAINING (Optional)
    # ========================================================================
    ENABLE_PRETRAINING = True  # Set to True to enable
    PRETRAIN_EPOCHS = 2500
    PRETRAIN_LR = 0.01

    if ENABLE_PRETRAINING:

        delta = 0.1
        params.constraints.pretrain_goal_target = params.constraints.beta_s - delta
        params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
        params.constraints.pretrain_init_target = params.constraints.beta_s + delta
        params.constraints.pretrain_phi_target = 1.0

        pretrain_network_samples(
            model=V_net,
            goal_region=goal,
            unsafe_region=unsafe,
            init_region=init,
            full_region=full,
            params=params,
            GV_net=GV_net,  # Pass GV_net if you want GV loss, None otherwise
            num_epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            device=params.training.device,
            # control_net=u_nn
        )

        print(f"Pretraining completed!\n")

    # ========================================================================
    # 5. DISCRETIZE REGIONS
    # ========================================================================
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=True  # Use radial + clipping for generator
    )

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
        # control_net=u_nn,   # [control synthesis]
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

    # ========================================================================
    # 9. SAVE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    # save the certificate
    save_path = OUTPUT_DIR / "trained_model_bounds.pth"
    torch.save({
        'model_state_dict': V_net.state_dict(),
        'hyperparameters': params.to_dict(),
        # 'dynamics': {'F': F_CL, 'G': g},
        'regions': regions.to_dict(),
        'final_results': results,
        'loss_history': loss_history,
        'training_method': 'bounds',
        'final_beta_s': final_beta_s  # Save the final beta_s value (learned or constant)
    }, save_path)
    print(f"Certificate Network saved to: {save_path}")

    control_save_path = OUTPUT_DIR / "control_net.pth"
    torch.save({
        'model_state_dict': u_nn.state_dict(),
    }, control_save_path)
    print(f"Control Network saved to: {control_save_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - Certificate Network: {save_path}")
    print(f"  - Control Network: {control_save_path}")

    print(f"  - Plots: results/")
    if loss_history:
        print(f"  - Training progress: training_progress/")


if __name__ == '__main__':
    main()
