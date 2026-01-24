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
from src.control_network import LorentzLinearControlNN # [control synthesis]
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
from src.save_load_utils import save_eval_bundle_opt, load_eval_bundle, log_loaded_training_epochs, enable_terminal_logging
from src.utils import cleanup_and_setup_directories, print_training_config
from src.visualization import (
    # visualize_training_progress,
    create_summary_plots
)
from main import(
    animate_closed_loop_3d_state_control
)

# Set random seed immediately after imports (matching testing_simple3.py)
torch.manual_seed(0)


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
    region_order_V = ['init', 'goal', 'unsafe', 'outside', 'boundary']  # Added boundary

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
            'w_soft': 2000.0,
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
                'V_outside_upper': bounds['outside'][1],
                'V_boundary_lower': bounds['boundary'][0],
                'V_boundary_upper': bounds['boundary'][1]
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
                REFINE_INTERVAL = 100
                REFINE_FACTOR = 2
                MAX_CELLS = 100000
                if (epoch == 1 or (epoch + 1) % REFINE_INTERVAL == 0) and \
                    len(region_cells['outside']) < MAX_CELLS:
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['outside'],
                        outside_failing_mask,
                        REFINE_FACTOR,
                        N_to_refine=100,
                    )
                    region_cells['outside'] = new_cells
                    print(f"[Refine-Outside] Epoch {epoch+1}: {num_outside_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['outside'].append(epoch + 1)

            if((epoch + 1) % 501 == 0):
                outside_failing_mask_relax = bounds_updated['outside'][0] <= 8.0
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

        # Unsafe refinement
        if params.compute_V and len(bounds_updated['unsafe'][0]) > 0:
            # Track failing cells in outside region
            unsafe_failing_mask = bounds_updated['unsafe'][0] <= params.constraints.beta_ra
            num_unsafe_failing = unsafe_failing_mask.sum().item()
            # Ensure bounds match current cell count
            if num_unsafe_failing > 0:
                REFINE_INTERVAL = 100
                REFINE_FACTOR = 2
                MAX_CELLS = 100000
                if (epoch == 1 or (epoch + 1) % REFINE_INTERVAL == 0) and \
                    len(region_cells['unsafe']) < MAX_CELLS:
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['unsafe'],
                        unsafe_failing_mask,
                        REFINE_FACTOR,
                        scores=-bounds_updated['unsafe'][0],
                        N_to_refine=100,
                    )
                    region_cells['unsafe'] = new_cells
                    print(f"[Refine-Unsafe] Epoch {epoch+1}: {num_unsafe_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True

        # Goal refinement
        if params.compute_V and len(bounds_updated['goal'][0]) > 0:
            # Track failing cells in outside region
            goal_failing_mask = bounds_updated['goal'][0] <= 0.0
            num_goal_failing = goal_failing_mask.sum().item()
            # Ensure bounds match current cell count
            if num_goal_failing > 0:
                REFINE_INTERVAL = 100
                REFINE_FACTOR = 2
                MAX_CELLS = 100000
                if (epoch == 1 or (epoch + 1) % REFINE_INTERVAL == 0) and \
                    len(region_cells['goal']) < MAX_CELLS:
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['goal'],
                        goal_failing_mask,
                        REFINE_FACTOR,
                        scores=-bounds_updated['goal'][0],
                        N_to_refine=100,
                    )
                    region_cells['goal'] = new_cells
                    print(f"[Refine-Goal] Epoch {epoch+1}: {num_goal_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True

        # Init refinement
        if params.compute_V and len(bounds_updated['init'][0]) > 0:
            # Track failing cells in outside region
            init_failing_mask = bounds_updated['init'][1] > 1.0
            num_init_failing = init_failing_mask.sum().item()
            # Ensure bounds match current cell count
            if num_init_failing > 0:
                REFINE_INTERVAL = 100
                REFINE_FACTOR = 2
                MAX_CELLS = 100000
                if (epoch == 1 or (epoch + 1) % REFINE_INTERVAL == 0) and \
                    len(region_cells['init']) < MAX_CELLS:
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['init'],
                        init_failing_mask,
                        REFINE_FACTOR,
                        scores=bounds_updated['init'][1],
                        N_to_refine=100,
                    )
                    region_cells['init'] = new_cells
                    print(f"[Refine-Init] Epoch {epoch+1}: {num_init_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True


        # Adaptive refinement for generator cells (before optimizer step)
        if params.compute_GV:
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):
                # Refinement parameters (matching testing_simple3.py)
                REFINE_INTERVAL = 100  # Refine every 100k epochs
                REFINE_FACTOR = 2  # Split into 2x2 subcells
                MAX_CELLS = 100000  # Don't refine if we already have too many cells
                # Check if it's time to refine
                if (epoch == 1 or (epoch + 1) % REFINE_INTERVAL == 0) and \
                    len(region_cells['generator']) < MAX_CELLS:
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
                phi_upper_failing_mask_relax = phi_uppers > -500.0
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
                    print(f"  Generator bounds: Φ ∈ [{phi_uppers.min().item():.6e}, {phi_uppers.max().item():.6e}]")
                    failing_cells = (phi_uppers > 0).sum().item()
                    print(f"  Generator failing cells: {failing_cells}/{len(phi_uppers)} ({100*failing_cells/len(phi_uppers):.1f}%)")

        # Early stopping check
        with torch.no_grad():
            # Check V constraints
            all_satisfied = True
            if params.compute_V:
                show = (epoch % 10 == 0)
                _, goal_satisfied = compute_loss_goal_bounds(V_net, regions.goal, bounds_updated["goal"][0], bounds_updated["goal"][1], beta_s_check, 
                                                             bounds_updated['outside'][0], device=device, show=show, check=True, n_samples=10000)
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
                print("ALL CONSTRAINTS SATISFIED")
                print("="*80)
                print(f"Training converged at epoch {epoch}")

                # Print relevant losses
                loss_parts = []
                if params.compute_V:
                    loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}, Boundary={loss_dict['boundary']:.4f}")
                if params.compute_GV:
                    loss_parts.append(f"Gen={loss_dict['generator']:.4f}")
                print(f"Final losses: {', '.join(loss_parts)}")

                # if control_net is not None:
                #     for name, param in control_net.named_parameters():
                #         if param.requires_grad:
                #             print(f" [Controller Params] {name} = {param.data}")

                final_beta_s = bounds_updated['outside'][0].min()
                print("final beta_s: {:.4f}".format(final_beta_s))

                # Evaluate the SAT models
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

                # Save the SAT models
                save_eval_bundle_opt(
                    OUTPUT_DIR,
                    V_net=V_net,
                    GV_net=GV_net,
                    control_net=control_net,
                    params=params,
                    regions=regions,
                    region_cells=region_cells,
                    final_beta_s=final_beta_s,
                    loss_history=loss_history,
                    refinement_epochs=refinement_epochs,
                    results=results,
                )

                # optimization-star
                if(params.constraints.beta_ra >= 20.0):
                    break
                params.constraints.beta_ra += 0.2
        
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
    params.discretization.n_unsafe = 10
    params.discretization.n_init = 16

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

    # Option 2: Neural network control (implements same K @ x)
    # u_nn = LinearControlNN(prior_knowledge=True, 
    #                        input_dim=params.network.n_inputs)
    # u_nn = NonlinearControlNN()
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
        training_config=params.training
    )

    # ========================================================================
    # 6.0 Setup saving
    # ========================================================================
    device = params.training.device
    params.constraints.pretrain_goal_target = params.constraints.beta_s
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = 1.0
    params.constraints.pretrain_phi_target = 0.0
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        # ========================================================================
        # 6.1. Opt bound
        # ========================================================================
        cleanup_and_setup_directories(["results_opt", "training_progress_opt"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log_opt.txt")
        print_training_config(params)
        print(f"\n{dynamics}")
        print(f"\nUsing device: {device}")

        bundle = load_eval_bundle(bundle_path, map_location="cpu")
        # rebuild region_cells (already cpu tensors)
        region_cells = bundle["region_cells"]
        # rebuild networks & load
        V_net.load_state_dict(bundle["V_state_dict"])
        u_nn.load_state_dict(bundle["control_state_dict"])

        train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            visualize_interval=1000,
            control_net=u_nn,   # [control synthesis]
        )
    else:
        print("\n" + "=" * 80)
        print("LOADING SAVED BUNDLE (skip training)")
        print("=" * 80)

        # create the controller from bound training
        bundle = load_eval_bundle(bundle_path, map_location="cpu")
        u_nn.load_state_dict(bundle["control_state_dict"])
        f_cl_module = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)

        # create the controller from opt-bound training
        bundle_path_opt = OUTPUT_DIR / "eval_bundle_opt.pth"
        bundle_opt = load_eval_bundle(bundle_path_opt, map_location="cpu")
        u_nn_opt = LorentzLinearControlNN()
        u_nn_opt.load_state_dict(bundle_opt["control_state_dict"])
        f_cl_module_opt = ClosedLoopDrift(f_ol, u_nn_opt).to(params.training.device)

        # create a constant zero open-loop control for comparison
        u_const = LorentzLinearControlNN()
        f_cl_module_no_control = ClosedLoopDrift(f_ol, u_const).to(params.training.device)

        # create the controller from pretraining
        u_nn_pretrain = LorentzLinearControlNN()
        u_nn_pretrain.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
        f_cl_module_pretrain = ClosedLoopDrift(f_ol, u_nn_pretrain).to(params.training.device)

        # animation
        # open-loop constant u_eq animation
        animate_closed_loop_3d_state_control(
            f_cl_module=f_cl_module_no_control, g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
            device=device,
            dt=0.02, T=120.0,
            seed=0, save_path=None,
            show=True,
            controller_label="open-loop"
        )
        # pretrain animation
        animate_closed_loop_3d_state_control(
            f_cl_module=f_cl_module_pretrain, g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
            device=device,
            dt=0.02, T=120.0,
            seed=0, save_path=None,
            show=True,
            controller_label="pre-train"
        )
        # control-synthesis animation
        animate_closed_loop_3d_state_control(
            f_cl_module=f_cl_module, g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
            device=device,
            dt=0.02, T=120.0,
            seed=0, save_path=None,
            show=True,
            controller_label="certified-synthesis"
        )
        # control-synthesis-opt animation
        animate_closed_loop_3d_state_control(
            f_cl_module=f_cl_module_opt, g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
            device=device,
            dt=0.02, T=120.0,
            seed=0, save_path=None,
            show=True,
            controller_label="Opt. certified-synthesis"
        )

        print("\n" + "=" * 80)
        print("Recreating Plots")
        print("=" * 80)
        params = Hyperparameters.from_dict(bundle_opt["hyperparameters"])
        region_cells = bundle_opt["region_cells"]
        # move cells to device
        region_cells = {
            k: [(lo.to(device), hi.to(device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }

        # rebuild networks & load
        V_net.load_state_dict(bundle_opt["V_state_dict"])
        dynamics = Dynamics.dynamics(f=f_cl_module_opt, g=g, state_dim=3)
        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network,
            training_config=params.training,
        ).to(device)
        final_beta_s = bundle_opt["final_beta_s"]
        loss_history = bundle_opt["loss_history"]
        refinement_epochs = bundle_opt["refinement_epochs"]
        results = bundle_opt.get("final_results", None)
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_s=final_beta_s,
                beta_ra=params.constraints.beta_ra,
                device=device,
                n_samples=5000
            )

        print("\n" + "=" * 80)
        print("FINAL EVALUATION (LOADED)")
        print("=" * 80)
        print_constraint_summary(results)

        print("\n" + "=" * 80)
        print("CREATING FINAL VISUALIZATIONS (LOADED)")
        print("=" * 80)
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
            output_dir="results_opt"
        )


if __name__ == '__main__':
    main()
