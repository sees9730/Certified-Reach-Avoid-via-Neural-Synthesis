"""
Main training loop for neural certificate learning with CROWN bounds.

This module provides the core training function that is shared across
all experiments in the repository.
"""
import torch
import torch.nn as nn
import time

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.training_utils import (
    compute_total_loss_bounds,
    evaluate_constraints,
    print_loss_summary,
    print_constraint_summary,
    refine_failing_cells,
    merge_passing_neighbor_cells
)
from src.visualization import visualize_training_progress
from src.regions import Regions
from src.hyperparameters import Hyperparameters


def train_network_bounds(
    V_net,
    GV_net,
    region_cells: dict,
    regions: Regions,
    params: Hyperparameters,
    device: str = 'cpu',
    control_net: nn.Module = None,
    create_scheduler = None
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
        control_net: Optional control network for control synthesis
        create_scheduler: Optional scheduler factory function

    Returns:
        loss_history: List of loss dictionaries per epoch
        final_beta_s: Final beta_s value
        refinement_epochs: Dictionary tracking refinement epochs
    """
    print("\n" + "="*20)
    print("Bound-based training")
    print("="*20)

    # Move models to device
    V_net = V_net.to(device)

    # Collect ALL cells in the same order as original (init, goal, unsafe, outside, generator)
    print("=== Total cells per region for V ===")
    all_cells_V = []
    cell_counts_V = {}
    region_order_V = ['init', 'goal', 'unsafe', 'outside']

    for name in region_order_V:
        cells = region_cells[name]
        all_cells_V.extend(cells)
        cell_counts_V[name] = len(cells)
        print(f"{name}: {len(cells)} cells")

    total_cells_V = len(all_cells_V)
    print(f"Total V cells: {total_cells_V}")

    # Prepare all input bounds at once
    if total_cells_V > 0:
        input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)
    else:
        input_lowers_all = torch.empty(0, params.network.n_inputs, device=device)
        input_uppers_all = torch.empty(0, params.network.n_inputs, device=device)

    # Create CROWN cache for all V cells
    crown_cache_all = SymbolicCROWNCache(
        model=V_net,
        num_cells=total_cells_V,
        input_dim=params.network.n_inputs,
        device=device
    )
    print(f"Created CROWN cache for V with {total_cells_V} cells")

    # Create CROWN cache for generator (Phi) - separate cache
    crown_cache_phi = None
    input_lowers_gen = None
    input_uppers_gen = None
    if len(region_cells['generator']) > 0 and params.training.generator_weight > 0:
        crown_cache_phi = SymbolicCROWNCache_Phi(
            phi_module=GV_net,
            num_cells=len(region_cells['generator']),
            input_dim=params.network.n_inputs,
            device=device
        )
        print(f"Created {len(region_cells['generator'])} CROWN caches for GV")
        # Prepare generator input bounds
        input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

    # Prepare optimizer with all trainable parameters
    opt_params = list(V_net.parameters())

    # If we are doing control synthesis, add control net parameters
    if control_net is not None:
        opt_params += list(control_net.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)

    # Scheduler is optional and created via factory function from main.py
    scheduler = create_scheduler(optimizer) if create_scheduler is not None else None

    # Training loop
    loss_history = []
    refinement_epochs = {'outside': [], 'generator': []}
    start_time = time.time()
    final_beta_s = None

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
                phi_uppers = torch.tensor([], device=device)
                current_gen_weight = 0.0
                num_total_failing = 0

        # Compute total loss from bounds
        loss_kwargs = {
            'beta_ra': params.constraints.beta_ra,
            'device': device,
            'compute_V': params.compute_V,
            'compute_GV': params.compute_GV,
        }

        # Add V bounds if computing V
        if params.compute_V:
            loss_kwargs.update({
                'V_goal_lower': bounds['goal'][0],
                'V_unsafe_lower': bounds['unsafe'][0],
                'V_init_upper': bounds['init'][1],
                'V_outside_lower': bounds['outside'][0],
            })

        # Add GV bounds if computing GV
        if params.compute_GV:
            loss_kwargs.update({
                'Phi_upper': phi_uppers,
                'generator_weight': current_gen_weight
            })

        total_loss, loss_dict, sat_dict = compute_total_loss_bounds(**loss_kwargs)

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

        # Update scheduler after bounds recomputation (if provided)
        if scheduler is not None:
            scheduler.step(total_loss.item())

        # Adaptive refinement for V outside region cells
        needs_cache_rebuild = False
        if params.compute_V and len(bounds_updated['outside'][0]) > 0:
            # Track failing cells in outside region
            v_cfg = params.refinement.v_outside
            outside_failing_mask = bounds_updated['outside'][0] < 0.0
            num_outside_failing = outside_failing_mask.sum().item()

            # Ensure bounds match current cell count
            if len(outside_failing_mask) == len(region_cells['outside']) and num_outside_failing > 0:
                refine_interval = (v_cfg.refine_interval_late
                                 if epoch > v_cfg.late_epoch_threshold
                                 else v_cfg.refine_interval)

                if (v_cfg.enable_refinement and
                    (epoch + 1) % refine_interval == 0 and
                    len(region_cells['outside']) < v_cfg.max_cells):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['outside'],
                        outside_failing_mask,
                        v_cfg.refine_factor,
                        N_to_refine=v_cfg.N_to_refine
                    )
                    region_cells['outside'] = new_cells
                    print(f"Refining outside cells: {num_outside_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['outside'].append(epoch + 1)

            if (v_cfg.enable_merging and
                (epoch + 1) % v_cfg.merge_interval == 0):
                outside_failing_mask_relax = bounds_updated['outside'][0] < (v_cfg.merge_relax_margin)
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['outside'],
                    outside_failing_mask_relax,
                    max_passes=v_cfg.merge_max_passes,
                    max_merges=None,
                )
                region_cells['outside'] = merged_cells
                print(f"Merging outside cells: merged {num_merges} pairs, {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Adaptive refinement for generator cells
        if params.compute_GV:
            gv_cfg = params.refinement.gv_generator
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):

                # Get refinement interval based on epoch
                refine_interval = (gv_cfg.refine_interval_late
                                 if epoch > gv_cfg.late_epoch_threshold
                                 else gv_cfg.refine_interval)

                # Check if it's time to refine
                if (gv_cfg.enable_refinement and
                    (epoch + 1) % refine_interval == 0 and
                    len(region_cells['generator']) < gv_cfg.max_cells):

                    new_cells, num_refined = refine_failing_cells(
                        region_cells['generator'],
                        phi_upper_failing_mask,
                        gv_cfg.refine_factor,
                        N_to_refine=gv_cfg.N_to_refine,
                        scores=phi_uppers
                    )
                    region_cells['generator'] = new_cells
                    print(f"Refining generator cells: {num_total_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['generator'].append(epoch + 1)

            if (gv_cfg.enable_merging and
                (epoch + 1) % gv_cfg.merge_interval == 0):
                phi_upper_failing_mask_relax = phi_uppers > gv_cfg.merge_relax_margin
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['generator'],
                    phi_upper_failing_mask_relax,
                    max_passes=gv_cfg.merge_max_passes,
                    max_merges=None
                )
                region_cells['generator'] = merged_cells
                print(f"Merging generator cells: merged {num_merges} pairs, {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Logging
        if epoch % params.logging.loss_log_interval == 0 or epoch == params.training.num_epochs - 1 or epoch == 0:
            elapsed = time.time() - start_time
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV, elapsed_time=elapsed)
            loss_dict['epoch'] = epoch
            loss_history.append(loss_dict.copy())

        # Early stopping check
        with torch.no_grad():
            # Check V constraints
            all_satisfied = True
            if params.compute_V:
                for key in ['goal', 'unsafe', 'init', 'outside']:
                    if sat_dict[key] is False:
                        all_satisfied = False
                        break

            # Check GV constraints
            if params.compute_GV:
                if sat_dict['generator'] is False:
                    all_satisfied = False

            # Early stop if all active constraints are satisfied
            if all_satisfied:
                print("\n" + "="*20)
                print("All constraints satisfied, early stopping!")
                print("="*20)
                print(f"Training converged at epoch {epoch}")

                # Print relevant losses
                loss_parts = []
                if params.compute_V:
                    loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
                if params.compute_GV:
                    loss_parts.append(f"Gen={loss_dict['generator']:.4f}")
                print(f"Final losses: {', '.join(loss_parts)}")

                # Print final beta_s
                final_beta_s = bounds_updated['outside'][0].min()
                print(f"Final beta_s: {final_beta_s:.4f}")
                break

        # Detailed evaluation and visualization
        if (epoch % params.logging.detailed_eval_interval == 0) or epoch == params.training.num_epochs - 1:
            print(f"=== Detailed Evaluation ===")
            # For evaluation, we can just create temporary caches (not in the hot path)
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                crown_cache_all=crown_cache_all,
                crown_cache_phi=crown_cache_phi,
                input_bounds_all=(input_lowers_all, input_uppers_all),
                cell_counts_V=cell_counts_V,
                input_bounds_gen=(input_lowers_gen, input_uppers_gen) if input_lowers_gen is not None else None,
                beta_ra=params.constraints.beta_ra,
                device=device
            )

            # Pass bounds and phi_uppers to show cell statistics
            print_constraint_summary(
                results,
                prefix="",
                bounds=bounds if params.compute_V else None,
                phi_uppers=phi_uppers if params.compute_GV else None,
                region_cells=region_cells,
                beta_ra=params.constraints.beta_ra
            )

            # Visualize progress
            if params.logging.visualize_interval > 0 and epoch % params.logging.visualize_interval == 0:
                visualize_training_progress(
                    V_net, GV_net, regions, region_cells,
                    epoch=epoch,
                    output_dir="training_progress"
                )

        # Optimizer step
        optimizer.step()

        # Rebuild CROWN caches after optimizer step if needed (after adaptive refinement)
        if needs_cache_rebuild:
            # Rebuild V cache with all cells (in order: init, goal, unsafe, outside)
            if params.compute_V:
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])

                # Rebuild V CROWN cache
                total_cells_V = len(all_cells_V)
                input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)
                crown_cache_all = SymbolicCROWNCache(
                    model=V_net,
                    num_cells=total_cells_V,
                    input_dim=params.network.n_inputs,
                    device=device
                )
                print(f"Rebuilt V cache with {total_cells_V} cells")

                # Update cell counts after refinement
                for name in region_order_V:
                    cell_counts_V[name] = len(region_cells[name])

                # Rebuild concatenated input bounds for V
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])
                    input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

            # Rebuild Phi CROWN cache (only for generator region)
            if params.compute_GV:
                total_cells_GV = len(region_cells['generator'])
                input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)
                crown_cache_phi = SymbolicCROWNCache_Phi(
                    phi_module=GV_net,
                    num_cells=total_cells_GV,
                    input_dim=params.network.n_inputs,
                    device=device
                )
                print(f"Rebuilt Phi cache with {total_cells_GV} cells")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return loss_history, final_beta_s, refinement_epochs
