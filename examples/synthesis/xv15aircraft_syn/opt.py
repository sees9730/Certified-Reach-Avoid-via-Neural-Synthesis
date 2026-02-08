from __future__ import annotations

import math
import argparse
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Arc

# -----------------------------------------------------------------------------
# Repo paths (match your Lorentz script)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # repo_root
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.hyperparameters import Hyperparameters
from src.dynamics import Dynamics
from src.regions import Regions, Region
from src.network import create_V
from src.phi_module import create_GV
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
    create_summary_plots
)
from main import(
    XV15Constants, XV15AeroCoefficients, XV15KLinearAeroTorch,
    find_xv15_equilibrium_for_tilt_min_thrust,
    XV15EqMLPControl, ConstantControl, ClosedLoopDrift,
    animate_xv15_aircraft_state_control, mc_reach_avoid,
    check_gv_matches_autograd_full_range
)
from animate_minimal import animate_xv15_minimal

# -----------------------------------------------------------------------------
# Reuse your training functions from the Lorentz example
#   - adjust this import to wherever those functions live in your repo
# -----------------------------------------------------------------------------
# Example (if you saved the Lorentz file as scripts/main_lorentz.py):
# from scripts.main_lorentz import pretrain_network_samples, train_network_bounds


torch.manual_seed(0)
np.random.seed(0)

DEG = np.pi / 180.0
pi = np.pi
RHO = 1.225
G = 9.81


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
    v_eq = V_net.output_offset.detach()

    def _log_V_zero_at_offset(V_net, epoch: int, device: str, every: int = 100, atol: float = 1e-6):
        if epoch % every != 0:
            return
        V_net.eval()
        with torch.no_grad():
            x0 = V_net.input_offset.to(device=device, dtype=next(V_net.parameters()).dtype)
            y0 = V_net(x0).item()  # (out,) or scalar-ish
            print(f"[Check] epoch={epoch:3d}, {y0:.3e}, {params.constraints.beta_ra :.2f} ")
        V_net.train()

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

    # Scheduler (matching testing_simple3.py)
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

        # Outside Refinement
        if params.compute_V and len(bounds_updated['outside'][0]) > 0:
            # Track failing cells in outside region
            outside_failing_mask = bounds_updated['outside'][0] <= 0.0
            num_outside_failing = outside_failing_mask.sum().item()

            # Ensure bounds match current cell count
            if len(outside_failing_mask) == len(region_cells['outside']) and num_outside_failing > 0:
                REFINE_INTERVAL = 100
                REFINE_FACTOR = 2
                MAX_CELLS = 100000

                # if epoch > 2500:
                #     REFINE_INTERVAL = 50

                if (epoch == 1 or (epoch + 1) % REFINE_INTERVAL == 0) and \
                    len(region_cells['outside']) < MAX_CELLS:
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['outside'],
                        outside_failing_mask,
                        REFINE_FACTOR,
                        scores=-bounds_updated['outside'][0],
                        N_to_refine=100,
                    )
                    region_cells['outside'] = new_cells
                    print(f"[Refine-Outside] Epoch {epoch+1}: {num_outside_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['outside'].append(epoch + 1)

            if((epoch + 1) % 501 == 0):
                outside_failing_mask_relax = bounds_updated['outside'][0] <= 8.0 + v_eq
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['outside'],
                    outside_failing_mask_relax,
                    max_passes=4,
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
                        N_to_refine=100,
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
                print("ALL CONSTRAINTS SATISFIED")
                print("="*80)
                print(f"Training converged at epoch {epoch}")

                # Print relevant losses
                loss_parts = []
                if params.compute_V:
                    loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
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
                    beta_ra=params.constraints.beta_ra,
                    device=device,
                )
                print_constraint_summary(results, prefix="  ")

                # Save the SAT models
                save_eval_bundle(
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
                    file_name="eval_bundle_opt.pth"
                )

                # optimization-star
                if(params.constraints.beta_ra >= 20.0):
                    break
                params.constraints.beta_ra += 0.2
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        _log_V_zero_at_offset(V_net, epoch, device, every=10, atol=1e-6)  # before step

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


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    args = parser.parse_args()

    print("=" * 80)
    print("XV-15 CERTIFICATE-BASED CONTROL SYNTHESIS (V + Phi)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1) Hyperparameters (clone your Lorentz defaults, then adjust)
    # -------------------------------------------------------------------------
    params = Hyperparameters.default()

    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64

    # Scaling: pick something roughly comparable to variable ranges
    # v ~ [20,100] (span 80), gamma ~ [-0.26,0.26], beta ~ [0,1.57]
    # scale choices affect training conditioning; tune if needed.
    params.network.input_scale = [100.0, 20*DEG, 90*DEG]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 30000
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0
    params.training.learnable_beta_s = False
    params.constraints.beta_s = 0.00
    params.constraints.beta_ra = 5.0

    params.compute_V = True
    params.compute_GV = True

    device = params.training.device

    # -------------------------------------------------------------------------
    # 3) Regions (init, goal, unsafe union, full)
    # -------------------------------------------------------------------------
    full_range = np.array([
        [0.5, 100.0], # airspeed
        [-20.0 * DEG, 20.0 * DEG], # flight path angle
        [0.0 * DEG, 90.0 * DEG], # tilt angle
    ], dtype=np.float32)

    init_range = np.array([
        [28.0, 32.0],
        [8.5 * DEG,  10.5 * DEG],
        [58.0 * DEG, 62.0 * DEG],
    ], dtype=np.float32)

    goal_range = np.array([
        [65.0, 85.0],
        [-2.0 * DEG, 10.0 * DEG],
        [25.0 * DEG, 35.0 * DEG],
    ], dtype=np.float32)

    unsafe_up = np.array([
        full_range[0, :],
        [full_range[1,1]-1*DEG, full_range[1,1]],
        full_range[2, :],
    ], dtype=np.float32)

    unsafe_dn = np.array([
        full_range[0, :],
        [full_range[1,0], full_range[1,0]+1*DEG],
        full_range[2, :],
    ], dtype=np.float32)

    unsafe_min_vel = np.array([
        [full_range[0,0], full_range[0,0]+0.5],
        full_range[1, :],
        full_range[2, :],
    ], dtype=np.float32)

    unsafe_max_vel = np.array([
        [full_range[0,1]-0.5, full_range[0,1]],
        full_range[1, :],
        full_range[2, :],
    ], dtype=np.float32)

    unsafe_min_beta = np.array([
        full_range[0, :],
        full_range[1, :],
        [full_range[2,0], full_range[2,0]+1*DEG],
    ], dtype=np.float32)

    unsafe_max_beta = np.array([
        full_range[0, :],
        full_range[1, :],
        [full_range[2,1]-1*DEG, full_range[2,1]],
    ], dtype=np.float32)

    unsafe_range = np.vstack((unsafe_up, unsafe_dn, unsafe_min_beta, unsafe_max_beta,
                              unsafe_min_vel, unsafe_max_vel
                             ))
    init = Region(init_range)
    goal = Region(goal_range)
    full = Region(full_range)

    unsafe_up_reg = Region(unsafe_up)
    unsafe_dn_reg = Region(unsafe_dn)
    unsafe_min_beta_reg = Region(unsafe_min_beta)
    unsafe_max_beta_reg = Region(unsafe_max_beta)
    unsafe_min_vel_reg = Region(unsafe_min_vel)
    unsafe_max_vel_reg = Region(unsafe_max_vel)
    # unsafe_max_vel_dn_reg = Region(unsafe_max_vel_dn)

    unsafe = Region.union(unsafe_up_reg, unsafe_dn_reg, 
                          unsafe_min_beta_reg, unsafe_max_beta_reg,
                          unsafe_min_vel_reg, unsafe_max_vel_reg)
    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # Discretization: start modest; refine happens during training
    params.discretization.n_goal = 32
    params.discretization.n_outside_goal = 8
    params.discretization.n_generator = 4
    params.discretization.n_unsafe = 16
    params.discretization.n_init = 28

    # -------------------------------------------------------------------------
    # 2) Dynamics (closed-loop, torch, differentiable)
    # -------------------------------------------------------------------------
    # Create diffusion coefficients as a persistent tensor to avoid TracerWarnings
    g_coeffs = torch.tensor([0.5, 0.1 * DEG, 0.1 * DEG], dtype=torch.float32)

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

    aero = XV15KLinearAeroTorch().to(device)
    x_eq, u_eq, info = find_xv15_equilibrium_for_tilt_min_thrust(
        beta_eq_deg=30.0,
        aero=aero,
        v_min=0.5,
        v_max=80,
        gamma_min_deg=0.0,
        gamma_max_deg=15.0,
        device="cpu",
    )
    print(info)
    print("x_eq (m/s, Deg, Deg)=", x_eq[0], x_eq[1] / DEG, x_eq[2] / DEG)  # [v, gamma, beta]
    print("u_eq (N, Deg, Deg/s)=", u_eq[0], u_eq[1] / DEG, u_eq[2] / DEG)  # [T, alpha, 0]

    u_nn = XV15EqMLPControl(
        x_eq=x_eq,          # (3,) torch tensor
        u_eq=u_eq,          # (3,) torch tensor
        T_min=XV15Constants.MASS * 9.81 * 0.1,
        T_max=XV15Constants.MASS * 9.81 * 1.8,
        alpha_max=XV15Constants.AOA_MAX,
        delta_max=XV15Constants.MAX_TILT_ANGLE_RATE,
        hidden_dim=64,
        act="tanh",
    ).to(x_eq.device)
    u_nn.verify_u_at_equilibrium()

    f_cl_module = ClosedLoopDrift(aero=aero, controller=u_nn).to(device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=3)

    # -------------------------------------------------------------------------
    # 4) Networks (V and GV)
    # -------------------------------------------------------------------------
    input_offset = [x_eq[0], x_eq[1], x_eq[2]]
    output_offset = np.float32(0.1)
    V_net = create_V(params.network, input_offset=input_offset, output_offset=output_offset)
    V_net.verify_zero_at_offset(atol=1e-6, rtol=1e-6)

    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        input_offset=input_offset
    )

    res = check_gv_matches_autograd_full_range(
        V_net, GV_net, dynamics,
        full_range=full_range,
        num_points=1024,
        batch_size=64,
        device=device,
    )

    # -------------------------------------------------------------------------
    # 6) Pretrain + train (or load)
    # -------------------------------------------------------------------------
    params.constraints.pretrain_goal_target = params.constraints.beta_s
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = 1.0
    params.constraints.pretrain_phi_target = 0.0
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        cleanup_and_setup_directories(["results_opt", "training_progress_opt"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log_opt.txt")
        print(f"\n{dynamics}")
        print(f"\nUsing device: {device}")

        bundle = load_eval_bundle(bundle_path, map_location="cpu")
        # rebuild region_cells (already cpu tensors)
        region_cells = bundle["region_cells"]
        # rebuild networks & load
        V_net.load_state_dict(bundle["V_state_dict"])
        u_nn.load_state_dict(bundle["control_state_dict"])
        
        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            visualize_interval=1000,
            control_net=u_nn,  # CONTROL SYNTHESIS
        )
    else:
        print("\n" + "=" * 80)
        print("LOADING SAVED BUNDLE (skip training)")
        print("=" * 80)

        # create the controller from bound training
        bundle = load_eval_bundle(bundle_path, map_location="cpu")
        u_nn.load_state_dict(bundle["control_state_dict"])
        u_nn.verify_u_at_equilibrium()
        f_cl_module = ClosedLoopDrift(aero=aero, controller=u_nn).to(device)

        # create the controller from opt-bound training
        bundle_path_opt = OUTPUT_DIR / "eval_bundle_opt.pth"
        bundle_opt = load_eval_bundle(bundle_path_opt, map_location="cpu")
        u_nn_opt = XV15EqMLPControl(
            x_eq=x_eq,          # (3,) torch tensor
            u_eq=u_eq,          # (3,) torch tensor
            T_min=XV15Constants.MASS * 9.81 * 0.1,
            T_max=XV15Constants.MASS * 9.81 * 1.8,
            alpha_max=XV15Constants.AOA_MAX,
            delta_max=XV15Constants.MAX_TILT_ANGLE_RATE,
            hidden_dim=64,
            act="tanh",
        ).to(x_eq.device)
        u_nn_opt.load_state_dict(bundle_opt["control_state_dict"])
        u_nn_opt.verify_u_at_equilibrium()
        f_cl_module_opt = ClosedLoopDrift(aero=aero, controller=u_nn_opt).to(device)

        # create a constant u_eq open-loop control for comparison
        u_eq_device = u_eq.to(device=device, dtype=torch.float32)  # (3,)
        u_const = ConstantControl(u_eq_device).to(device)
        f_cl_module_no_control = ClosedLoopDrift(aero=aero, controller=u_const).to(device)

        # create the controller from pretraining
        u_nn_pretrain = XV15EqMLPControl(
            x_eq=x_eq,          # (3,) torch tensor
            u_eq=u_eq,          # (3,) torch tensor
            T_min=XV15Constants.MASS * 9.81 * 0.1,
            T_max=XV15Constants.MASS * 9.81 * 1.8,
            alpha_max=XV15Constants.AOA_MAX,
            delta_max=XV15Constants.MAX_TILT_ANGLE_RATE,
            hidden_dim=64,
            act="tanh",
        ).to(x_eq.device)
        u_nn_pretrain.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
        u_nn_pretrain.verify_u_at_equilibrium()
        f_cl_module_pretrain = ClosedLoopDrift(aero=aero, controller=u_nn_pretrain).to(device)

        # # Minimal publication-ready animations
        # print("\n" + "=" * 80)
        # print("GENERATING PUBLICATION-READY ANIMATIONS")
        # print("=" * 80)

        # # Open-loop
        # animate_xv15_minimal(
        #     f_cl_module=f_cl_module_no_control, g_fn=g,
        #     init_range=init_range, goal_range=goal_range,
        #     full_range=full_range, unsafe_boxes=unsafe_range,
        #     device=device,
        #     dt=0.02, T=120.0,
        #     seed=0,
        #     save_path=None,
        #     save_final_frame=HERE / "results_opt" / "animation_open_loop.pdf",
        #     show=True,
        #     controller_label="Open-Loop Control"
        # )

        # # Pretrain
        # animate_xv15_minimal(
        #     f_cl_module=f_cl_module_pretrain, g_fn=g,
        #     init_range=init_range, goal_range=goal_range,
        #     full_range=full_range, unsafe_boxes=unsafe_range,
        #     device=device,
        #     dt=0.02, T=120.0,
        #     seed=0,
        #     save_path=None,
        #     save_final_frame=HERE / "results_opt" / "animation_pretrain.pdf",
        #     show=True,
        #     controller_label="Pre-trained Controller"
        # )

        # # Certified synthesis
        # animate_xv15_minimal(
        #     f_cl_module=f_cl_module, g_fn=g,
        #     init_range=init_range, goal_range=goal_range,
        #     full_range=full_range, unsafe_boxes=unsafe_range,
        #     device=device,
        #     dt=0.02, T=120.0,
        #     seed=0,
        #     save_path=None,
        #     save_final_frame=HERE / "results_opt" / "animation_synthesis.pdf",
        #     show=True,
        #     controller_label="Certified Synthesis"
        # )

        # Optimized certified synthesis
        animate_xv15_minimal(
            f_cl_module=f_cl_module_opt,
            f_open_module=f_cl_module_no_control,
            f_pretrain_module=f_cl_module_pretrain,
            g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,
            device=device,
            dt=0.02, T=120.0,
            seed=0,
            save_path=None,
            save_final_frame=HERE / "results_opt" / "animation_synthesis_opt.pdf",
            show=True,
            controller_label=""
        )

        print("\n" + "=" * 80)
        print("Monte Carlo results")
        print("=" * 80)

        # monte-carlo (open-loop)
        mc = mc_reach_avoid(
            f_cl_module=f_cl_module_no_control,
            g_fn=g,  # or None
            init_range=init_range,
            goal_range=goal_range,
            full_range=full_range,
            unsafe_boxes=unsafe_range,
            N_trials=100,
            dt=0.02,
            T=120.0,
            seed=0,
            device=device,
        )
        print("Open Loop Control")
        print(mc)

        # monte-carlo (pretrain)
        mc = mc_reach_avoid(
            f_cl_module=f_cl_module_pretrain,
            g_fn=g,  # or None
            init_range=init_range,
            goal_range=goal_range,
            full_range=full_range,
            unsafe_boxes=unsafe_range,
            N_trials=100,
            dt=0.02,
            T=120.0,
            seed=0,
            device=device,
        )
        print("Pretrain Control")
        print(mc)

        # monte-carlo (synthesis)
        mc = mc_reach_avoid(
            f_cl_module=f_cl_module,
            g_fn=g,  # or None
            init_range=init_range,
            goal_range=goal_range,
            full_range=full_range,
            unsafe_boxes=unsafe_range,
            N_trials=100,
            dt=0.02,
            T=120.0,
            seed=0,
            device=device,
        )
        print("Synthesized Control")
        print(mc)

        # monte-carlo (Opt synthesis)
        mc = mc_reach_avoid(
            f_cl_module=f_cl_module_opt,
            g_fn=g,  # or None
            init_range=init_range,
            goal_range=goal_range,
            full_range=full_range,
            unsafe_boxes=unsafe_range,
            N_trials=100,
            dt=0.02,
            T=120.0,
            seed=0,
            device=device,
        )
        print("Opt. Synthesized Control")
        print(mc)

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
            input_offset=input_offset
        ).to(device)
        final_beta_s = bundle_opt["final_beta_s"]
        loss_history = bundle_opt["loss_history"]
        refinement_epochs = bundle_opt["refinement_epochs"]
        results = None
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_ra=params.constraints.beta_ra,
                device=device,
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


if __name__ == "__main__":
    main()