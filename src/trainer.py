"""
Main training loop for neural certificate learning with CROWN bounds.

This module provides the core training function that is shared across
all experiments in the repository.
"""
import torch
import torch.nn as nn
import time
import select
import gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.training_utils import (
    GENERATOR_MARGIN,
    generator_bound_masks,
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
from src.save_load_utils import save_eval_bundle


def _cells_to_cpu(region_cells: dict):
    out = {}
    for name, cells in region_cells.items():
        out[name] = [(lo.detach().cpu(), hi.detach().cpu()) for (lo, hi) in cells]
    return out


def train_network_bounds(
    V_net,
    GV_net,
    region_cells: dict,
    regions: Regions,
    params: Hyperparameters,
    control_net: nn.Module = None,
    create_scheduler = None,
    start_time: float = None
):
    """
    Train the value network using CROWN bounds.

    Args:
        V_net: Value network
        GV_net: Generator network
        region_cells: Dictionary of discretized cells
        regions: Regions object
        params: Hyperparameters
        control_net: Optional control network for control synthesis
        create_scheduler: Optional scheduler factory function
        start_time: Optional start time for timing purposes

    Returns:
        loss_history: List of loss dictionaries per epoch
        final_beta_s: Final beta_s value
        refinement_epochs: Dictionary tracking refinement epochs
    """
    print("\n" + "="*20)
    print("Bound-based training")
    print("="*20)

    # Move models to device
    V_net = V_net.to(params.training.device)

    # Collect all cells 
    print("=== Total cells per region for V ===")
    cell_counts_V = {}
    region_order_V = ['init', 'goal', 'unsafe', 'outside']

    # Pre-calculate total size and pre-allocate
    total_cells_V = sum(len(region_cells[name]) for name in region_order_V)
    all_cells_V = [None] * total_cells_V
    idx = 0

    for name in region_order_V:
        cells = region_cells[name]
        n = len(cells)
        cell_counts_V[name] = n
        print(f"{name}: {n} cells")
        for cell in cells:
            all_cells_V[idx] = cell
            idx += 1
    print(f"Total V cells: {total_cells_V}")

    # Prepare all input bounds at once
    if total_cells_V > 0:
        input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, params.training.device, input_dim=params.network.n_inputs)
    else:
        input_lowers_all = torch.empty(0, params.network.n_inputs, device=params.training.device)
        input_uppers_all = torch.empty(0, params.network.n_inputs, device=params.training.device)

    # Create CROWN cache for all V cells
    crown_cache_all = SymbolicCROWNCache(
        model=V_net,
        num_cells=total_cells_V,
        input_dim=params.network.n_inputs,
        device=params.training.device
    )
    print(f"Created CROWN cache for V with {total_cells_V} cells")

    # Create CROWN cache for generator (Phi) - separate cache
    crown_cache_phi = None
    crown_cache_v_gen = None
    input_lowers_gen = None
    input_uppers_gen = None
    if len(region_cells['generator']) > 0 and params.training.generator_weight > 0:
        crown_cache_phi = SymbolicCROWNCache_Phi(
            phi_module=GV_net,
            num_cells=len(region_cells['generator']),
            input_dim=params.network.n_inputs,
            device=params.training.device
        )
        print(f"Created {len(region_cells['generator'])} CROWN caches for GV")
        # Prepare generator input bounds
        input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], params.training.device, input_dim=params.network.n_inputs)
        crown_cache_v_gen = SymbolicCROWNCache(
            model=V_net,
            num_cells=len(region_cells['generator']),
            input_dim=params.network.n_inputs,
            device=params.training.device
        )

    # Prepare optimizer with all trainable parameters
    opt_params = list(V_net.parameters())

    # If we are doing control synthesis, add control net parameters
    if control_net is not None:
        opt_params += list(control_net.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)

    # Scheduler is optional and created via factory function from main.py
    scheduler = create_scheduler(optimizer) if create_scheduler is not None else None

    # Pre-compute region slice indices for fast bound splitting
    region_slice_indices = {}
    start_idx = 0
    for name in region_order_V:
        num = cell_counts_V[name]
        region_slice_indices[name] = (start_idx, start_idx + num)
        start_idx += num

    # Pre-allocate empty tensors for reuse (avoid repeated allocation)
    empty_tensor = torch.tensor([], device=params.training.device)

    # Pre-allocate dictionaries that are rebuilt every epoch
    bounds = {}
    bounds_updated = {}
    loss_kwargs = {
        'beta_ra': params.constraints.beta_ra,
        'device': params.training.device,
        'compute_V': params.compute_V,
        'compute_GV': params.compute_GV,
    }

    # Training loop
    loss_history = []
    refinement_epochs = {'goal': [], 'init': [], 'outside': [], 'unsafe': [], 'generator': []}
    if start_time is None:
        start_time = time.time()
    final_beta_s = None
    first_sat_refine_interval_applied = False
    refine_interval_after_first_sat = getattr(params.refinement, 'refine_interval_after_first_sat', None)
    curriculum_mode = str(getattr(params.training, 'curriculum_mode', 'beta')).lower()
    if curriculum_mode not in {"none", "beta"}:
        raise ValueError("curriculum_mode must be 'none' or 'beta'")
    async_stop_requested = False

    def _poll_async_stop_command() -> bool:
        """
        Non-blocking check for user command on stdin.
        Returns True once a line equal to 'stop' is received.
        """
        try:
            if sys.stdin is None or sys.stdin.closed or (not hasattr(sys.stdin, "fileno")):
                return False
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return False
            line = sys.stdin.readline()
            return isinstance(line, str) and (line.strip().lower() == "stop")
        except Exception:
            return False

    # For beta curriculum runs, directly use the post-first-SAT
    # refinement interval from the start.
    if curriculum_mode == "beta" and refine_interval_after_first_sat is not None:
        new_interval = int(refine_interval_after_first_sat)
        if new_interval > 0:
            for cfg in (
                params.refinement.v_goal,
                params.refinement.v_init,
                params.refinement.v_outside,
                params.refinement.v_unsafe,
                params.refinement.gv_generator,
            ):
                if cfg.enable_refinement:
                    cfg.refine_interval = new_interval
                    cfg.refine_interval_late = new_interval
            first_sat_refine_interval_applied = True
            print(f"[Curriculum] Using refine_interval_after_first_sat={new_interval} for all enabled refinement regions")

    def _save_resume_checkpoint(epoch_idx: int, stop_requested: bool = False):
        checkpoint_path = Path(getattr(params.training, 'resume_checkpoint_path', "resume_checkpoint.pth"))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        latest_sat_beta_ra = getattr(params.training, "latest_sat_beta_ra", None)
        torch.save({
            "V_state_dict": {k: v.detach().cpu() for k, v in V_net.state_dict().items()},
            "GV_state_dict": {k: v.detach().cpu() for k, v in GV_net.state_dict().items()} if GV_net is not None else None,
            "control_state_dict": {k: v.detach().cpu() for k, v in control_net.state_dict().items()} if control_net is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "region_cells": _cells_to_cpu(region_cells),
            "hyperparameters": params.to_dict(),
            "epoch": int(epoch_idx),
            "loss_history": loss_history,
            "refinement_epochs": refinement_epochs,
            "stop_requested": bool(stop_requested),
            "latest_sat_beta_ra": latest_sat_beta_ra,
        }, checkpoint_path)
        print(f"Saved resume checkpoint -> {checkpoint_path}")

    def _save_eval_bundle_snapshot():
        output_dir = Path(getattr(params.training, "eval_bundle_output_dir", "outputs"))
        latest_sat_beta_ra = getattr(params.training, "latest_sat_beta_ra", None)
        save_eval_bundle(
            output_dir,
            V_net=V_net,
            GV_net=GV_net,
            control_net=control_net,
            params=params,
            regions=regions,
            region_cells=region_cells,
            final_beta_s=None,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=None,
            latest_sat_beta_ra=latest_sat_beta_ra,
        )

    def _compute_all_satisfied(sat_dict_local: dict) -> bool:
        ok = True
        if params.compute_V:
            for key in ['goal', 'unsafe', 'init', 'outside']:
                if sat_dict_local[key] is False:
                    ok = False
                    break
        if ok and params.compute_GV and sat_dict_local['generator'] is False:
            ok = False
        return ok

    for epoch in range(params.training.num_epochs):
        V_net.train()
        needs_v_cache_rebuild = False
        needs_gv_cache_rebuild = False

        optimizer.zero_grad()

        if params.compute_V:
            # Compute bounds for ALL V cells at once
            if total_cells_V > 0:
                v_lowers_all, v_uppers_all = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all = empty_tensor
                v_uppers_all = empty_tensor

            # Split bounds by region using pre-computed indices
            for name in region_order_V:
                start, end = region_slice_indices[name]
                if start < end:
                    bounds[name] = (v_lowers_all[start:end], v_uppers_all[start:end])
                else:
                    bounds[name] = (empty_tensor, empty_tensor)

        if params.compute_GV:
            # Compute generator bounds if enabled
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None):
                phi_uppers = crown_cache_phi.compute_bounds(input_lowers_gen, input_uppers_gen)
                v_gen_lowers, _ = crown_cache_v_gen.compute_bounds(input_lowers_gen, input_uppers_gen)
                current_gen_weight = params.training.generator_weight
                # Refine every active cell that misses the same margin used
                # in the loss and SAT checks, including V_lower == beta.
                _, phi_upper_failing_mask = generator_bound_masks(
                    phi_uppers, params.constraints.beta_ra, v_gen_lowers)
                num_total_failing = phi_upper_failing_mask.sum().item()
            else:
                phi_uppers = empty_tensor
                v_gen_lowers = empty_tensor
                current_gen_weight = 0.0
                num_total_failing = 0

        # Update loss kwargs with current bounds
        if params.compute_V:
            loss_kwargs['beta_ra'] = params.constraints.beta_ra
            loss_kwargs['V_goal_lower'] = bounds['goal'][0]
            loss_kwargs['V_unsafe_lower'] = bounds['unsafe'][0]
            loss_kwargs['V_init_upper'] = bounds['init'][1]
            loss_kwargs['V_outside_lower'] = bounds['outside'][0]

        if params.compute_GV:
            loss_kwargs['Phi_upper'] = phi_uppers
            loss_kwargs['V_generator_lower'] = v_gen_lowers
            loss_kwargs['generator_weight'] = current_gen_weight

        total_loss, loss_dict, sat_dict = compute_total_loss_bounds(**loss_kwargs)

        # Backward pass
        total_loss.backward()

        # Recompute bounds after optimizer step for verification
        V_net.eval()
        with torch.no_grad():
            if total_cells_V > 0:
                v_lowers_all_updated, v_uppers_all_updated = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all_updated = empty_tensor
                v_uppers_all_updated = empty_tensor

            # Split updated bounds by region using pre-computed indices
            for name in region_order_V:
                start, end = region_slice_indices[name]
                if start < end:
                    bounds_updated[name] = (v_lowers_all_updated[start:end], v_uppers_all_updated[start:end])
                else:
                    bounds_updated[name] = (empty_tensor, empty_tensor)

        # Adaptive refinement for V outside region cells
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
                        N_to_refine=v_cfg.N_to_refine,
                        scores=-bounds_updated['outside'][0],
                    )
                    region_cells['outside'] = new_cells
                    print(f"Refining outside cells: {num_outside_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                    needs_v_cache_rebuild = True
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
                if num_merges > 0:
                    needs_v_cache_rebuild = True

        # Adaptive refinement for V goal region cells
        if params.compute_V and len(bounds_updated['goal'][0]) > 0:
            v_cfg = params.refinement.v_goal
            goal_failing_mask = bounds_updated['goal'][0] < 0.0
            num_goal_failing = goal_failing_mask.sum().item()

            if len(goal_failing_mask) == len(region_cells['goal']) and num_goal_failing > 0:
                refine_interval = (v_cfg.refine_interval_late
                                 if epoch > v_cfg.late_epoch_threshold
                                 else v_cfg.refine_interval)

                if (v_cfg.enable_refinement and
                    (epoch + 1) % refine_interval == 0 and
                    len(region_cells['goal']) < v_cfg.max_cells):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['goal'],
                        goal_failing_mask,
                        v_cfg.refine_factor,
                        N_to_refine=v_cfg.N_to_refine,
                        scores=-bounds_updated['goal'][0],
                    )
                    region_cells['goal'] = new_cells
                    print(f"Refining goal cells: {num_goal_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                    needs_v_cache_rebuild = True
                    refinement_epochs['goal'].append(epoch + 1)

            if (v_cfg.enable_merging and
                (epoch + 1) % v_cfg.merge_interval == 0):
                goal_failing_mask_relax = bounds_updated['goal'][0] < (v_cfg.merge_relax_margin)
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['goal'],
                    goal_failing_mask_relax,
                    max_passes=v_cfg.merge_max_passes,
                    max_merges=None,
                )
                region_cells['goal'] = merged_cells
                print(f"Merging goal cells: merged {num_merges} pairs, {len(merged_cells)} total")
                if num_merges > 0:
                    needs_v_cache_rebuild = True

        # Adaptive refinement for V init region cells
        if params.compute_V and len(bounds_updated['init'][1]) > 0:
            v_cfg = params.refinement.v_init
            init_failing_mask = bounds_updated['init'][1] > 1.0
            num_init_failing = init_failing_mask.sum().item()

            if len(init_failing_mask) == len(region_cells['init']) and num_init_failing > 0:
                refine_interval = (v_cfg.refine_interval_late
                                 if epoch > v_cfg.late_epoch_threshold
                                 else v_cfg.refine_interval)

                if (v_cfg.enable_refinement and
                    (epoch + 1) % refine_interval == 0 and
                    len(region_cells['init']) < v_cfg.max_cells):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['init'],
                        init_failing_mask,
                        v_cfg.refine_factor,
                        N_to_refine=v_cfg.N_to_refine,
                        scores=bounds_updated['init'][1],
                    )
                    region_cells['init'] = new_cells
                    print(f"Refining init cells: {num_init_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                    needs_v_cache_rebuild = True
                    refinement_epochs['init'].append(epoch + 1)

            if (v_cfg.enable_merging and
                (epoch + 1) % v_cfg.merge_interval == 0):
                init_failing_mask_relax = bounds_updated['init'][1] > (1.0 - v_cfg.merge_relax_margin)
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['init'],
                    init_failing_mask_relax,
                    max_passes=v_cfg.merge_max_passes,
                    max_merges=None,
                )
                region_cells['init'] = merged_cells
                print(f"Merging init cells: merged {num_merges} pairs, {len(merged_cells)} total")
                if num_merges > 0:
                    needs_v_cache_rebuild = True

        # Adaptive refinement for V unsafe region cells
        if params.compute_V and len(bounds_updated['unsafe'][0]) > 0:
            # Track failing cells in outside region
            v_cfg = params.refinement.v_unsafe
            unsafe_failing_mask = bounds_updated['unsafe'][0] < params.constraints.beta_ra
            num_unsafe_failing = unsafe_failing_mask.sum().item()

            # Ensure bounds match current cell count
            if len(unsafe_failing_mask) == len(region_cells['unsafe']) and num_unsafe_failing > 0:
                refine_interval = (v_cfg.refine_interval_late
                                 if epoch > v_cfg.late_epoch_threshold
                                 else v_cfg.refine_interval)

                if (v_cfg.enable_refinement and
                    (epoch + 1) % refine_interval == 0 and
                    len(region_cells['unsafe']) < v_cfg.max_cells):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['unsafe'],
                        unsafe_failing_mask,
                        v_cfg.refine_factor,
                        N_to_refine=v_cfg.N_to_refine,
                        scores=-bounds_updated['unsafe'][0],
                    )
                    region_cells['unsafe'] = new_cells
                    print(f"Refining unsafe cells: {num_unsafe_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                    needs_v_cache_rebuild = True
                    refinement_epochs['unsafe'].append(epoch + 1)

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
                        scores=phi_uppers,
                    )
                    region_cells['generator'] = new_cells
                    print(f"Refining generator cells: {num_total_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                    needs_gv_cache_rebuild = True
                    refinement_epochs['generator'].append(epoch + 1)
                
                elif (gv_cfg.enable_refinement and
                    (epoch + 1) % refine_interval == 0 and
                    len(region_cells['generator']) > gv_cfg.max_cells):
                    print(f"Generator cells exceed max cells threshold: {len(region_cells['generator'])} > {gv_cfg.max_cells}")    

            if (gv_cfg.enable_merging and
                (epoch + 1) % gv_cfg.merge_interval == 0):
                phi_upper_failing_mask_relax = phi_uppers > min(gv_cfg.merge_relax_margin, -GENERATOR_MARGIN)
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['generator'],
                    phi_upper_failing_mask_relax,
                    max_passes=gv_cfg.merge_max_passes,
                    max_merges=None,
                )
                region_cells['generator'] = merged_cells
                print(f"Merging generator cells: merged {num_merges} pairs, {len(merged_cells)} total")
                if num_merges > 0:
                    needs_gv_cache_rebuild = True

        # Logging
        if epoch % params.logging.loss_log_interval == 0 or epoch == params.training.num_epochs - 1 or epoch == 0:
            elapsed = time.time() - start_time
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV, elapsed_time=elapsed)
            loss_dict['epoch'] = epoch
            loss_history.append(loss_dict.copy())

        # Early stopping check
        with torch.no_grad():
            if curriculum_mode == "beta":
                async_stop_requested = async_stop_requested or _poll_async_stop_command()

            # Immediate stop for curriculum modes: main.py will reload latest SAT eval_bundle.
            if async_stop_requested and curriculum_mode == "beta":
                print("User requested stop. Exiting training and loading latest SAT bundle in main.")
                break

            # A partition changed after these bounds were computed. Recheck
            # its new cells before accepting or saving a SAT pair.
            all_satisfied = (_compute_all_satisfied(sat_dict)
                             and not needs_v_cache_rebuild and not needs_gv_cache_rebuild
                             and (not params.compute_GV or current_gen_weight > 0))

            # Optional one-time refinement interval update at first SAT.
            if (all_satisfied and
                (not first_sat_refine_interval_applied) and
                (refine_interval_after_first_sat is not None)):
                new_interval = int(refine_interval_after_first_sat)
                if new_interval > 0:
                    for cfg in (
                        params.refinement.v_goal,
                        params.refinement.v_init,
                        params.refinement.v_outside,
                        params.refinement.v_unsafe,
                        params.refinement.gv_generator,
                    ):
                        if cfg.enable_refinement:
                            cfg.refine_interval = new_interval
                            cfg.refine_interval_late = new_interval
                    first_sat_refine_interval_applied = True
                    print(f"Applied refine_interval_after_first_sat={new_interval} to enabled refinement regions")

            if all_satisfied:
                if curriculum_mode == "none":
                    _save_resume_checkpoint(epoch)
                    _save_eval_bundle_snapshot()
                    final_beta_s = bounds_updated['outside'][0].min() if params.compute_V else None
                    print("No curriculum mode provided; saved checkpoint and stopped at first SAT.")
                    break

                if curriculum_mode == "beta":
                    # Save SAT artifacts for the current beta_ra first.
                    params.training.latest_sat_beta_ra = float(params.constraints.beta_ra)
                    _save_resume_checkpoint(epoch, stop_requested=False)
                    _save_eval_bundle_snapshot()
                    beta_ra_max = float(getattr(params.constraints, "beta_ra_max", 20.0))
                    if async_stop_requested:
                        _save_resume_checkpoint(epoch, stop_requested=True)
                        print("User requested early stop for beta curriculum.")
                    elif params.constraints.beta_ra < beta_ra_max:
                        params.constraints.beta_ra += float(getattr(params.constraints, 'beta_increment', 0.2))
                        if params.constraints.beta_ra > beta_ra_max:
                            params.constraints.beta_ra = beta_ra_max
                        all_satisfied = False
                        print(f"Incremented beta_ra to {params.constraints.beta_ra:.2f}")
                    else:
                        print(f"beta_ra reached maximum of {beta_ra_max:.2f}")

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
        # If refinement/curriculum changed cell counts in this epoch, caches are stale until
        # the rebuild section below runs. Skip detailed eval to avoid cache/cell-count mismatch.
        if (epoch % params.logging.detailed_eval_interval == 0) or epoch == params.training.num_epochs - 1:
            if needs_v_cache_rebuild or needs_gv_cache_rebuild:
                print("Skipping detailed evaluation this epoch (cache rebuild pending after cell updates).")
            else:
                print(f"=== Detailed Evaluation ===")
                # bounds_updated/phi_uppers/v_gen_lowers were already computed this epoch
                # (above, before optimizer.step()) on the exact same cells/weights - reuse
                # them instead of recomputing via a second compute_bounds() pass.
                generator_bounds_valid = (
                    params.compute_GV
                    and epoch >= params.training.generator_start_epoch
                    and params.training.generator_weight > 0
                    and crown_cache_phi is not None
                )
                results = evaluate_constraints(
                    V_net, GV_net, region_cells,
                    crown_cache_all=crown_cache_all,
                    crown_cache_phi=crown_cache_phi,
                    crown_cache_v_gen=crown_cache_v_gen,
                    input_bounds_all=(input_lowers_all, input_uppers_all),
                    cell_counts_V=cell_counts_V,
                    input_bounds_gen=(input_lowers_gen, input_uppers_gen) if input_lowers_gen is not None else None,
                    beta_ra=params.constraints.beta_ra,
                    device=params.training.device,
                    precomputed_region_bounds=bounds_updated,
                    precomputed_phi_uppers=phi_uppers if generator_bounds_valid else None,
                    precomputed_v_gen_lowers=v_gen_lowers if generator_bounds_valid else None,
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
        if curriculum_mode == "none" and params.logging.visualize_interval > 0 and epoch % params.logging.visualize_interval == 0:
            progress_dir = getattr(params.training, "progress_output_dir", "training_progress")
            visualize_training_progress(
                V_net, GV_net, regions, region_cells,
                epoch=epoch,
                output_dir=progress_dir
            )

        # Optimizer step
        optimizer.step()

        # Update scheduler after optimizer step
        if scheduler is not None:
            # ReduceLROnPlateau needs metrics, StepLR/etc don't
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss.item())
            else:
                scheduler.step()

        # Rebuild CROWN caches after optimizer step if needed (after adaptive refinement)
        if needs_v_cache_rebuild:
            # Rebuild V cache with all cells (in order: init, goal, unsafe, outside)
            if params.compute_V:
                # Pre-allocate and rebuild all_cells_V
                total_cells_V = sum(len(region_cells[name]) for name in region_order_V)
                all_cells_V = [None] * total_cells_V
                idx = 0
                for name in region_order_V:
                    cell_counts_V[name] = len(region_cells[name])
                    for cell in region_cells[name]:
                        all_cells_V[idx] = cell
                        idx += 1

                # Rebuild V CROWN cache and input bounds
                input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, params.training.device, input_dim=params.network.n_inputs)
                del crown_cache_all
                crown_cache_all = SymbolicCROWNCache(
                    model=V_net,
                    num_cells=total_cells_V,
                    input_dim=params.network.n_inputs,
                    device=params.training.device
                )
                print(f"Rebuilt V cache with {total_cells_V} cells")

                # Rebuild region slice indices after refinement
                region_slice_indices = {}
                start_idx = 0
                for name in region_order_V:
                    num = cell_counts_V[name]
                    region_slice_indices[name] = (start_idx, start_idx + num)
                    start_idx += num

        # Rebuild Phi CROWN cache (only for generator region, separate from V cache)
        if needs_gv_cache_rebuild:
            if params.compute_GV:
                total_cells_GV = len(region_cells['generator'])
                input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], params.training.device, input_dim=params.network.n_inputs)
                del crown_cache_phi, crown_cache_v_gen
                crown_cache_phi = SymbolicCROWNCache_Phi(
                    phi_module=GV_net,
                    num_cells=total_cells_GV,
                    input_dim=params.network.n_inputs,
                    device=params.training.device
                )
                crown_cache_v_gen = SymbolicCROWNCache(
                    model=V_net,
                    num_cells=total_cells_GV,
                    input_dim=params.network.n_inputs,
                    device=params.training.device
                )
                print(f"Rebuilt Phi cache with {total_cells_GV} cells")

        if needs_v_cache_rebuild or needs_gv_cache_rebuild:
            gc.collect()
            if str(params.training.device).startswith('cuda'):
                torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return loss_history, final_beta_s, refinement_epochs
