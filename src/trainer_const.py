"""
Training loop for constant-but-unknown set-valued drift parameters.

Generator loss uses a global theta-cell adversary:
    max_{theta_cell} sum_i ReLU(Phi_upper_i(theta_cell) + delta)
"""
import itertools
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.dynamics import Dynamics
from src.phi_module import create_GV
from src.set_values import ClosedLoopSetValuedDrift, InvertedPendulumSetDrift
from src.training_utils import (
    compute_total_loss_bounds,
    print_loss_summary,
    refine_failing_cells,
    merge_passing_neighbor_cells,
)


def _theta_edges(theta_ranges: dict, splits_per_dim, device: str):
    keys = ("g", "L", "b", "m")
    if len(splits_per_dim) != 4:
        raise ValueError(f"splits_per_dim must be length 4, got {len(splits_per_dim)}")

    edges = {}
    for i, k in enumerate(keys):
        lo = float(theta_ranges[k][0])
        hi = float(theta_ranges[k][1])
        if splits_per_dim[i] <= 1 or hi <= lo:
            edges[k] = torch.tensor([lo, hi], dtype=torch.float32, device=device)
        else:
            edges[k] = torch.linspace(lo, hi, steps=int(splits_per_dim[i]) + 1, device=device)
    return edges


def _theta_boxes(theta_ranges: dict, splits_per_dim, device: str):
    edges = _theta_edges(theta_ranges, splits_per_dim, device=device)
    keys = ("g", "L", "b", "m")
    ranges = [range(edges[k].numel() - 1) for k in keys]

    boxes = []
    for idx in itertools.product(*ranges):
        lo = {}
        hi = {}
        valid = True
        for d, k in enumerate(keys):
            l = float(edges[k][idx[d]].item())
            h = float(edges[k][idx[d] + 1].item())
            if h < l:
                valid = False
                break
            lo[k], hi[k] = l, h
        if valid:
            boxes.append((lo, hi))
    return boxes


def _build_phi_caches_over_theta_cells(
    V_net: nn.Module,
    GV_net: nn.Module,
    params,
    input_dim: int,
    num_gen_cells: int,
    theta_boxes,
    device: str,
):
    f_orig = GV_net.dynamics.get_f()
    g_fn = GV_net.dynamics.get_g()
    caches = []
    for lo, hi in theta_boxes:
        f_ol = InvertedPendulumSetDrift(
            g_range=(lo["g"], hi["g"]),
            L_range=(lo["L"], hi["L"]),
            b_range=(lo["b"], hi["b"]),
            m_range=(lo["m"], hi["m"]),
        ).to(device)
        if hasattr(f_orig, "controller"):
            f_cl = ClosedLoopSetValuedDrift(f_ol, f_orig.controller).to(device)
        else:
            f_cl = f_ol
        dyn = Dynamics.dynamics(f=f_cl, g=g_fn, state_dim=input_dim)
        GV_theta = create_GV(V_net=V_net, dynamics=dyn, network_config=params.network, verify=False).to(device)
        cache = SymbolicCROWNCache_Phi(
            phi_module=GV_theta,
            num_cells=num_gen_cells,
            input_dim=input_dim,
            device=device,
        )
        caches.append({
            "lo": lo,
            "hi": hi,
            "cache": cache,
        })
    return caches


def train_network_bounds_const_theta(
    V_net,
    GV_net,
    region_cells: dict,
    regions,
    params,
    theta_ranges: dict,
    theta_grid_splits=None,
    control_net: nn.Module = None,
    create_scheduler=None,
    start_time: float = None,
    adv_delta: float = 1e-4,
):
    print("\n" + "=" * 20)
    print("Bound-based training (constant theta_f adversary)")
    print("=" * 20)
    _ = regions

    if theta_grid_splits is None:
        theta_grid_splits = [1, 8, 1, 1]

    device = params.training.device
    input_dim = params.network.n_inputs
    V_net = V_net.to(device)

    region_order_V = ["init", "goal", "unsafe", "outside"]
    cell_counts_V = {name: len(region_cells[name]) for name in region_order_V}
    total_cells_V = sum(cell_counts_V.values())

    all_cells_V = [None] * total_cells_V
    idx = 0
    for name in region_order_V:
        for cell in region_cells[name]:
            all_cells_V[idx] = cell
            idx += 1

    if total_cells_V > 0:
        input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device=device, input_dim=input_dim)
        crown_cache_all = SymbolicCROWNCache(V_net, total_cells_V, input_dim=input_dim, device=device)
    else:
        input_lowers_all = torch.empty(0, input_dim, device=device)
        input_uppers_all = torch.empty(0, input_dim, device=device)
        crown_cache_all = None

    def _build_generator_side():
        if len(region_cells["generator"]) == 0 or params.training.generator_weight <= 0:
            return None, None, []
        in_lo, in_hi = prepare_cell_bounds(region_cells["generator"], device=device, input_dim=input_dim)
        theta_boxes = _theta_boxes(theta_ranges, theta_grid_splits, device=device)
        caches = _build_phi_caches_over_theta_cells(
            V_net=V_net,
            GV_net=GV_net,
            params=params,
            input_dim=input_dim,
            num_gen_cells=len(region_cells["generator"]),
            theta_boxes=theta_boxes,
            device=device,
        )
        return in_lo, in_hi, caches

    input_lowers_gen, input_uppers_gen, theta_phi_caches = _build_generator_side()
    if len(theta_phi_caches) > 0:
        print(f"Built {len(theta_phi_caches)} theta-cell GV caches")

    opt_params = list(V_net.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)
    scheduler = create_scheduler(optimizer) if create_scheduler is not None else None

    region_slice_indices = {}
    start_idx = 0
    for name in region_order_V:
        n = cell_counts_V[name]
        region_slice_indices[name] = (start_idx, start_idx + n)
        start_idx += n

    if start_time is None:
        start_time = time.time()

    empty_tensor = torch.tensor([], device=device)
    loss_history = []
    final_beta_s = None
    refinement_epochs = {"outside": [], "generator": []}

    for epoch in range(params.training.num_epochs):
        V_net.train()
        optimizer.zero_grad()
        needs_v_cache_rebuild = False
        needs_gv_cache_rebuild = False

        if total_cells_V > 0:
            v_lowers_all, v_uppers_all = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
        else:
            v_lowers_all, v_uppers_all = empty_tensor, empty_tensor

        bounds = {}
        for name in region_order_V:
            s, e = region_slice_indices[name]
            bounds[name] = (v_lowers_all[s:e], v_uppers_all[s:e]) if s < e else (empty_tensor, empty_tensor)

        phi_uppers_for_refine = empty_tensor
        if (
            params.compute_GV
            and epoch >= params.training.generator_start_epoch
            and params.training.generator_weight > 0
            and len(theta_phi_caches) > 0
        ):
            # Global constant-theta adversary over theta cells.
            best_idx = None
            best_obj = float("-inf")
            for j, entry in enumerate(theta_phi_caches):
                with torch.no_grad():
                    phi_u_j = entry["cache"].compute_bounds(input_lowers_gen, input_uppers_gen)
                    obj_j = float(F.relu(phi_u_j + adv_delta).sum().item())
                if obj_j > best_obj:
                    best_obj = obj_j
                    best_idx = j

            chosen = theta_phi_caches[best_idx]
            phi_uppers = chosen["cache"].compute_bounds(input_lowers_gen, input_uppers_gen)
            phi_uppers_for_refine = phi_uppers.detach()
            current_gen_weight = params.training.generator_weight
            theta_diag = {
                "idx": int(best_idx),
                "obj": float(best_obj),
                "lo": chosen["lo"],
                "hi": chosen["hi"],
            }
        else:
            phi_uppers = empty_tensor
            current_gen_weight = 0.0
            theta_diag = None
            phi_uppers_for_refine = empty_tensor

        total_loss, loss_dict, sat_dict = compute_total_loss_bounds(
            beta_ra=params.constraints.beta_ra,
            device=device,
            compute_V=params.compute_V,
            compute_GV=params.compute_GV,
            V_goal_lower=bounds["goal"][0],
            V_unsafe_lower=bounds["unsafe"][0],
            V_init_upper=bounds["init"][1],
            V_outside_lower=bounds["outside"][0],
            Phi_upper=phi_uppers,
            generator_weight=current_gen_weight,
        )
        total_loss.backward()
        optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss.item())
            else:
                scheduler.step()

        # Recompute V bounds after update for adaptive refinement on outside region.
        V_net.eval()
        with torch.no_grad():
            if total_cells_V > 0:
                v_lowers_all_updated, v_uppers_all_updated = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all_updated, v_uppers_all_updated = empty_tensor, empty_tensor

        bounds_updated = {}
        for name in region_order_V:
            s, e = region_slice_indices[name]
            bounds_updated[name] = (v_lowers_all_updated[s:e], v_uppers_all_updated[s:e]) if s < e else (empty_tensor, empty_tensor)

        # Adaptive refinement for V outside cells.
        if params.compute_V and len(bounds_updated["outside"][0]) > 0:
            v_cfg = params.refinement.v_outside
            outside_failing_mask = bounds_updated["outside"][0] < 0.0
            num_outside_failing = int(outside_failing_mask.sum().item())
            if len(outside_failing_mask) == len(region_cells["outside"]) and num_outside_failing > 0:
                refine_interval = (
                    v_cfg.refine_interval_late if epoch > v_cfg.late_epoch_threshold else v_cfg.refine_interval
                )
                if (
                    v_cfg.enable_refinement
                    and (epoch + 1) % refine_interval == 0
                    and len(region_cells["outside"]) < v_cfg.max_cells
                ):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells["outside"],
                        outside_failing_mask,
                        v_cfg.refine_factor,
                        N_to_refine=v_cfg.N_to_refine,
                    )
                    region_cells["outside"] = new_cells
                    needs_v_cache_rebuild = True
                    refinement_epochs["outside"].append(epoch + 1)
                    print(
                        f"Refining outside cells: {num_outside_failing} failing, "
                        f"refined {num_refined} cells, {len(new_cells)} total"
                    )

            if v_cfg.enable_merging and (epoch + 1) % v_cfg.merge_interval == 0:
                outside_failing_mask_relax = bounds_updated["outside"][0] < v_cfg.merge_relax_margin
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells["outside"],
                    outside_failing_mask_relax,
                    max_passes=v_cfg.merge_max_passes,
                    max_merges=None,
                )
                region_cells["outside"] = merged_cells
                if num_merges > 0:
                    needs_v_cache_rebuild = True
                print(f"Merging outside cells: merged {num_merges} pairs, {len(merged_cells)} total")

        # Adaptive refinement for generator cells using worst-case theta-cell Phi as scores/margins.
        if (
            params.compute_GV
            and epoch >= params.training.generator_start_epoch
            and params.training.generator_weight > 0
            and phi_uppers_for_refine.numel() > 0
        ):
            gv_cfg = params.refinement.gv_generator
            phi_upper_failing_mask = phi_uppers_for_refine > 0.0
            num_total_failing = int(phi_upper_failing_mask.sum().item())
            refine_interval = (
                gv_cfg.refine_interval_late if epoch > gv_cfg.late_epoch_threshold else gv_cfg.refine_interval
            )

            if (
                gv_cfg.enable_refinement
                and num_total_failing > 0
                and (epoch + 1) % refine_interval == 0
                and len(region_cells["generator"]) < gv_cfg.max_cells
            ):
                new_cells, num_refined = refine_failing_cells(
                    region_cells["generator"],
                    phi_upper_failing_mask,
                    gv_cfg.refine_factor,
                    N_to_refine=gv_cfg.N_to_refine,
                    scores=phi_uppers_for_refine,
                )
                region_cells["generator"] = new_cells
                needs_gv_cache_rebuild = True
                refinement_epochs["generator"].append(epoch + 1)
                print(
                    f"Refining generator cells: {num_total_failing} failing, "
                    f"refined {num_refined} cells, {len(new_cells)} total"
                )

            if gv_cfg.enable_merging and (epoch + 1) % gv_cfg.merge_interval == 0:
                # Use worst-case-theta-cell phi as merging margin signal.
                phi_upper_failing_mask_relax = phi_uppers_for_refine > gv_cfg.merge_relax_margin
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells["generator"],
                    phi_upper_failing_mask_relax,
                    max_passes=gv_cfg.merge_max_passes,
                    max_merges=None,
                )
                region_cells["generator"] = merged_cells
                if num_merges > 0:
                    needs_gv_cache_rebuild = True
                print(f"Merging generator cells: merged {num_merges} pairs, {len(merged_cells)} total")

        # Rebuild caches after adaptive partition updates.
        if needs_v_cache_rebuild:
            total_cells_V = sum(len(region_cells[name]) for name in region_order_V)
            all_cells_V = [None] * total_cells_V
            idx = 0
            for name in region_order_V:
                cell_counts_V[name] = len(region_cells[name])
                for cell in region_cells[name]:
                    all_cells_V[idx] = cell
                    idx += 1

            if total_cells_V > 0:
                input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device=device, input_dim=input_dim)
                crown_cache_all = SymbolicCROWNCache(V_net, total_cells_V, input_dim=input_dim, device=device)
            else:
                input_lowers_all = torch.empty(0, input_dim, device=device)
                input_uppers_all = torch.empty(0, input_dim, device=device)
                crown_cache_all = None

            region_slice_indices = {}
            start_idx = 0
            for name in region_order_V:
                n = cell_counts_V[name]
                region_slice_indices[name] = (start_idx, start_idx + n)
                start_idx += n
            print(f"Rebuilt V cache with {total_cells_V} cells")

        if needs_gv_cache_rebuild:
            input_lowers_gen, input_uppers_gen, theta_phi_caches = _build_generator_side()
            print(
                f"Rebuilt generator caches: cells={len(region_cells['generator'])}, "
                f"theta_cells={len(theta_phi_caches)}"
            )

        if epoch % params.logging.loss_log_interval == 0 or epoch == 0 or epoch == params.training.num_epochs - 1:
            elapsed = time.time() - start_time
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV, elapsed_time=elapsed)
            if theta_diag is not None:
                print(
                    f"theta_cell_worst[{theta_diag['idx']}]: "
                    f"g[{theta_diag['lo']['g']:.4f},{theta_diag['hi']['g']:.4f}] "
                    f"L[{theta_diag['lo']['L']:.4f},{theta_diag['hi']['L']:.4f}] "
                    f"b[{theta_diag['lo']['b']:.4f},{theta_diag['hi']['b']:.4f}] "
                    f"m[{theta_diag['lo']['m']:.4f},{theta_diag['hi']['m']:.4f}] "
                    f"obj={theta_diag['obj']:.6f}"
                )
                loss_dict["theta_cell_idx"] = theta_diag["idx"]
                loss_dict["theta_cell_obj"] = theta_diag["obj"]
            loss_dict["epoch"] = epoch
            loss_history.append(loss_dict.copy())

        all_satisfied = True
        for key in ("goal", "unsafe", "init", "outside"):
            if params.compute_V and (not sat_dict[key]):
                all_satisfied = False
                break
        if params.compute_GV and (not sat_dict["generator"]):
            all_satisfied = False

        if all_satisfied:
            print("\n" + "=" * 20)
            print("All constraints satisfied, early stopping!")
            print("=" * 20)
            final_beta_s = bounds["outside"][0].min() if params.compute_V else None
            break

    return loss_history, final_beta_s, refinement_epochs
