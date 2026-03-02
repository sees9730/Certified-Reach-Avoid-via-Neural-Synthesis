"""
Main training loop for neural certificate learning with CROWN bounds.

This module provides the core training function that is shared across
all experiments in the repository.
"""
import torch
import torch.nn as nn
import time
import sys
import select

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
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
from src.smt_cell_verifier import SMTCellVerifier
from src.visualization import visualize_training_progress
from src.regions import Regions
from src.hyperparameters import Hyperparameters


def _check_user_stop_command() -> bool:
    """
    Return True if the user typed `stop` in the terminal (non-blocking).
    """
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return False
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return False
        return sys.stdin.readline().strip().lower() == "stop"
    except Exception:
        # Keep training robust in environments where stdin/select behaves differently.
        return False


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
    beta_ra_current = None
    if getattr(params.constraints, "use_beta_ra_curriculum", False):
        beta_ra_current = float(params.constraints.beta_ra_init)
    beta_ra_terminal_current = None
    if getattr(params.constraints, "use_terminal_beta_curriculum", False):
        beta_ra_terminal_current = float(params.constraints.beta_ra_terminal_init)

    # Training loop
    loss_history = []
    refinement_epochs = {'goal': [], 'unsafe': [], 'init': [], 'outside': [], 'generator': []}
    if start_time is None:
        start_time = time.time()
    final_beta_s = None
    has_sat_certificate = False
    has_stage3_sat_certificate = False
    stop_prompt_shown = False
    user_stop_requested = False
    latest_sat_checkpoint = None
    latest_stage3_sat_checkpoint = None
    last_refine_epoch = {'goal': None, 'unsafe': None, 'init': None, 'outside': None, 'generator': None}
    last_refine_loss = {'goal': None, 'unsafe': None, 'init': None, 'outside': None, 'generator': None}
    simple_refine_mode = bool(getattr(params.refinement, "use_simple_refinement", False))
    simple_refine_every = int(getattr(params.refinement, "simple_refine_every", 500))
    simple_loss_improve_tol = float(getattr(params.refinement, "simple_loss_improve_tol", 0.01))
    simple_refine_budget = int(getattr(params.refinement, "simple_refine_budget", 100))
    simple_regions_per_trigger = int(getattr(params.refinement, "simple_regions_per_trigger", 2))
    simple_last_refine_loss = None
    stage3_force_refine_interval = int(getattr(params.refinement, "stage3_force_refine_interval", 1000))
    stage3_window_start_epoch = None
    stage3_window_start_beta = None
    stage3_refined_in_window = False
    generator_after_v_sat = bool(getattr(params.training, "generator_after_v_sat", False))
    generator_ramp_epochs = int(getattr(params.training, "generator_ramp_epochs", 0))
    generator_target_weight = float(params.training.generator_weight)
    use_gen_thresh_curr = bool(getattr(params.training, "use_generator_threshold_curriculum", False))
    generator_threshold_init_cfg = getattr(params.training, "generator_threshold_init", None)
    generator_threshold_current = (None if generator_threshold_init_cfg is None
                                   else float(generator_threshold_init_cfg))
    generator_threshold_final = float(getattr(params.training, "generator_threshold_final", 0.0))
    generator_threshold_step = float(getattr(params.training, "generator_threshold_step", 0.1))
    generator_threshold_step_fine = float(getattr(params.training, "generator_threshold_step_fine", 0.02))
    generator_threshold_switch = float(getattr(params.training, "generator_threshold_switch", 0.2))
    generator_threshold_margin = 1e-3
    gv_phase_active = (not generator_after_v_sat)
    gv_phase_start_epoch = int(params.training.generator_start_epoch) if gv_phase_active else None
    smt_filter_enabled = bool(getattr(params.training, "enable_smt_cell_filter", False))
    smt_filter_start_epoch = int(getattr(params.training, "smt_cell_filter_start_epoch", 0))
    smt_filter_interval = int(getattr(params.training, "smt_cell_filter_interval", 200))
    smt_filter_max_cells = int(getattr(params.training, "smt_cell_filter_max_cells", 20))
    smt_filter_timeout_ms = int(getattr(params.training, "smt_cell_filter_timeout_ms", 200))
    _regions_raw = getattr(params.training, "smt_cell_filter_regions", None)
    if _regions_raw is None:
        smt_filter_regions = {"goal", "unsafe", "init", "outside", "generator"}
    else:
        smt_filter_regions = {str(r).strip().lower() for r in _regions_raw}
    smt_check_all_regions_in_stage3 = bool(
        getattr(params.training, "smt_check_all_regions_in_stage3", True)
    )
    stage3_smt_full_check_interval = int(
        getattr(params.training, "stage3_smt_full_check_interval", smt_filter_interval)
    )
    stage3_smt_full_check_start_epoch = int(
        getattr(params.training, "stage3_smt_full_check_start_epoch", smt_filter_start_epoch)
    )
    smt_parallel_workers = int(getattr(params.training, "smt_parallel_workers", 1))
    smt_verifier = (
        SMTCellVerifier(timeout_ms=smt_filter_timeout_ms, num_workers=smt_parallel_workers)
        if smt_filter_enabled else None
    )
    smt_missing_solver_warned = False
    smt_ce_replay_enabled = bool(getattr(params.training, "enable_smt_counterexample_replay", False))
    smt_ce_replay_max_points = int(getattr(params.training, "smt_ce_replay_max_points_per_region", 256))
    smt_ce_replay_batch_size = int(getattr(params.training, "smt_ce_replay_batch_size", 32))
    smt_ce_replay_noise_radius = float(getattr(params.training, "smt_ce_replay_noise_radius", 0.0))
    smt_ce_replay_weight_v = float(getattr(params.training, "smt_ce_replay_weight_v", 0.2))
    smt_ce_replay_weight_gv = float(getattr(params.training, "smt_ce_replay_weight_gv", 0.2))
    smt_ce_buffers = {
        'goal': [],
        'unsafe': [],
        'init': [],
        'outside': [],
        'generator': [],
    }
    smt_ce_added_total = {k: 0 for k in smt_ce_buffers.keys()}
    # Policy mode: when SMT is generator-only, activate SMT only after
    # generator-threshold reduction starts and keep it active through beta-ra stage.
    generator_only_smt = (smt_filter_regions == {"generator"})
    smt_generator_phase_active = (not generator_only_smt)

    def _push_smt_counterexample(region_name: str, x_point: torch.Tensor) -> None:
        if region_name not in smt_ce_buffers:
            return
        buf = smt_ce_buffers[region_name]
        buf.append(x_point.detach().cpu().reshape(-1))
        if len(buf) > smt_ce_replay_max_points:
            del buf[: len(buf) - smt_ce_replay_max_points]

    def _sample_smt_counterexamples(region_name: str, k: int, device: str) -> torch.Tensor:
        buf = smt_ce_buffers.get(region_name, [])
        if len(buf) == 0 or k <= 0:
            return torch.empty(0, params.network.n_inputs, device=device)
        kk = min(int(k), len(buf))
        idx = torch.randint(0, len(buf), (kk,))
        x = torch.stack([buf[int(i)] for i in idx], dim=0).to(device=device, dtype=torch.float32)
        if smt_ce_replay_noise_radius > 0.0:
            x = x + smt_ce_replay_noise_radius * torch.randn_like(x)
        return x

    def _reset_v_refinement_baselines():
        """Reset V-region refinement baselines after a curriculum phase change."""
        nonlocal simple_last_refine_loss, stage3_window_start_epoch, stage3_window_start_beta, stage3_refined_in_window
        simple_last_refine_loss = None
        stage3_window_start_epoch = None
        stage3_window_start_beta = None
        stage3_refined_in_window = False
        for k in ('goal', 'unsafe', 'init', 'outside'):
            last_refine_loss[k] = None
            last_refine_epoch[k] = None

    def _should_refine_region(cfg, region_key: str, epoch_1b: int, total_loss_val: float, num_failing: int) -> bool:
        """Gate refinement by interval, cooldown, loss progress, and failing-cell count."""
        if not cfg.enable_refinement:
            return False
        if num_failing < int(getattr(cfg, "refine_min_failing_cells", 1)):
            return False
        refine_interval = (cfg.refine_interval_late
                         if (epoch_1b - 1) > cfg.late_epoch_threshold
                         else cfg.refine_interval)
        if refine_interval <= 0 or (epoch_1b % refine_interval) != 0:
            return False
        last_ep = last_refine_epoch[region_key]
        cooldown = int(getattr(cfg, "refine_cooldown_epochs", 0))
        if last_ep is not None and (epoch_1b - last_ep) < cooldown:
            return False
        prev_loss = last_refine_loss[region_key]
        rel_improve = float(getattr(cfg, "refine_loss_rel_improve", 0.0))
        if prev_loss is not None and total_loss_val >= prev_loss * (1.0 - rel_improve):
            return False
        return True

    def _stage3_active() -> bool:
        """Return True when training is in stage-3 (beta curriculum phase)."""
        if not getattr(params.constraints, "use_beta_ra_curriculum", False):
            return True
        if params.compute_GV and use_gen_thresh_curr:
            return (
                generator_threshold_current is not None and
                generator_threshold_current <= generator_threshold_final + 1e-8
            )
        return True

    def _snapshot_region_cells(cells_dict: dict) -> dict:
        """Shallow-copy region cell lists (tuples/tensors are treated as immutable here)."""
        return {k: list(v) for k, v in cells_dict.items()}

    def _apply_smt_filter_if_enabled(
        *,
        epoch_1b: int,
        region_name: str,
        region_cells_local,
        failing_mask_local: torch.Tensor,
        scores_local: torch.Tensor,
        mode: str,
        threshold_low: float,
        threshold_high: float = None,
        target_net: nn.Module = None,
        max_cells_override: int = None,
        force_run: bool = False,
        check_any_region: bool = False,
    ) -> torch.Tensor:
        nonlocal smt_missing_solver_warned, smt_generator_phase_active
        if not smt_filter_enabled or smt_verifier is None:
            return failing_mask_local
        in_stage3 = _stage3_active()
        allow_all_regions_now = smt_check_all_regions_in_stage3 and in_stage3
        if (
            (not check_any_region)
            and (str(region_name).strip().lower() not in smt_filter_regions)
            and (not allow_all_regions_now)
        ):
            return failing_mask_local
        if (not force_run) and epoch_1b < smt_filter_start_epoch:
            return failing_mask_local
        if (not force_run) and (smt_filter_interval <= 0 or (epoch_1b % smt_filter_interval) != 0):
            return failing_mask_local
        if (
            str(region_name).strip().lower() == "generator"
            and generator_only_smt
            and (not smt_generator_phase_active)
        ):
            return failing_mask_local
        if int(failing_mask_local.sum().item()) <= 0:
            return failing_mask_local

        filtered_mask, smt_stats = smt_verifier.filter_failing_mask(
            V_net=V_net,
            region_cells=region_cells_local,
            failing_mask=failing_mask_local,
            mode=mode,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            scores=scores_local,
            max_cells_to_check=(smt_filter_max_cells if max_cells_override is None else int(max_cells_override)),
            GV_net=target_net,
        )
        ce_added_now = 0
        if smt_ce_replay_enabled and smt_stats.sat_witnesses:
            for _, witness in smt_stats.sat_witnesses:
                _push_smt_counterexample(region_name, witness)
                ce_added_now += 1
            smt_ce_added_total[region_name] += ce_added_now

        if smt_stats.missing_solver:
            reason = getattr(smt_stats, "missing_reason", "") or "solver capability is unavailable"
            raise RuntimeError(f"SMT solver is enabled but unavailable: {reason}")

        if smt_stats.checked > 0:
            select_policy = "top-violating" if scores_local is not None else "first-k"
            score_span = ""
            if smt_stats.selected_score_min is not None and smt_stats.selected_score_max is not None:
                score_span = (
                    f", selected_score_range=[{smt_stats.selected_score_min:.4f}, "
                    f"{smt_stats.selected_score_max:.4f}]"
                )
            print(
                f"SMT solver [{region_name}] @ epoch {epoch_1b}: "
                f"failing_total={smt_stats.failing_total}, selected={smt_stats.selected} ({select_policy}), "
                f"checked={smt_stats.checked}, unsat={smt_stats.unsat}, sat={smt_stats.sat}, unknown={smt_stats.unknown}"
                f"{score_span}, ce_added={ce_added_now}, ce_total={smt_ce_added_total[region_name]}"
            )
        return filtered_mask

    def _beta_target_reached() -> bool:
        if getattr(params.constraints, "use_beta_ra_curriculum", False):
            if beta_ra_current is None:
                return False
            return float(beta_ra_current) >= float(params.constraints.beta_ra) - 1e-8
        if getattr(params.constraints, "use_terminal_beta_curriculum", False):
            if beta_ra_terminal_current is None:
                return False
            return float(beta_ra_terminal_current) >= float(params.constraints.beta_ra) - 1e-8
        return True

    for epoch in range(params.training.num_epochs):
        if _check_user_stop_command():
            if has_stage3_sat_certificate and latest_stage3_sat_checkpoint is not None:
                print("\n" + "="*20)
                print("User requested stop. Restoring latest SAT certificate and stopping.")
                print("="*20)
                user_stop_requested = True
                break
            print("Stop ignored: stage-3 SAT certificate/controller not available yet.")

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
            can_start_gv = (
                epoch >= params.training.generator_start_epoch and
                generator_target_weight > 0 and
                crown_cache_phi is not None and
                gv_phase_active
            )
            if can_start_gv:
                phi_uppers = crown_cache_phi.compute_bounds(input_lowers_gen, input_uppers_gen)
                if gv_phase_start_epoch is None:
                    gv_phase_start_epoch = epoch
                if generator_ramp_epochs > 0:
                    alpha = min(1.0, float(epoch - gv_phase_start_epoch + 1) / float(max(1, generator_ramp_epochs)))
                    current_gen_weight = generator_target_weight * alpha
                else:
                    current_gen_weight = generator_target_weight

                # Track failing cells for adaptive refinement
                generator_fail_threshold = (
                    float(generator_threshold_current)
                    if (use_gen_thresh_curr and generator_threshold_current is not None)
                    else 0.0
                )
                phi_upper_failing_mask = phi_uppers > generator_fail_threshold
                generator_scores = torch.relu(phi_uppers - float(generator_fail_threshold))
                phi_upper_failing_mask = _apply_smt_filter_if_enabled(
                    epoch_1b=epoch + 1,
                    region_name='generator',
                    region_cells_local=region_cells['generator'],
                    failing_mask_local=phi_upper_failing_mask,
                    scores_local=generator_scores,
                    mode='le',
                    threshold_low=float(generator_fail_threshold),
                    target_net=GV_net,
                )
                num_total_failing = phi_upper_failing_mask.sum().item()

            else:
                phi_uppers = empty_tensor
                current_gen_weight = 0.0
                num_total_failing = 0

        # Update loss kwargs with current bounds
        if params.compute_V:
            if getattr(params.constraints, "use_beta_ra_curriculum", False):
                loss_kwargs['beta_ra'] = beta_ra_current
            else:
                loss_kwargs['beta_ra'] = params.constraints.beta_ra
            loss_kwargs['V_goal_lower'] = bounds['goal'][0]
            if (getattr(params.constraints, "use_terminal_beta_curriculum", False)
                and not getattr(params.constraints, "use_beta_ra_curriculum", False)):
                n_unsafe_total = len(bounds['unsafe'][0])
                n_terminal = int(getattr(params.constraints, "unsafe_terminal_cell_count", 0))
                n_terminal = min(max(n_terminal, 0), n_unsafe_total)
                n_tube = n_unsafe_total - n_terminal
                loss_kwargs['V_unsafe_lower_tube'] = bounds['unsafe'][0][:n_tube]
                loss_kwargs['V_unsafe_lower_terminal'] = bounds['unsafe'][0][n_tube:]
                loss_kwargs['beta_ra_terminal'] = beta_ra_terminal_current
                loss_kwargs['V_unsafe_lower'] = None
            else:
                loss_kwargs['V_unsafe_lower'] = bounds['unsafe'][0]
                loss_kwargs['V_unsafe_lower_tube'] = None
                loss_kwargs['V_unsafe_lower_terminal'] = None
                loss_kwargs['beta_ra_terminal'] = None
            loss_kwargs['V_init_upper'] = bounds['init'][1]
            loss_kwargs['V_outside_lower'] = bounds['outside'][0]

        if params.compute_GV:
            loss_kwargs['Phi_upper'] = phi_uppers
            loss_kwargs['generator_weight'] = current_gen_weight
            loss_kwargs['generator_threshold'] = float(generator_threshold_current) if (use_gen_thresh_curr and generator_threshold_current is not None) else 0.0

        total_loss, loss_dict, sat_dict = compute_total_loss_bounds(**loss_kwargs)

        # CEGIS-style replay on SMT counterexamples: add local pointwise losses.
        ce_loss_v = torch.tensor(0.0, device=params.training.device)
        ce_loss_gv = torch.tensor(0.0, device=params.training.device)
        if smt_ce_replay_enabled:
            if params.compute_V and smt_ce_replay_weight_v > 0.0:
                v_terms = []
                for region_name in ('goal', 'unsafe', 'init', 'outside'):
                    x_ce = _sample_smt_counterexamples(region_name, smt_ce_replay_batch_size, params.training.device)
                    if x_ce.numel() == 0:
                        continue
                    v_val = V_net(x_ce).reshape(-1)
                    if region_name in ('goal', 'outside'):
                        v_terms.append(torch.relu(0.0 - v_val).mean())
                    elif region_name == 'unsafe':
                        beta_unsafe_current = (
                            float(beta_ra_current)
                            if (getattr(params.constraints, "use_beta_ra_curriculum", False) and beta_ra_current is not None)
                            else float(params.constraints.beta_ra)
                        )
                        v_terms.append(torch.relu(beta_unsafe_current - v_val).mean())
                    elif region_name == 'init':
                        v_terms.append((torch.relu(0.0 - v_val) + torch.relu(v_val - 1.0)).mean())
                if len(v_terms) > 0:
                    ce_loss_v = torch.stack(v_terms).mean()
                    total_loss = total_loss + smt_ce_replay_weight_v * ce_loss_v

            if params.compute_GV and smt_ce_replay_weight_gv > 0.0:
                x_ce_g = _sample_smt_counterexamples('generator', smt_ce_replay_batch_size, params.training.device)
                if x_ce_g.numel() > 0:
                    gv_val = GV_net(x_ce_g).reshape(-1)
                    gen_thr = float(generator_threshold_current) if (
                        use_gen_thresh_curr and generator_threshold_current is not None
                    ) else 0.0
                    ce_loss_gv = torch.relu(gv_val - gen_thr + 1e-4).mean()
                    total_loss = total_loss + smt_ce_replay_weight_gv * ce_loss_gv

        loss_dict['smt_ce_v'] = float(ce_loss_v.detach().item())
        loss_dict['smt_ce_generator'] = float(ce_loss_gv.detach().item())
        loss_dict['total'] = float(total_loss.detach().item())
        total_loss_val = float(loss_dict['total'])
        refinement_candidates = []
        pre_refine_all_satisfied = True
        forced_refinement_done = False
        refinement_done_this_epoch = False
        if params.compute_V:
            for key in ['goal', 'unsafe', 'init', 'outside']:
                if sat_dict[key] is False:
                    pre_refine_all_satisfied = False
                    break
        if params.compute_GV and sat_dict.get('generator', True) is False:
            pre_refine_all_satisfied = False
        skip_refinement_this_epoch = (
            params.compute_GV and gv_phase_active and use_gen_thresh_curr and
            pre_refine_all_satisfied and generator_threshold_current is not None and
            generator_threshold_current > generator_threshold_final + 1e-8
        )
        if skip_refinement_this_epoch:
            print(
                f"Skipping refinement at epoch {epoch}: "
                "generator threshold will be decreased this epoch."
            )

        # Backward pass
        total_loss.backward()

        # Reuse current-epoch V bounds for refinement/logging to avoid an extra
        # full bound pass per epoch (major runtime cost).
        if params.compute_V and not skip_refinement_this_epoch:
            for name in region_order_V:
                bounds_updated[name] = bounds[name]

        # Adaptive refinement for V region cells: goal / unsafe / init / outside
        if params.compute_V:
            # Goal: V >= 0
            if len(bounds_updated['goal'][0]) > 0:
                v_cfg = params.refinement.v_goal
                failing_mask = bounds_updated['goal'][0] < 0.0
                scores = torch.relu(0.0 - bounds_updated['goal'][0])
                failing_mask = _apply_smt_filter_if_enabled(
                    epoch_1b=epoch + 1,
                    region_name='goal',
                    region_cells_local=region_cells['goal'],
                    failing_mask_local=failing_mask,
                    scores_local=scores,
                    mode='ge',
                    threshold_low=0.0,
                )
                num_failing = failing_mask.sum().item()
                if len(failing_mask) == len(region_cells['goal']) and num_failing > 0:
                    if simple_refine_mode:
                        if len(region_cells['goal']) < v_cfg.max_cells:
                            sev = float(scores[failing_mask].mean().item()) if num_failing > 0 else 0.0
                            refinement_candidates.append({
                                'name': 'goal', 'cfg': v_cfg, 'failing_mask': failing_mask,
                                'scores': scores, 'num_failing': num_failing, 'severity': sev,
                            })
                    elif (_should_refine_region(v_cfg, 'goal', epoch + 1, total_loss_val, num_failing) and
                          len(region_cells['goal']) < v_cfg.max_cells):
                        new_cells, num_refined = refine_failing_cells(region_cells['goal'], failing_mask, v_cfg.refine_factor, N_to_refine=v_cfg.N_to_refine, scores=scores)
                        region_cells['goal'] = new_cells
                        print(f"Refining goal cells: {num_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                        needs_v_cache_rebuild = True
                        refinement_done_this_epoch = True
                        refinement_epochs['goal'].append(epoch + 1)
                        last_refine_epoch['goal'] = epoch + 1
                        last_refine_loss['goal'] = total_loss_val

            # Unsafe: V >= beta
            if len(bounds_updated['unsafe'][0]) > 0:
                v_cfg = params.refinement.v_unsafe
                beta_unsafe_current = (
                    float(beta_ra_current)
                    if (getattr(params.constraints, "use_beta_ra_curriculum", False) and beta_ra_current is not None)
                    else float(params.constraints.beta_ra)
                )
                if getattr(params.constraints, "use_beta_ra_curriculum", False) and not getattr(params.constraints, "use_terminal_beta_curriculum", False):
                    failing_mask = bounds_updated['unsafe'][0] < beta_unsafe_current
                    scores = torch.relu(beta_unsafe_current - bounds_updated['unsafe'][0])
                elif getattr(params.constraints, "use_terminal_beta_curriculum", False):
                    n_total = len(bounds_updated['unsafe'][0])
                    n_terminal = min(max(int(getattr(params.constraints, "unsafe_terminal_cell_count", 0)), 0), n_total)
                    n_tube = n_total - n_terminal
                    tube = bounds_updated['unsafe'][0][:n_tube]
                    term = bounds_updated['unsafe'][0][n_tube:]
                    beta_terminal_current = (
                        float(beta_ra_terminal_current)
                        if beta_ra_terminal_current is not None
                        else beta_unsafe_current
                    )
                    tube_fail = tube < beta_unsafe_current
                    term_fail = term < beta_terminal_current
                    failing_mask = torch.cat([tube_fail, term_fail], dim=0)
                    tube_scores = torch.relu(beta_unsafe_current - tube)
                    term_scores = torch.relu(beta_terminal_current - term)
                    scores = torch.cat([tube_scores, term_scores], dim=0)
                else:
                    failing_mask = bounds_updated['unsafe'][0] < beta_unsafe_current
                    scores = torch.relu(beta_unsafe_current - bounds_updated['unsafe'][0])

                failing_mask = _apply_smt_filter_if_enabled(
                    epoch_1b=epoch + 1,
                    region_name='unsafe',
                    region_cells_local=region_cells['unsafe'],
                    failing_mask_local=failing_mask,
                    scores_local=scores,
                    mode='ge',
                    threshold_low=float(beta_unsafe_current),
                )
                num_failing = failing_mask.sum().item()
                if len(failing_mask) == len(region_cells['unsafe']) and num_failing > 0:
                    if simple_refine_mode:
                        if len(region_cells['unsafe']) < v_cfg.max_cells:
                            sev = float(scores[failing_mask].mean().item()) if num_failing > 0 else 0.0
                            refinement_candidates.append({
                                'name': 'unsafe', 'cfg': v_cfg, 'failing_mask': failing_mask,
                                'scores': scores, 'num_failing': num_failing, 'severity': sev,
                            })
                    elif (_should_refine_region(v_cfg, 'unsafe', epoch + 1, total_loss_val, num_failing) and
                          len(region_cells['unsafe']) < v_cfg.max_cells):
                        new_cells, num_refined = refine_failing_cells(region_cells['unsafe'], failing_mask, v_cfg.refine_factor, N_to_refine=v_cfg.N_to_refine, scores=scores)
                        region_cells['unsafe'] = new_cells
                        print(f"Refining unsafe cells: {num_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                        needs_v_cache_rebuild = True
                        refinement_done_this_epoch = True
                        refinement_epochs['unsafe'].append(epoch + 1)
                        last_refine_epoch['unsafe'] = epoch + 1
                        last_refine_loss['unsafe'] = total_loss_val

            # Init: V in [0, 1]
            if len(bounds_updated['init'][0]) > 0 and len(bounds_updated['init'][1]) > 0:
                v_cfg = params.refinement.v_init
                lower = bounds_updated['init'][0]
                upper = bounds_updated['init'][1]
                failing_mask = (lower < 0.0) | (upper > 1.0)
                scores = torch.relu(0.0 - lower) + torch.relu(upper - 1.0)
                failing_mask = _apply_smt_filter_if_enabled(
                    epoch_1b=epoch + 1,
                    region_name='init',
                    region_cells_local=region_cells['init'],
                    failing_mask_local=failing_mask,
                    scores_local=scores,
                    mode='interval',
                    threshold_low=0.0,
                    threshold_high=1.0,
                )
                num_failing = failing_mask.sum().item()
                if len(failing_mask) == len(region_cells['init']) and num_failing > 0:
                    if simple_refine_mode:
                        if len(region_cells['init']) < v_cfg.max_cells:
                            sev = float(scores[failing_mask].mean().item()) if num_failing > 0 else 0.0
                            refinement_candidates.append({
                                'name': 'init', 'cfg': v_cfg, 'failing_mask': failing_mask,
                                'scores': scores, 'num_failing': num_failing, 'severity': sev,
                            })
                    elif (_should_refine_region(v_cfg, 'init', epoch + 1, total_loss_val, num_failing) and
                          len(region_cells['init']) < v_cfg.max_cells):
                        new_cells, num_refined = refine_failing_cells(region_cells['init'], failing_mask, v_cfg.refine_factor, N_to_refine=v_cfg.N_to_refine, scores=scores)
                        region_cells['init'] = new_cells
                        print(f"Refining init cells: {num_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                        needs_v_cache_rebuild = True
                        refinement_done_this_epoch = True
                        refinement_epochs['init'].append(epoch + 1)
                        last_refine_epoch['init'] = epoch + 1
                        last_refine_loss['init'] = total_loss_val

            # Outside: V >= 0
            if len(bounds_updated['outside'][0]) > 0:
                v_cfg = params.refinement.v_outside
                failing_mask = bounds_updated['outside'][0] < 0.0
                scores = torch.relu(0.0 - bounds_updated['outside'][0])
                failing_mask = _apply_smt_filter_if_enabled(
                    epoch_1b=epoch + 1,
                    region_name='outside',
                    region_cells_local=region_cells['outside'],
                    failing_mask_local=failing_mask,
                    scores_local=scores,
                    mode='ge',
                    threshold_low=0.0,
                )
                num_failing = failing_mask.sum().item()
                if len(failing_mask) == len(region_cells['outside']) and num_failing > 0:
                    if simple_refine_mode:
                        if len(region_cells['outside']) < v_cfg.max_cells:
                            sev = float(scores[failing_mask].mean().item()) if num_failing > 0 else 0.0
                            refinement_candidates.append({
                                'name': 'outside', 'cfg': v_cfg, 'failing_mask': failing_mask,
                                'scores': scores, 'num_failing': num_failing, 'severity': sev,
                            })
                    elif (_should_refine_region(v_cfg, 'outside', epoch + 1, total_loss_val, num_failing) and
                          len(region_cells['outside']) < v_cfg.max_cells):
                        new_cells, num_refined = refine_failing_cells(region_cells['outside'], failing_mask, v_cfg.refine_factor, N_to_refine=v_cfg.N_to_refine, scores=scores)
                        region_cells['outside'] = new_cells
                        print(f"Refining outside cells: {num_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                        needs_v_cache_rebuild = True
                        refinement_done_this_epoch = True
                        refinement_epochs['outside'].append(epoch + 1)
                        last_refine_epoch['outside'] = epoch + 1
                        last_refine_loss['outside'] = total_loss_val

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
                    needs_v_cache_rebuild = True

        # Adaptive refinement for generator cells
        if params.compute_GV and not skip_refinement_this_epoch:
            gv_cfg = params.refinement.gv_generator
            if (epoch >= params.training.generator_start_epoch and
                current_gen_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):

                # Get refinement interval based on epoch
                # Check if it's time to refine
                if simple_refine_mode:
                    if len(region_cells['generator']) < gv_cfg.max_cells:
                        sev = float(phi_uppers[phi_upper_failing_mask].mean().item()) if num_total_failing > 0 else 0.0
                        refinement_candidates.append({
                            'name': 'generator', 'cfg': gv_cfg, 'failing_mask': phi_upper_failing_mask,
                            'scores': phi_uppers, 'num_failing': num_total_failing, 'severity': sev,
                        })
                elif (_should_refine_region(gv_cfg, 'generator', epoch + 1, total_loss_val, num_total_failing) and
                      len(region_cells['generator']) < gv_cfg.max_cells):
                    new_cells, num_refined = refine_failing_cells(region_cells['generator'], phi_upper_failing_mask, gv_cfg.refine_factor, N_to_refine=gv_cfg.N_to_refine, scores=phi_uppers)
                    region_cells['generator'] = new_cells
                    print(f"Refining generator cells: {num_total_failing} failing, refined {num_refined} cells, {len(new_cells)} total")
                    needs_gv_cache_rebuild = True
                    refinement_done_this_epoch = True
                    refinement_epochs['generator'].append(epoch + 1)
                    last_refine_epoch['generator'] = epoch + 1
                    last_refine_loss['generator'] = total_loss_val
                
                elif (gv_cfg.enable_refinement and
                    (epoch + 1) % gv_cfg.refine_interval == 0 and
                    len(region_cells['generator']) > gv_cfg.max_cells):
                    print(f"Generator cells exceed max cells threshold: {len(region_cells['generator'])} > {gv_cfg.max_cells}")    

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
                needs_gv_cache_rebuild = True

        if (
            stage3_force_refine_interval > 0
            and _stage3_active()
            and getattr(params.constraints, "use_beta_ra_curriculum", False)
            and not skip_refinement_this_epoch
        ):
            epoch_1b = epoch + 1
            current_beta_k = (
                float(beta_ra_current)
                if beta_ra_current is not None
                else float(params.constraints.beta_ra)
            )
            if stage3_window_start_epoch is None:
                stage3_window_start_epoch = epoch_1b
                stage3_window_start_beta = current_beta_k
                stage3_refined_in_window = False
            if refinement_done_this_epoch:
                stage3_refined_in_window = True

            window_len = epoch_1b - stage3_window_start_epoch + 1
            if window_len >= stage3_force_refine_interval:
                beta_increased = current_beta_k > float(stage3_window_start_beta) + 1e-8
                no_refine = not stage3_refined_in_window
                if (no_refine or (not beta_increased)) and len(refinement_candidates) > 0:
                    refinement_candidates.sort(key=lambda c: c['severity'], reverse=True)
                    selected = refinement_candidates[:max(1, simple_regions_per_trigger)]
                    per_region_budget = max(1, simple_refine_budget // max(1, len(selected)))
                    reason = "no refinement" if no_refine else "beta_ra_k did not increase"
                    print(
                        f"Stage-3 forced refinement at epoch {epoch_1b} ({reason} in last {stage3_force_refine_interval} epochs)."
                    )
                    for c in selected:
                        new_cells, num_refined = refine_failing_cells(
                            region_cells[c['name']],
                            c['failing_mask'],
                            c['cfg'].refine_factor,
                            N_to_refine=per_region_budget,
                            scores=c['scores'],
                        )
                        region_cells[c['name']] = new_cells
                        print(
                            f"Forced refining {c['name']} cells: {c['num_failing']} failing, "
                            f"refined {num_refined} cells, {len(new_cells)} total"
                        )
                        refinement_epochs[c['name']].append(epoch_1b)
                        if c['name'] == 'generator':
                            needs_gv_cache_rebuild = True
                        else:
                            needs_v_cache_rebuild = True
                        refinement_done_this_epoch = True
                    simple_last_refine_loss = None
                    forced_refinement_done = True

                stage3_window_start_epoch = epoch_1b + 1
                stage3_window_start_beta = current_beta_k
                stage3_refined_in_window = False

        if simple_refine_mode and not skip_refinement_this_epoch and not forced_refinement_done:
            epoch_1b = epoch + 1
            on_refine_tick = simple_refine_every > 0 and (epoch_1b % simple_refine_every == 0)
            improved_enough = (
                simple_last_refine_loss is None or
                total_loss_val < simple_last_refine_loss * (1.0 - simple_loss_improve_tol)
            )
            if on_refine_tick and improved_enough and len(refinement_candidates) > 0:
                refinement_candidates.sort(key=lambda c: c['severity'], reverse=True)
                selected = refinement_candidates[:max(1, simple_regions_per_trigger)]
                per_region_budget = max(1, simple_refine_budget // max(1, len(selected)))
                for c in selected:
                    new_cells, num_refined = refine_failing_cells(
                        region_cells[c['name']],
                        c['failing_mask'],
                        c['cfg'].refine_factor,
                        N_to_refine=per_region_budget,
                        scores=c['scores'],
                    )
                    region_cells[c['name']] = new_cells
                    print(
                        f"Refining {c['name']} cells (simple mode): {c['num_failing']} failing, "
                        f"refined {num_refined} cells, {len(new_cells)} total"
                    )
                    refinement_epochs[c['name']].append(epoch_1b)
                    if c['name'] == 'generator':
                        needs_gv_cache_rebuild = True
                    else:
                        needs_v_cache_rebuild = True
                    refinement_done_this_epoch = True
                simple_last_refine_loss = total_loss_val

        # Logging
        if epoch % params.logging.loss_log_interval == 0 or epoch == params.training.num_epochs - 1 or epoch == 0:
            elapsed = time.time() - start_time
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV, elapsed_time=elapsed)
            if params.compute_GV:
                print(f"generator_weight_k: {current_gen_weight:.4f}")
                if use_gen_thresh_curr and generator_threshold_current is not None:
                    print(f"generator_threshold_k: {generator_threshold_current:.4f}")
            if getattr(params.constraints, "use_beta_ra_curriculum", False):
                print(f"beta_ra_k: {beta_ra_current:.4f}")
            elif getattr(params.constraints, "use_terminal_beta_curriculum", False):
                print(f"beta_ra_k (terminal): {beta_ra_terminal_current:.4f}")
            loss_dict['epoch'] = epoch
            if getattr(params.constraints, "use_beta_ra_curriculum", False):
                loss_dict['beta_ra_k'] = float(beta_ra_current)
            elif getattr(params.constraints, "use_terminal_beta_curriculum", False):
                loss_dict['beta_ra_k_terminal'] = float(beta_ra_terminal_current)
            loss_history.append(loss_dict.copy())

        # Early stopping check
        with torch.no_grad():
            # Check V constraints
            v_only_satisfied = True
            if params.compute_V:
                for key in ['goal', 'unsafe', 'init', 'outside']:
                    if sat_dict[key] is False:
                        v_only_satisfied = False
                        break

            generator_bound_satisfied = True
            if params.compute_GV:
                generator_bound_satisfied = bool(sat_dict.get('generator', True))

            # Bound-level status (kept for diagnostics/snapshot fallbacks).
            all_satisfied = (v_only_satisfied and generator_bound_satisfied)
            # Strict bound-loss gate for generator-threshold decrement:
            # only allow decrement when all active bound losses are zero.
            active_bound_loss_keys = []
            if params.compute_V:
                active_bound_loss_keys.extend(['goal', 'unsafe', 'init', 'outside'])
            if params.compute_GV:
                active_bound_loss_keys.append('generator')
            bound_losses_all_zero = True
            for _k in active_bound_loss_keys:
                if float(loss_dict.get(_k, 0.0)) > 0.0:
                    bound_losses_all_zero = False
                    break

            # SMT curriculum criterion for generator phase:
            # run a full generator SMT pass over all currently failing generator cells.
            generator_smt_clean = generator_bound_satisfied
            if (
                params.compute_GV
                and gv_phase_active
                and use_gen_thresh_curr
                and isinstance(phi_uppers, torch.Tensor)
                and phi_uppers.numel() > 0
            ):
                generator_fail_threshold = (
                    float(generator_threshold_current)
                    if (generator_threshold_current is not None)
                    else 0.0
                )
                gen_fail_mask_full = phi_uppers > generator_fail_threshold
                if int(gen_fail_mask_full.sum().item()) > 0:
                    gen_scores_full = torch.relu(phi_uppers - float(generator_fail_threshold))
                    gen_filtered_full = _apply_smt_filter_if_enabled(
                        epoch_1b=epoch + 1,
                        region_name='generator',
                        region_cells_local=region_cells['generator'],
                        failing_mask_local=gen_fail_mask_full,
                        scores_local=gen_scores_full,
                        mode='le',
                        threshold_low=float(generator_fail_threshold),
                        target_net=GV_net,
                        max_cells_override=int(gen_fail_mask_full.sum().item()),
                        force_run=True,
                        check_any_region=True,
                    )
                    generator_smt_clean = int(gen_filtered_full.sum().item()) == 0
                else:
                    generator_smt_clean = True

            # Stage-3 SMT criterion for unsafe/beta curriculum and final stop:
            # all active constraints must have no SMT-confirmed violating cells.
            stage3_all_constraints_smt_clean = False
            stage3_smt_checked_this_epoch = False
            run_stage3_full_smt_now = (
                _stage3_active()
                and smt_filter_enabled
                and all_satisfied
                and ((epoch + 1) >= stage3_smt_full_check_start_epoch)
                and (
                    stage3_smt_full_check_interval <= 1
                    or (((epoch + 1) - stage3_smt_full_check_start_epoch) % stage3_smt_full_check_interval == 0)
                )
            )
            if run_stage3_full_smt_now:
                stage3_smt_checked_this_epoch = True
                stage3_all_constraints_smt_clean = True

                if params.compute_V and len(bounds_updated.get('goal', (empty_tensor, empty_tensor))[0]) > 0:
                    goal_lower = bounds_updated['goal'][0]
                    goal_fail = goal_lower < 0.0
                    goal_scores = torch.relu(0.0 - goal_lower)
                    if int(goal_fail.sum().item()) > 0:
                        goal_filtered = _apply_smt_filter_if_enabled(
                            epoch_1b=epoch + 1,
                            region_name='goal',
                            region_cells_local=region_cells['goal'],
                            failing_mask_local=goal_fail,
                            scores_local=goal_scores,
                            mode='ge',
                            threshold_low=0.0,
                            max_cells_override=int(goal_fail.sum().item()),
                            force_run=True,
                            check_any_region=True,
                        )
                        if int(goal_filtered.sum().item()) > 0:
                            stage3_all_constraints_smt_clean = False

                if stage3_all_constraints_smt_clean and params.compute_V and len(bounds_updated.get('unsafe', (empty_tensor, empty_tensor))[0]) > 0:
                    unsafe_lower = bounds_updated['unsafe'][0]
                    beta_unsafe_current = (
                        float(beta_ra_current)
                        if (getattr(params.constraints, "use_beta_ra_curriculum", False) and beta_ra_current is not None)
                        else float(params.constraints.beta_ra)
                    )
                    unsafe_threshold_high = None
                    if getattr(params.constraints, "use_terminal_beta_curriculum", False):
                        n_total = len(unsafe_lower)
                        n_terminal = min(max(int(getattr(params.constraints, "unsafe_terminal_cell_count", 0)), 0), n_total)
                        n_tube = n_total - n_terminal
                        tube = unsafe_lower[:n_tube]
                        term = unsafe_lower[n_tube:]
                        beta_terminal_current = (
                            float(beta_ra_terminal_current)
                            if (beta_ra_terminal_current is not None)
                            else beta_unsafe_current
                        )
                        tube_fail = tube < beta_unsafe_current
                        term_fail = term < beta_terminal_current
                        unsafe_fail = torch.cat([tube_fail, term_fail], dim=0)
                        tube_scores = torch.relu(beta_unsafe_current - tube)
                        term_scores = torch.relu(beta_terminal_current - term)
                        unsafe_scores = torch.cat([tube_scores, term_scores], dim=0)
                        unsafe_threshold_high = float(beta_terminal_current)
                    else:
                        unsafe_fail = unsafe_lower < beta_unsafe_current
                        unsafe_scores = torch.relu(beta_unsafe_current - unsafe_lower)
                    if int(unsafe_fail.sum().item()) > 0:
                        unsafe_filtered = _apply_smt_filter_if_enabled(
                            epoch_1b=epoch + 1,
                            region_name='unsafe',
                            region_cells_local=region_cells['unsafe'],
                            failing_mask_local=unsafe_fail,
                            scores_local=unsafe_scores,
                            mode='ge',
                            threshold_low=float(beta_unsafe_current),
                            threshold_high=unsafe_threshold_high,
                            max_cells_override=int(unsafe_fail.sum().item()),
                            force_run=True,
                            check_any_region=True,
                        )
                        if int(unsafe_filtered.sum().item()) > 0:
                            stage3_all_constraints_smt_clean = False

                if (
                    stage3_all_constraints_smt_clean
                    and params.compute_V
                    and len(bounds_updated.get('init', (empty_tensor, empty_tensor))[0]) > 0
                    and len(bounds_updated.get('init', (empty_tensor, empty_tensor))[1]) > 0
                ):
                    init_lower = bounds_updated['init'][0]
                    init_upper = bounds_updated['init'][1]
                    init_fail = (init_lower < 0.0) | (init_upper > 1.0)
                    init_scores = torch.relu(0.0 - init_lower) + torch.relu(init_upper - 1.0)
                    if int(init_fail.sum().item()) > 0:
                        init_filtered = _apply_smt_filter_if_enabled(
                            epoch_1b=epoch + 1,
                            region_name='init',
                            region_cells_local=region_cells['init'],
                            failing_mask_local=init_fail,
                            scores_local=init_scores,
                            mode='range',
                            threshold_low=0.0,
                            threshold_high=1.0,
                            max_cells_override=int(init_fail.sum().item()),
                            force_run=True,
                            check_any_region=True,
                        )
                        if int(init_filtered.sum().item()) > 0:
                            stage3_all_constraints_smt_clean = False

                if stage3_all_constraints_smt_clean and params.compute_V and len(bounds_updated.get('outside', (empty_tensor, empty_tensor))[0]) > 0:
                    outside_lower = bounds_updated['outside'][0]
                    outside_fail = outside_lower < 0.0
                    outside_scores = torch.relu(0.0 - outside_lower)
                    if int(outside_fail.sum().item()) > 0:
                        outside_filtered = _apply_smt_filter_if_enabled(
                            epoch_1b=epoch + 1,
                            region_name='outside',
                            region_cells_local=region_cells['outside'],
                            failing_mask_local=outside_fail,
                            scores_local=outside_scores,
                            mode='ge',
                            threshold_low=0.0,
                            max_cells_override=int(outside_fail.sum().item()),
                            force_run=True,
                            check_any_region=True,
                        )
                        if int(outside_filtered.sum().item()) > 0:
                            stage3_all_constraints_smt_clean = False

                if (
                    stage3_all_constraints_smt_clean
                    and params.compute_GV
                    and gv_phase_active
                    and isinstance(phi_uppers, torch.Tensor)
                    and phi_uppers.numel() > 0
                ):
                    generator_fail_threshold = (
                        float(generator_threshold_current)
                        if (use_gen_thresh_curr and generator_threshold_current is not None)
                        else 0.0
                    )
                    gen_fail_mask_full = phi_uppers > generator_fail_threshold
                    if int(gen_fail_mask_full.sum().item()) > 0:
                        gen_scores_full = torch.relu(phi_uppers - float(generator_fail_threshold))
                        gen_filtered_full = _apply_smt_filter_if_enabled(
                            epoch_1b=epoch + 1,
                            region_name='generator',
                            region_cells_local=region_cells['generator'],
                            failing_mask_local=gen_fail_mask_full,
                            scores_local=gen_scores_full,
                            mode='le',
                            threshold_low=float(generator_fail_threshold),
                            target_net=GV_net,
                            max_cells_override=int(gen_fail_mask_full.sum().item()),
                            force_run=True,
                            check_any_region=True,
                        )
                        if int(gen_filtered_full.sum().item()) > 0:
                            stage3_all_constraints_smt_clean = False
            elif _stage3_active() and all_satisfied and (epoch % params.logging.loss_log_interval == 0):
                print(
                    f"Stage-3 SMT full check skipped at epoch {epoch + 1} "
                    f"(interval={stage3_smt_full_check_interval}, start={stage3_smt_full_check_start_epoch})."
                )

            if (params.compute_GV and generator_after_v_sat and not gv_phase_active and
                epoch >= params.training.generator_start_epoch and v_only_satisfied):
                gv_phase_active = True
                gv_phase_start_epoch = epoch + 1
                if (use_gen_thresh_curr and generator_threshold_current is None and
                    crown_cache_phi is not None and input_lowers_gen is not None):
                    phi_uppers_transition = crown_cache_phi.compute_bounds(input_lowers_gen, input_uppers_gen)
                    transition_max = float(phi_uppers_transition.max().item()) if phi_uppers_transition.numel() > 0 else 0.0
                    generator_threshold_current = 0.5 * transition_max
                # Reset refinement improvement baselines at stage transition so
                # stage-2 refinement is not blocked by the expected generator-loss jump.
                simple_last_refine_loss = None
                last_refine_loss['generator'] = None
                last_refine_epoch['generator'] = None
                print(
                    f"V constraints SAT at epoch {epoch}. Enabling generator loss "
                    f"with ramp over {generator_ramp_epochs} epochs."
                )
                if use_gen_thresh_curr and generator_threshold_current is not None:
                    print(f"Auto-set generator_threshold_k to {generator_threshold_current:.4f} at transition.")

            # Snapshot latest SAT certificate before curriculum threshold updates.
            checkpoint_satisfied = all_satisfied
            if _stage3_active():
                checkpoint_satisfied = stage3_smt_checked_this_epoch and stage3_all_constraints_smt_clean
            if checkpoint_satisfied:
                has_sat_certificate = True
                latest_sat_checkpoint = {
                    'V_state_dict': {k: v.detach().cpu().clone() for k, v in V_net.state_dict().items()},
                    'GV_state_dict': {k: v.detach().cpu().clone() for k, v in GV_net.state_dict().items()} if GV_net is not None else None,
                    'control_state_dict': {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()} if control_net is not None else None,
                    'region_cells': _snapshot_region_cells(region_cells),
                    'beta_ra_k': (float(beta_ra_current) if beta_ra_current is not None else float(params.constraints.beta_ra)),
                    'beta_ra_terminal_k': (float(beta_ra_terminal_current) if beta_ra_terminal_current is not None else None),
                    'beta_s': float(bounds_updated['outside'][0].min().item()) if len(bounds_updated['outside'][0]) > 0 else 0.0,
                    'epoch': epoch,
                }
                controller_ready = (
                    control_net is None or latest_sat_checkpoint['control_state_dict'] is not None
                )
                if _stage3_active() and controller_ready:
                    has_stage3_sat_certificate = True
                    latest_stage3_sat_checkpoint = latest_sat_checkpoint
                    if not stop_prompt_shown:
                        print("Stage-3 SAT certificate found. You can now type 'stop' + Enter to early stop.")
                        stop_prompt_shown = True

            if (
                params.compute_GV
                and gv_phase_active
                and use_gen_thresh_curr
                and v_only_satisfied
                and generator_smt_clean
                and bound_losses_all_zero
            ):
                if generator_threshold_current > generator_threshold_final + 1e-8:
                    current_generator_upper = float(generator_threshold_current)
                    dec_step = (
                        generator_threshold_step
                        if current_generator_upper > generator_threshold_switch
                        else generator_threshold_step_fine
                    )
                    generator_threshold_current = max(
                        generator_threshold_final,
                        current_generator_upper - dec_step
                    )
                    # New generator threshold defines a new optimization phase; reset
                    # refinement improvement baselines so refinement can trigger again.
                    simple_last_refine_loss = None
                    last_refine_loss['generator'] = None
                    last_refine_epoch['generator'] = None
                    smt_generator_phase_active = True
                    all_satisfied = False
                    print(f"Decremented generator_threshold_k to {generator_threshold_current:.4f}")
                else:
                    print(f"generator_threshold_k reached target {generator_threshold_current:.4f}")

            if (getattr(params.constraints, "use_beta_ra_curriculum", False)
                and _stage3_active()
                and stage3_smt_checked_this_epoch
                and stage3_all_constraints_smt_clean):
                step = float(getattr(params.constraints, "beta_ra_step", 0.0))
                if beta_ra_current is None:
                    beta_ra_current = float(params.constraints.beta_ra)
                if beta_ra_current + 1e-8 < float(params.constraints.beta_ra):
                    beta_ra_current = min(
                        float(params.constraints.beta_ra),
                        beta_ra_current + step
                    )
                    _reset_v_refinement_baselines()
                    print(f"Incremented beta_ra_k to {beta_ra_current:.2f}")
                else:
                    print(f"beta_ra_k reached target {beta_ra_current:.2f}")
            elif (getattr(params.constraints, "use_terminal_beta_curriculum", False)
                and _stage3_active()
                and stage3_smt_checked_this_epoch
                and stage3_all_constraints_smt_clean):
                step = float(getattr(params.constraints, "beta_ra_terminal_step", 0.0))
                if beta_ra_terminal_current is None:
                    beta_ra_terminal_current = float(params.constraints.beta_ra)
                if beta_ra_terminal_current + 1e-8 < float(params.constraints.beta_ra):
                    beta_ra_terminal_current = min(
                        float(params.constraints.beta_ra),
                        beta_ra_terminal_current + step
                    )
                    _reset_v_refinement_baselines()
                    print(f"Incremented beta_ra_terminal to {beta_ra_terminal_current:.2f}")
                else:
                    print(f"beta_ra_terminal reached target {beta_ra_terminal_current:.2f}")
            smt_early_stop_ready = (
                _stage3_active()
                and all_satisfied
                and stage3_smt_checked_this_epoch
                and _beta_target_reached()
                and stage3_all_constraints_smt_clean
            )
            if (
                smt_early_stop_ready
                and params.compute_GV
                and use_gen_thresh_curr
                and generator_threshold_current is not None
                and generator_threshold_current > generator_threshold_final + 1e-8
            ):
                smt_early_stop_ready = False

            if smt_early_stop_ready:
                print("\n" + "="*20)
                print("SMT certifies all constraints in stage-3, early stopping!")
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
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                crown_cache_all=crown_cache_all,
                crown_cache_phi=crown_cache_phi,
                input_bounds_all=(input_lowers_all, input_uppers_all),
                cell_counts_V=cell_counts_V,
                input_bounds_gen=(input_lowers_gen, input_uppers_gen) if input_lowers_gen is not None else None,
                beta_ra=params.constraints.beta_ra,
                device=params.training.device
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
                crown_cache_phi = SymbolicCROWNCache_Phi(
                    phi_module=GV_net,
                    num_cells=total_cells_GV,
                    input_dim=params.network.n_inputs,
                    device=params.training.device
                )
                print(f"Rebuilt Phi cache with {total_cells_GV} cells")

    if user_stop_requested and latest_stage3_sat_checkpoint is not None:
        V_net.load_state_dict(latest_stage3_sat_checkpoint['V_state_dict'])
        if GV_net is not None and latest_stage3_sat_checkpoint['GV_state_dict'] is not None:
            GV_net.load_state_dict(latest_stage3_sat_checkpoint['GV_state_dict'])
        if control_net is not None and latest_stage3_sat_checkpoint.get('control_state_dict') is not None:
            control_net.load_state_dict(latest_stage3_sat_checkpoint['control_state_dict'])
        restored_cells = latest_stage3_sat_checkpoint.get('region_cells')
        if restored_cells is not None:
            for k in list(region_cells.keys()):
                if k in restored_cells:
                    region_cells[k] = list(restored_cells[k])
        if latest_stage3_sat_checkpoint.get('beta_ra_k') is not None:
            params.constraints.beta_ra = float(latest_stage3_sat_checkpoint['beta_ra_k'])
        if latest_stage3_sat_checkpoint.get('beta_ra_terminal_k') is not None:
            params.constraints.beta_ra_terminal_init = float(latest_stage3_sat_checkpoint['beta_ra_terminal_k'])
        final_beta_s = latest_stage3_sat_checkpoint['beta_s']
        print(
            f"Loaded SAT certificate from epoch {latest_stage3_sat_checkpoint['epoch']} "
            f"(beta_s={final_beta_s:.4f}, beta_ra_k={params.constraints.beta_ra:.4f})"
        )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return loss_history, final_beta_s, refinement_epochs
