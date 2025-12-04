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
from dynamics import Dynamics
from regions import Regions
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
    print_constraint_summary
)
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
    

def pretrain_structure_aware(model, x_goal_range, x_unsafe_range, x_init_range, x_range,
                              A=None, R=None, scale_factor=1.0, num_epochs=1000, lr=0.01, device='cpu'):
    """
    Pre-train V and GV (Φ) jointly to match constraint structure:

    V constraints:
    - Goal: V ≈ 0.4 (below BETA_S = 0.9)
    - Unsafe: V ≈ 12 (above BETA_RA = 10)
    - Init: V ≈ 0.95 (between BETA_S and 1.0)
    - Outside: V ≈ 1.2 (above BETA_S)

    GV (Φ) constraint (if A and R provided):
    - Generator region (X \ (Goal ∪ Unsafe)): Φ ≤ 0

    This gives the network a good starting point that respects both constraint structures.
    """
    print("\n" + "="*80)
    print("PRE-TRAINING: Structure-Aware Initialization (V + GV)")
    print("="*80)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create Phi module if dynamics are provided
    phi_module = None
    if A is not None and R is not None:
        from phi_module import create_GV
        # Create a simple network config for the phi module
        class NetworkConfig:
            def __init__(self):
                self.input_scale = [100.0, 100.0]
                self.scale_factor = scale_factor
        network_config = NetworkConfig()

        class TrainingConfig:
            def __init__(self):
                self.learnable_scale = False
                self.learnable_input_scale = False
        training_config = TrainingConfig()

        # Create dynamics object
        from dynamics import Dynamics
        if isinstance(A, np.ndarray):
            A_np = A
        else:
            A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A

        if isinstance(R, np.ndarray):
            G_matrix = R
        else:
            G_matrix = R.cpu().numpy() if isinstance(R, torch.Tensor) else R

        dynamics = Dynamics.dynamics(f=A, g=R)
        phi_module = create_GV(model, dynamics, network_config, training_config).to(device)
        # print(phi_module)
        print(f"  GV (Φ) pre-training ENABLED with dynamics")
    else:
        print(f"  GV (Φ) pre-training DISABLED (no dynamics provided)")

    for epoch in range(num_epochs):
        # Sample points uniformly across state space
        x = torch.rand(1000, 2, device=device)
        x[:, 0] = x[:, 0] * (x_range[0, 1] - x_range[0, 0]) + x_range[0, 0]
        x[:, 1] = x[:, 1] * (x_range[1, 1] - x_range[1, 0]) + x_range[1, 0]

        # ===== V LOSS =====
        # Define target V based on region
        target_v = torch.zeros(1000, device=device)

        for i in range(1000):
            x1, x2 = x[i, 0].item(), x[i, 1].item()

            # Check which region this point is in
            in_goal = (x_goal_range[0, 0] <= x1 <= x_goal_range[0, 1] and
                      x_goal_range[1, 0] <= x2 <= x_goal_range[1, 1])

            in_unsafe = (x_unsafe_range[0, 0] <= x1 <= x_unsafe_range[0, 1] and
                        x_unsafe_range[1, 0] <= x2 <= x_unsafe_range[1, 1])

            in_init = (x_init_range[0, 0] <= x1 <= x_init_range[0, 1] and
                      x_init_range[1, 0] <= x2 <= x_init_range[1, 1])

            if in_goal:
                # Goal: want V small (< BETA_S = 0.9)
                target_v[i] = 0.3 +  0.001 * torch.rand(1, device=device).item()  # Random in [0.3, 0.5]
            elif in_unsafe:
                # Unsafe: want V large (> BETA_RA = 10)
                target_v[i] = 20.0 + 3.0 * torch.rand(1, device=device).item()  # Random in [12, 15]
            elif in_init:
                # Init: want BETA_S < V < 1.0
                target_v[i] = 0.92 + 0.05 * torch.rand(1, device=device).item()  # Random in [0.92, 0.97]
            else:
                # Outside goal: want V > BETA_S
                target_v[i] = 0.3 + 0.3 * torch.rand(1, device=device).item()  # Random in [1.1, 1.4]

        # Forward pass for V
        v_output = model(x).squeeze()

        # V MSE Loss
        loss_v = F.mse_loss(v_output, target_v)

        # ===== GV (Φ) LOSS =====
        loss_phi = torch.tensor(0.0, device=device)
        if phi_module is not None:
            # Sample points from generator region (X \ (Goal ∪ Unsafe))
            x_gen = torch.rand(100, 2, device=device)
            x_gen[:, 0] = x_gen[:, 0] * (x_range[0, 1] - x_range[0, 0]) + x_range[0, 0]
            x_gen[:, 1] = x_gen[:, 1] * (x_range[1, 1] - x_range[1, 0]) + x_range[1, 0]

            # Filter out points in goal or unsafe
            in_goal_mask = ((x_gen[:, 0] >= x_goal_range[0, 0]) & (x_gen[:, 0] <= x_goal_range[0, 1]) &
                           (x_gen[:, 1] >= x_goal_range[1, 0]) & (x_gen[:, 1] <= x_goal_range[1, 1]))
            in_unsafe_mask = ((x_gen[:, 0] >= x_unsafe_range[0, 0]) & (x_gen[:, 0] <= x_unsafe_range[0, 1]) &
                             (x_gen[:, 1] >= x_unsafe_range[1, 0]) & (x_gen[:, 1] <= x_unsafe_range[1, 1]))

            in_generator = ~(in_goal_mask | in_unsafe_mask)
            x_gen_filtered = x_gen[in_generator]

            if len(x_gen_filtered) > 0:
                # Compute Φ
                phi_output = phi_module(x_gen_filtered).squeeze()

                # Loss: penalize positive Φ (want Φ ≤ 0 in generator region)
                loss_phi = torch.nn.functional.relu(phi_output + 1.0).sum()

        # print(f"loss_phi = {loss_phi.item()}")
        # Combined loss (weight V more heavily initially)
        total_loss = loss_v + 1.0 * loss_phi  # Start with low GV weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            if phi_module is not None:
                print(f"  Pre-train Epoch [{epoch}/{num_epochs}]: V_loss = {loss_v.item():.6f}, Φ_loss = {loss_phi.item():.6f}, Total = {total_loss.item():.6f}")
            else:
                print(f"  Pre-train Epoch [{epoch}/{num_epochs}]: V_loss = {loss_v.item():.6f}")

    print("="*80)
    if phi_module is not None:
        print("Pre-training complete. V and GV (Φ) now have structure matching constraints.")
    else:
        print("Pre-training complete. V now has structure matching constraints.")
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
        if params.compute_V and not params.compute_GV: 
            total_loss, loss_dict = compute_total_loss_bounds(
                model=V_net,
                goal_region=regions.goal,
                V_goal_lower=bounds['goal'][0],
                V_goal_upper=bounds['goal'][1],
                V_unsafe_lower=bounds['unsafe'][0],
                V_unsafe_upper=bounds['unsafe'][1],
                V_init_lower=bounds['init'][0],
                V_init_upper=bounds['init'][1],
                V_outside_lower=bounds['outside'][0],
                V_outside_upper=bounds['outside'][1],
                beta_s=current_beta_s,
                beta_ra=params.constraints.beta_ra,
                device=device,
                compute_V=params.compute_V,
                compute_GV=params.compute_GV
            )
        elif not params.compute_V and params.compute_GV:
            total_loss, loss_dict = compute_total_loss_bounds(
            model=V_net,
            goal_region=regions.goal,
            Phi_lower=phi_lowers,
            Phi_upper=phi_uppers,
            beta_s=current_beta_s,
            beta_ra=params.constraints.beta_ra,
            generator_weight=current_gen_weight,
            device=device,
            compute_V=params.compute_V,
            compute_GV=params.compute_GV
            )
        else:
            total_loss, loss_dict = compute_total_loss_bounds(
            model=V_net,
            goal_region=regions.goal,
            V_goal_lower=bounds['goal'][0],
            V_goal_upper=bounds['goal'][1],
            V_unsafe_lower=bounds['unsafe'][0],
            V_unsafe_upper=bounds['unsafe'][1],
            V_init_lower=bounds['init'][0],
            V_init_upper=bounds['init'][1],
            V_outside_lower=bounds['outside'][0],
            V_outside_upper=bounds['outside'][1],
            Phi_lower=phi_lowers,
            Phi_upper=phi_uppers,
            beta_s=current_beta_s,
            beta_ra=params.constraints.beta_ra,
            generator_weight=current_gen_weight,
            device=device,
            compute_V=params.compute_V,
            compute_GV=params.compute_GV,
            epoch=epoch
            )

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
                    print(f"\n[Adaptive Refinement - Outside] Refining failing cells at epoch {epoch+1}")
                    print(f"  Before: {len(region_cells['outside'])} cells")
                    print(f"  Failing cells: {num_outside_failing}/{len(region_cells['outside'])}")

                    from discretization import discretize_region
                    from regions import Region
                    new_cells = []
                    num_refined = 0

                    for i, (cell_lower, cell_upper) in enumerate(region_cells['outside']):
                        if i < len(outside_failing_mask) and outside_failing_mask[i]:
                            cell_bounds = np.array([
                                [cell_lower[0].item(), cell_upper[0].item()],
                                [cell_lower[1].item(), cell_upper[1].item()]
                            ], dtype=np.float32)
                            cell_region = Region(cell_bounds)
                            refined = discretize_region(cell_region, REFINE_FACTOR)
                            new_cells.extend(refined)
                            num_refined += 1
                        else:
                            new_cells.append((cell_lower, cell_upper))

                    region_cells['outside'] = new_cells
                    print(f"  After: {len(region_cells['outside'])} cells")
                    print(f"  Refined {num_refined} failing cells into {num_refined * REFINE_FACTOR**2} subcells")
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
                MAX_CELLS = 1500  # Don't refine if we already have too many cells

                # Adjust interval for later epochs
                if epoch > 2500:
                    REFINE_INTERVAL = 100

                # Check if it's time to refine
                if ((epoch + 1) % REFINE_INTERVAL == 0 and
                    len(region_cells['generator']) < MAX_CELLS):
                    print(f"\n[Adaptive Refinement] Refining failing cells at epoch {epoch+1}")
                    print(f"  Before: {len(region_cells['generator'])} cells")
                    print(f"  Failing cells: {num_total_failing}/{len(region_cells['generator'])}")

                    # Refine failing cells
                    from discretization import discretize_region
                    from regions import Region
                    new_cells = []
                    num_refined = 0

                    for i, (cell_lower, cell_upper) in enumerate(region_cells['generator']):
                        if phi_upper_failing_mask[i]:
                            # Refine this failing cell into subcells
                            # Create Region object from cell bounds
                            cell_bounds = np.array([
                                [cell_lower[0].item(), cell_upper[0].item()],
                                [cell_lower[1].item(), cell_upper[1].item()]
                            ], dtype=np.float32)
                            cell_region = Region(cell_bounds)
                            refined = discretize_region(cell_region, REFINE_FACTOR)
                            new_cells.extend(refined)
                            num_refined += 1
                        else:
                            # Keep original cell
                            new_cells.append((cell_lower, cell_upper))

                    region_cells['generator'] = new_cells
                    print(f"  After: {len(region_cells['generator'])} cells")
                    print(f"  Refined {num_refined} failing cells into {num_refined * REFINE_FACTOR**2} subcells")
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

        # Early stopping check (every epoch) - check if all constraints are satisfied
        # Moved here to match testing_simple3.py's order (after logging, before detailed evaluation)
        # Use bounds_updated (computed AFTER optimizer step) to match testing_simple3.py!
        with torch.no_grad():
            if params.compute_V:
                if epoch % 10 == 0:
                    show = True
                else:
                    show = False
                _, goal_satisfied = compute_loss_goal_bounds(V_net, regions.goal, bounds_updated["goal"][0], bounds_updated["goal"][1], beta_s_check, device=device, show=show, check=True, n_samples=10000)
                unsafe_satisfied = (bounds_updated['unsafe'][0].min() >= params.constraints.beta_ra)
                init_satisfied = (bounds_updated['init'][0].min() >= beta_s_check and bounds_updated['init'][1].max() <= 1.0)
                outside_satisfied = (bounds_updated['outside'][0].min() >= beta_s_check)
            if params.compute_GV:
                generator_satisfied = (phi_uppers.max() <= 0.0)

            if params.compute_V and not params.compute_GV:
                if goal_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied:
                    print("\n" + "="*80)
                    print("🎉 ALL CONSTRAINTS SATISFIED - EARLY STOPPING!")
                    print("="*80)
                    print(f"Training converged at epoch {epoch}")
                    print(f"Final losses: Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}, Gen={loss_dict['generator']:.4f}")
                    break
            elif not params.compute_V and params.compute_GV:
                if generator_satisfied:
                    print("\n" + "="*80)
                    print("🎉 ALL CONSTRAINTS SATISFIED - EARLY STOPPING!")
                    print("="*80)
                    print(f"Training converged at epoch {epoch}")
                    print(f"Final losses: Gen={loss_dict['generator']:.4f}")
                    break
            elif params.compute_V and params.compute_GV:
                if goal_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied and generator_satisfied:
                    print("\n" + "="*80)
                    print("🎉 ALL CONSTRAINTS SATISFIED - EARLY STOPPING!")
                    print("="*80)
                    print(f"Training converged at epoch {epoch}")
                    print(f"Final losses: Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}, Gen={loss_dict['generator']:.4f}")
                    if control_net is not None:
                        print("control_net parameters:")
                        for name, param in control_net.named_parameters():
                            if param.requires_grad:
                                print(f"  {name} =\n{param.data}")
                    break

        # Detailed evaluation and visualization
        # Skip epoch 0 to avoid affecting random state and maintain exact reproducibility with testing_simple3.py
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


def main():
    """Main training function."""
    print("="*80)
    print("MODULAR RL VERIFICATION - BOUND-BASED TRAINING")
    print("="*80)

    # Random seed is set at module level (line 17) to match testing_simple3.py

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
    params.network.input_scale = [100.0, 100.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000  # Adjust as needed
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 1.0  # Enable generator constraint
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 7
    params.discretization.n_outside_goal = 5
    params.discretization.n_generator = 1  # Will be overridden by radial discretization
    params.discretization.n_unsafe = 3
    params.discretization.n_init = 2

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
    import shutil
    results_dir = Path("results")
    training_progress_dir = Path("training_progress")

    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"Cleaned up old results directory")

    if training_progress_dir.exists():
        shutil.rmtree(training_progress_dir)
        print(f"Cleaned up old training_progress directory")

    # Recreate directories
    results_dir.mkdir(exist_ok=True)
    training_progress_dir.mkdir(exist_ok=True)

    print(f"Network: {params.network.n_inputs} -> {params.network.n_hidden_1} -> "
          f"{params.network.n_hidden_2} -> {params.network.n_outputs}")
    print(f"Training: {params.training.num_epochs} epochs, LR={params.training.learning_rate}")
    if params.training.learnable_beta_s or params.constraints.beta_s is None:
        init_beta_s = params.constraints.beta_s if params.constraints.beta_s is not None else 0.6
        print(f"Constraints: beta_s=LEARNABLE (init={init_beta_s}), beta_ra={params.constraints.beta_ra}")
    else:
        print(f"Constraints: beta_s={params.constraints.beta_s}, beta_ra={params.constraints.beta_ra}")
    print(f"Generator: weight={params.training.generator_weight}, start_epoch={params.training.generator_start_epoch}")
    print(f"Training method: CROWN bounds (rigorous)")

    # ========================================================================
    # 2. SYSTEM DYNAMICS
    # ========================================================================
    print("\n" + "="*80)
    print("SYSTEM DYNAMICS")
    print("="*80)

    # Option 2: Neural network control (implements same K @ x)
    u_nn = LinearControlNN()

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        # Batch
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = -0.5 * x1 + 1.0 * x2
        f2 = -1.0 * x1 + -0.5 * x2
        return torch.stack([f1, f2], dim=1)
    
    # [control synthesis] F_CL needs to be an nn Module if u_control is being trained as well
    class ClosedLoopDrift(nn.Module):
        """
        the open-loop dynamics f_ol is specified in forward()
        """
        def __init__(self, controller: nn.Module):
            super().__init__()
            self.controller = controller   # <- registered as submodule

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = x[:, 0]
            x2 = x[:, 1]
            f1 = -0.5 * x1 + 1.0 * x2
            f2 = -1.0 * x1 + -0.5 * x2
            f_ol = torch.stack([f1, f2], dim=1)
            return f_ol + self.controller(x)

    # if we do [control synthesis], then
    #   u_fn and F_CL are necessary only for the current pre_training
    #   in other words, the controller is kept fixed during pre_training
    #   future work will remove this dependence such that we can use pre_training to train or not train controller.
    u_fn = u_control(u_nn)
    def F_CL(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """Closed-loop drift: f_cl(x) = f(x) + u(x)"""
        return f_ol(x) + u_fn(x) 

    G_matrix = np.array([
        [0.2, 0.0],
        [0.0, 0.2]
    ], dtype=np.float32)

    # Create custom callable diffusion function: g(x) = diag(sigma * x)
    sigma_diag = np.diag(G_matrix)  # [0.2, 0.2]
    sigma_torch = torch.from_numpy(sigma_diag).float()

    def g(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.2, 0.2], device=x.device, dtype=x.dtype) * x

    # dynamics = Dynamics.dynamics(f=F_CL, g=g)
    # new f_cl wrapper for [control synthesis]
    f_cl_module = ClosedLoopDrift(u_nn).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    print(f"\n{dynamics}")

    # ========================================================================
    # 3. SPATIAL REGIONS
    # ========================================================================
    print("\n" + "="*80)
    print("SPATIAL REGIONS")
    print("="*80)

    init_range = np.array([[45.0, 55.0], [-55.0, -45.0]], dtype=np.float32)
    goal_range = np.array([[-25.0, 25.0], [-25.0, 25.0]], dtype=np.float32)
    unsafe_range = np.array([[-100.0, -80.0], [-100.0, 100.0]], dtype=np.float32)
    full_range = np.array([[-100.0, 100.0], [-100.0, 100.0]], dtype=np.float32)

    regions = Regions.from_numpy_ranges(
        init_range=init_range,
        goal_range=goal_range,
        unsafe_range=unsafe_range,
        full_range=full_range
    )

    # ========================================================================
    # 4. CREATE NETWORKS
    # ========================================================================
    print("\n" + "="*80)
    print("NETWORKS")
    print("="*80)

    V_net = create_V(params.network)
    print(f"V network: {V_net}")

    # Debug: Print initial weight statistics
    print(f"\nDEBUG - Initial V_net weights:")
    for name, param in V_net.named_parameters():
        print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")

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
    PRETRAIN_EPOCHS = 2000
    PRETRAIN_LR = 0.01

    if ENABLE_PRETRAINING:
        # Check weights BEFORE pretraining
        print(f"\nDEBUG - V_net weights BEFORE pretraining:")
        for name, param in V_net.named_parameters():
            print(f"  {name}: mean={param.data.mean().item():.6f}")

        pretrain_structure_aware(
            model=V_net,
            x_goal_range=goal_range,
            x_unsafe_range=unsafe_range,
            x_init_range=init_range,
            x_range=full_range,
            A=F_CL,
            R=G_matrix,
            scale_factor=params.network.scale_factor,
            num_epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            device=params.training.device
        )

        # Check weights AFTER pretraining
        print(f"\nDEBUG - V_net weights AFTER pretraining:")
        for name, param in V_net.named_parameters():
            print(f"  {name}: mean={param.data.mean().item():.6f}")
        print(f"Pretraining completed - weights have been updated!\n")

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
