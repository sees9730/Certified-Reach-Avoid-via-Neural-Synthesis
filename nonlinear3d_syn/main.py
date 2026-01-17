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
from src.control_network import LinearControlNN, LorentzLinearControlNN
from src.phi_module import create_GV
from src.discretization import discretize_regions
from src.training_utils import (
    evaluate_constraints,
    print_constraint_summary,
)
from src.trainer import train_network_bounds
from src.save_load_utils import save_eval_bundle, load_eval_bundle, log_loaded_training_epochs, enable_terminal_logging
from src.utils import cleanup_and_setup_directories, print_training_config
from src.visualization import (
    # visualize_training_progress,
    create_summary_plots
)

# Set random seed immediately after imports (matching testing_simple3.py)
torch.manual_seed(0)


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
):
    """
    Pre-train V network using sampled points to match constraint structure.

    v_loss = v_loss_full + v_loss_init + v_loss_unsafe + v_loss_inside_goal + v_loss_others
    and optionally:
      loss_phi = relu(phi_output).sum() on the SAME "others" samples (x in X \ (goal ∪ unsafe)).
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

        # ---------------------------------------------------------------------
        # 1) full-range samples -> enforce v(x) >= 0
        # ---------------------------------------------------------------------
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # ---------------------------------------------------------------------
        # 2) init-range samples -> enforce v(x) <= 1
        # ---------------------------------------------------------------------
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # ---------------------------------------------------------------------
        # 3) unsafe-range samples -> enforce v(x) >= pretrain_unsafe_target
        # ---------------------------------------------------------------------
        x_unsafe = _sample_in_box(unsafe_t, n_each)
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.pretrain_unsafe_target - v_unsafe).sum()

        # ---------------------------------------------------------------------
        # 5) samples inside full-range but outside (goal ∪ unsafe)
        #    -> enforce v(x) >= pretrain_goal_target
        # ---------------------------------------------------------------------
        # Rejection sampling (simple + robust). Oversample and filter.
        # If goal/unsafe are large, loop a few times to collect enough.
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
            # Fallback: if we cannot find any "others", just use random full samples.
            # (Keeps code from crashing; also signals geometry is degenerate.)
            x_others = x_full.detach()
        else:
            x_others = torch.cat(x_others_list, dim=0)

        v_others = model(x_others).squeeze(-1)
        v_loss_others = F.relu(params.constraints.pretrain_goal_target - v_others).sum()

        # ---------------------------------------------------------------------
        # 4) goal-range samples -> enforce v(x) <= pretrain_goal_target
        # ---------------------------------------------------------------------
        x_goal = _sample_in_box(goal_t, n_each)
        v_goal = model(x_goal).squeeze(-1)
        v_loss_inside_goal = F.relu(params.constraints.pretrain_goal_target - v_goal).sum()

        # ---------------------------------------------------------------------
        # Total V loss
        # ---------------------------------------------------------------------
        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
            + v_loss_inside_goal
            + v_loss_others
        )

        # ---------------------------------------------------------------------
        # Phi loss on SAME x_others -> enforce phi(x) <= 0
        # ---------------------------------------------------------------------
        loss_phi = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            # Make leaf for autograd inside GV (∇V / Hessian)
            x_phi = x_others.detach().clone().requires_grad_(True)
            phi_output = GV_net(x_phi).squeeze(-1)
            loss_phi = F.relu(phi_output).sum()

        total_loss = loss_v + loss_phi

        # Track best
        if total_loss.item() <= best_loss:
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

    print("="*80)
    networks_trained = []
    networks_trained.append("V")
    if GV_net is not None:
        networks_trained.append("GV (Φ)")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")
    print("="*80 + "\n")

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
    params.training.generator_weight = 1.0  # Enable generator constraint
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 10
    params.discretization.n_outside_goal = 4
    params.discretization.n_generator = 1  # Will be overridden by radial discretization
    params.discretization.n_unsafe = 10
    params.discretization.n_init = 10

    # Set beta_s to a value (constant), or set to None to make it learnable
    # If learnable_beta_s is True, this value will be used as initialization
    params.constraints.beta_s = 0.00
    params.constraints.beta_ra = 10.0

    # Control what to compute during training
    params.compute_V = True
    params.compute_GV = True

    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_interval = 250
    params.refinement.v_outside.late_epoch_threshold = 2500
    params.refinement.v_outside.refine_interval_late = 50
    params.refinement.v_outside.refine_factor = 2
    params.refinement.v_outside.max_cells = 30000
    params.refinement.v_outside.N_to_refine = 100

    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 501
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 0.3

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_interval = 500
    params.refinement.gv_generator.late_epoch_threshold = 2500
    params.refinement.gv_generator.refine_interval_late = 200
    params.refinement.gv_generator.refine_factor = 2
    params.refinement.gv_generator.max_cells = 60000
    params.refinement.gv_generator.N_to_refine = 100

    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -200.0

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
    [-1.2, 1.2],
    [-1.2, 1.2],
    [-1.2, 1.2],
    ], dtype=np.float32)

    goal_range = np.array([
        [-0.3, 0.3],
        [-0.3, 0.3],
        [-0.3, 0.3],
    ], dtype=np.float32)

    unsafe_range = np.array([
        [-6.0, -5.0],
        [-6.0, -5.0],
        [-6.0, -5.0],
    ], dtype=np.float32)

    full_range = np.array([
        [-6.0, 6.0],
        [-6.0, 6.0],
        [-6.0, 6.0],
    ], dtype=np.float32)

    # regions = Regions.from_numpy_ranges(
    #     init_range=init_range,
    #     goal_range=goal_range,
    #     unsafe_range=unsafe_range,
    #     full_range=full_range
    # )

    init = Region(init_range)
    goal = Region(goal_range)
    # unsafe_down1 = Region(unsafe_down1)
    # unsafe_down2 = Region(unsafe_down2)
    # unsafe_up = Region(unsafe_up)
    # unsafe = Region.union(unsafe_down1, unsafe_down2, unsafe_up)
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
        use_radial_generator=True
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
        print_training_config(params)
        print(f"\n{dynamics}")
        print(f"V network: {V_net}")
        print(f"\nUsing device: {device}")

        ENABLE_PRETRAINING = True  # Set to True to enable
        PRETRAIN_EPOCHS = 3000
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
                n_each=1000
            )

            print(f"Pretraining completed!\n")

        # ========================================================================
        # 6. TRAIN WITH BOUNDS
        # ========================================================================
        device = params.training.device
        print(f"\nUsing device: {device}")

        def create_scheduler(optimizer):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=300,
                verbose=True,
                min_lr=1e-6,
                threshold=1e-3
            )

        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            create_scheduler=create_scheduler,
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
            device=device
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
            training_config=params.training
        ).to(device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"])

        # Control net (only needed if your plots depend on it)
        u_nn = LinearControlNN(input_dim=3).to(device)
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

        # You can reuse saved results, or recompute
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
