
"""
2D Geometric Brownian Motion Control Synthesis, 
Goal is centered at origin (the equilibrium points)
"""
import torch
import torch.nn.functional as F
import numpy as np
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
from src.control_network import LinearControlNN # [control synthesis]
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
    create_summary_plots
)

# Set random seed
torch.manual_seed(0)

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
    device="cpu",
    control_net=None,
    n_each: int = 400,  # samples per region per epoch
):
    # Print header so it is obvious we are in pretraining
    print("\n" + "=" * 20)
    print("PRE-TRAINING: Constraint-Structured Initialization (Sample-Based)")
    print("=" * 20)

    # Build optimizer over V network and controller if it exists
    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    # Log whether the GV network is being trained
    if GV_net is not None:
        print("GV pre-training ENABLED")
    else:
        print("GV pre-training DISABLED")

    best_loss = float("inf")
    best_model_state = None
    best_control_state = None

    # Convert all state space ranges to torch tensors
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    unsafe_t = torch.as_tensor(x_unsafe_range, dtype=torch.float32, device=device)
    init_t = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    # Extract dimension and bounds of the full state space
    D = x_range_t.shape[0]
    low = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low

    # Sample uniformly from a box
    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        b_low = box[:, 0]
        b_high = box[:, 1]
        return torch.rand(N, D, device=device) * (b_high - b_low) + b_low

    # Check whether points are inside a box
    def _in_box(x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return ((x >= box[:, 0]) & (x <= box[:, 1])).all(dim=1)

    for epoch in range(num_epochs):
        # Put all networks in training mode
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # Sample from the full state space and enforce v(x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # Sample from the initial set and enforce v(x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # Sample from the unsafe set and enforce large values
        x_unsafe = _sample_in_box(unsafe_t, n_each)
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(
            params.constraints.pretrain_unsafe_target - v_unsafe
        ).sum()

        # Sample points that are not in the goal or unsafe sets
        x_others_list = []
        need = n_each
        max_tries = 20
        tries = 0

        while need > 0 and tries < max_tries:
            tries += 1
            x_cand = torch.rand(max(need * 4, 32), D, device=device) * span + low
            in_goal = _in_box(x_cand, goal_t)
            in_unsafe = _in_box(x_cand, unsafe_t)
            keep = ~(in_goal | in_unsafe)
            x_keep = x_cand[keep]
            if x_keep.numel() > 0:
                take = min(need, x_keep.shape[0])
                x_others_list.append(x_keep[:take])
                need -= take

        # Fall back to full-space samples if rejection sampling fails
        if len(x_others_list) == 0:
            x_others = x_full.detach()
        else:
            x_others = torch.cat(x_others_list, dim=0)

        # Combine the value losses exactly as in the original logic
        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
        )

        # Compute the GV loss on the same "other" states
        loss_gv = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_gv = x_others.detach().clone().requires_grad_(True)
            gv_output = GV_net(x_gv).squeeze(-1)
            loss_gv = F.relu(gv_output).sum()

        total_loss = loss_v + loss_gv

        # Save the best model seen so far
        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            if control_net is not None:
                best_control_state = {
                    k: v.cpu().clone()
                    for k, v in control_net.state_dict().items()
                }

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(
                f"  Epoch [{epoch}/{num_epochs}]: "
                f"V={loss_v.item():.6f}, "
                f"GV={loss_gv.item():.6f}, "
                f"Total={total_loss.item():.6f}"
            )

        # Backprop and update parameters
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Restore the best model weights after training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\n  Best loss: {best_loss:.6f}")

    print("=" * 20)
    networks_trained = ["V"]
    if GV_net is not None:
        networks_trained.append("GV")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")
    print("=" * 20 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    args = parser.parse_args()

    """Main training function."""
    print("="*20)
    print("MODULAR RL VERIFICATION - BOUND-BASED TRAINING")
    print("="*20)

    # === Hyperparameters ===
    print("\n" + "="*20)
    print("HYPERPARAMETERS")
    print("="*20)

    params = Hyperparameters.default()

    # Network architecture
    params.network.n_inputs = 2
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [100.0, 100.0]
    params.network.scale_factor = 20.0

    # Training
    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0

    # Initial region discretization
    params.discretization.n_goal = 7
    params.discretization.n_outside_goal = 4
    params.discretization.n_generator = 1
    params.discretization.n_unsafe = 7
    params.discretization.n_init = 7

    # Refinement
    params.refinement.v_outside.enable_refinement = params.refinement.gv_generator.enable_refinement = True
    params.refinement.v_outside.refine_factor = params.refinement.gv_generator.refine_factor = 2
    params.refinement.v_outside.refine_interval = params.refinement.gv_generator.refine_interval = 100
    params.refinement.v_outside.late_epoch_threshold = params.refinement.gv_generator.late_epoch_threshold = 2500
    params.refinement.v_outside.refine_interval_late = params.refinement.gv_generator.refine_interval_late = 100
    params.refinement.v_outside.max_cells = params.refinement.gv_generator.max_cells = 30000

    # Merging
    params.refinement.v_outside.enable_merging = params.refinement.gv_generator.enable_merging = True
    params.refinement.v_outside.merge_max_passes = params.refinement.gv_generator.merge_max_passes = 8 

    # Constraints
    params.constraints.beta_s = 0.0
    params.constraints.beta_ra = 20.0

    # Control what to compute during training
    params.compute_V = True
    params.compute_GV = True

    # Pretrain 
    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 1000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 500

    # === System dynamics ===
    print("\n" + "="*20)
    print("SYSTEM DYNAMICS")
    print("="*20)

    # Controller network for closed-loop dynamics
    u_nn = LinearControlNN(prior_knowledge=False)

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        # Batch
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = -0.5 * x1 + 1.0 * x2
        f2 = -1.0 * x1 + -0.5 * x2
        return torch.stack([f1, f2], dim=1)

    # Create diffusion coefficients as a persistent tensor to avoid TracerWarnings
    g_coeffs = torch.tensor([0.2, 0.2], dtype=torch.float32)
    def g(x: torch.Tensor) -> torch.Tensor:
        return g_coeffs.to(device=x.device, dtype=x.dtype) * x

    # Create closed-loop drift for control synthesis
    f_cl_module = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    # === Spatial regions ===
    print("\n" + "="*20)
    print("SPATIAL REGIONS")
    print("="*20)

    init_range = np.array([[45.0, 55.0], [-55.0, -45.0]], dtype=np.float32)
    goal_range = np.array([[-25.0, 25.0], [-25.0, 25.0]], dtype=np.float32)
    unsafe_range = np.array([[-100.0, -80.0], [-100.0, 100.0]], dtype=np.float32)
    full_range = np.array([[-100.0, 100.0], [-100.0, 100.0]], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe = Region(unsafe_range)
    full = Region(full_range)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # === Networks ===
    print("\n" + "="*20)
    print("NETWORKS")
    print("="*20)

    V_net = create_V(params.network)
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        training_config=params.training
    )

    # === Discretize regions ===
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=False
    )

    # === Training / Evaluation ===
    device = params.training.device
    params.constraints.pretrain_goal_target = params.constraints.beta_s
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = 1.0
    params.constraints.pretrain_phi_target = 0.0
    # Path to bundle we will save/load
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        # === Setup training ===
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")
        print_training_config(params)
        print(f"\n{dynamics}")
        print(f"V network: {V_net}")
        print(f"\nUsing device: {device}")
    
        if params.training.enable_pretraining:
            pretrain_network_samples(
                model=V_net,
                x_goal_range=goal_range,
                x_unsafe_range=unsafe_range,
                x_init_range=init_range,
                x_range=full_range,
                params=params,
                GV_net=GV_net,
                num_epochs=params.training.pretrain_epochs,
                lr=params.training.pretrain_lr,
                device=params.training.device,
                n_each=params.training.pretrain_n_samples,
                control_net=u_nn
            )

            print(f"Pretraining completed!\n")

        # ========================================================================
        # 6. TRAIN WITH BOUNDS
        # ========================================================================
        device = params.training.device
        print(f"\nUsing device: {device}")

        # Create scheduler factory (each main.py can customize this)
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
            control_net=u_nn,
            create_scheduler=create_scheduler
        )

        # ========================================================================
        # 7. FINAL EVALUATION
        # ========================================================================
        print("\n" + "="*20)
        print("FINAL EVALUATION")
        print("="*20)

        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_ra=params.constraints.beta_ra,
            device=device
        )
        print_constraint_summary(results)

        # ========================================================================
        # 8. FINAL VISUALIZATIONS
        # ========================================================================
        print("\n" + "="*20)
        print("CREATING FINAL VISUALIZATIONS")
        print("="*20)

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
        print("\n" + "="*20)
        print("SAVING EVAL BUNDLE")
        print("="*20)

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
        print("\n" + "="*20)
        print("LOADING SAVED BUNDLE (skip training)")
        print("="*20)

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
        u_nn = LinearControlNN(prior_knowledge=True).to(device)
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

        print("\n" + "="*20)
        print("FINAL EVALUATION (LOADED)")
        print("="*20)
        print_constraint_summary(results)

        print("\n" + "="*20)
        print("CREATING FINAL VISUALIZATIONS (LOADED)")
        print("="*20)
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
