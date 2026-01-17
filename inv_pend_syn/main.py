"""
2D Inverted Pendulum Control Synthesis
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
from src.control_network import WrapperConterlNN, InvertControlNN
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
    device='cpu',
    control_net=None,
    n_each: int = 400,   # samples per region per epoch
):
    """
    Pre-train V network using sampled points to match constraint structure.

    Supports x_unsafe_range as:
      1) (D,2) single box
      2) (K,D,2) union of K boxes
      3) (2*D,2) produced by np.vstack((box1, box2, ...))  <-- your case

    Losses:
      - full-range: enforce v(x) >= 0
      - init-range: enforce v(x) <= 1
      - unsafe-range: enforce v(x) >= pretrain_unsafe_target
      - goal-range: enforce v(x) <= pretrain_goal_target
      - others (outside goal ∪ unsafe): enforce v(x) >= pretrain_goal_target
      - optional phi loss on x_others: enforce phi(x) <= 0
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
    init_t    = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
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

    # -----------------------------
    # Unsafe region: allow union of boxes
    # -----------------------------
    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
        """
        Returns unsafe_boxes as torch.Tensor of shape (K,D,2).
        Accepts:
          - (D,2)
          - (K,D,2)
          - (K*D,2) from np.vstack((box1, box2, ...))
        """
        t = torch.as_tensor(x_unsafe, dtype=torch.float32, device=device)

        if t.dim() == 2:
            if t.shape == (D, 2):
                return t.unsqueeze(0)  # (1,D,2)
            if t.shape[1] == 2 and (t.shape[0] % D == 0):
                K = int(t.shape[0] // D)
                return t.view(K, D, 2)  # (K,D,2)  <-- handles vstack case
            raise ValueError(f"x_unsafe_range 2D must be (D,2) or (K*D,2); got {tuple(t.shape)}")

        if t.dim() == 3:
            if t.shape[1:] != (D, 2):
                raise ValueError(f"x_unsafe_range 3D must be (K,D,2) with D={D}; got {tuple(t.shape)}")
            return t

        raise ValueError(f"x_unsafe_range must be (D,2), (K,D,2), or (K*D,2); got {tuple(t.shape)}")

    unsafe_boxes = _unsafe_to_boxes(x_unsafe_range)  # (K,D,2)
    K_unsafe = int(unsafe_boxes.shape[0])

    def _in_unsafe_union(x_batch: torch.Tensor) -> torch.Tensor:
        """mask True if x is inside ANY unsafe box."""
        mask = torch.zeros(x_batch.shape[0], dtype=torch.bool, device=device)
        for k in range(K_unsafe):
            mask |= _in_box(x_batch, unsafe_boxes[k])
        return mask

    def _sample_in_unsafe_union(N: int) -> torch.Tensor:
        """
        Sample N points from union of K unsafe boxes using torch only.
        Picks a box index ~ Categorical(probs) where probs ∝ volume, then samples uniformly in that box.
        """
        if K_unsafe == 1:
            return _sample_in_box(unsafe_boxes[0], N)

        # probs ∝ volume
        widths = (unsafe_boxes[:, :, 1] - unsafe_boxes[:, :, 0]).clamp_min(0.0)  # (K,D)
        vols = widths.prod(dim=1)                                                # (K,)
        probs = vols / vols.sum().clamp_min(1e-12)                               # safe normalize

        # choose box for each sample (N,)
        choices = torch.multinomial(probs, num_samples=N, replacement=True)      # (N,)

        # gather bounds for chosen boxes: (N,D)
        b_low  = unsafe_boxes[choices, :, 0]
        b_high = unsafe_boxes[choices, :, 1]

        # uniform in chosen box
        u = torch.rand(N, D, device=device)
        return u * (b_high - b_low) + b_low

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # 1) full-range samples -> enforce v(x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # 2) init-range samples -> enforce v(x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # 3) unsafe-range samples -> enforce v(x) >= pretrain_unsafe_target
        x_unsafe = _sample_in_unsafe_union(n_each)
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.pretrain_unsafe_target - v_unsafe).sum()

        # 4) goal-range samples -> enforce v(x) >= 0.0
        x_goal = _sample_in_box(goal_t, n_each)
        v_goal = model(x_goal).squeeze(-1)
        v_loss_inside_goal = F.relu(0.0 - v_goal).sum()

        # 5) samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= pretrain_goal_target
        x_others_list = []
        need = n_each
        max_tries = 20
        tries = 0
        while need > 0 and tries < max_tries:
            tries += 1
            x_cand = torch.rand(max(need * 4, 32), D, device=device) * span + low
            cand_in_goal = _in_box(x_cand, goal_t)
            cand_in_unsafe = _in_unsafe_union(x_cand)
            keep = ~(cand_in_goal | cand_in_unsafe)
            x_keep = x_cand[keep]
            if x_keep.shape[0] > 0:
                take = min(need, x_keep.shape[0])
                x_others_list.append(x_keep[:take])
                need -= take

        if len(x_others_list) == 0:
            x_others = x_full.detach()
        else:
            x_others = torch.cat(x_others_list, dim=0)

        v_others = model(x_others).squeeze(-1)
        v_loss_others = F.relu(params.constraints.pretrain_goal_target - v_others).sum()

        # Total V loss
        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
            # + v_loss_inside_goal
            # + v_loss_others
        )

        # Phi loss on SAME x_others -> enforce phi(x) <= 0
        loss_phi = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_phi = x_others.detach().clone().requires_grad_(True)
            phi_output = GV_net(x_phi).squeeze(-1)
            loss_phi = F.relu(phi_output).sum()

        total_loss = loss_v + loss_phi

        # Track best
        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

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
    networks_trained = ["V"]
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
    params.network.n_inputs = 2
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 16
    params.network.input_scale = [2*np.pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 10
    params.discretization.n_outside_goal = 10
    params.discretization.n_generator = 1
    params.discretization.n_unsafe = 10
    params.discretization.n_init = 10

    params.constraints.beta_s = 0.0
    params.constraints.beta_ra = 20.0

    # Control what to compute during training
    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 20000
    params.training.pretrain_lr = 0.001
    params.training.pretrain_n_samples = 1000

    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_interval = 500
    params.refinement.v_outside.late_epoch_threshold = 2500
    params.refinement.v_outside.refine_interval_late = 100
    params.refinement.v_outside.refine_factor = 2
    params.refinement.v_outside.max_cells = 10000
    params.refinement.v_outside.N_to_refine = 100

    params.refinement.v_outside.enable_merging = False  # Add this feature
    params.refinement.v_outside.merge_interval = 500  # Suggested default
    params.refinement.v_outside.merge_max_passes = 8  # Suggested default
    params.refinement.v_outside.merge_relax_margin = 0.1  # Suggested default

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_interval = 150
    params.refinement.gv_generator.late_epoch_threshold = 2500
    params.refinement.gv_generator.refine_interval_late = 250
    params.refinement.gv_generator.refine_factor = 3
    params.refinement.gv_generator.max_cells = 10000
    params.refinement.gv_generator.N_to_refine = 100

    params.refinement.gv_generator.enable_merging = False  # Add this feature
    params.refinement.gv_generator.merge_interval = 500  # Suggested default
    params.refinement.gv_generator.merge_max_passes = 8  # Suggested default
    params.refinement.gv_generator.merge_relax_margin = -100.0  # Suggested default



    # ========================================================================
    # 2. SYSTEM DYNAMICS
    # ========================================================================
    print("\n" + "="*80)
    print("SYSTEM DYNAMICS")
    print("="*80)

    # pre-trained in RL deterministic env
    # rl_policy_net = TanhPolicy(2, 1, 64, device=device)
    # rl_policy_net.load_state_dict(torch.load(
    #     OUTPUT_DIR / "pendulum_policy.pt",
    #     map_location=device,
    #     weights_only=True
    # ))
    # rl_policy_net.requires_grad_(False)
    # create a u_nn wrapper to wrap around the rl_policy_net and retrun [rl_policy_net(x), 0] vector
    # u_nn = WrapperConterlNN(rl_policy_net)

    rl_policy_net = InvertControlNN()
    u_nn = WrapperConterlNN(rl_policy_net) # wrapper around InvertControlNN to create u_nn for ClosedLoopDrift

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        g = 9.81
        L = 0.5
        b = 0.1
        m = 0.15
        # Batch
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x2
        f2 = (g/L)*torch.sin(x1) - (b/(m*L**2))*x2
        return torch.stack([f1, f2], dim=1)


    # Create diffusion coefficients as a persistent tensor to avoid TracerWarnings
    g_coeffs = torch.tensor([0.0, 2.0], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term g(x) that is constant: [0.0, 0.2] for every x.

        If x has shape (2,), returns (2,).
        If x has shape (N, 2), returns (N, 2) with each row [0.0, 0.2].
        """
        base = g_coeffs.to(device=x.device, dtype=x.dtype)  # shape (2,)

        if x.dim() == 1:
            # x is shape (2,)
            return base
        elif x.dim() == 2:
            # x is shape (N, 2)
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)  # (N, 2)
        else:
            raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")


    # Create closed-loop drift for control synthesis
    f_cl_module = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)

    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    # ========================================================================
    # 3. SPATIAL REGIONS
    # ========================================================================
    print("\n" + "="*80)
    print("SPATIAL REGIONS")
    print("="*80)

    init_range = np.array([[(3/4)*np.pi, (5/4)*np.pi], [-1.0, 1.0]], dtype=np.float32)
    goal_range = np.array([[-0.5*np.pi, 0.5*np.pi], [-4.0, 4.0]], dtype=np.float32)

    # Create unsafe region as union of two rectangles (matching paper exactly)
    unsafe_down1 = np.array([[-2*np.pi, -(3/2)*np.pi], [-20.0, -10.0]], dtype=np.float32)
    unsafe_down2 = np.array([[(3/2)*np.pi, 2*np.pi], [10.0, 20.0]], dtype=np.float32)
    unsafe_range = np.vstack((unsafe_down1, unsafe_down2))
    full_range = np.array([[-2*np.pi, 2*np.pi], [-20.0, 20.0]], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe_down1 = Region(unsafe_down1)
    unsafe_down2 = Region(unsafe_down2)
    unsafe = Region.union(unsafe_down1, unsafe_down2)
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
        use_radial_generator=False
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
        PRETRAIN_EPOCHS = 2000
        PRETRAIN_LR = 0.001

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
                n_each=100,
            )

            print(f"Pretraining completed!\n")

        # ========================================================================
        # 6.2 TRAIN WITH BOUNDS
        # ========================================================================
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
        rl_policy_net = InvertControlNN(input_dim=2, hidden_dim=8, output_dim=1)
        u_nn =  WrapperConterlNN(rl_policy_net).to(device)
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