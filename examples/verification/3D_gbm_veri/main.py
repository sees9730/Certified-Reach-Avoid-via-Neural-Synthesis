"""
3D Geometric Brownian Motion Verification
"""
import argparse
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Set up directories
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import LinearControlNN
from src.discretization import discretize_regions
from src.dynamics import ClosedLoopDrift, Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.regions import Region, Regions
from src.save_load_utils import (enable_terminal_logging, load_eval_bundle,
                                  log_loaded_training_epochs, save_eval_bundle)
from src.trainer import train_network_bounds
from src.training_utils import evaluate_constraints, print_constraint_summary
from src.utils import cleanup_and_setup_directories
from src.visualization import create_summary_plots

# Set random seed
torch.manual_seed(4)

def pretrain_network_samples(
    model,
    x_goal_range,
    x_unsafe_range,
    x_init_range,
    x_range,
    params,
    GV_net=None,
    num_epochs=1000,
    lr=1e-4,
    device='cpu',
    control_net=None,
    n_each: int = 400,
    lambda_w = 0.1,
    save_v_path=None,
    save_control_path=None,
):
    """
    Pre-train V and GV networks using sampled points.
    """
    print("\n" + "="*20)
    print("Pre-training using samples")
    print("="*20)

    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    if GV_net is not None:
        print("GV pre-training enabled")
    else:
        print("GV pre-training disabled")

    best_loss = float('inf')
    best_model_state = None
    best_control_state = None

    # Convert ranges to torch once: each is (D,2)
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    init_t = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
    low  = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low

    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        b_low = box[:, 0]
        b_high = box[:, 1]
        return torch.rand(N, D, device=device) * (b_high - b_low) + b_low

    def _in_box(x_batch: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return ((x_batch >= box[:, 0]) & (x_batch <= box[:, 1])).all(dim=1)

    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
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
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")

        if K_unsafe == 1:
            return _sample_in_box(unsafe_boxes[0], N)

        xs = []
        for k in range(K_unsafe):
            xs.append(_sample_in_box(unsafe_boxes[k], N))  # (N, D) per box

        x = torch.cat(xs, dim=0)  # (K_unsafe * N, D)
        x = x[torch.randperm(x.shape[0], device=device)]  # shuffle
        return x
    
    def _l2_weight_penalty(model: torch.nn.Module, exclude_bias: bool = True) -> torch.Tensor:
        reg = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_bias and (p.dim() == 1 or name.endswith("bias")):
                continue
            reg = reg + (p ** 2).sum()
        return reg

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # full-range samples -> enforce v(x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # init-range samples -> enforce v(x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # unsafe-range samples -> enforce v(x) >= beta_ra
        x_unsafe = _sample_in_unsafe_union(int(n_each /6))
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.beta_ra - v_unsafe).sum()

        # samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= 0.0
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

        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
        )

        loss_gv = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_gv = x_others.detach().clone().requires_grad_(True)
            gv_output = GV_net(x_gv).squeeze(-1)
            loss_gv = F.relu(gv_output).sum()

        total_loss = loss_v + loss_gv

        # L2 weight penalty
        reg_w = _l2_weight_penalty(model, exclude_bias=True)
        total_loss = total_loss + lambda_w * reg_w

        # Track best
        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        if epoch % 100 == 0:
            if GV_net is not None:
                print(f"Epoch {epoch} | V_loss={loss_v.item():8.4f} | GV_loss={loss_gv.item():8.4f}")
            else:
                print(f"Epoch {epoch} | V_loss={loss_v.item():8.4f}")

        # Optimize/update networks
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\nBest loss: {best_loss:.6f}")

        # Save best pretrained weights (state_dict)
        if save_v_path is not None:
            torch.save(best_model_state, save_v_path)
            print(f"Saved pretrained V_net to: {save_v_path}")
        if (control_net is not None) and (best_control_state is not None) and (save_control_path is not None):
            torch.save(best_control_state, save_control_path)
            print(f"Saved pretrained Controller_net to: {save_control_path}")

    networks_trained = ["V"]
    if GV_net is not None:
        networks_trained.append("GV")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")


def main(benchmark_mode=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    parser.add_argument("--benchmark", type=int, default=0, choices=[0, 1],
                        help="1: run benchmark (5 runs), 0: normal run")
    args = parser.parse_args()

    if benchmark_mode:
        args.benchmark = 1

    print("="*20)
    print("3D Geometric Brownian Motion Verification")
    print("="*20)

    # Return value for benchmark mode
    training_time_result = None

    # === Hyperparameters ===
    params = Hyperparameters.default()

    # Customize configuration
    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [100.0, 100.0, 100.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 200000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 14
    params.discretization.n_outside_goal = 4
    params.discretization.n_generator = 1
    params.discretization.n_unsafe = 5
    params.discretization.n_init = 4

    params.constraints.beta_ra = 20.0

    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 3000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 400

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
    params.refinement.gv_generator.refine_interval_late = 100
    params.refinement.gv_generator.refine_factor = 2
    params.refinement.gv_generator.max_cells = 30000
    params.refinement.gv_generator.N_to_refine = 100

    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -100.0

    # === System Dynamics ===
    u_nn = LinearControlNN(prior_knowledge=True, 
                           input_dim=params.network.n_inputs)

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        f1 = -1.5 * x1 + 1.0 * x2 + 0.0 * x3
        f2 = -1.0 * x1 - 1.5 * x2 + 1.0 * x3
        f3 =  0.0 * x1 - 1.0 * x2 - 1.5 * x3
        return torch.stack([f1, f2, f3], dim=1)

    g_coeffs = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        return g_coeffs.to(device=x.device, dtype=x.dtype) * x

    f_cl_module = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    # === Regions ===
    init_range = np.array([[45.0, 55.0], 
                           [-55.0, -45.0],
                           [50.0, 60.0]
                           ], dtype=np.float32)
    goal_range = np.array([[-25.0, 25.0], [-25.0, 25.0], [-25.0, 25.0]
                           ], dtype=np.float32)
    unsafe_range = np.array([[-100.0, -80.0], [-100.0, 100.0], [-100.0, -80.0]
                             ], dtype=np.float32)
    full_range = np.array([[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]
                           ], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe = Region(unsafe_range)
    full = Region(full_range)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # === Networks ===
    V_net = create_V(params.network)
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network
    )

    # === Discretize Regions ===
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=False
    )

    # === Setup ===
    device = params.training.device
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        # === Pre-training ===
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")

        if params.training.enable_pretraining:
            # Start timer before pretraining
            training_start_time = time.time()

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
                n_each=params.training.pretrain_n_samples,
                device=params.training.device,
                save_v_path=OUTPUT_DIR / "V_pretrained.pth",
            )

        # === Training ===
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
            create_scheduler=create_scheduler,
            start_time=training_start_time
        )

        # Record training end time
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        # === Final Evaluation ===
        print("\n" + "="*20)
        print("Final Evaluation")
        print("="*20)

        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_ra=params.constraints.beta_ra,
            device=device
        )
        print_constraint_summary(results)

        # === Visualizations ===
        print("\n" + "="*20)
        print("Creating Visualizations")
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

        # === Save Bundle ===
        print("\n" + "="*20)
        print("Saving Evaluation Bundle")
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

        # Set training time result for benchmark
        training_time_result = total_training_time

    else:
        # === Load Bundle ===
        print("\n" + "="*20)
        print("Loading Saved Bundle")
        print("="*20)

        bundle = load_eval_bundle(bundle_path, map_location="cpu")

        # Rebuild params/regions from dicts
        params = Hyperparameters.from_dict(bundle["hyperparameters"])

        # Rebuild discretization cells
        region_cells = bundle["region_cells"]

        # Rebuild networks and load weights
        V_net = create_V(params.network).to(device)
        V_net.load_state_dict(bundle["V_state_dict"])

        # Recreate dynamics + GV_net
        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network,
            training_config=params.training
        ).to(device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"])

        # Control net (only needed if your plots depend on it)
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
                beta_ra=params.constraints.beta_ra,
                device=device
            )

        print("\n" + "="*20)
        print("Final Evaluation (Loaded)")
        print("="*20)
        print_constraint_summary(results)

        print("\n" + "="*20)
        print("Creating Visualizations (Loaded)")
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

    return training_time_result


if __name__ == '__main__':
    import sys
    is_benchmark = '--benchmark=1' in sys.argv or '--benchmark' in sys.argv and '1' in sys.argv

    if is_benchmark:
        n_runs = 2
        times = []
        cells = []

        print("="*20)
        print(f"Benchmark mode: {n_runs} runs")
        print("="*20)

        for i in range(n_runs):
            print(f"\n*** Run {i+1}/{n_runs} ***")
            training_time = main(benchmark_mode=True)
            if training_time is not None:
                times.append(training_time)

            bundle_path = OUTPUT_DIR / "eval_bundle.pth"
            if bundle_path.exists():
                import torch
                bundle = torch.load(bundle_path, map_location='cpu')
                cell_counts = {
                    'init': len(bundle['region_cells']['init']),
                    'goal': len(bundle['region_cells']['goal']),
                    'unsafe': len(bundle['region_cells']['unsafe']),
                    'outside': len(bundle['region_cells']['outside']),
                    'generator': len(bundle['region_cells']['generator']),
                }
                cell_counts['total_v'] = cell_counts['init'] + cell_counts['goal'] + cell_counts['unsafe'] + cell_counts['outside']
                cell_counts['total'] = cell_counts['total_v'] + cell_counts['generator']
                cells.append(cell_counts)

        print("\n" + "="*20)
        print("Benchmark results")
        print("="*20)
        avg_time = stats.mean(times)
        std_time = stats.stdev(times)
        print(f"Training time (pretrain+train): {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"  Individual times: {[f'{t:.2f}s' for t in times]}")

        if cells:
            categories = ['init', 'goal', 'unsafe', 'outside', 'generator', 'total_v', 'total']
            print("\nCell counts:")
            for cat in categories:
                values = [c[cat] for c in cells]
                avg = stats.mean(values)
                std = stats.stdev(values)
                print(f"  {cat:12s}: {avg:.0f} ± {std:.0f}")
    else:
        main()
