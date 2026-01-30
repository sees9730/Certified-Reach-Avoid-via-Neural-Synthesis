"""
2D Inverted Pendulum Verification
"""
import argparse
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN
from src.discretization import discretize_regions
from src.dynamics import ClosedLoopDrift, Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging, load_eval_bundle, log_loaded_training_epochs, save_eval_bundle
from src.trainer import train_network_bounds
from src.pretrainer import pretrain_network_samples
from src.training_utils import evaluate_constraints, print_constraint_summary
from src.utils import cleanup_and_setup_directories
from src.visualization import create_summary_plots

# Set random seed
torch.manual_seed(0)

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
    print("2D Inverted Pendulum Verification")
    print("="*20)

    # Return value for benchmark mode
    training_time_result = None

    # === Hyperparameters ===
    params = Hyperparameters.default()

    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 16
    params.network.input_scale = [2*np.pi, 20.0]
    params.network.scale_factor = 10.0

    params.training.learning_rate = 0.01
    params.training.num_epochs = 200000

    params.discretization.n_goal = 5
    params.discretization.n_outside_goal = 4
    params.discretization.n_generator = 4
    params.discretization.n_unsafe = 15
    params.discretization.n_init = 30

    params.constraints.beta_ra = 20.0

    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 1500
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 100

    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_interval = 250
    params.refinement.v_outside.late_epoch_threshold = 2500
    params.refinement.v_outside.refine_interval_late = 100
    params.refinement.v_outside.refine_factor = 2
    params.refinement.v_outside.max_cells = 50000
    params.refinement.v_outside.N_to_refine = 999999

    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 501
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 0.5

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_interval = 250
    params.refinement.gv_generator.late_epoch_threshold = 3500
    params.refinement.gv_generator.refine_interval_late = 100
    params.refinement.gv_generator.refine_factor = 2
    params.refinement.gv_generator.max_cells = 50000
    params.refinement.gv_generator.N_to_refine = 999999

    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -500.0

    # === Dynamics ===
    rl_policy_net = InvertControlNN()
    u_nn = WrapperConterlNN(rl_policy_net)
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"
    bundle = load_eval_bundle(bundle_path, map_location="cpu")
    u_nn.load_state_dict(bundle["control_state_dict"])

    def f_ol(x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        g = 9.81
        L = 0.5
        b = 0.1
        m = 0.15
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x2
        f2 = (g/L)*torch.sin(x1) - (b/(m*L**2))*x2
        return torch.stack([f1, f2], dim=1)

    g_coeffs = torch.tensor([0.0, 2.0], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        base = g_coeffs.to(device=x.device, dtype=x.dtype)
        if x.dim() == 1:
            return base
        elif x.dim() == 2:
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)
        else:
            raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")

    f_cl_module = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g)

    # === Regions ===
    init_range = np.array([[(3/4)*np.pi, (5/4)*np.pi], [-1.0, 1.0]], dtype=np.float32)
    goal_range = np.array([[-0.4*np.pi, 0.4*np.pi], [-4.0, 4.0]], dtype=np.float32)
    unsafe_down1 = np.array([[-2*np.pi, -2*np.pi+0.5*np.pi], [-20.0, -10.0]], dtype=np.float32)
    unsafe_down2 = np.array([[2*np.pi-0.5*np.pi, 2*np.pi], [10.0, 20.0]], dtype=np.float32)
    unsafe_range = np.vstack((unsafe_down1, unsafe_down2))
    full_range = np.array([[-2*np.pi, 2*np.pi], [-20.0, 20.0]], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe_down1 = Region(unsafe_down1)
    unsafe_down2 = Region(unsafe_down2)
    unsafe = Region.union(unsafe_down1, unsafe_down2)
    full = Region(full_range)

    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # === Networks ===
    V_net = create_V(params.network)
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network
    )

    # === Discretization ===
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=False
    )

    if args.train == 1:
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")

        training_start_time = time.time()

        # === Pretraining ===
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
                lambda_w=1e-4,
                unsafe_sample_fraction=1.0/6.0,
                save_v_path=OUTPUT_DIR / "V_pretrained.pth",
            )

        def create_scheduler(optimizer):
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=2000,
                gamma=0.9
            )

        # === Training ===
        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            create_scheduler=create_scheduler,
            start_time=training_start_time
        )

        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        print("\n" + "="*20)
        print("Final Evaluation")
        print("="*20)

        # === Final Evaluation ===
        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_ra=params.constraints.beta_ra,
            device=params.training.device
        )
        print_constraint_summary(results)

        print("\n" + "="*20)
        print("Visualizations")
        print("="*20)

        # === Final Visualizations ===
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

        print("\n" + "="*20)
        print("Saving Bundle")
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

        training_time_result = total_training_time

    else:
        print("\n" + "="*20)
        print("Loading Saved Bundle")
        print("="*20)

        bundle = load_eval_bundle(bundle_path, map_location="cpu")

        params = Hyperparameters.from_dict(bundle["hyperparameters"])
        region_cells = bundle["region_cells"]

        V_net = create_V(params.network).to(params.training.device)
        V_net.load_state_dict(bundle["V_state_dict"])

        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network
        ).to(params.training.device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"])

        region_cells = {
            k: [(lo.to(params.training.device), hi.to(params.training.device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }

        final_beta_s = bundle["final_beta_s"]
        loss_history = bundle["loss_history"]
        refinement_epochs = bundle["refinement_epochs"]

        results = bundle.get("final_results", None)
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_ra=params.constraints.beta_ra,
                device=params.training.device
            )

        print("\n" + "="*20)
        print("Final Evaluation (loaded)")
        print("="*20)
        print_constraint_summary(results)

        print("\n" + "="*20)
        print("Creating Visualizations (loaded)")
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
    # Quick check for benchmark mode before full arg parse
    import sys
    is_benchmark = '--benchmark=1' in sys.argv or '--benchmark' in sys.argv and '1' in sys.argv

    if is_benchmark:
        n_runs = 2
        times = []
        cells = []

        print("="*20)
        print(f"Benchmark: {n_runs} runs")
        print("="*20)

        for i in range(n_runs):
            print(f"\n*** Run {i+1}/{n_runs} ***")
            torch.manual_seed(i)
            training_time = main(benchmark_mode=True)
            if training_time is not None:
                times.append(training_time)

            # Extract cell counts from last run
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
        print("Benchmark Results")
        print("="*20)
        avg_time = stats.mean(times)
        std_time = stats.stdev(times)
        print(f"Training time (pretrain+train): {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"  Individual times: {[f'{t:.2f}s' for t in times]}")

        if cells:
            # Compute averages for each category
            categories = ['init', 'goal', 'unsafe', 'outside', 'generator', 'total_v', 'total']
            print("\nCell counts:")
            for cat in categories:
                values = [c[cat] for c in cells]
                avg = stats.mean(values)
                std = stats.stdev(values)
                print(f"  {cat:12s}: {avg:.0f} ± {std:.0f}")
    else:
        main()
