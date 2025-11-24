"""
Main training script for RL verification using modular framework.

This script:
1. Loads configuration from hyperparameters
2. Defines system dynamics (F and G matrices)
3. Defines spatial regions (init, goal, unsafe)
4. Creates value network and generator module
5. Discretizes regions into cells
6. Trains the network with constraint losses
7. Evaluates and saves the trained model
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

from hyperparameters import Hyperparameters
from dynamics import Dynamics
from regions import Regions
from network import create_V
from phi_module import create_GV
from discretization import discretize_regions
from training_utils import (
    sample_from_cells,
    compute_total_loss,
    evaluate_constraints,
    print_loss_summary,
    print_constraint_summary
)
from visualization import (
    visualize_training_progress,
    create_summary_plots
)


def train_network(
    V_net,
    GV_net,
    region_cells: dict,
    regions: Regions,
    params: Hyperparameters,
    device: str = 'cpu',
    visualize_interval: int = 1000
):
    """
    Train the value network with constraint losses.

    Args:
        V_net: Value network
        GV_net: Generator network
        region_cells: Dictionary of discretized cells
        regions: Regions object
        params: Hyperparameters
        device: Device for training
        visualize_interval: Interval for visualization (0 to disable)
    """
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    # Move models to device
    V_net = V_net.to(device)
    GV_net = GV_net.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        V_net.parameters(),
        lr=params.training.learning_rate
    )

    # Training loop
    n_samples_per_region = 256  # Samples per region per iteration
    loss_history = []  # Track loss history for plotting

    start_time = time.time()

    for epoch in range(params.training.num_epochs):
        V_net.train()
        GV_net.train()

        # Sample from regions
        x_goal = sample_from_cells(region_cells['goal'], n_samples_per_region, device)
        x_unsafe = sample_from_cells(region_cells['unsafe'], n_samples_per_region, device)
        x_init = sample_from_cells(region_cells['init'], n_samples_per_region, device)
        x_outside = sample_from_cells(region_cells['outside'], n_samples_per_region, device)
        x_gen = sample_from_cells(region_cells['generator'], n_samples_per_region, device)

        # Forward pass
        V_goal = V_net(x_goal)
        V_unsafe = V_net(x_unsafe)
        V_init = V_net(x_init)
        V_outside = V_net(x_outside)

        # Compute generator (only if weight > 0 and after start epoch)
        if epoch >= params.training.generator_start_epoch and params.training.generator_weight > 0:
            Phi_gen = GV_net(x_gen)
            current_gen_weight = params.training.generator_weight
        else:
            Phi_gen = torch.zeros(x_gen.shape[0], 1, device=device)
            current_gen_weight = 0.0

        # Compute loss
        total_loss, loss_dict = compute_total_loss(
            V_goal=V_goal,
            V_unsafe=V_unsafe,
            V_init=V_init,
            V_outside=V_outside,
            Phi=Phi_gen,
            beta_s_goal=params.constraints.beta_s / 2.0,  # beta_s_goal
            beta_s=params.constraints.beta_s,
            beta_ra=params.constraints.beta_ra,
            generator_weight=current_gen_weight
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Logging
        if epoch % 100 == 0 or epoch == params.training.num_epochs - 1:
            print_loss_summary(epoch, loss_dict)
            # Track loss history
            loss_dict['epoch'] = epoch
            loss_history.append(loss_dict.copy())

        # Detailed evaluation and visualization
        if epoch % 1000 == 0 or epoch == params.training.num_epochs - 1:
            print(f"\nEpoch {epoch} - Detailed Evaluation:")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_s=params.constraints.beta_s,
                beta_ra=params.constraints.beta_ra,
                device=device,
                n_samples=1000
            )
            print_constraint_summary(results, prefix="  ")
            print()

            # Visualize progress
            if visualize_interval > 0 and epoch % visualize_interval == 0:
                print(f"  Creating visualization for epoch {epoch}...")
                visualize_training_progress(
                    V_net, GV_net, regions, region_cells,
                    epoch=epoch,
                    output_dir="training_progress"
                )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return loss_history


def main():
    """Main training function."""
    print("="*80)
    print("MODULAR RL VERIFICATION TRAINING")
    print("="*80)

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
    params.network.input_scale = 100.0
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.001
    params.training.num_epochs = 100000  # Adjust as needed
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 0.0  # Set > 0 to enable generator constraint
    params.training.generator_start_epoch = 0

    params.discretization.n_goal = 3
    params.discretization.n_outside_goal = 3
    params.discretization.n_generator = 1
    params.discretization.n_unsafe = 3
    params.discretization.n_init = 3

    params.constraints.beta_s = 0.6
    params.constraints.beta_ra = 20.0

    print(f"Network: {params.network.n_inputs} -> {params.network.n_hidden_1} -> "
          f"{params.network.n_hidden_2} -> {params.network.n_outputs}")
    print(f"Training: {params.training.num_epochs} epochs, LR={params.training.learning_rate}")
    print(f"Constraints: beta_s={params.constraints.beta_s}, beta_ra={params.constraints.beta_ra}")
    print(f"Generator: weight={params.training.generator_weight}, start_epoch={params.training.generator_start_epoch}")

    # ========================================================================
    # 2. SYSTEM DYNAMICS
    # ========================================================================
    print("\n" + "="*80)
    print("SYSTEM DYNAMICS")
    print("="*80)

    # Define drift matrix F
    F_matrix = np.array([
        [-0.5, 1.0],
        [-1.0, -0.5]
    ], dtype=np.float32)

    # Define diffusion matrix G
    G_matrix = np.array([
        [0.2, 0.0],
        [0.0, 0.2]
    ], dtype=np.float32)

    dynamics = Dynamics.from_matrices(F=F_matrix, G=G_matrix)
    print(dynamics)

    # ========================================================================
    # 3. SPATIAL REGIONS
    # ========================================================================
    print("\n" + "="*80)
    print("SPATIAL REGIONS")
    print("="*80)

    init_range = np.array([
        [45.0, 55.0],
        [-55.0, -45.0]
    ], dtype=np.float32)

    goal_range = np.array([
        [-25.0, 25.0],
        [-25.0, 25.0]
    ], dtype=np.float32)

    unsafe_range = np.array([
        [-100.0, -80.0],
        [-100.0, 100.0]
    ], dtype=np.float32)

    full_range = np.array([
        [-100.0, 100.0],
        [-100.0, 100.0]
    ], dtype=np.float32)

    regions = Regions.from_numpy_ranges(
        init_range=init_range,
        goal_range=goal_range,
        unsafe_range=unsafe_range,
        full_range=full_range
    )

    print(regions)

    # ========================================================================
    # 4. CREATE NETWORKS
    # ========================================================================
    print("\n" + "="*80)
    print("NETWORKS")
    print("="*80)

    # Set random seed
    torch.manual_seed(params.training.random_seed)
    np.random.seed(params.training.random_seed)

    # Create value network
    V_net = create_V(params.network)
    print(f"V network: {V_net}")

    # Create generator network
    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        training_config=params.training
    )
    print(f"GV network created")

    # ========================================================================
    # 5. DISCRETIZE REGIONS
    # ========================================================================
    region_cells = discretize_regions(regions, params.discretization, use_radial_generator = True)

    # ========================================================================
    # 6. TRAIN
    # ========================================================================
    device = params.training.device
    print(f"\nUsing device: {device}")

    loss_history = train_network(
        V_net=V_net,
        GV_net=GV_net,
        region_cells=region_cells,
        regions=regions,
        params=params,
        device=device,
        visualize_interval=5000  # Visualize every 5000 epochs (0 to disable)
    )

    # ========================================================================
    # 7. FINAL EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    results = evaluate_constraints(
        V_net, GV_net, region_cells,
        beta_s=params.constraints.beta_s,
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
        beta_s=params.constraints.beta_s,
        beta_ra=params.constraints.beta_ra,
        loss_history=loss_history,
        output_dir="results"
    )

    # ========================================================================
    # 9. SAVE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    save_path = Path("trained_model.pth")
    torch.save({
        'model_state_dict': V_net.state_dict(),
        'hyperparameters': params.to_dict(),
        'dynamics': {
            'F': F_matrix,
            'G': G_matrix
        },
        'regions': regions.to_dict(),
        'final_results': results,
        'loss_history': loss_history
    }, save_path)

    print(f"Model saved to: {save_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - Model: {save_path}")
    print(f"  - Plots: results/")
    if loss_history:
        print(f"  - Training progress: training_progress/")


if __name__ == '__main__':
    main()
