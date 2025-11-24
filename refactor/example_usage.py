"""
Example usage of the modular RL verification framework.

This script demonstrates how to:
1. Set up hyperparameters
2. Define system dynamics (F and G matrices)
3. Define spatial regions (init, goal, unsafe)
4. Create a value network
5. Create a Phi module for generator computation
6. Use the components together

This replaces the monolithic testing_simple3.py with a clean, modular approach.
"""

import numpy as np
import torch

from hyperparameters import Hyperparameters
from regions import Regions
from dynamics import Dynamics
from network import create_V
from phi_module import create_GV


def main():
    print("="*80)
    print("SETTING UP HYPERPARAMETERS")
    print("="*80)

    # Create default hyperparameters
    params = Hyperparameters.default()

    # Customize if needed
    params.network.n_hidden_1 = 256
    params.network.n_hidden_2 = 32
    params.network.input_scale = 100.0
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.001
    params.training.num_epochs = 200000
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False

    params.discretization.n_goal = 3
    params.discretization.n_outside_goal = 3
    params.discretization.n_generator = 1
    params.discretization.n_unsafe = 3
    params.discretization.n_init = 3

    params.constraints.beta_s = 0.6
    params.constraints.beta_ra = 20.0

    print(f"Network: {params.network.n_inputs} -> {params.network.n_hidden_1} -> "
          f"{params.network.n_hidden_2} -> {params.network.n_outputs}")
    print(f"Input scale: {params.network.input_scale}")
    print(f"Scale factor: {params.network.scale_factor}")
    print(f"Learning rate: {params.training.learning_rate}")
    print(f"Num epochs: {params.training.num_epochs}")
    print(f"Beta_s: {params.constraints.beta_s}")
    print(f"Beta_ra: {params.constraints.beta_ra}")
    if params.training.learnable_scale:
        print("Output scale is learnable.")
    else:
        print(f"Output scale: {params.network.scale_factor}.")
    if params.training.learnable_input_scale:
        print("Input scale is learnable.")
    else:
        print(f"Input scale: {params.network.input_scale}.")

    print("\n" + "="*80)
    print("SYSTEM DYNAMICS")
    print("="*80)

    F_matrix = np.array([
        [-0.5, 1.0],
        [-1.0, -0.5]
    ], dtype=np.float32)

    G_matrix = np.array([
        [0.2, 0.0],
        [0.0, 0.2]
    ], dtype=np.float32)

    dynamics = Dynamics.from_matrices(F=F_matrix, G=G_matrix)
    print(dynamics)

    print("\n" + "="*80)
    print("DEFINING SPATIAL REGIONS")
    print("="*80)

    # Define regions using numpy arrays (state_dim x 2)
    init_range = np.array([
        [45.0, 55.0],    # x1 range
        [-55.0, -45.0]   # x2 range
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

    print("\n" + "="*80)
    print("CREATING V NETWORK")
    print("="*80)

    # Create network from config
    V = create_V(params.network)
    print(V)

    print("\n" + "="*80)
    print("CREATING GV (PHI MODULE)")
    print("="*80)

    GV = create_GV(
        V_net=V,              # GV-specific: the value network
        dynamics=dynamics,    # GV-specific: system dynamics
        network_config=params.network,    # Shared with V: scale_factor, input_scale
        training_config=params.training   # Shared with V: learnable flags
    )
    print(GV)

    

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    main()
