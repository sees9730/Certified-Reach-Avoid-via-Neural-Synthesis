"""Utility functions for training and setup."""

import shutil
import torch.nn as nn
from pathlib import Path
from hyperparameters import Hyperparameters


def cleanup_and_setup_directories(dirs: list) -> None:
    """Remove and recreate directories for fresh training run."""
    for dir_path in dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Cleaned up old {dir_path.name} directory")
        dir_path.mkdir(exist_ok=True)


def print_training_config(params: Hyperparameters) -> None:
    """Print training configuration summary."""
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
