"""
Hyperparameters for RL verification training.

This module centralizes all configuration settings for the neural network training,
discretization, constraints, and verification.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    n_inputs: int = 2
    n_hidden_1: int = 256
    n_hidden_2: int = 32
    n_outputs: int = 1

    # Scaling parameters
    input_scale: list = field(default_factory=lambda: [100.0, 100.0])  # Input normalization each index corresponds to a dimension
    scale_factor: float = 20.0  # Output layer scaling


@dataclass
class DiscretizationConfig:
    """Discretization parameters for different regions."""
    n_goal: int = 3
    n_outside_goal: int = 3  # For V constraint: X \ Goal
    n_generator: int = 1  # For Φ constraint: X \ (Goal ∪ Unsafe)
    n_unsafe: int = 3
    n_init: int = 3


@dataclass
class ConstraintConfig:
    """Constraint parameters for training."""
    beta_s: Optional[float] = 0.6  # Separation threshold (None = learnable)
    beta_ra: float = 20.0  # Reachability-avoid threshold
    all_v_lower_target: float = 0.0


@dataclass
class TrainingConfig:
    """Training parameters."""
    learning_rate: float = 0.001
    num_epochs: int = 200000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed: int = 0

    # Pre-training
    enable_pretraining: bool = True
    pretrain_epochs: int = 2000
    pretrain_lr: float = 0.01

    # Generator constraint
    generator_weight: float = 0.0
    generator_start_epoch: int = 0

    # Learnable parameters
    learnable_scale: bool = False
    learnable_input_scale: bool = False
    learnable_beta_s: bool = False  # If True, beta_s becomes a learnable parameter


@dataclass
class Hyperparameters:
    """
    Complete hyperparameter configuration.

    This class combines all configuration settings into a single interface.
    """
    network: NetworkConfig
    discretization: DiscretizationConfig
    constraints: ConstraintConfig
    training: TrainingConfig

    # Flags for what to compute
    compute_V: bool = True
    compute_GV: bool = True

    @classmethod
    def default(cls):
        """Create default hyperparameter configuration."""
        return cls(
            network=NetworkConfig(),
            discretization=DiscretizationConfig(),
            constraints=ConstraintConfig(),
            training=TrainingConfig()
        )

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create hyperparameters from dictionary."""
        network = NetworkConfig(**config_dict.get('network', {}))
        discretization = DiscretizationConfig(**config_dict.get('discretization', {}))
        constraints = ConstraintConfig(**config_dict.get('constraints', {}))
        training = TrainingConfig(**config_dict.get('training', {}))

        return cls(
            network=network,
            discretization=discretization,
            constraints=constraints,
            training=training,
            compute_V=config_dict.get('compute_V', True),
            compute_GV=config_dict.get('compute_GV', True)
        )

    def to_dict(self):
        """Convert hyperparameters to dictionary."""
        return {
            'network': {
                'n_inputs': self.network.n_inputs,
                'n_hidden_1': self.network.n_hidden_1,
                'n_hidden_2': self.network.n_hidden_2,
                'n_outputs': self.network.n_outputs,
                'input_scale': self.network.input_scale,
                'scale_factor': self.network.scale_factor
            },
            'discretization': {
                'n_goal': self.discretization.n_goal,
                'n_outside_goal': self.discretization.n_outside_goal,
                'n_generator': self.discretization.n_generator,
                'n_unsafe': self.discretization.n_unsafe,
                'n_init': self.discretization.n_init
            },
            'constraints': {
                'beta_s': self.constraints.beta_s,
                'beta_ra': self.constraints.beta_ra,
                'all_v_lower_target': self.constraints.all_v_lower_target
            },
            'training': {
                'learning_rate': self.training.learning_rate,
                'num_epochs': self.training.num_epochs,
                'device': self.training.device,
                'random_seed': self.training.random_seed,
                'enable_pretraining': self.training.enable_pretraining,
                'pretrain_epochs': self.training.pretrain_epochs,
                'pretrain_lr': self.training.pretrain_lr,
                'generator_weight': self.training.generator_weight,
                'generator_start_epoch': self.training.generator_start_epoch,
                'learnable_scale': self.training.learnable_scale,
                'learnable_input_scale': self.training.learnable_input_scale,
                'learnable_beta_s': self.training.learnable_beta_s
            },
            'compute_V': self.compute_V,
            'compute_GV': self.compute_GV
        }
