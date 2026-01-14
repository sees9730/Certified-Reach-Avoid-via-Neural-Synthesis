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
    beta_s: Optional[float] = 0.6
    beta_ra: float = 20.0
    all_v_lower_target: float = 0.0

    # Pre-training target values
    pretrain_goal_target: float = 0.3
    pretrain_unsafe_target: float = 20.0
    pretrain_init_target: float = 0.92
    pretrain_phi_target: float = 1.0


@dataclass
class RefinementConfigRegion:
    """Refinement parameters for a specific region."""
    enable_refinement: bool = True
    refine_interval: int = 200  # Epochs between refinement checks
    refine_interval_late: int = 100  # Interval after late_epoch_threshold
    late_epoch_threshold: int = 2500  # When to switch to faster refinement
    refine_factor: int = 2  # Split cells into refine_factor^D subcells
    max_cells: int = 30000  # Maximum cells per region
    N_to_refine: int = 100  # Number of failing cells to refine at once

    # Merging parameters
    enable_merging: bool = True
    merge_interval: int = 502  # Epochs between merge checks
    merge_max_passes: int = 8  # Maximum merge passes
    merge_relax_margin: float = 0.3  # Relaxation margin for merging


@dataclass
class RefinementConfig:
    """Adaptive refinement parameters for V and GV regions."""
    # V region refinement (outside, init, goal, unsafe)
    v_outside: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
        merge_relax_margin=0.3
    ))

    # GV region refinement (generator)
    gv_generator: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
        merge_relax_margin=-200.0
    ))


@dataclass
class LoggingConfig:
    """Logging and visualization parameters."""
    loss_log_interval: int = 10  # Epochs between loss prints
    detailed_eval_interval: int = 1000  # Epochs between detailed evaluations
    visualize_interval: int = 1000  # Epochs between visualizations (0 to disable)
    print_controller_interval: int = 100  # Epochs between controller param prints

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
    pretrain_n_samples: int = 1000  # Samples per region per epoch during pretraining

    # Generator constraint
    generator_weight: float = 0.0
    generator_start_epoch: int = 0

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
    refinement: RefinementConfig
    logging: LoggingConfig

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
            training=TrainingConfig(),
            refinement=RefinementConfig(),
            logging=LoggingConfig()
        )

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create hyperparameters from dictionary."""
        network = NetworkConfig(**config_dict.get('network', {}))
        discretization = DiscretizationConfig(**config_dict.get('discretization', {}))
        constraints = ConstraintConfig(**config_dict.get('constraints', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        logging = LoggingConfig(**config_dict.get('logging', {}))

        # Handle nested refinement config
        refinement_dict = config_dict.get('refinement', {})
        v_outside = RefinementConfigRegion(**refinement_dict.get('v_outside', {}))
        gv_generator = RefinementConfigRegion(**refinement_dict.get('gv_generator', {}))
        refinement = RefinementConfig(v_outside=v_outside, gv_generator=gv_generator)

        return cls(
            network=network,
            discretization=discretization,
            constraints=constraints,
            training=training,
            refinement=refinement,
            logging=logging,
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
                'beta_ra': self.constraints.beta_ra,
                'all_v_lower_target': self.constraints.all_v_lower_target,
                'pretrain_goal_target': self.constraints.pretrain_goal_target,
                'pretrain_unsafe_target': self.constraints.pretrain_unsafe_target,
                'pretrain_init_target': self.constraints.pretrain_init_target,
                'pretrain_phi_target': self.constraints.pretrain_phi_target
            },
            'training': {
                'learning_rate': self.training.learning_rate,
                'num_epochs': self.training.num_epochs,
                'device': self.training.device,
                'random_seed': self.training.random_seed,
                'enable_pretraining': self.training.enable_pretraining,
                'pretrain_epochs': self.training.pretrain_epochs,
                'pretrain_lr': self.training.pretrain_lr,
                'pretrain_n_samples': self.training.pretrain_n_samples,
                'generator_weight': self.training.generator_weight,
                'generator_start_epoch': self.training.generator_start_epoch
            },
            'refinement': {
                'v_outside': {
                    'enable_refinement': self.refinement.v_outside.enable_refinement,
                    'refine_interval': self.refinement.v_outside.refine_interval,
                    'refine_interval_late': self.refinement.v_outside.refine_interval_late,
                    'late_epoch_threshold': self.refinement.v_outside.late_epoch_threshold,
                    'refine_factor': self.refinement.v_outside.refine_factor,
                    'max_cells': self.refinement.v_outside.max_cells,
                    'N_to_refine': self.refinement.v_outside.N_to_refine,
                    'enable_merging': self.refinement.v_outside.enable_merging,
                    'merge_interval': self.refinement.v_outside.merge_interval,
                    'merge_max_passes': self.refinement.v_outside.merge_max_passes,
                    'merge_relax_margin': self.refinement.v_outside.merge_relax_margin
                },
                'gv_generator': {
                    'enable_refinement': self.refinement.gv_generator.enable_refinement,
                    'refine_interval': self.refinement.gv_generator.refine_interval,
                    'refine_interval_late': self.refinement.gv_generator.refine_interval_late,
                    'late_epoch_threshold': self.refinement.gv_generator.late_epoch_threshold,
                    'refine_factor': self.refinement.gv_generator.refine_factor,
                    'max_cells': self.refinement.gv_generator.max_cells,
                    'N_to_refine': self.refinement.gv_generator.N_to_refine,
                    'enable_merging': self.refinement.gv_generator.enable_merging,
                    'merge_interval': self.refinement.gv_generator.merge_interval,
                    'merge_max_passes': self.refinement.gv_generator.merge_max_passes,
                    'merge_relax_margin': self.refinement.gv_generator.merge_relax_margin
                }
            },
            'logging': {
                'loss_log_interval': self.logging.loss_log_interval,
                'detailed_eval_interval': self.logging.detailed_eval_interval,
                'visualize_interval': self.logging.visualize_interval,
                'print_controller_interval': self.logging.print_controller_interval
            },
            'compute_V': self.compute_V,
            'compute_GV': self.compute_GV
        }
