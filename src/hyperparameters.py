"""
Hyperparameters for verification training.

This module centralizes all configuration settings for the neural network training,
discretization, constraints, and verification.
"""
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")

def _filter_kwargs(cls: Type[T], d: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    d = d or {}
    return {k: v for k, v in d.items() if k in allowed}

def _dc_from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
    return cls(**_filter_kwargs(cls, d))


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    n_inputs: int = 2
    n_hidden_1: int = 256
    n_hidden_2: int = 32
    n_outputs: int = 1

    # Scaling parameters
    input_scale: list = field(default_factory=lambda: [100.0, 100.0])
    scale_factor: float = 20.0

@dataclass
class DiscretizationConfig:
    """Discretization parameters for different regions."""
    axis_weights: list = field(default_factory=list)
    max_region_budget: int = 0
    max_generator_budget: int = 0
    n_goal: int = 3
    n_outside_goal: int = 3
    n_generator: int = 1 
    n_unsafe: int = 3
    n_init: int = 3

@dataclass
class ConstraintConfig:
    """Constraint parameters for training."""
    beta_ra: float = 20.0
    beta_increment: float = 0.2

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
    v_goal: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
        merge_relax_margin=0.3
    ))
    v_init: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
        merge_relax_margin=0.3
    ))
    v_outside: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
        merge_relax_margin=0.3
    ))
    v_unsafe: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
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

@dataclass
class TrainingConfig:
    """Training parameters."""
    learning_rate: float = 0.001
    num_epochs: int = 200000
    device: str = 'cpu'
    random_seed: int = 0

    # Pre-training
    enable_pretraining: bool = True
    pretrain_epochs: int = 2000
    pretrain_lr: float = 0.01
    pretrain_n_samples: int = 1000

    # Generator constraint
    generator_weight: float = 1.0
    generator_start_epoch: int = 0

    # Step decay, for scheduler factories that honour it: the rate is
    # multiplied by lr_decay_gamma every lr_decay_step epochs.
    lr_decay_gamma: float = 0.95
    lr_decay_step: int = 2000

    # Learning-rate floor, for scheduler factories that honour it: hold
    # lr_floor from lr_floor_epoch onwards instead of decaying forever.
    # lr_floor_epoch = 0 disables the floor.
    lr_floor: float = 0.0
    lr_floor_epoch: int = 0

    # auto_LiRPA bound method for the V and GV bounds used in the loss, the SAT
    # checks and the diagnostics. See src.crown_bounds.BOUND_METHODS. Kept at
    # 'IBP' by default so existing examples are unaffected.
    bound_method: str = 'IBP'
    generator_bound_method: str = 'IBP'

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
    include_time: bool = False

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

    # --- in Hyperparameters.from_dict ---
    @classmethod
    def from_dict(cls, config_dict: dict):
        network = _dc_from_dict(NetworkConfig, config_dict.get("network"))
        discretization = _dc_from_dict(DiscretizationConfig, config_dict.get("discretization"))
        constraints = _dc_from_dict(ConstraintConfig, config_dict.get("constraints"))  # beta_s gets dropped here
        training = _dc_from_dict(TrainingConfig, config_dict.get("training"))
        logging = _dc_from_dict(LoggingConfig, config_dict.get("logging"))

        refinement_dict = config_dict.get("refinement") or {}
        v_goal = _dc_from_dict(RefinementConfigRegion, refinement_dict.get("v_goal"))
        v_init = _dc_from_dict(RefinementConfigRegion, refinement_dict.get("v_init"))
        v_outside = _dc_from_dict(RefinementConfigRegion, refinement_dict.get("v_outside"))
        v_unsafe = _dc_from_dict(RefinementConfigRegion, refinement_dict.get("v_unsafe"))
        gv_generator = _dc_from_dict(RefinementConfigRegion, refinement_dict.get("gv_generator"))
        refinement = RefinementConfig(
            v_goal=v_goal,
            v_init=v_init,
            v_outside=v_outside,
            v_unsafe=v_unsafe,
            gv_generator=gv_generator,
        )

        return cls(
            network=network,
            discretization=discretization,
            constraints=constraints,
            training=training,
            refinement=refinement,
            logging=logging,
            compute_V=config_dict.get("compute_V", True),
            compute_GV=config_dict.get("compute_GV", True),
            include_time=config_dict.get("include_time", False),
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
                'axis_weights': self.discretization.axis_weights,
                'max_region_budget': self.discretization.max_region_budget,
                'max_generator_budget': self.discretization.max_generator_budget,
                'n_goal': self.discretization.n_goal,
                'n_outside_goal': self.discretization.n_outside_goal,
                'n_generator': self.discretization.n_generator,
                'n_unsafe': self.discretization.n_unsafe,
                'n_init': self.discretization.n_init,
            },
            'constraints': {
                'beta_ra': self.constraints.beta_ra,
                'beta_increment': self.constraints.beta_increment,
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
                'generator_start_epoch': self.training.generator_start_epoch,
                'lr_decay_gamma': self.training.lr_decay_gamma,
                'lr_decay_step': self.training.lr_decay_step,
                'lr_floor': self.training.lr_floor,
                'lr_floor_epoch': self.training.lr_floor_epoch,
                'bound_method': self.training.bound_method,
                'generator_bound_method': self.training.generator_bound_method
            },
            'refinement': {
                'v_goal': {
                    'enable_refinement': self.refinement.v_goal.enable_refinement,
                    'refine_interval': self.refinement.v_goal.refine_interval,
                    'refine_interval_late': self.refinement.v_goal.refine_interval_late,
                    'late_epoch_threshold': self.refinement.v_goal.late_epoch_threshold,
                    'refine_factor': self.refinement.v_goal.refine_factor,
                    'max_cells': self.refinement.v_goal.max_cells,
                    'N_to_refine': self.refinement.v_goal.N_to_refine,
                    'enable_merging': self.refinement.v_goal.enable_merging,
                    'merge_interval': self.refinement.v_goal.merge_interval,
                    'merge_max_passes': self.refinement.v_goal.merge_max_passes,
                    'merge_relax_margin': self.refinement.v_goal.merge_relax_margin
                },
                'v_init': {
                    'enable_refinement': self.refinement.v_init.enable_refinement,
                    'refine_interval': self.refinement.v_init.refine_interval,
                    'refine_interval_late': self.refinement.v_init.refine_interval_late,
                    'late_epoch_threshold': self.refinement.v_init.late_epoch_threshold,
                    'refine_factor': self.refinement.v_init.refine_factor,
                    'max_cells': self.refinement.v_init.max_cells,
                    'N_to_refine': self.refinement.v_init.N_to_refine,
                    'enable_merging': self.refinement.v_init.enable_merging,
                    'merge_interval': self.refinement.v_init.merge_interval,
                    'merge_max_passes': self.refinement.v_init.merge_max_passes,
                    'merge_relax_margin': self.refinement.v_init.merge_relax_margin
                },
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
                'v_unsafe': {
                    'enable_refinement': self.refinement.v_unsafe.enable_refinement,
                    'refine_interval': self.refinement.v_unsafe.refine_interval,
                    'refine_interval_late': self.refinement.v_unsafe.refine_interval_late,
                    'late_epoch_threshold': self.refinement.v_unsafe.late_epoch_threshold,
                    'refine_factor': self.refinement.v_unsafe.refine_factor,
                    'max_cells': self.refinement.v_unsafe.max_cells,
                    'N_to_refine': self.refinement.v_unsafe.N_to_refine,
                    'enable_merging': self.refinement.v_unsafe.enable_merging,
                    'merge_interval': self.refinement.v_unsafe.merge_interval,
                    'merge_max_passes': self.refinement.v_unsafe.merge_max_passes,
                    'merge_relax_margin': self.refinement.v_unsafe.merge_relax_margin
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
            },
            'compute_V': self.compute_V,
            'compute_GV': self.compute_GV,
            'include_time': self.include_time
        }
