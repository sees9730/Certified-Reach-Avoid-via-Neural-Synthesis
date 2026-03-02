"""
Hyperparameters for verification training.

This module centralizes all configuration settings for the neural network training,
discretization, constraints, and verification.
"""
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

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
    # Each field can be:
    # - int: isotropic split count for all dimensions
    # - list[int]: per-dimension split counts (anisotropic)
    n_goal: Union[int, List[int]] = 3
    n_outside_goal: Union[int, List[int]] = 3
    n_generator: Union[int, List[int]] = 1 
    n_unsafe: Union[int, List[int]] = 3
    n_init: Union[int, List[int]] = 3

@dataclass
class ConstraintConfig:
    """Constraint parameters for training."""
    beta_ra: float = 20.0
    all_v_lower_target: float = 0.0
    # Global unsafe curriculum: use beta_ra_current for all unsafe cells and
    # increase until beta_ra.
    use_beta_ra_curriculum: bool = False
    beta_ra_init: float = 1.0
    beta_ra_step: float = 0.2
    # Terminal-unsafe curriculum (optional). If enabled, unsafe cells are split
    # into [tube | terminal] using unsafe_terminal_cell_count and terminal cells
    # use beta_ra_terminal_current, which is increased until beta_ra.
    use_terminal_beta_curriculum: bool = False
    beta_ra_terminal_init: float = 1.0
    beta_ra_terminal_step: float = 0.2
    unsafe_terminal_cell_count: int = 0

@dataclass
class RefinementConfigRegion:
    """Refinement parameters for a specific region."""
    enable_refinement: bool = True
    refine_interval: int = 200  # Epochs between refinement checks
    refine_interval_late: int = 100  # Interval after late_epoch_threshold
    late_epoch_threshold: int = 2500  # When to switch to faster refinement
    # int: isotropic split; list[int]: per-dimension split
    refine_factor: Union[int, List[int]] = 2
    max_cells: int = 30000  # Maximum cells per region
    N_to_refine: int = 100  # Number of failing cells to refine at once
    refine_cooldown_epochs: int = 0  # Minimum epochs between refinements
    refine_loss_rel_improve: float = 0.0  # Require total loss improve vs last refine
    refine_min_failing_cells: int = 1  # Minimum failing cells required to refine

    # Merging parameters
    enable_merging: bool = True
    merge_interval: int = 502  # Epochs between merge checks
    merge_max_passes: int = 8  # Maximum merge passes
    merge_relax_margin: float = 0.3  # Relaxation margin for merging


@dataclass
class RefinementConfig:
    """Adaptive refinement parameters for V and GV regions."""
    # Global simple refinement mode (reduces per-region trigger tuning).
    use_simple_refinement: bool = False
    simple_refine_every: int = 500
    simple_loss_improve_tol: float = 0.01
    simple_refine_budget: int = 100
    simple_regions_per_trigger: int = 2
    enable_force_refinement: bool = True
    stage2_force_refine_interval: int = 0
    stage3_force_refine_interval: int = 1000

    # V region refinement
    v_goal: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
        enable_refinement=False,
        enable_merging=False,
        merge_relax_margin=0.0
    ))
    v_unsafe: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
        enable_refinement=False,
        enable_merging=False,
        merge_relax_margin=0.0
    ))
    v_init: RefinementConfigRegion = field(default_factory=lambda: RefinementConfigRegion(
        enable_refinement=False,
        enable_merging=False,
        merge_relax_margin=0.0
    ))
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
    generator_after_v_sat: bool = False
    generator_ramp_epochs: int = 0
    use_generator_threshold_curriculum: bool = False
    generator_threshold_init: Optional[float] = None
    generator_threshold_final: float = 0.0
    generator_threshold_step: float = 0.1
    generator_threshold_step_fine: float = 0.02
    generator_threshold_switch: float = 0.2
    use_last_layer_lp: bool = False
    last_layer_lp_interval: int = 0
    last_layer_lp_max_cells: int = 5000
    last_layer_lp_timeout_sec: float = 30.0
    last_layer_lp_verbose: bool = False
    verify_lp_apply_sat: bool = True

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

    # --- in Hyperparameters.from_dict ---
    @classmethod
    def from_dict(cls, config_dict: dict):
        network = _dc_from_dict(NetworkConfig, config_dict.get("network"))
        discretization = _dc_from_dict(DiscretizationConfig, config_dict.get("discretization"))
        constraints = _dc_from_dict(ConstraintConfig, config_dict.get("constraints"))  # beta_s gets dropped here
        training = _dc_from_dict(TrainingConfig, config_dict.get("training"))
        logging = _dc_from_dict(LoggingConfig, config_dict.get("logging"))

        refinement_dict = config_dict.get("refinement") or {}
        defaults = RefinementConfig()
        v_goal = (
            _dc_from_dict(RefinementConfigRegion, refinement_dict.get("v_goal"))
            if "v_goal" in refinement_dict else defaults.v_goal
        )
        v_unsafe = (
            _dc_from_dict(RefinementConfigRegion, refinement_dict.get("v_unsafe"))
            if "v_unsafe" in refinement_dict else defaults.v_unsafe
        )
        v_init = (
            _dc_from_dict(RefinementConfigRegion, refinement_dict.get("v_init"))
            if "v_init" in refinement_dict else defaults.v_init
        )
        v_outside = _dc_from_dict(RefinementConfigRegion, refinement_dict.get("v_outside"))
        gv_generator = _dc_from_dict(RefinementConfigRegion, refinement_dict.get("gv_generator"))
        refinement = RefinementConfig(
            use_simple_refinement=refinement_dict.get("use_simple_refinement", defaults.use_simple_refinement),
            simple_refine_every=refinement_dict.get("simple_refine_every", defaults.simple_refine_every),
            simple_loss_improve_tol=refinement_dict.get("simple_loss_improve_tol", defaults.simple_loss_improve_tol),
            simple_refine_budget=refinement_dict.get("simple_refine_budget", defaults.simple_refine_budget),
            simple_regions_per_trigger=refinement_dict.get("simple_regions_per_trigger", defaults.simple_regions_per_trigger),
            enable_force_refinement=refinement_dict.get("enable_force_refinement", defaults.enable_force_refinement),
            stage2_force_refine_interval=refinement_dict.get("stage2_force_refine_interval", defaults.stage2_force_refine_interval),
            stage3_force_refine_interval=refinement_dict.get("stage3_force_refine_interval", defaults.stage3_force_refine_interval),
            v_goal=v_goal,
            v_unsafe=v_unsafe,
            v_init=v_init,
            v_outside=v_outside,
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
                'use_beta_ra_curriculum': self.constraints.use_beta_ra_curriculum,
                'beta_ra_init': self.constraints.beta_ra_init,
                'beta_ra_step': self.constraints.beta_ra_step,
                'use_terminal_beta_curriculum': self.constraints.use_terminal_beta_curriculum,
                'beta_ra_terminal_init': self.constraints.beta_ra_terminal_init,
                'beta_ra_terminal_step': self.constraints.beta_ra_terminal_step,
                'unsafe_terminal_cell_count': self.constraints.unsafe_terminal_cell_count,
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
                'generator_after_v_sat': self.training.generator_after_v_sat,
                'generator_ramp_epochs': self.training.generator_ramp_epochs,
                'use_generator_threshold_curriculum': self.training.use_generator_threshold_curriculum,
                'generator_threshold_init': self.training.generator_threshold_init,
                'generator_threshold_final': self.training.generator_threshold_final,
                'generator_threshold_step': self.training.generator_threshold_step,
                'generator_threshold_step_fine': self.training.generator_threshold_step_fine,
                'generator_threshold_switch': self.training.generator_threshold_switch,
                'use_last_layer_lp': self.training.use_last_layer_lp,
                'last_layer_lp_interval': self.training.last_layer_lp_interval,
                'last_layer_lp_max_cells': self.training.last_layer_lp_max_cells,
                'last_layer_lp_timeout_sec': self.training.last_layer_lp_timeout_sec,
                'last_layer_lp_verbose': self.training.last_layer_lp_verbose,
                'verify_lp_apply_sat': self.training.verify_lp_apply_sat,
            },
            'refinement': {
                'use_simple_refinement': self.refinement.use_simple_refinement,
                'simple_refine_every': self.refinement.simple_refine_every,
                'simple_loss_improve_tol': self.refinement.simple_loss_improve_tol,
                'simple_refine_budget': self.refinement.simple_refine_budget,
                'simple_regions_per_trigger': self.refinement.simple_regions_per_trigger,
                'enable_force_refinement': self.refinement.enable_force_refinement,
                'stage2_force_refine_interval': self.refinement.stage2_force_refine_interval,
                'stage3_force_refine_interval': self.refinement.stage3_force_refine_interval,
                'v_goal': {
                    'enable_refinement': self.refinement.v_goal.enable_refinement,
                    'refine_interval': self.refinement.v_goal.refine_interval,
                    'refine_interval_late': self.refinement.v_goal.refine_interval_late,
                    'late_epoch_threshold': self.refinement.v_goal.late_epoch_threshold,
                    'refine_factor': self.refinement.v_goal.refine_factor,
                    'max_cells': self.refinement.v_goal.max_cells,
                    'N_to_refine': self.refinement.v_goal.N_to_refine,
                    'refine_cooldown_epochs': self.refinement.v_goal.refine_cooldown_epochs,
                    'refine_loss_rel_improve': self.refinement.v_goal.refine_loss_rel_improve,
                    'refine_min_failing_cells': self.refinement.v_goal.refine_min_failing_cells,
                    'enable_merging': self.refinement.v_goal.enable_merging,
                    'merge_interval': self.refinement.v_goal.merge_interval,
                    'merge_max_passes': self.refinement.v_goal.merge_max_passes,
                    'merge_relax_margin': self.refinement.v_goal.merge_relax_margin
                },
                'v_unsafe': {
                    'enable_refinement': self.refinement.v_unsafe.enable_refinement,
                    'refine_interval': self.refinement.v_unsafe.refine_interval,
                    'refine_interval_late': self.refinement.v_unsafe.refine_interval_late,
                    'late_epoch_threshold': self.refinement.v_unsafe.late_epoch_threshold,
                    'refine_factor': self.refinement.v_unsafe.refine_factor,
                    'max_cells': self.refinement.v_unsafe.max_cells,
                    'N_to_refine': self.refinement.v_unsafe.N_to_refine,
                    'refine_cooldown_epochs': self.refinement.v_unsafe.refine_cooldown_epochs,
                    'refine_loss_rel_improve': self.refinement.v_unsafe.refine_loss_rel_improve,
                    'refine_min_failing_cells': self.refinement.v_unsafe.refine_min_failing_cells,
                    'enable_merging': self.refinement.v_unsafe.enable_merging,
                    'merge_interval': self.refinement.v_unsafe.merge_interval,
                    'merge_max_passes': self.refinement.v_unsafe.merge_max_passes,
                    'merge_relax_margin': self.refinement.v_unsafe.merge_relax_margin
                },
                'v_init': {
                    'enable_refinement': self.refinement.v_init.enable_refinement,
                    'refine_interval': self.refinement.v_init.refine_interval,
                    'refine_interval_late': self.refinement.v_init.refine_interval_late,
                    'late_epoch_threshold': self.refinement.v_init.late_epoch_threshold,
                    'refine_factor': self.refinement.v_init.refine_factor,
                    'max_cells': self.refinement.v_init.max_cells,
                    'N_to_refine': self.refinement.v_init.N_to_refine,
                    'refine_cooldown_epochs': self.refinement.v_init.refine_cooldown_epochs,
                    'refine_loss_rel_improve': self.refinement.v_init.refine_loss_rel_improve,
                    'refine_min_failing_cells': self.refinement.v_init.refine_min_failing_cells,
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
                    'refine_cooldown_epochs': self.refinement.v_outside.refine_cooldown_epochs,
                    'refine_loss_rel_improve': self.refinement.v_outside.refine_loss_rel_improve,
                    'refine_min_failing_cells': self.refinement.v_outside.refine_min_failing_cells,
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
                    'refine_cooldown_epochs': self.refinement.gv_generator.refine_cooldown_epochs,
                    'refine_loss_rel_improve': self.refinement.gv_generator.refine_loss_rel_improve,
                    'refine_min_failing_cells': self.refinement.gv_generator.refine_min_failing_cells,
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


def load_hparams(hp_dict: dict) -> Hyperparameters:
    # start from defaults (so newly-added fields get sensible values)
    base = Hyperparameters.default().to_dict()

    # shallow merge at top-level + sub-dicts
    # (for your structure, a simple recursive merge is safer)
    def deep_update(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    merged = deep_update(base, hp_dict or {})
    return Hyperparameters.from_dict(merged)
