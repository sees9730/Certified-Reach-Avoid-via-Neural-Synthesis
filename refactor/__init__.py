"""
Neural Network Verification Framework

A clean, modular implementation for neural network-based verification
of stochastic control systems.

Main components:
- hyperparameters: Configuration management
- dynamics: System dynamics (drift F and diffusion G)
- regions: Spatial regions (init, goal, unsafe)
- network: Value function neural networks
- phi_module: Infinitesimal generator computation
"""

from .hyperparameters import (
    Hyperparameters,
    NetworkConfig,
    DiscretizationConfig,
    ConstraintConfig,
    TrainingConfig
)

from .dynamics import Dynamics, diagonal_state_diffusion

from .regions import Region, Regions

from .network import V, VDeep, create_value_network

from .phi_module import GV, create_GV

from . import discretization
from . import training_utils
from . import visualization
from . import crown_bounds


__version__ = '1.0.0'

__all__ = [
    # Hyperparameters
    'Hyperparameters',
    'NetworkConfig',
    'DiscretizationConfig',
    'ConstraintConfig',
    'TrainingConfig',

    # Dynamics
    'Dynamics',
    'diagonal_state_diffusion',

    # Regions
    'Region',
    'Regions',

    # Network
    'V',
    'VDeep',
    'create_value_network',

    # Phi module
    'GV',
    'create_GV',

    # Utilities
    'discretization',
    'training_utils',
    'visualization',
    'crown_bounds',
]
