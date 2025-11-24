# Modular RL Verification Framework

A clean, modular refactoring of the neural network-based verification code for stochastic control systems.

## Overview

This framework provides a modular implementation for learning Lyapunov functions using neural networks to verify safety and reachability properties of stochastic dynamical systems.

**Key improvements over `testing_simple3.py`:**
- ✅ Clean separation of concerns (hyperparameters, dynamics, regions, network)
- ✅ Properly handles **generalized diffusion matrices G** (not just R[0,0])
- ✅ Flexible dynamics specification (easy to change F and G)
- ✅ Supports both diagonal and non-diagonal diffusion
- ✅ Configurable and maintainable code structure

## Structure

```
modular_rl_verification/
├── __init__.py           # Package initialization
├── hyperparameters.py    # Configuration management
├── dynamics.py           # System dynamics (F and G matrices)
├── regions.py            # Spatial regions (init, goal, unsafe)
├── network.py            # Value function neural networks
├── phi_module.py         # Infinitesimal generator computation
├── discretization.py     # Region discretization utilities (with radial refinement!)
├── training_utils.py     # Loss computation (sampling & bounds)
├── crown_bounds.py       # CROWN bound computation for training
├── visualization.py      # Plotting and visualization
├── train.py              # Sample-based training script (fast)
├── train_bounds.py       # Bound-based training script (rigorous, like original)
├── example_usage.py      # Example demonstrating usage
├── README.md             # This file
└── MIGRATION_GUIDE.md    # Migration guide from testing_simple3.py
```

## Components

### 1. Hyperparameters (`hyperparameters.py`)

Centralizes all configuration settings:

```python
from hyperparameters import Hyperparameters

# Create default configuration
params = Hyperparameters.default()

# Customize
params.network.n_hidden_1 = 256
params.network.input_scale = 100.0
params.training.learning_rate = 0.001
```

**Modules:**
- `NetworkConfig`: Architecture (layers, activation, scaling)
- `DiscretizationConfig`: Grid discretization parameters
- `ConstraintConfig`: Constraint thresholds (beta_s, beta_ra)
- `TrainingConfig`: Optimization settings (LR, epochs, CROWN)

### 2. Dynamics (`dynamics.py`)

Defines the stochastic differential equation:
```
dx = f(x)dt + g(x)dW
```

where:
- `f(x) = F @ x` (linear drift with matrix F)
- `g(x) = G` or `g(x) = G(x)` (constant or state-dependent diffusion)

**Usage:**

```python
from dynamics import Dynamics
import numpy as np

# Drift matrix
F = np.array([
    [-1.5, 1.0],
    [-1.0, -1.5]
], dtype=np.float32)

# Diffusion matrix
G = np.array([
    [0.2, 0.0],
    [0.0, 0.2]
], dtype=np.float32)

# Create dynamics
dynamics = Dynamics.from_matrices(F=F, G=G)

# Or use helper methods
dynamics = Dynamics.scalar_diffusion(F=F, sigma=0.2)  # G = sigma * I
```

**Key fix:** The original code only used `R[0,0]` for sigma. This properly handles the full G matrix and computes `G @ G^T` correctly.

### 3. Regions (`regions.py`)

Defines rectangular regions in state space:

```python
from regions import Regions
import numpy as np

# Define region bounds (state_dim x 2)
init_range = np.array([[45.0, 55.0], [-55.0, -45.0]])
goal_range = np.array([[-25.0, 25.0], [-25.0, 25.0]])
unsafe_range = np.array([[-100.0, -80.0], [-100.0, 100.0]])

regions = Regions.from_numpy_ranges(
    init_range=init_range,
    goal_range=goal_range,
    unsafe_range=unsafe_range
)

# Check containment
regions.goal.contains(x)  # Returns boolean
```

### 4. Network (`network.py`)

Value function approximation neural network:

```python
from network import create_value_network

# Create from config
V_net = create_value_network(params.network)

# Forward pass
V = V_net(x)  # x: (batch_size, state_dim) -> V: (batch_size, 1)
```

**Features:**
- Configurable activation functions (sigmoid, ReLU, GELU, arctan)
- Input normalization
- Output scaling

### 5. Phi Module (`phi_module.py`)

Computes the infinitesimal generator:
```
Φ(x) = f(x) · ∇V + 0.5 · Tr(g(x)g(x)^T @ H_V)
```

**Usage:**

```python
from phi_module import create_phi_module

phi_module = create_phi_module(
    V_net=V_net,
    dynamics=dynamics,
    scale_factor=20.0,
    learnable_scale=True
)

# Compute generator
Phi = phi_module(x)  # x: (batch_size, state_dim) -> Φ: (batch_size, 1)
```

**Key improvements:**
- Takes `Dynamics` object instead of hardcoded A and sigma
- Properly computes diffusion term using full G matrix
- Optimized path for diagonal diffusion (only computes H_diag)
- General path for non-diagonal diffusion (computes full Hessian)

### 6. Discretization (`discretization.py`)

Region discretization utilities:

```python
from discretization import discretize_regions

# Discretize all regions
region_cells = discretize_regions(regions, params.discretization)

# Returns dictionary with cells for:
# - 'goal': Goal region cells
# - 'unsafe': Unsafe region cells
# - 'init': Init region cells
# - 'outside': Outside goal cells (for V constraint)
# - 'generator': Outside goal ∪ unsafe cells (for Φ constraint)
```

### 7. Training Utilities (`training_utils.py`)

Loss computation and sampling:

```python
from training_utils import (
    sample_from_cells,
    compute_total_loss,
    evaluate_constraints
)

# Sample from discretized cells
x_goal = sample_from_cells(region_cells['goal'], n_samples=256)

# Compute total loss
total_loss, loss_dict = compute_total_loss(
    V_goal, V_unsafe, V_init, V_outside, Phi,
    beta_s_goal, beta_s, beta_ra, generator_weight
)

# Evaluate constraint satisfaction
results = evaluate_constraints(V_net, GV_net, region_cells, ...)
```

### 8. Visualization (`visualization.py`)

Plotting and visualization utilities:

```python
from visualization import (
    visualize_value_function,
    visualize_generator,
    create_summary_plots
)

# Plot value function
visualize_value_function(
    V_net, regions,
    title="Value Function V(x)",
    show_regions=True,
    filename="value_function.png"
)

# Plot generator
visualize_generator(
    V_net, GV_net, regions,
    title="Generator Φ(x)",
    show_regions=True,
    filename="generator.png"
)

# Create all summary plots
create_summary_plots(
    V_net, GV_net, regions, region_cells,
    beta_s, beta_ra,
    loss_history=loss_history,
    output_dir="results"
)
```

**Available plots:**
- Value function V(x) contour plots
- Generator Φ(x) contour plots (with Φ=0 boundary)
- Constraint boundaries (beta_s, beta_ra levels)
- Training loss history
- Discretization overlays

## Quick Start

### Training Options

**Two training modes available:**

#### 1. Sample-Based Training (Fast, Approximate)
```bash
cd modular_rl_verification
python train.py
```

- **Method:** Samples points from discretized cells
- **Speed:** Fast (no CROWN overhead)
- **Guarantees:** Approximate (only checked at sampled points)
- **Use case:** Quick prototyping and debugging

#### 2. Bound-Based Training (Rigorous, Slower)
```bash
cd modular_rl_verification
python train_bounds.py
```

- **Method:** Computes CROWN bounds over cells (matches original `testing_simple3.py`)
- **Speed:** Slower (CROWN computation overhead)
- **Guarantees:** Rigorous (bounds cover entire cells)
- **Use case:** Final verification and publication results

**Recommended workflow:**
1. Use `train.py` for fast experimentation
2. Use `train_bounds.py` for final results

### What Happens During Training

Both scripts will:
1. Load configuration and define dynamics
2. Create value network and generator
3. Discretize regions into cells (with adaptive refinement for generator)
4. Train with constraint losses
5. Visualize progress during training
6. Create final summary plots
7. Evaluate and save the model

**Output:**
- `trained_model.pth` (or `trained_model_bounds.pth`) - Saved model with metadata
- `results/` - Final visualization plots:
  - `value_function.png` - V(x) contour plot
  - `generator.png` - Φ(x) contour plot
  - `constraint_regions.png` - V(x) with constraint levels
  - `loss_history.png` - Training loss curves
  - `value_function_discretization.png` - V(x) with cells
- `training_progress/` - Intermediate plots (if enabled)

### Exploring the Framework

See `example_usage.py` for component demonstrations:

```bash
cd modular_rl_verification
python example_usage.py
```

## Usage in Training

Basic training loop structure:

```python
# 1. Setup
params = Hyperparameters.default()
dynamics = Dynamics.from_matrices(F, G)
regions = Regions.from_numpy_ranges(...)
V_net = create_value_network(params.network)
phi_module = create_phi_module(V_net, dynamics)

# 2. Optimizer
optimizer = torch.optim.Adam(V_net.parameters(), lr=params.training.learning_rate)

# 3. Training loop
for epoch in range(params.training.num_epochs):
    # Sample from regions
    x_goal = sample_from_region(regions.goal)
    x_unsafe = sample_from_region(regions.unsafe)
    x_outside = sample_outside_goal(regions)

    # Compute constraints
    V_goal = V_net(x_goal)
    V_unsafe = V_net(x_unsafe)
    Phi = phi_module(x_outside)

    # Loss
    loss_goal = F.relu(V_goal - params.constraints.beta_s_goal).mean()
    loss_unsafe = F.relu(params.constraints.beta_ra - V_unsafe).mean()
    loss_generator = F.relu(Phi).mean()

    loss = loss_goal + loss_unsafe + loss_generator

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Key Differences from `testing_simple3.py`

### Original Code Issues

1. **Hardcoded sigma extraction:**
   ```python
   # Line 474-477 in testing_simple3.py
   if isinstance(sigma, np.ndarray):
       sigma = float(sigma[0, 0])  # Only uses R[0,0]!
   self.sigma = sigma
   ```

2. **Only diagonal diffusion:**
   ```python
   # Lines 577-578
   g11_sq = (self.sigma * x1) ** 2
   g22_sq = (self.sigma * x2) ** 2
   # Cannot handle off-diagonal terms in G!
   ```

### New Modular Code

1. **Proper diffusion matrix handling:**
   ```python
   # phi_module.py
   self.register_buffer('G', G_constant)
   self.register_buffer('GGT', G_constant @ G_constant.T)
   ```

2. **Supports non-diagonal diffusion:**
   ```python
   # Computes full Hessian when needed
   H_full = compute_full_hessian(...)
   diffusion_term = torch.einsum('ij,bij->b', self.GGT, H_full)
   ```

## Changing Dynamics

Easy to modify system dynamics:

```python
# 1. Change drift matrix
F_new = np.array([[-2.0, 0.5], [-0.5, -2.0]])
dynamics = Dynamics.from_matrices(F=F_new, G=G)

# 2. Change diffusion to non-diagonal
G_coupled = np.array([[0.2, 0.05], [0.05, 0.2]])
dynamics = Dynamics.from_matrices(F=F, G=G_coupled)

# 3. Use scalar diffusion
dynamics = Dynamics.scalar_diffusion(F=F, sigma=0.3)

# 4. State-dependent diffusion (advanced)
def custom_diffusion(x):
    # g(x) = sigma * diag(x)
    return diagonal_state_diffusion(sigma=0.2)(x)

dynamics = Dynamics.state_dependent_diffusion(F=F, diffusion_fn=custom_diffusion)
```

## Changing Regions

Easily modify spatial bounds:

```python
# Larger unsafe region
unsafe_range = np.array([[-120.0, -70.0], [-120.0, 120.0]])

# Different goal
goal_range = np.array([[-30.0, 30.0], [-30.0, 30.0]])

regions = Regions.from_numpy_ranges(
    init_range=init_range,
    goal_range=goal_range,
    unsafe_range=unsafe_range
)
```

## Next Steps

To complete the refactoring:

1. **Add discretization utilities** - Port discretization functions from `testing_simple3.py`
2. **Add training loop** - Adapt training logic to use modular components
3. **Add visualization** - Port visualization functions
4. **Add CROWN verification** - Integrate symbolic CROWN caching
5. **Add pre-training** - Port structure-aware pre-training

## Benefits

- 📦 **Modular**: Each component has a single responsibility
- 🔧 **Configurable**: Easy to change hyperparameters and system settings
- 🎯 **Correct**: Properly handles general F and G matrices
- 📖 **Readable**: Clean, well-documented code
- 🧪 **Testable**: Each component can be tested independently
- 🚀 **Extensible**: Easy to add new features (3D systems, different activations, etc.)

## References

Based on research in:
- Neural Lyapunov control
- Barrier certificates
- CROWN verification for neural networks
- Stochastic reachability analysis
