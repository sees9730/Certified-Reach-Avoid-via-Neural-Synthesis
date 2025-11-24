# Migration Guide: From testing_simple3.py to Modular Framework

This guide explains how to migrate from the monolithic `testing_simple3.py` to the new modular framework.

## Key Changes

### 1. System Dynamics (MOST IMPORTANT FIX)

**BEFORE (testing_simple3.py):**
```python
# Lines 2372-2382
A_matrix = np.array([
    [-1.5, 1.0],
    [-1.0, -1.5]
], dtype=np.float32)

R_matrix = np.array([
    [0.2, 0.0],
    [0.0, 0.2]
], dtype=np.float32)

# Problem: Only R[0,0] is used!
# Line 476: sigma = float(sigma[0, 0])
# Lines 577-578: Only diagonal diffusion
```

**AFTER (Modular Framework):**
```python
from dynamics import Dynamics

F_matrix = np.array([
    [-1.5, 1.0],
    [-1.0, -1.5]
], dtype=np.float32)

G_matrix = np.array([
    [0.2, 0.0],
    [0.0, 0.2]
], dtype=np.float32)

dynamics = Dynamics.from_matrices(F=F_matrix, G=G_matrix)
# Properly computes G @ G^T for the full matrix!
```

### 2. Hyperparameters

**BEFORE:**
```python
# Lines 19-60: Scattered configuration
N_INPUTS = 2
N_HIDDEN_1 = 256
N_HIDDEN_2 = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 200000
INPUT_SCALE = 100.0
SCALE_FACTOR = 20.0
BETA_S = 0.6
# ... many more scattered variables
```

**AFTER:**
```python
from hyperparameters import Hyperparameters

params = Hyperparameters.default()

# Customize
params.network.n_hidden_1 = 256
params.network.n_hidden_2 = 32
params.training.learning_rate = 0.001
params.training.num_epochs = 200000
params.constraints.beta_s = 0.6
```

### 3. Regions

**BEFORE:**
```python
# Lines 27-30: Direct numpy arrays
x_init_range = np.array([[45.0, 55.0], [-55.0, -45.0]])
x_goal_range = np.array([[-25.0, 25.0], [-25.0, 25.0]])
x_unsafe_range = np.array([[-100.0, -80.0], [-100.0, 100.0]])
x_range = np.array([[-100.0, 100.0], [-100.0, 100.0]])
```

**AFTER:**
```python
from regions import Regions

regions = Regions.from_numpy_ranges(
    init_range=np.array([[45.0, 55.0], [-55.0, -45.0]]),
    goal_range=np.array([[-25.0, 25.0], [-25.0, 25.0]]),
    unsafe_range=np.array([[-100.0, -80.0], [-100.0, 100.0]]),
    full_range=np.array([[-100.0, 100.0], [-100.0, 100.0]])
)

# Access via clean interface
regions.init.contains(x)
regions.goal.lower
regions.unsafe.upper
```

### 4. Network Creation

**BEFORE:**
```python
# Lines 71-106: Direct class instantiation
model = SimpleNN(
    n_inputs=N_INPUTS,
    n_hidden_1=N_HIDDEN_1,
    n_hidden_2=N_HIDDEN_2,
    n_hidden_3=None,  # Confusing!
    n_outputs=N_OUTPUTS
)
```

**AFTER:**
```python
from network import create_value_network

V_net = create_value_network(params.network)
# All configuration from params, no confusion about unused layers
```

### 5. Phi Module

**BEFORE:**
```python
# Lines 437-588: Long class with hardcoded A and sigma
phi_module = _PhiModuleTrainable(
    V_net=model,
    A=A_matrix,  # Hardcoded
    sigma=R_matrix[0, 0],  # ONLY uses R[0,0]!
    scale_factor=SCALE_FACTOR
)
```

**AFTER:**
```python
from phi_module import create_phi_module

phi_module = create_phi_module(
    V_net=V_net,
    dynamics=dynamics,  # Uses full F and G matrices!
    scale_factor=params.network.scale_factor,
    learnable_scale=params.training.learnable_scale
)
```

## Critical Bug Fix: Generalized Diffusion

### The Problem in testing_simple3.py

The original code has a **critical limitation** in how it handles the diffusion matrix:

```python
# Line 474-477
if isinstance(sigma, np.ndarray):
    sigma = float(sigma[0, 0])  # ❌ ONLY USES R[0,0]!
self.sigma = sigma

# Lines 577-578
g11_sq = (self.sigma * x1) ** 2  # ❌ Assumes diagonal
g22_sq = (self.sigma * x2) ** 2  # ❌ Assumes diagonal
```

**Issues:**
1. Only `R[0,0]` is extracted, ignoring `R[1,1]` and off-diagonal terms
2. Assumes state-dependent diagonal diffusion: `g(x) = sigma * diag(x)`
3. Cannot handle constant non-diagonal diffusion
4. Cannot handle different diffusion coefficients per dimension

### The Fix in Modular Framework

```python
# phi_module.py
# Properly registers full G matrix
self.register_buffer('G', G_constant)
self.register_buffer('GGT', G_constant @ G_constant.T)

# For diagonal diffusion (optimized):
# Correctly uses GGT[i,i] for each dimension
diffusion_sum = 0.5 * Σ_i [GGT]_{ii} * H_{ii}

# For non-diagonal diffusion (general):
# Computes full Hessian and proper trace
diffusion_term = 0.5 * Tr(GGT @ H_V)
```

**Benefits:**
- ✅ Uses full G matrix, not just G[0,0]
- ✅ Handles diagonal diffusion efficiently
- ✅ Handles non-diagonal diffusion correctly
- ✅ Supports different sigma per dimension
- ✅ Computes `G @ G^T` properly

## Example: Non-Diagonal Diffusion

This would **NOT** work with testing_simple3.py but **DOES** work with the modular framework:

```python
# Coupled diffusion (non-diagonal)
G_coupled = np.array([
    [0.2, 0.05],   # Off-diagonal coupling!
    [0.05, 0.2]
], dtype=np.float32)

dynamics = Dynamics.from_matrices(F=F_matrix, G=G_coupled)
phi_module = create_phi_module(V_net, dynamics)

# This correctly computes:
# GG^T = [[0.0425, 0.020], [0.020, 0.0425]]
# Φ(x) = f·∇V + 0.5·Tr(GG^T @ H_V)
#      = f·∇V + 0.5·(GG^T[0,0]·H[0,0] + GG^T[0,1]·H[0,1]
#                    + GG^T[1,0]·H[1,0] + GG^T[1,1]·H[1,1])
```

## Step-by-Step Migration

1. **Create a new file** (don't modify testing_simple3.py)

2. **Import modular components:**
   ```python
   from modular_rl_verification import (
       Hyperparameters,
       Dynamics,
       Regions,
       create_value_network,
       create_phi_module
   )
   ```

3. **Setup hyperparameters:**
   ```python
   params = Hyperparameters.default()
   # Customize as needed
   ```

4. **Define dynamics** (IMPORTANT - use G matrix, not just sigma):
   ```python
   F = np.array([...])  # Your drift matrix
   G = np.array([...])  # Your diffusion matrix (FULL matrix!)
   dynamics = Dynamics.from_matrices(F=F, G=G)
   ```

5. **Define regions:**
   ```python
   regions = Regions.from_numpy_ranges(
       init_range=...,
       goal_range=...,
       unsafe_range=...
   )
   ```

6. **Create network and phi module:**
   ```python
   V_net = create_value_network(params.network)
   phi_module = create_phi_module(V_net, dynamics)
   ```

7. **Adapt training loop** (copy from testing_simple3.py, update to use new components)

## What to Copy from testing_simple3.py

You still need these parts (not yet modularized):

1. **Discretization functions** (lines 112-294)
   - `discretize_region()`
   - `subtract_rectangle()`
   - `refine_cells_by_mask()`
   - etc.

2. **Bound computation** (lines 359-430)
   - `softplus_bounds()`
   - `compute_bounds_manual_softplus()`

3. **CROWN caching** (lines 590-697)
   - `SymbolicCROWNCache_Phi`
   - `SymbolicCROWNCache`

4. **Training loop** (lines 1400-2200)
   - Constraint computation
   - Loss functions
   - Optimizer logic

5. **Visualization** (lines 1100-1300)
   - `visualize_network_output()`
   - `visualize_gv_output()`

6. **Pre-training** (lines 1500-1600)
   - `pretrain_structure_aware()`

## Testing the Migration

Run the example to verify everything works:

```bash
cd modular_rl_verification
python example_usage.py
```

Expected output:
- Hyperparameters setup ✓
- Dynamics creation with full G matrix ✓
- Regions definition ✓
- Network forward pass ✓
- Phi computation with generalized diffusion ✓

## Summary of Benefits

| Aspect | testing_simple3.py | Modular Framework |
|--------|-------------------|-------------------|
| **Diffusion handling** | Only R[0,0] | Full G matrix ✓ |
| **Non-diagonal diffusion** | ❌ Not supported | ✓ Supported |
| **Code organization** | Monolithic (~2700 lines) | Modular (~7 files) ✓ |
| **Configurability** | Scattered variables | Centralized config ✓ |
| **Readability** | Mixed concerns | Clean separation ✓ |
| **Extensibility** | Difficult | Easy ✓ |
| **Changing dynamics** | Hardcoded | Flexible API ✓ |

## Next Steps

1. Run `example_usage.py` to understand the new structure
2. Copy training logic from testing_simple3.py
3. Adapt it to use modular components
4. Test with your specific problem
5. Extend as needed (3D systems, different networks, etc.)

## Questions?

- See `README.md` for detailed component documentation
- See `example_usage.py` for working code examples
- Each module has detailed docstrings
