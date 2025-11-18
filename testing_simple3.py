import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.gelu import GELU
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch.func import jacrev, hessian, vmap

# ============================================================================
# CONFIGURATION
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('gpu')

# Network architecture (3 hidden layers for deeper composition)
N_INPUTS = 2
N_HIDDEN_1 = 128
N_HIDDEN_2 = 128
# N_HIDDEN_3 = 64
N_OUTPUTS = 1

# Regions
x_init_range = np.array([[45.0, 55.0], [-55.0, -45.0]], dtype=np.float32)
x_goal_range = np.array([[-25.0, 25.0], [-25.0, 25.0]], dtype=np.float32)
x_unsafe_range = np.array([[-100.0, -80.0], [-100.0, 100.0]], dtype=np.float32)
x_range = np.array([[-100.0, 100.0], [-100.0, 100.0]], dtype=np.float32)

# Training parameters
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.001
NUM_EPOCHS = 200000

# Discretization
# NOTE: With state-dependent diffusion (σ²x²), we need finer discretization
# to catch extreme corners where Φ can explode!
N_DISCRETIZE_GOAL = 1
N_DISCRETIZE_OUTSIDE_GOAL = 4  # For V constraint: X \ Goal
N_DISCRETIZE_GENERATOR = 1  # For Φ constraint: X \ (Goal ∪ Unsafe)
# N_DISCRETIZE_ALL = 1
N_DISCRETIZE_UNSAFE = 7
N_DISCRETIZE_INIT = 7

# Constraints
use_tan = False
use_arctan = False
# use_gelu = True
use_gelu = False
use_relu = False
# use_relu = True
BETA_S = 0.3
BETA_S_GOAL = BETA_S/2.0
BETA_RA = 20.0
SCALE_FACTOR = 10.0
ALL_V_LOWER_TARGET = 0.0

# Input normalization scale (maps [-INPUT_SCALE, INPUT_SCALE] -> [-1, 1])
INPUT_SCALE = 100.0


compute_V = True
compute_GV = True

# ============================================================================
# SIMPLE NEURAL NETWORK (No transformations, just basic network)
# ============================================================================

class SimpleNN(nn.Module):
    """Basic feedforward network with 3 hidden layers"""

    def __init__(self, n_inputs, n_hidden_1, n_hidden_2, n_hidden_3, n_outputs):
        super(SimpleNN, self).__init__()
        # Input normalization: maps [-INPUT_SCALE, INPUT_SCALE] -> [-1, 1]
        # Register as buffer so it's properly tracked in the computational graph
        self.register_buffer('input_scale', torch.tensor(INPUT_SCALE))

        self.layer1 = nn.Linear(n_inputs, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        # self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        # self.output = nn.Linear(n_hidden_3, n_outputs)
        self.output = nn.Linear(n_hidden_2, n_outputs)
        # Activation selection
        if use_arctan:
            self.activation_fn = torch.tan  # Use arctan for bound computation
        elif use_gelu:
            self.activation_fn = GELU()
        elif use_relu:
            self.activation_fn = F.relu
        else:
            self.activation_fn = torch.sigmoid

    def forward(self, x):
        # Normalize input: [-INPUT_SCALE, INPUT_SCALE] -> [-1, 1]
        x = x / self.input_scale

        x = self.layer1(x)
        x = self.activation_fn(x)
        x = self.layer2(x)
        x = self.activation_fn(x)
        # x = self.layer3(x)
        # x = self.activation_fn(x)
        x = self.output(x * SCALE_FACTOR)
        return x

# ============================================================================
# DISCRETIZATION
# ============================================================================

def discretize_region(region, n_squares):
    """Discretize a region into n_squares x n_squares grid cells"""
    x1_edges = np.linspace(region[0, 0], region[0, 1], n_squares + 1)
    x2_edges = np.linspace(region[1, 0], region[1, 1], n_squares + 1)

    cells = []
    for i in range(n_squares):
        for j in range(n_squares):
            cell_lower = np.array([x1_edges[i], x2_edges[j]], dtype=np.float32)
            cell_upper = np.array([x1_edges[i+1], x2_edges[j+1]], dtype=np.float32)
            cells.append((
                torch.tensor(cell_lower, dtype=torch.float32),
                torch.tensor(cell_upper, dtype=torch.float32)
            ))
    return cells

def subtract_rectangle(cell_lower, cell_upper, excl_lower, excl_upper):
    """
    Subtract exclusion rectangle from cell rectangle.
    Returns list of rectangles representing cell \ exclusion.

    Returns:
        List of (lower, upper) tuples representing the non-overlapping parts
    """
    # Extract bounds
    x1_min, x2_min = cell_lower
    x1_max, x2_max = cell_upper
    ex1_min, ex2_min = excl_lower
    ex1_max, ex2_max = excl_upper

    # Check for no overlap
    if (x1_max <= ex1_min or x1_min >= ex1_max or
        x2_max <= ex2_min or x2_min >= ex2_max):
        # No overlap - return cell as-is
        return [(cell_lower, cell_upper)]

    # Check for complete containment (cell entirely inside exclusion)
    if (ex1_min <= x1_min and x1_max <= ex1_max and
        ex2_min <= x2_min and x2_max <= ex2_max):
        # Cell completely inside exclusion - return empty list
        return []

    # Partial overlap - decompose into up to 4 rectangles around the exclusion
    # Strategy: Create strips that don't overlap with exclusion
    result = []

    # Bottom strip: below the exclusion zone
    if x2_min < ex2_min:
        result.append((
            np.array([x1_min, x2_min], dtype=np.float32),
            np.array([x1_max, min(x2_max, ex2_min)], dtype=np.float32)
        ))

    # Top strip: above the exclusion zone
    if x2_max > ex2_max:
        result.append((
            np.array([x1_min, max(x2_min, ex2_max)], dtype=np.float32),
            np.array([x1_max, x2_max], dtype=np.float32)
        ))

    # Middle vertical range (overlaps with exclusion in x2 direction)
    y_mid_min = max(x2_min, ex2_min)
    y_mid_max = min(x2_max, ex2_max)

    # Left strip: left of exclusion zone, within middle vertical range
    if x1_min < ex1_min:
        result.append((
            np.array([x1_min, y_mid_min], dtype=np.float32),
            np.array([min(x1_max, ex1_min), y_mid_max], dtype=np.float32)
        ))

    # Right strip: right of exclusion zone, within middle vertical range
    if x1_max > ex1_max:
        result.append((
            np.array([max(x1_min, ex1_max), y_mid_min], dtype=np.float32),
            np.array([x1_max, y_mid_max], dtype=np.float32)
        ))

    return result

def clip_cell_against_exclusions(cell_lower, cell_upper, exclusion_regions):
    """
    Clip a cell against multiple exclusion regions.
    Returns list of cells with all exclusions removed.
    """
    current_cells = [(cell_lower, cell_upper)]

    for excl_region in exclusion_regions:
        excl_lower = excl_region[:, 0]
        excl_upper = excl_region[:, 1]

        new_cells = []
        for lower, upper in current_cells:
            # Subtract this exclusion from each current cell
            new_cells.extend(subtract_rectangle(lower, upper, excl_lower, excl_upper))

        current_cells = new_cells
        if not current_cells:  # All removed
            break

    return current_cells

def discretize_region_radial(region, radius_thresholds, n_splits, n_subdivide=20):
    """
    Discretize a region with adaptive refinement based on rectangular distance from origin.

    Args:
        region: np.array([[x1_min, x1_max], [x2_min, x2_max]])
        radius_thresholds: list of absolute distance thresholds, e.g., [30.0, 70.0]
        n_splits: list of n_split values for each zone, e.g., [10, 30, 100]
        n_subdivide: number of subdivisions for radial classification (default: 20)
    """
    assert len(n_splits) == len(radius_thresholds) + 1

    x1_edges = np.linspace(region[0, 0], region[0, 1], n_subdivide + 1)
    x2_edges = np.linspace(region[1, 0], region[1, 1], n_subdivide + 1)

    all_cells = []
    zone_counts = {i: 0 for i in range(len(n_splits))}

    for i in range(n_subdivide):
        for j in range(n_subdivide):
            sub_rect = np.array([[x1_edges[i], x1_edges[i+1]],
                                [x2_edges[j], x2_edges[j+1]]], dtype=np.float32)

            # Rectangular distance: max(|x|, |y|) of closest point to origin
            # Closest point to origin: clamp origin coordinates to rectangle bounds
            x1_closest = np.clip(0.0, sub_rect[0, 0], sub_rect[0, 1])
            x2_closest = np.clip(0.0, sub_rect[1, 0], sub_rect[1, 1])
            rect_dist = max(abs(x1_closest), abs(x2_closest))

            # Determine n_split based on thresholds
            n_split = n_splits[-1]
            zone_idx = len(n_splits) - 1
            for k, threshold in enumerate(radius_thresholds):
                if rect_dist < threshold:
                    n_split = n_splits[k]
                    zone_idx = k
                    break

            zone_counts[zone_idx] += 1
            all_cells.extend(discretize_region(sub_rect, n_split))

    print(f"  Zone distribution across {n_subdivide}x{n_subdivide} = {n_subdivide**2} macro-cells:")
    for zone_idx, count in zone_counts.items():
        if zone_idx < len(radius_thresholds):
            print(f"    Zone {zone_idx} (dist < {radius_thresholds[zone_idx]}): {count} macro-cells with {n_splits[zone_idx]}x{n_splits[zone_idx]} subdivision")
        else:
            print(f"    Zone {zone_idx} (dist >= {radius_thresholds[-1]}): {count} macro-cells with {n_splits[zone_idx]}x{n_splits[zone_idx]} subdivision")

    return all_cells

def refine_cells_by_mask(cells, failing_mask, refinement_factor=2):
    """
    Refine cells marked as failing.

    Args:
        cells: list of (lower, upper) tuples
        failing_mask: boolean tensor indicating which cells are failing
        refinement_factor: split each failing cell into refinement_factor^2 subcells

    Returns:
        new_cells: list of refined cells
    """
    new_cells = []
    num_refined = 0

    for i, (cell_lower, cell_upper) in enumerate(cells):
        if failing_mask[i]:
            # Refine this failing cell
            cell_region = np.array([
                [cell_lower[0].item(), cell_upper[0].item()],
                [cell_lower[1].item(), cell_upper[1].item()]
            ], dtype=np.float32)
            refined = discretize_region(cell_region, refinement_factor)
            new_cells.extend(refined)
            num_refined += 1
        else:
            # Keep original cell
            new_cells.append((cell_lower, cell_upper))

    print(f"    Refined {num_refined} failing cells into {num_refined * refinement_factor**2} subcells")
    return new_cells

def compute_rectangular_partition_outside_goal(full_region, goal_region):
    """Compute rectangles covering full_region \ goal_region"""
    # Collect boundaries
    x_boundaries = set([full_region[0, 0], full_region[0, 1],
                        goal_region[0, 0], goal_region[0, 1]])
    y_boundaries = set([full_region[1, 0], full_region[1, 1],
                        goal_region[1, 0], goal_region[1, 1]])

    x_sorted = sorted(x_boundaries)
    y_sorted = sorted(y_boundaries)

    rectangles = []
    for i in range(len(x_sorted) - 1):
        for j in range(len(y_sorted) - 1):
            rect = np.array([[x_sorted[i], x_sorted[i+1]],
                           [y_sorted[j], y_sorted[j+1]]], dtype=np.float32)

            # Check if center is inside goal
            center_x = (rect[0, 0] + rect[0, 1]) / 2
            center_y = (rect[1, 0] + rect[1, 1]) / 2

            in_goal = (goal_region[0, 0] <= center_x <= goal_region[0, 1] and
                      goal_region[1, 0] <= center_y <= goal_region[1, 1])

            if not in_goal:
                rectangles.append(rect)

    return rectangles

def compute_rectangular_partition_outside_goal_and_unsafe(full_region, goal_region, unsafe_region):
    """Compute rectangles covering full_region \ (goal_region ∪ unsafe_region)"""
    # Collect all boundaries
    x_boundaries = set([full_region[0, 0], full_region[0, 1],
                        goal_region[0, 0], goal_region[0, 1],
                        unsafe_region[0, 0], unsafe_region[0, 1]])
    y_boundaries = set([full_region[1, 0], full_region[1, 1],
                        goal_region[1, 0], goal_region[1, 1],
                        unsafe_region[1, 0], unsafe_region[1, 1]])

    x_sorted = sorted(x_boundaries)
    y_sorted = sorted(y_boundaries)

    rectangles = []
    for i in range(len(x_sorted) - 1):
        for j in range(len(y_sorted) - 1):
            rect = np.array([[x_sorted[i], x_sorted[i+1]],
                           [y_sorted[j], y_sorted[j+1]]], dtype=np.float32)

            # Check if center is inside goal or unsafe
            center_x = (rect[0, 0] + rect[0, 1]) / 2
            center_y = (rect[1, 0] + rect[1, 1]) / 2

            in_goal = (goal_region[0, 0] <= center_x <= goal_region[0, 1] and
                      goal_region[1, 0] <= center_y <= goal_region[1, 1])
            in_unsafe = (unsafe_region[0, 0] <= center_x <= unsafe_region[0, 1] and
                        unsafe_region[1, 0] <= center_y <= unsafe_region[1, 1])

            if not in_goal and not in_unsafe:
                rectangles.append(rect)

    return rectangles

# ============================================================================
# MANUAL SOFTPLUS BOUNDS (bypasses auto_LiRPA)
# ============================================================================

def softplus_bounds(x_lower, x_upper, beta=1.0):
    """
    Compute linear relaxation bounds for Softplus: (1/beta)*log(1 + exp(beta*x))
    Returns: (lower_slope, lower_intercept, upper_slope, upper_intercept)
    """
    y_lower = torch.nn.functional.softplus(x_lower, beta=beta)
    y_upper = torch.nn.functional.softplus(x_upper, beta=beta)

    # Upper bound: secant line (overestimates convex function)
    delta = x_upper - x_lower
    delta = torch.where(delta < 1e-8, torch.ones_like(delta), delta)

    upper_slope = (y_upper - y_lower) / delta
    upper_intercept = y_lower - upper_slope * x_lower

    # Lower bound: tangent parallel to secant
    lower_slope = upper_slope
    z0 = -torch.log(1.0 / (lower_slope + 1e-8) - 1.0 + 1e-8)
    z0 = torch.clamp(z0, x_lower, x_upper)
    lower_intercept = torch.nn.functional.softplus(z0, beta=beta) - lower_slope * z0

    return lower_slope, lower_intercept, upper_slope, upper_intercept


def compute_bounds_manual_softplus(model, cells):
    """
    Manually compute bounds for network with Softplus activation (auto-detects layers).
    Works with any number of hidden layers.
    """
    if len(cells) == 0:
        return torch.tensor([]), torch.tensor([])

    input_lowers = torch.stack([cell[0] for cell in cells])
    input_uppers = torch.stack([cell[1] for cell in cells])

    with torch.no_grad():
        # Auto-detect hidden layers (layer1, layer2, layer3, ...)
        hidden_layers = []
        i = 1
        while hasattr(model, f'layer{i}'):
            hidden_layers.append(getattr(model, f'layer{i}'))
            i += 1

        # Start with input bounds
        a_L = input_lowers.T  # (input_dim, N)
        a_U = input_uppers.T

        # Propagate through hidden layers
        for layer in hidden_layers:
            W, b = layer.weight, layer.bias
            W_pos, W_neg = torch.clamp(W, min=0), torch.clamp(W, max=0)

            # Linear layer
            z_L = W_pos @ a_L + W_neg @ a_U + b.unsqueeze(1)
            z_U = W_pos @ a_U + W_neg @ a_L + b.unsqueeze(1)

            # Activation (Softplus bounds)
            s_L, i_L, s_U, i_U = softplus_bounds(z_L, z_U)
            a_L = s_L * z_L + i_L
            a_U = s_U * z_U + i_U

        # Output layer (no activation)
        W_out, b_out = model.output.weight, model.output.bias
        W_out_pos, W_out_neg = torch.clamp(W_out, min=0), torch.clamp(W_out, max=0)

        v_L = W_out_pos @ a_L + W_out_neg @ a_U + b_out.unsqueeze(1)
        v_U = W_out_pos @ a_U + W_out_neg @ a_L + b_out.unsqueeze(1)

    return v_L.squeeze(0), v_U.squeeze(0)


# ============================================================================
# BOUND COMPUTATION WITH CACHING
# ============================================================================

class _PhiModuleTrainable(nn.Module):
    """
    Computes Φ(x) = f·∇V + 0.5·(γ² ⊙ H_diag) using closed-form derivatives.

    Key difference from verify_crown_direct_2layer._Phi:
    - Uses REFERENCES to V_net's layers (not detached copies)
    - Gradients flow back to V_net's parameters during training
    - Compatible with auto_LiRPA's BoundedModule
    """
    def __init__(self, V_net, A, sigma, scale_factor=1.0):
        super().__init__()
        self.V_net = V_net
        self.scale_factor = scale_factor
        # Input normalization: maps [-INPUT_SCALE, INPUT_SCALE] -> [-1, 1]
        # Register as buffer so CROWN can properly track it in computational graph
        self.register_buffer('input_scale', torch.tensor(INPUT_SCALE))
        self.register_buffer('input_scale_sq', torch.tensor(INPUT_SCALE ** 2))

        # Convert A to tensor if needed
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
        self.register_buffer('A', A)  # (2, 2) drift matrix

        # Extract sigma from R matrix if needed
        if isinstance(sigma, np.ndarray):
            sigma = float(sigma[0, 0])
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Φ(x) using closed-form derivatives (no autograd.grad).

        For network: h0 = σ(W0·x_norm + b0), h1 = σ(W1·h0 + b1), V = W2·(SCALE·h1) + b2
        where x_norm = x / input_scale

        Derivatives:
            ∇V w.r.t x_norm = SCALE · W2·diag(σ'(z1))·W1·diag(σ'(z0))·W0
            ∇V w.r.t x = (1/input_scale) · ∇V w.r.t x_norm  (chain rule)
            H_ii w.r.t x = (1/input_scale²) · H_ii w.r.t x_norm  (chain rule)
        """
        # Store original x for dynamics computation (f(x) = A·x uses UNNORMALIZED coords)
        # Use clone() to avoid reference issues with CROWN bound propagation
        x_orig = x.clone()

        # Normalize input: [-100, 100] -> [-1, 1]
        x_norm = x / self.input_scale

        # Get weights from V_net (TRAINABLE - not detached!)
        W0 = self.V_net.layer1.weight  # (m0, 2)
        b0 = self.V_net.layer1.bias    # (m0,)
        W1 = self.V_net.layer2.weight  # (m1, m0)
        b1 = self.V_net.layer2.bias    # (m1,)
        W2 = self.V_net.output.weight  # (1, m1)
        b2 = self.V_net.output.bias    # (1,)

        # Forward pass through first hidden layer (using NORMALIZED input)
        z0 = F.linear(x_norm, W0, b0)     # (N, m0)
        h0 = torch.sigmoid(z0)             # (N, m0)
        d0 = h0 * (1.0 - h0)              # σ'(z0): (N, m0)
        q0 = (1.0 - 2.0 * h0) * d0        # σ''(z0): (N, m0)

        # Forward pass through second hidden layer
        z1 = F.linear(h0, W1, b1)         # (N, m1)
        h1 = torch.sigmoid(z1)             # (N, m1)
        d1 = h1 * (1.0 - h1)              # σ'(z1): (N, m1)
        q1 = (1.0 - 2.0 * h1) * d1        # σ''(z1): (N, m1)

        # Compute ∇V w.r.t normalized coordinates
        # ∂V/∂x_norm_i = SCALE · Σ_j W2[j]·σ'(z1[j])·(Σ_k W1[j,k]·σ'(z0[k])·W0[k,i])

        # Pre-compute common terms to avoid redundant operations
        # W1_scaled[j,k] = W1[j,k] · σ'(z0[k])  (shared for both x1 and x2)
        W1_scaled = W1.unsqueeze(0) * d0.unsqueeze(1)  # (N, m1, m0)

        # For x1 (i=0):
        sum_over_k1 = (W1_scaled * W0[:, 0].view(1, 1, -1)).sum(dim=2)  # (N, m1)
        dVdx1_norm = self.scale_factor * (W2 * d1 * sum_over_k1).sum(dim=1, keepdim=True)  # (N, 1)

        # For x2 (i=1):
        sum_over_k2 = (W1_scaled * W0[:, 1].view(1, 1, -1)).sum(dim=2)  # (N, m1)
        dVdx2_norm = self.scale_factor * (W2 * d1 * sum_over_k2).sum(dim=1, keepdim=True)  # (N, 1)

        # Apply chain rule: ∇V w.r.t x = (1/input_scale) · ∇V w.r.t x_norm
        dVdx1 = dVdx1_norm / self.input_scale  # (N, 1)
        dVdx2 = dVdx2_norm / self.input_scale  # (N, 1)

        # Compute Hessian diagonal w.r.t normalized coordinates
        # Pre-compute common term for Hessian: W1[j,k] · σ''(z0[k])
        W1_q0 = W1.unsqueeze(0) * q0.unsqueeze(1)  # (N, m1, m0)

        # For x1 (i=0):
        # Cross-term: σ''(z1[j]) · (Σ_k W1[j,k]·σ'(z0[k])·W0[k,0])²
        cross_term1 = q1 * (sum_over_k1 ** 2)  # (N, m1)

        # Direct term: σ'(z1[j]) · Σ_k W1[j,k]·σ''(z0[k])·W0[k,0]²
        W0_0_sq = W0[:, 0] ** 2  # Pre-compute squared weights
        direct_term1 = d1 * (W1_q0 * W0_0_sq.view(1, 1, -1)).sum(dim=2)  # (N, m1)

        H11_norm = self.scale_factor * (W2 * (cross_term1 + direct_term1)).sum(dim=1, keepdim=True)  # (N, 1)

        # For x2 (i=1):
        cross_term2 = q1 * (sum_over_k2 ** 2)  # (N, m1)
        W0_1_sq = W0[:, 1] ** 2  # Pre-compute squared weights
        direct_term2 = d1 * (W1_q0 * W0_1_sq.view(1, 1, -1)).sum(dim=2)  # (N, m1)

        H22_norm = self.scale_factor * (W2 * (cross_term2 + direct_term2)).sum(dim=1, keepdim=True)  # (N, 1)

        # Apply chain rule: H_ii w.r.t x = (1/input_scale²) · H_ii w.r.t x_norm
        H11 = H11_norm / self.input_scale_sq  # (N, 1)
        H22 = H22_norm / self.input_scale_sq  # (N, 1)

        # Compute f(x) = A·x using ORIGINAL UNNORMALIZED coordinates
        # (Dynamics operate on the physical state space, not the normalized NN input)
        fx = x_orig @ self.A.T  # (N, 2)
        f1 = fx[:, 0:1]    # (N, 1)
        f2 = fx[:, 1:2]    # (N, 1)

        # State-dependent diffusion: g(x) = σ·x using ORIGINAL UNNORMALIZED coordinates
        x1 = x_orig[:, 0:1]  # (N, 1)
        x2 = x_orig[:, 1:2]  # (N, 1)
        g11_sq = (self.sigma * x1) ** 2  # (N, 1)
        g22_sq = (self.sigma * x2) ** 2  # (N, 1)
        # g11_sq = (self.sigma)**2 * torch.ones_like(x[:, 0:1])
        # g22_sq = (self.sigma)**2 * torch.ones_like(x[:, 1:2])
        
        # Φ(x) = f·∇V + 0.5·(g²·H_diag)
        drift = f1 * dVdx1 + f2 * dVdx2
        diff = 0.5 * (g11_sq * H11 + g22_sq * H22)

        return drift + diff  # (N, 1)

class SymbolicCROWNCache_Phi:
    """
    Cache for symbolic CROWN bounds on Φ(x).
    Similar to SymbolicCROWNCache but wraps _PhiModuleTrainable instead of V.

    Computes differentiable bounds that can be used in training loss!
    """
    def __init__(self, V_net, A, sigma, num_cells, scale_factor=1.0, input_dim=2, device='cpu'):
        self.V_net = V_net
        self.num_cells = num_cells
        self.input_dim = input_dim
        self.device = device

        # Extract sigma from R matrix if needed
        if isinstance(sigma, np.ndarray):
            sigma = float(sigma[0, 0])

        print(f"[SymbolicCROWNCache_Phi] Creating Phi module with sigma={sigma}")

        # Create Phi module (trainable - uses V_net's weights)
        self.phi_module = _PhiModuleTrainable(V_net, A, sigma, scale_factor=scale_factor).to(device)

        # Save original training mode
        self.was_training = V_net.training
        V_net.eval()
        self.phi_module.eval()

        # Create dummy batch input
        dummy_batch = torch.zeros(num_cells, input_dim, dtype=torch.float32, device=device)

        # Create BoundedModule ONCE - this builds the symbolic computation graph
        print(f"[SymbolicCROWNCache_Phi] Creating BoundedModule for {num_cells} cells...")
        self.lirpa_model = BoundedModule(self.phi_module, dummy_batch[:1], device=device)

        # Initialize with dummy bounds
        dummy_lower = torch.zeros(num_cells, input_dim, device=device)
        dummy_upper = torch.ones(num_cells, input_dim, device=device)

        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=dummy_lower,
            x_U=dummy_upper
        )
        bounded_input = BoundedTensor(dummy_batch, ptb)

        # Forward pass to build symbolic computation graph
        _ = self.lirpa_model(bounded_input)

        # Compute bounds once to initialize symbolic structure
        _ = self.lirpa_model.compute_bounds(
            x=(bounded_input,),
            method='IBP',
            # method = 'alpha-CROWN',
            # IBP=True,
            bound_lower=True,
            bound_upper=True
        )

        print(f"[SymbolicCROWNCache_Phi] Initialized for {num_cells} cells - symbolic structure cached!")

    def compute_bounds(self, input_lowers, input_uppers):
        """
        Compute differentiable CROWN bounds on Φ(x) using cached symbolic structure.

        Args:
            input_lowers: (N, D) lower bounds on inputs
            input_uppers: (N, D) upper bounds on inputs

        Returns:
            phi_lowers: (N,) lower bounds on Φ(x)
            phi_uppers: (N,) upper bounds on Φ(x)
        """
        assert input_lowers.shape[0] == self.num_cells

        # Create dummy batch input (center of each cell)
        # Use in-place operations to reduce memory allocations
        dummy_batch = input_lowers.clone().to(self.device)
        dummy_batch.add_(input_uppers.to(self.device)).mul_(0.5)
        # dummy_batch = input_lowers.detach().to(self.device)
        # dummy_batch = dummy_batch.add_(input_uppers.detach().to(self.device)).mul_(0.5)

        # Create new perturbation with updated bounds
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=input_lowers.detach().to(self.device),
            x_U=input_uppers.detach().to(self.device)
        )
        bounded_input = BoundedTensor(dummy_batch, ptb)

        # Compute bounds directly - compute_bounds() does its own forward pass internally
        # No need for explicit model(bounded_input) call - that would be redundant
        lb, ub = self.lirpa_model.compute_bounds(
            x=(bounded_input,),
            method='IBP'
            # method = 'alpha-CROWN'
        )

        phi_lowers = lb.squeeze(-1)  # (N,)
        phi_uppers = ub.squeeze(-1)  # (N,)

        return phi_lowers, phi_uppers

    def __del__(self):
        """Restore model training mode on cleanup"""
        if hasattr(self, 'was_training') and self.was_training:
            self.V_net.train()

class SymbolicCROWNCache:
    """
    Cache for symbolic CROWN computation. Computes symbolic backward bounds once,
    then just numerically evaluates during training by plugging in new bound values.

    This is MUCH faster than creating a new BoundedModule every iteration!
    """
    def __init__(self, model, num_cells, input_dim=2):
        self.model = model
        self.num_cells = num_cells
        self.input_dim = input_dim

        # Save original training mode
        self.was_training = model.training
        model.eval()

        # Create dummy batch input
        dummy_batch = torch.zeros(num_cells, input_dim, dtype=torch.float32)

        # Create BoundedModule ONCE - this builds the symbolic computation graph
        self.lirpa_model = BoundedModule(model, dummy_batch[:1], device='cpu')

        # Initialize with dummy bounds
        dummy_lower = torch.zeros(num_cells, input_dim)
        dummy_upper = torch.ones(num_cells, input_dim)

        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=dummy_lower,
            x_U=dummy_upper
        )
        bounded_input = BoundedTensor(dummy_batch, ptb)

        # Forward pass to build symbolic computation graph
        _ = self.lirpa_model(bounded_input)

        # Compute bounds once to initialize symbolic structure
        _ = self.lirpa_model.compute_bounds(
            x=(bounded_input,),
            method='IBP',
            # IBP=False,
            forward=True,
            bound_lower=True,
            bound_upper=True
        )

        print(f"[SymbolicCROWNCache] Initialized for {num_cells} cells - symbolic structure cached!")

    def compute_bounds(self, input_lowers, input_uppers):
        """
        Numerically evaluate bounds by plugging in new bound values.
        Uses the cached symbolic structure - NO symbolic re-propagation!
        """
        assert input_lowers.shape[0] == self.num_cells

        # Create dummy batch input (center of each cell)
        dummy_batch = ((input_lowers + input_uppers) * 0.5).detach()

        # Create new perturbation with updated bounds
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=input_lowers.detach(),
            x_U=input_uppers.detach()
        )
        bounded_input = BoundedTensor(dummy_batch, ptb)

        # Compute bounds directly - compute_bounds() does its own forward pass internally
        # No need for explicit model(bounded_input) call - that would be redundant
        lb, ub = self.lirpa_model.compute_bounds(
            x=(bounded_input,),
            method='IBP',
            # IBP=False,
            forward=True,
            bound_lower=True,
            bound_upper=True
        )

        v_lowers = lb.squeeze(-1)
        v_uppers = ub.squeeze(-1)

        return v_lowers, v_uppers

    def __del__(self):
        """Restore model training mode on cleanup"""
        if hasattr(self, 'was_training') and self.was_training:
            self.model.train()


def compute_bounds_differentiable_batch(model, cells):
    """Compute differentiable bounds for multiple cells using CROWN"""
    if len(cells) == 0:
        return torch.tensor([]), torch.tensor([])

    input_lowers = torch.stack([cell[0] for cell in cells])  # (N, D)
    input_uppers = torch.stack([cell[1] for cell in cells])  # (N, D)

    # Center point for each cell
    x0 = ((input_lowers + input_uppers) / 2).detach()

    # Create BoundedModule with single dummy input
    lirpa_model = BoundedModule(model, torch.zeros(1, input_lowers.shape[1]), device='cpu')

    # Perturbation specification
    ptb = PerturbationLpNorm(
        norm=np.inf,
        eps=None,
        x_L=input_lowers.detach(),
        x_U=input_uppers.detach()
    )
    bounded_input = BoundedTensor(x0, ptb)

    # Compute bounds using CROWN method
    lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method="CROWN")

    v_lowers = lb.squeeze(-1)  # (N,)
    v_uppers = ub.squeeze(-1)  # (N,)

    return v_lowers, v_uppers

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def loss_goal_sampled(model, goal_region, v_lowers, show, n_samples=10):
    """
    Goal: ∃x ∈ goal s.t. V(x) < beta_s
    SOFT constraint: Sample points in goal and minimize V at those points
    This avoids the boundary contradiction with the hard constraint
    """
    # Sample random points in goal region
    x1_samples = torch.rand(n_samples) * (goal_region[0, 1] - goal_region[0, 0]) + goal_region[0, 0]
    x2_samples = torch.rand(n_samples) * (goal_region[1, 1] - goal_region[1, 0]) + goal_region[1, 0]
    samples = torch.stack([x1_samples, x2_samples], dim=1)

    # Evaluate V at samples
    v_samples = model(samples).squeeze()

    # Minimize V at these points (push toward 0, but penalize going below 0)
    # Want: some V < beta_s, and all V >= 0
    
    
    # loss_soft = torch.relu(v_samples - BETA_S_GOAL).mean()  # Push below beta_s
    loss_soft = torch.mean(v_samples)
    # loss_soft = torch.sum(v_samples)


    # loss_nonneg = torch.relu(-v_samples).mean()  # Keep >= 0
    # loss_nonneg = torch.relu(-v_lowers).sum()

    if (v_samples.min() < BETA_S).item() and (v_lowers.min() >= 0).item():
        passed = True
    else:
        if show:
            print(f"  ✗ Goal sample failed: min V = {v_samples.min().item():.4f}, min V_lower = {v_lowers.min().item():.4f}")
        passed = False

    # return loss_soft + loss_nonneg, passed
    return loss_soft, passed

def loss_outside_goal_bounds(v_lowers, v_uppers):
    """
    Outside goal: V(x) >= beta_s for all x
    """
    loss_lower = torch.relu(BETA_S - v_lowers).sum()
    return loss_lower

def loss_unsafe_bounds(v_lowers, v_uppers):
    """
    Outside goal: V(x) >= beta_s for all x
    """
    loss_lower = torch.relu(BETA_RA - v_lowers).sum()

    # print(f'  Number of unsafe losses: {len(torch.relu(BETA_RA - v_lowers) > 0)}')
    return loss_lower

def loss_init_bounds(v_lowers, v_uppers):
    """
    Loss for init region: We want 0.9 <= V < 1

    Init must satisfy TWO constraints:
    1. V < 1.0 (its own upper bound)
    2. V >= 0.9 (general constraint that applies everywhere except goal)

    Minimize:
    - RELU(V_upper - 1) to push V_upper <= 1.0
    - RELU(0.9 - V_lower) to push V_lower >= 0.9

    v_lowers and v_uppers should have gradients (from differentiable bound propagation)
    """
    loss_upper = torch.relu(v_uppers - 1.0).sum()
    loss_lower = torch.relu(BETA_S - v_lowers).sum()
    return loss_upper * 1.0 + loss_lower * 1.0

def compute_gradient_and_hessian_fast(model, x):
    """
    FAST version using torch.func for vectorized gradient/Hessian computation.

    PERFORMANCE: ~3-5x faster than the manual autograd version.

    WHY IT'S FASTER:
    - torch.func uses optimized functional transformations (vmap, jacrev, hessian)
    - vmap vectorizes operations across batch dimension (no Python loops)
    - jacrev computes Jacobian with reverse-mode AD (optimized for scalar outputs)
    - Avoids creating/destroying computation graphs repeatedly

    FURTHER OPTIMIZATIONS:
    1. Reduce generator loss frequency: Only compute every N epochs (e.g., N=5)
    2. Use torch.compile() (PyTorch 2.0+): Compile this function for ~2x additional speedup
    3. Reduce n_samples: Use fewer samples per epoch (e.g., 50 instead of 100)
    4. Use mixed precision (torch.autocast): Trade accuracy for speed on GPU

    Args:
        model: neural network
        x: (N, 2) input tensor with requires_grad=True

    Returns:
        grad_v: (N, 2) gradient [∂V/∂x1, ∂V/∂x2]
        hess_diag: (N, 2) Hessian diagonal [∂²V/∂x1², ∂²V/∂x2²]
    """
    # Define function that takes single input and returns scalar
    def v_func(x_single):
        return model(x_single.unsqueeze(0)).squeeze()

    # Compute Jacobian (gradient) for each sample in batch
    # jacrev returns (N, 1, 2) -> squeeze to (N, 2)
    grad_v = vmap(jacrev(v_func))(x).squeeze(1)  # (N, 2)

    # Compute Hessian diagonal for each sample
    def hess_func(x_single):
        h = hessian(v_func)(x_single)  # (2, 2)
        return torch.diag(h)  # Extract diagonal

    hess_diag = vmap(hess_func)(x)  # (N, 2)

    return grad_v, hess_diag

def loss_generator_sampled(model, A, R, x_goal_range, x_unsafe_range, x_range, n_samples=500):
    """
    FAST sampling-based generator loss (alternative to CROWN).

    Samples points from generator region and computes Φ directly using autograd.
    Much faster than CROWN (~50x) but less rigorous (point-wise instead of bounds).

    Args:
        model: neural network
        A: drift matrix
        R: diffusion matrix (or sigma scalar)
        x_goal_range, x_unsafe_range, x_range: region definitions
        n_samples: number of points to sample

    Returns:
        loss: mean positive Φ violation
    """
    # Sample points uniformly from state space
    x = torch.rand(n_samples, 2)
    x[:, 0] = x[:, 0] * (x_range[0, 1] - x_range[0, 0]) + x_range[0, 0]
    x[:, 1] = x[:, 1] * (x_range[1, 1] - x_range[1, 0]) + x_range[1, 0]

    # Filter to generator region (X \ (Goal ∪ Unsafe))
    in_goal = ((x[:, 0] >= x_goal_range[0, 0]) & (x[:, 0] <= x_goal_range[0, 1]) &
               (x[:, 1] >= x_goal_range[1, 0]) & (x[:, 1] <= x_goal_range[1, 1]))
    in_unsafe = ((x[:, 0] >= x_unsafe_range[0, 0]) & (x[:, 0] <= x_unsafe_range[0, 1]) &
                 (x[:, 1] >= x_unsafe_range[1, 0]) & (x[:, 1] <= x_unsafe_range[1, 1]))

    in_generator = ~(in_goal | in_unsafe)
    x_gen = x[in_generator]

    if len(x_gen) == 0:
        return torch.tensor(0.0)

    # Enable gradients for Φ computation
    x_gen.requires_grad_(True)

    # Compute V
    v = model(x_gen)

    # Compute ∇V using autograd
    grad_v = torch.autograd.grad(v.sum(), x_gen, create_graph=True)[0]  # (N, 2)

    # Compute Hessian diagonal (approximation using finite differences is faster)
    # Or use the fast function
    grad_v_full, hess_diag = compute_gradient_and_hessian_fast(model, x_gen)

    # Extract sigma
    if isinstance(R, np.ndarray):
        sigma = R[0, 0]
    else:
        sigma = R

    # Compute drift term: f·∇V where f = A·x
    if isinstance(A, np.ndarray):
        A_torch = torch.from_numpy(A).float()
    else:
        A_torch = A

    f = x_gen @ A_torch.T  # (N, 2)
    drift_term = (f * grad_v_full).sum(dim=1)  # (N,)

    # Compute diffusion term: 0.5·g²·H_diag where g = σ·x (state-dependent)
    g_sq = (sigma * x_gen) ** 2  # (N, 2)
    diffusion_term = 0.5 * (g_sq * hess_diag).sum(dim=1)  # (N,)

    # Φ(x) = drift + diffusion
    phi = drift_term + diffusion_term

    # Loss: penalize positive Φ
    loss = torch.nn.functional.relu(phi + 0.0).mean()

    return loss

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_network_output(model, region, title="Network Output",
                            show_regions=False, x_init=None, x_unsafe=None, x_goal=None,
                            v_lower_range=None, v_upper_range=None, training_cells=None,
                            filename=None, show_discretization=False):
    """
    Visualize the network output over the region as a heatmap.

    Args:
        model: neural network
        region: region to visualize
        title: plot title
        show_regions: whether to overlay region boundaries
        x_init: init region (if show_regions=True)
        x_unsafe: unsafe region (if show_regions=True)
        x_goal: goal region (if show_regions=True)
        v_lower_range: tuple of (min, max) for V_lower bounds from CROWN (optional)
        v_upper_range: tuple of (min, max) for V_upper bounds from CROWN (optional)
        training_cells: list of (lower, upper) tuples for training cells (optional)
    """
    from matplotlib.patches import Rectangle

    # Always create fine uniform grid for smooth visualization
    x1_vals = np.linspace(region[0, 0], region[0, 1], 100)
    x2_vals = np.linspace(region[1, 0], region[1, 1], 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Flatten and create input
    x1_flat = X1.flatten()
    x2_flat = X2.flatten()
    x_grid = torch.tensor(np.stack([x1_flat, x2_flat], axis=1), dtype=torch.float32)

    # Evaluate network on fine grid
    model.eval()
    with torch.no_grad():
        output = model(x_grid).numpy().reshape(X1.shape)

    # Get actual output range from fine grid
    output_min = output.min()
    output_max = output.max()

    # If training cells are provided, also sample at cell corners to capture extremes
    if training_cells is not None:
        corner_samples = []
        for cell_lower, cell_upper in training_cells:
            # Sample all 4 corners of each cell
            corners = [
                [cell_lower[0].item(), cell_lower[1].item()],  # bottom-left
                [cell_lower[0].item(), cell_upper[1].item()],  # top-left
                [cell_upper[0].item(), cell_lower[1].item()],  # bottom-right
                [cell_upper[0].item(), cell_upper[1].item()],  # top-right
            ]
            corner_samples.extend(corners)

        # Evaluate at corners
        corner_samples = np.array(corner_samples)
        corner_grid = torch.tensor(corner_samples, dtype=torch.float32)
        with torch.no_grad():
            corner_outputs = model(corner_grid).numpy().flatten()

        # Update min/max with corner samples
        corner_min = corner_outputs.min()
        corner_max = corner_outputs.max()
        output_min = min(output_min, corner_min)
        output_max = max(output_max, corner_max)

    # Use provided CROWN bounds for colormap if available
    if v_lower_range is not None and v_upper_range is not None:
        vmin = v_lower_range[0]  # min of V_lower
        vmax = v_upper_range[1]  # max of V_upper
        range_info = f"\nActual: [{output_min:.4f}, {output_max:.4f}] | CROWN: [{vmin:.4f}, {vmax:.4f}]"
    else:
        vmin = output_min
        vmax = output_max
        range_info = f"\nOutput Range: [{vmin:.4f}, {vmax:.4f}]"

    # Plot as contour
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X1, X2, output, levels=20, cmap='PiYG', vmin=vmin, vmax=vmax)
    plt.colorbar(contour, label='V(x1, x2)')

    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(f"{title}{range_info}", fontsize=14, fontweight='bold')

    # Overlay region rectangles if requested
    if show_regions:
        # Draw x_init region (green)
        if x_init is not None:
            x_init_rect = Rectangle(
                (x_init[0, 0], x_init[1, 0]),
                x_init[0, 1] - x_init[0, 0],
                x_init[1, 1] - x_init[1, 0],
                linewidth=3, edgecolor='green', facecolor='none', label='Init'
            )
            plt.gca().add_patch(x_init_rect)

        # Draw x_unsafe region (red)
        if x_unsafe is not None:
            x_unsafe_rect = Rectangle(
                (x_unsafe[0, 0], x_unsafe[1, 0]),
                x_unsafe[0, 1] - x_unsafe[0, 0],
                x_unsafe[1, 1] - x_unsafe[1, 0],
                linewidth=3, edgecolor='red', facecolor='none', label='Unsafe'
            )
            plt.gca().add_patch(x_unsafe_rect)

        # Draw x_goal region (blue)
        if x_goal is not None:
            x_goal_rect = Rectangle(
                (x_goal[0, 0], x_goal[1, 0]),
                x_goal[0, 1] - x_goal[0, 0],
                x_goal[1, 1] - x_goal[1, 0],
                linewidth=3, edgecolor='blue', facecolor='none', label='Goal'
            )
            plt.gca().add_patch(x_goal_rect)

        plt.legend(loc='upper right', fontsize=10)

    # Draw discretization grid if requested
    if show_discretization and training_cells is not None:
        for cell_lower, cell_upper in training_cells:
            # Draw rectangle for each cell
            from matplotlib.patches import Rectangle
            cell_rect = Rectangle(
                (cell_lower[0].item(), cell_lower[1].item()),
                cell_upper[0].item() - cell_lower[0].item(),
                cell_upper[1].item() - cell_lower[1].item(),
                linewidth=0.5, edgecolor='black', facecolor='none', alpha=0.5
            )
            plt.gca().add_patch(cell_rect)

    plt.tight_layout()

    # Auto-generate filename based on title if not provided
    if filename is None:
        # Extract region name from title
        if "Full State Space" in title:
            filename = "network_output_simple_full.png"
        elif "Init" in title:
            filename = "network_output_simple_init.png"
        elif "Goal" in title:
            filename = "network_output_simple_goal.png"
        elif "Unsafe" in title:
            filename = "network_output_simple_unsafe.png"
        else:
            filename = "network_output_simple.png"

    plt.savefig(filename, dpi=150)
    print(f"  → Saved to '{filename}'")
    plt.close()

def visualize_gv_output(model, region, A_matrix, R_matrix, title="GV Output",
                        show_regions=False, x_init=None, x_unsafe=None, x_goal=None,
                        phi_lower_range=None, phi_upper_range=None, training_cells=None,
                        filename=None, show_discretization=False, phi_module=None):
    """
    Visualize the GV (Φ) function over the region as a heatmap.

    Φ(x) = f(x)ᵀ∇V(x) + 0.5·Tr(g(x)g(x)ᵀH_V(x))

    where:
    - f(x) = Ax (drift term)
    - g(x) = Rx (diffusion term)
    - ∇V(x) = gradient of V
    - H_V(x) = Hessian of V

    Args:
        model: neural network for V(x)
        region: region to visualize
        A_matrix: drift matrix (numpy array or torch tensor)
        R_matrix: diffusion matrix (numpy array or torch tensor)
        title: plot title
        show_regions: whether to overlay region boundaries
        x_init: init region (if show_regions=True)
        x_unsafe: unsafe region (if show_regions=True)
        x_goal: goal region (if show_regions=True)
        phi_lower_range: tuple of (min, max) for Φ_lower bounds from CROWN (optional)
        phi_upper_range: tuple of (min, max) for Φ_upper bounds from CROWN (optional)
        training_cells: list of (lower, upper) tuples for training cells (optional)
        filename: output filename (optional)
        show_discretization: whether to show cell boundaries
        phi_module: optional _PhiModuleTrainable to use for computing Φ (uses same computation as training)
    """
    from matplotlib.patches import Rectangle

    # Always create fine uniform grid for smooth visualization
    x1_vals = np.linspace(region[0, 0], region[0, 1], 10)
    x2_vals = np.linspace(region[1, 0], region[1, 1], 10)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Flatten and create input
    x1_flat = X1.flatten()
    x2_flat = X2.flatten()
    x_grid = torch.tensor(np.stack([x1_flat, x2_flat], axis=1), dtype=torch.float32)
    x_grid.requires_grad_(True)

    # Evaluate GV (Φ) on fine grid
    model.eval()

    if phi_module is not None:
        # Use the provided phi_module (same computation as training!)
        print(f"[visualize_gv_output] Using phi_module for computation (same as training)")
        phi_module.eval()
        with torch.no_grad():
            phi_output = phi_module(x_grid).detach().numpy()  # (N, 1)
            phi = phi_output.squeeze().reshape(X1.shape)  # Squeeze to (N,) then reshape
            print(f"[visualize_gv_output] Φ range: [{phi.min():.4f}, {phi.max():.4f}]")
    else:
        # Fallback: compute Φ manually using gradient and Hessian
        # Convert A_matrix and R_matrix to torch tensors if they're numpy arrays
        if isinstance(A_matrix, np.ndarray):
            A_torch = torch.from_numpy(A_matrix).float()
        else:
            A_torch = A_matrix.float() if A_matrix.dtype != torch.float32 else A_matrix

        if isinstance(R_matrix, np.ndarray):
            R_torch = torch.from_numpy(R_matrix).float()
            sigma = R_matrix[0, 0]  # Extract sigma from numpy
        else:
            R_torch = R_matrix.float() if R_matrix.dtype != torch.float32 else R_matrix
            sigma = R_matrix[0, 0].item()  # Extract sigma from tensor

        # Compute f(x) = Ax
        fx = x_grid @ A_torch.T  # (N, 2)
        f1 = fx[:, 0:1]
        f2 = fx[:, 1:2]

        # Compute g(x) = Rx (assuming diagonal R for simplicity: g_ii = R_ii * x_i)
        x1 = x_grid[:, 0:1]
        x2 = x_grid[:, 1:2]
        g11 = sigma * x1
        g22 = sigma * x2

        # Compute gradient and Hessian
        grad_v, hess_diag = compute_gradient_and_hessian_fast(model, x_grid)
        dVdx1 = grad_v[:, 0:1]
        dVdx2 = grad_v[:, 1:2]
        H11 = hess_diag[:, 0:1]
        H22 = hess_diag[:, 1:2]

        # Compute Φ = drift + diffusion
        drift_term = f1 * dVdx1 + f2 * dVdx2
        diffusion_term = 0.5 * (g11**2 * H11 + g22**2 * H22)
        phi = (drift_term + diffusion_term).detach().numpy().reshape(X1.shape)

    # Get actual output range from fine grid
    output_min = phi.min()
    output_max = phi.max()

    # If training cells are provided, also sample at cell corners to capture extremes
    if training_cells is not None:
        corner_samples = []
        for cell_lower, cell_upper in training_cells:
            # Sample all 4 corners of each cell
            corners = [
                [cell_lower[0].item(), cell_lower[1].item()],  # bottom-left
                [cell_lower[0].item(), cell_upper[1].item()],  # top-left
                [cell_upper[0].item(), cell_lower[1].item()],  # bottom-right
                [cell_upper[0].item(), cell_upper[1].item()],  # top-right
            ]
            corner_samples.extend(corners)

        # Evaluate at corners
        corner_samples = np.array(corner_samples)
        corner_grid = torch.tensor(corner_samples, dtype=torch.float32)
        corner_grid.requires_grad_(True)

        if phi_module is not None:
            # Use phi_module for corners too
            with torch.no_grad():
                phi_corner = phi_module(corner_grid).detach().numpy().flatten()
        else:
            # Compute Φ at corners manually
            fx_corner = corner_grid @ A_torch.T
            f1_corner = fx_corner[:, 0:1]
            f2_corner = fx_corner[:, 1:2]

            x1_corner = corner_grid[:, 0:1]
            x2_corner = corner_grid[:, 1:2]
            g11_corner = sigma * x1_corner
            g22_corner = sigma * x2_corner

            grad_v_corner, hess_diag_corner = compute_gradient_and_hessian_fast(model, corner_grid)
            dVdx1_corner = grad_v_corner[:, 0:1]
            dVdx2_corner = grad_v_corner[:, 1:2]
            H11_corner = hess_diag_corner[:, 0:1]
            H22_corner = hess_diag_corner[:, 1:2]

            drift_corner = f1_corner * dVdx1_corner + f2_corner * dVdx2_corner
            diffusion_corner = 0.5 * (g11_corner**2 * H11_corner + g22_corner**2 * H22_corner)
            phi_corner = (drift_corner + diffusion_corner).detach().numpy().flatten()

        # Update min/max with corner samples
        corner_min = phi_corner.min()
        corner_max = phi_corner.max()
        output_min = min(output_min, corner_min)
        output_max = max(output_max, corner_max)

    # Use provided CROWN bounds for colormap if available
    if phi_lower_range is not None and phi_upper_range is not None:
        print('aaa')
        vmin = phi_lower_range[0]  # min of Φ_lower
        vmax = phi_upper_range[1]  # max of Φ_upper
        range_info = f"\nCROWN for GV: [{vmin:.4f}, {vmax:.4f}]"
    else:
        vmin = output_min
        vmax = output_max
        range_info = f"\nOutput Range: [{vmin:.4f}, {vmax:.4f}]"

    # Plot as contour
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X1, X2, phi, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(contour, label='Φ(x1, x2)')

    # Add zero contour line (critical boundary where Φ = 0)
    plt.contour(X1, X2, phi, levels=[0], colors='black', linewidths=2, linestyles='--')

    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(f"{title}{range_info}", fontsize=14, fontweight='bold')

    # Overlay region rectangles if requested
    if show_regions:
        # Draw x_init region (green)
        if x_init is not None:
            x_init_rect = Rectangle(
                (x_init[0, 0], x_init[1, 0]),
                x_init[0, 1] - x_init[0, 0],
                x_init[1, 1] - x_init[1, 0],
                linewidth=3, edgecolor='green', facecolor='none', label='Init'
            )
            plt.gca().add_patch(x_init_rect)

        # Draw x_unsafe region (red)
        if x_unsafe is not None:
            x_unsafe_rect = Rectangle(
                (x_unsafe[0, 0], x_unsafe[1, 0]),
                x_unsafe[0, 1] - x_unsafe[0, 0],
                x_unsafe[1, 1] - x_unsafe[1, 0],
                linewidth=3, edgecolor='red', facecolor='none', label='Unsafe'
            )
            plt.gca().add_patch(x_unsafe_rect)

        # Draw x_goal region (blue)
        if x_goal is not None:
            x_goal_rect = Rectangle(
                (x_goal[0, 0], x_goal[1, 0]),
                x_goal[0, 1] - x_goal[0, 0],
                x_goal[1, 1] - x_goal[1, 0],
                linewidth=3, edgecolor='blue', facecolor='none', label='Goal'
            )
            plt.gca().add_patch(x_goal_rect)

        plt.legend(loc='upper right', fontsize=10)

    # Draw discretization grid if requested
    if show_discretization and training_cells is not None:
        for cell_lower, cell_upper in training_cells:
            # Draw rectangle for each cell
            from matplotlib.patches import Rectangle
            cell_rect = Rectangle(
                (cell_lower[0].item(), cell_lower[1].item()),
                cell_upper[0].item() - cell_lower[0].item(),
                cell_upper[1].item() - cell_lower[1].item(),
                linewidth=0.5, edgecolor='black', facecolor='none', alpha=0.5
            )
            plt.gca().add_patch(cell_rect)

    plt.tight_layout()

    # Auto-generate filename based on title if not provided
    if filename is None:
        # Extract region name from title
        if "Full State Space" in title:
            filename = "gv_output_simple_full.png"
        elif "Init" in title:
            filename = "gv_output_simple_init.png"
        elif "Goal" in title:
            filename = "gv_output_simple_goal.png"
        elif "Unsafe" in title:
            filename = "gv_output_simple_unsafe.png"
        elif "Generator" in title:
            filename = "gv_output_simple_generator.png"
        else:
            filename = "gv_output_simple.png"

    plt.savefig(filename, dpi=150)
    print(f"  → Saved to '{filename}'")
    plt.close()

# ============================================================================
# TRAINING
# ============================================================================

def pretrain_structure_aware(model, x_goal_range, x_unsafe_range, x_init_range, x_range,
                              A=None, R=None, scale_factor=1.0, num_epochs=1000, lr=0.01):
    """
    Pre-train V and GV (Φ) jointly to match constraint structure:

    V constraints:
    - Goal: V ≈ 0.4 (below BETA_S = 0.9)
    - Unsafe: V ≈ 12 (above BETA_RA = 10)
    - Init: V ≈ 0.95 (between BETA_S and 1.0)
    - Outside: V ≈ 1.2 (above BETA_S)

    GV (Φ) constraint (if A and R provided):
    - Generator region (X \ (Goal ∪ Unsafe)): Φ ≤ 0

    This gives the network a good starting point that respects both constraint structures.
    """
    print("\n" + "="*80)
    print("PRE-TRAINING: Structure-Aware Initialization (V + GV)")
    print("="*80)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create Phi module if dynamics are provided
    phi_module = None
    if A is not None and R is not None:
        sigma = R[0, 0] if isinstance(R, np.ndarray) else R
        phi_module = _PhiModuleTrainable(model, A, sigma, scale_factor=scale_factor)
        print(f"  GV (Φ) pre-training ENABLED with sigma={sigma}")
    else:
        print(f"  GV (Φ) pre-training DISABLED (no dynamics provided)")

    for epoch in range(num_epochs):
        # Sample points uniformly across state space
        x = torch.rand(1000, 2)
        x[:, 0] = x[:, 0] * (x_range[0, 1] - x_range[0, 0]) + x_range[0, 0]
        x[:, 1] = x[:, 1] * (x_range[1, 1] - x_range[1, 0]) + x_range[1, 0]

        # ===== V LOSS =====
        # Define target V based on region
        target_v = torch.zeros(1000)

        for i in range(1000):
            x1, x2 = x[i, 0].item(), x[i, 1].item()

            # Check which region this point is in
            in_goal = (x_goal_range[0, 0] <= x1 <= x_goal_range[0, 1] and
                      x_goal_range[1, 0] <= x2 <= x_goal_range[1, 1])

            in_unsafe = (x_unsafe_range[0, 0] <= x1 <= x_unsafe_range[0, 1] and
                        x_unsafe_range[1, 0] <= x2 <= x_unsafe_range[1, 1])

            in_init = (x_init_range[0, 0] <= x1 <= x_init_range[0, 1] and
                      x_init_range[1, 0] <= x2 <= x_init_range[1, 1])

            if in_goal:
                # Goal: want V small (< BETA_S = 0.9)
                target_v[i] = 0.3 + 0.2 * torch.rand(1).item()  # Random in [0.3, 0.5]
            elif in_unsafe:
                # Unsafe: want V large (> BETA_RA = 10)
                target_v[i] = 12.0 + 3.0 * torch.rand(1).item()  # Random in [12, 15]
            elif in_init:
                # Init: want BETA_S < V < 1.0
                target_v[i] = 0.92 + 0.05 * torch.rand(1).item()  # Random in [0.92, 0.97]
            else:
                # Outside goal: want V > BETA_S
                target_v[i] = 1.1 + 0.3 * torch.rand(1).item()  # Random in [1.1, 1.4]

        # Forward pass for V
        v_output = model(x).squeeze()

        # V MSE Loss
        loss_v = F.mse_loss(v_output, target_v)

        # ===== GV (Φ) LOSS =====
        loss_phi = torch.tensor(0.0)
        if phi_module is not None:
            # Sample points from generator region (X \ (Goal ∪ Unsafe))
            x_gen = torch.rand(100, 2)
            x_gen[:, 0] = x_gen[:, 0] * (x_range[0, 1] - x_range[0, 0]) + x_range[0, 0]
            x_gen[:, 1] = x_gen[:, 1] * (x_range[1, 1] - x_range[1, 0]) + x_range[1, 0]

            # Filter out points in goal or unsafe
            in_goal_mask = ((x_gen[:, 0] >= x_goal_range[0, 0]) & (x_gen[:, 0] <= x_goal_range[0, 1]) &
                           (x_gen[:, 1] >= x_goal_range[1, 0]) & (x_gen[:, 1] <= x_goal_range[1, 1]))
            in_unsafe_mask = ((x_gen[:, 0] >= x_unsafe_range[0, 0]) & (x_gen[:, 0] <= x_unsafe_range[0, 1]) &
                             (x_gen[:, 1] >= x_unsafe_range[1, 0]) & (x_gen[:, 1] <= x_unsafe_range[1, 1]))

            in_generator = ~(in_goal_mask | in_unsafe_mask)
            x_gen_filtered = x_gen[in_generator]

            if len(x_gen_filtered) > 0:
                # Compute Φ
                phi_output = phi_module(x_gen_filtered).squeeze()

                # Loss: penalize positive Φ (want Φ ≤ 0 in generator region)
                loss_phi = torch.nn.functional.relu(phi_output + 0.0).mean()

        # Combined loss (weight V more heavily initially)
        total_loss = loss_v + 1.0 * loss_phi  # Start with low GV weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            if phi_module is not None:
                print(f"  Pre-train Epoch [{epoch}/{num_epochs}]: V_loss = {loss_v.item():.6f}, Φ_loss = {loss_phi.item():.6f}, Total = {total_loss.item():.6f}")
            else:
                print(f"  Pre-train Epoch [{epoch}/{num_epochs}]: V_loss = {loss_v.item():.6f}")

    print("="*80)
    if phi_module is not None:
        print("Pre-training complete. V and GV (Φ) now have structure matching constraints.")
    else:
        print("Pre-training complete. V now has structure matching constraints.")
    print("="*80 + "\n")

def train_network(model, regions_dict, region_cells, num_epochs, lr, A=None, R=None, generator_weight=1.0, generator_start_epoch=1000,
                  crown_bounds_start_epoch=180000, enable_crown_verification=False, crown_verify_epoch_interval=500, crown_refine_N=2, crown_max_depth=2, scale_factor=1.0):
    """
    Train the network using regions_dict approach

    Args:
        model: neural network
        regions_dict: dictionary of regions and loss functions
        region_cells: dictionary of discretized cells per region

    Returns:
        tuple: (verified_boxes, verified_bounds) if CROWN verification succeeded, else (None, None)
        num_epochs: number of training epochs
        lr: learning rate
        A: (2, 2) drift matrix for generator condition (optional)
        R: (2, 2) diffusion matrix for generator condition (optional)
        generator_weight: weight for generator loss (0 = disabled)
        generator_start_epoch: epoch to start applying generator loss (curriculum learning)
        crown_bounds_start_epoch: epoch to switch from sampling to CROWN bounds for GV (default: 180000, ~90% of 200k)
        enable_crown_verification: if True, run CROWN verification when all constraints satisfied
        crown_verify_epoch_interval: run CROWN verification every N epochs (when constraints satisfied)
        crown_refine_N: refinement factor for CROWN verification (2 = 2x2 split)
        crown_max_depth: maximum CROWN refinement iterations (early stop for training feedback)
        scale_factor: SCALE_FACTOR from network architecture
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # minimize the loss
        factor=0.5,           # reduce LR by half when plateau detected
        patience=100,         # wait 100 epochs of no improvement before reducing
        verbose=True,         # print when LR changes
        min_lr=1e-6,          # minimum learning rate
        threshold=1e-3        # minimum change to qualify as improvement
    )

    # Detect activation type
    use_manual_softplus = isinstance(model.activation_fn, nn.Softplus)

    print(f"\nActivation: {model.activation_fn.__class__.__name__}")
    print(f"Bound method: {'Manual Softplus bounds' if use_manual_softplus else 'CROWN (auto_LiRPA)'}")
    print(f"\nRegions and cells:")
    for name, cells in region_cells.items():
        print(f"  {name}: {len(cells)} cells")
    print(f"\nEpochs: {num_epochs}, LR: {lr}\n")

    # Collect all cells
    all_cells = []
    cell_region_map = []
    for name in regions_dict.keys():
        cells = region_cells[name]
        all_cells.extend(cells)
        cell_region_map.extend([name] * len(cells))

    # Only create CROWN cache if not using manual softplus
    crown_cache = None
    if not use_manual_softplus:
        print(f"*** Creating SymbolicCROWNCache for {len(all_cells)} cells ***")
        print(f'    Number of cells: {len(all_cells)}')
        crown_cache = SymbolicCROWNCache(model, num_cells=len(all_cells), input_dim=2)
        print()

    # Create CROWN caches for generator (if enabled)
    crown_phi_cache = None  # For training (differentiable)
    dynamics_const = None  # Dynamics constants for verification

    if enable_crown_verification and A is not None and R is not None:
        # Create dynamics constants for verification
        class DynamicsConstants:
            def __init__(self, A, R, x_range, x_goal_range, x_unsafe_range):
                self.X_RANGE = x_range
                self.X_GOAL_RANGE = x_goal_range
                self.X_UNSAFE_RANGE = x_unsafe_range

                if isinstance(A, np.ndarray):
                    A = torch.from_numpy(A).float()
                if isinstance(R, np.ndarray):
                    R = torch.from_numpy(R).float()

                self.A = A.float()
                self.sigma = R[0, 0]  # σ for g(x) = σ·x

            def f_of_x(self, x: torch.Tensor) -> tuple:
                fx = x @ self.A.T
                return fx[:, 0:1], fx[:, 1:2]

            def gamma_sq_of_x(self, x: torch.Tensor) -> tuple:
                """State-dependent diffusion: g²(x) = (σ·x)² = σ²·x²"""
                x1 = x[:, 0:1]
                x2 = x[:, 1:2]
                g11_sq = (self.sigma * x1) ** 2
                g22_sq = (self.sigma * x2) ** 2
                return g11_sq, g22_sq

        dynamics_const = DynamicsConstants(A, R, x_range, x_goal_range, x_unsafe_range)
        print(f"*** Creating SymbolicCROWNCache_Phi for training ***")
        # Extract sigma from R matrix
        sigma = R[0, 0] if isinstance(R, np.ndarray) else R
        # GV only needs bounds over generator region, not all regions
        num_generator_cells = len(region_cells['generator'])
        crown_phi_cache = SymbolicCROWNCache_Phi(
            V_net=model,
            A=A,
            sigma=sigma,
            num_cells=num_generator_cells,
            scale_factor=scale_factor,
            input_dim=2,
            device='cpu'
        )

        # Note: We create a fresh CROWNVerifier each time we verify (in the training loop)
        # to ensure it uses the current network weights, not frozen initial weights
        print(f"*** Will create fresh CROWNVerifier during verification (uses current weights) ***")
        print()

    start_time = time.time()

    # Initialize return values
    verified_boxes = None
    verified_bounds = None

    # Pre-compute input bounds once (they don't change during training)
    if not use_manual_softplus:
        input_lowers = torch.stack([cell[0] for cell in all_cells])
        input_uppers = torch.stack([cell[1] for cell in all_cells])

    # Pre-compute GV input bounds (only for generator region)
    input_lowers_GV = None
    input_uppers_GV = None
    if compute_GV and crown_phi_cache is not None:
        input_lowers_GV = torch.stack([cell[0] for cell in region_cells['generator']])
        input_uppers_GV = torch.stack([cell[1] for cell in region_cells['generator']])
        print(f"[GV Optimization] Pre-computed input bounds for {input_lowers_GV.shape[0]} generator cells")

    # Helper function to split bounds by region
    def split_bounds_by_region(v_lowers, v_uppers):
        bounds = {}
        cell_idx = 0
        for name in regions_dict.keys():
            if name != 'generator':
                num_cells = len(region_cells[name])
                bounds[name] = (
                    v_lowers[cell_idx:cell_idx + num_cells],
                    v_uppers[cell_idx:cell_idx + num_cells]
                )
                cell_idx += num_cells
            # print(f"  {name}: {num_cells} cells")
            # if name == 'generator':
            #     print(f"  {name}: {num_cells} cells")
                # plt.plot()
        return bounds

    # Tally tracker for most frequently failing cells in generator region
    num_generator_cells = len(region_cells['generator'])
    failure_tally = torch.zeros(num_generator_cells, dtype=torch.int32)

    for epoch in range(num_epochs):
        # if epoch > 10000:
        #     break
        # Step 1: Compute differentiable bounds for training
        # import time
        epoch_start_time = time.time()

        model.train()
        optimizer.zero_grad()

        # Flag for adaptive refinement cache rebuild
        needs_cache_rebuild = False

        # if use_manual_softplus:
        #     v_lowers_train, v_uppers_train = compute_bounds_manual_softplus(model, all_cells)
        # else:
        if compute_V:
            t0 = time.time()
            v_lowers_train, v_uppers_train = crown_cache.compute_bounds(input_lowers, input_uppers)
            t_v_crown = time.time() - t0

            t0 = time.time()
            region_bounds_train = split_bounds_by_region(v_lowers_train, v_uppers_train)
            t_split = time.time() - t0
        else:
            t_v_crown = 0.0
            t_split = 0.0


        

        # Compute losses from training bounds
        t0 = time.time()
        total_loss = torch.tensor(0.0, device=device)
        region_losses = {}
        if compute_V:
            for name, (region, loss_fn) in regions_dict.items():
                # if False:
                # if True:
                    if name == 'goal':
                        v_lowers_region, v_uppers_region = region_bounds_train[name]
                        if epoch % 100 == 0:
                            show = True
                        else:
                            show = False
                        loss, passed = loss_goal_sampled(model, region, v_lowers_region, show=show, n_samples=1000)
                        region_losses[name] = loss
                        total_loss = total_loss + loss
                    elif loss_fn is not None:  # Skip regions with no V loss (e.g., 'generator')
                        v_lowers_region, v_uppers_region = region_bounds_train[name]
                        loss = loss_fn(v_lowers_region, v_uppers_region)
                        region_losses[name] = loss
                        total_loss = total_loss + loss
        t_v_loss = time.time() - t0

        # compute_generator_this_epoch = (generator_weight > 0 and A is not None and R is not None
                                    #    and (epoch % mod == 0) or epoch == 1)  # Every 5 epochs instead of every epoch
                                    #    and epoch >= generator_start_epoch)
        # compute_generator_this_epoch = True# if epoch >= generator_start_epoch else False
        # compute_generator_this_epoch = True if epoch == 1 or epoch % 19 == 0 else False
        # compute_generator_this_epoch = False
        # compute_generator_this_epoch = True
        if compute_GV:
            compute_generator_this_epoch = True if epoch == 1 or epoch % 1 == 0 else False

        # if compute_generator_this_epoch:
        #     # Apply generator constraint on 'outside' region (X \ Goal)
        #     # Use CELL-BASED BOUNDS (like V bounds) instead of sampling!
        #     for name, (region, _) in regions_dict.items():
        #         if name == 'outside':  # The main region to enforce generator
        #             # NEW: Use rigorous cell-based bounds (evaluates Φ at corners)
        #             loss_gen = loss_generator_bounds_cells(model, A, R, region_cells[name], use_fast_gradients=True)
        #             region_losses['generator'] = loss_gen
        #             total_loss += generator_weight * loss_gen
        #             break

        t_gv_crown = 0.0
        t_gv_loss = 0.0
        t_gv_sample = 0.0
        num_total_failing = 0  # Initialize for tracking (used in both sampling and CROWN modes)

        # Decide: use sampling (fast) or CROWN (rigorous) for GV loss
        # Sampling: epochs [generator_start_epoch, crown_bounds_start_epoch)
        # CROWN:    epochs [crown_bounds_start_epoch, num_epochs]
        use_crown_bounds = epoch >= 0
        use_sampling = not use_crown_bounds

        if compute_GV and epoch >= generator_start_epoch:
            if compute_generator_this_epoch:
                if use_crown_bounds and crown_phi_cache is not None:
                    # ===== CROWN BOUNDS (Rigorous, Slow) =====
                    # Use pre-computed input bounds for generator cells
                    # Compute Phi bounds only for generator region
                    t0 = time.time()
                    phi_lowers_gen, phi_uppers_gen = crown_phi_cache.compute_bounds(input_lowers_GV,
                    input_uppers_GV)
                    t_gv_crown = time.time() - t0

                    

                    t0 = time.time()
                    # Loss: penalize cells where Φ upper OR lower bound > 0
                    # Want: Φ(x) ≤ 0 for all x in 'generator' region (X \ (Goal ∪ Unsafe))
                    # This requires BOTH: phi_upper ≤ 0 AND phi_lower ≤ 0
                    # Strategy: Minimize upper bounds directly (always provides gradient signal)
                    # phi_upper_violation = phi_uppers_gen  # No relu! Minimize even when negative
                    phi_upper_violation = torch.nn.functional.relu(phi_uppers_gen + 0.0)
                    # phi_lower_violation = torch.nn.functional.relu(phi_lowers_gen + 10000.0)
                    # phi_violation = phi_upper_violation + phi_lower_violation
                    phi_violation = phi_upper_violation
                    # phi_violation = phi_lower_violation

                    # Track which bounds are failing
                    upper_failing_mask = phi_uppers_gen > 0.0
                    lower_failing_mask = phi_lowers_gen > 0.0
                    # any_failing_mask = upper_failing_mask | lower_failing_mask
                    any_failing_mask = upper_failing_mask

                    # num_upper_failing = upper_failing_mask.sum().item()
                    # num_lower_failing = lower_failing_mask.sum().item()
                    num_total_failing = any_failing_mask.sum().item()
                    total_cells = phi_uppers_gen.shape[0]

                    if num_total_failing > 0 and epoch % 100 == 0:
                        failing_indices = torch.where(any_failing_mask)[0]
                        worst_violations = phi_violation[any_failing_mask].topk(min(5, num_total_failing))

                        # Update failure tally
                        failure_tally[failing_indices] += 1

                        # Get most frequently failing cells
                        top_k_persistent = min(10, num_generator_cells)
                        most_failed_counts, most_failed_indices = failure_tally.topk(top_k_persistent)

                        print(f'\n[Generator Bounds] {num_total_failing}/{total_cells} cells failing:')
                        # print(f'  Upper bound violations: {num_upper_failing} cells')
                        # print(f'  Lower bound violations: {num_lower_failing} cells')
                        # print(f'  Worst violations (Φ_upper + Φ_lower): {worst_violations.values.tolist()}')
                        # print(f'  Failing cell indices (first 10): {failing_indices[:10].tolist()}')
                        print(f'  Φ_upper range: [{phi_uppers_gen.min().item():.6f}, {phi_uppers_gen.max().item():.6f}]')
                        print(f'  Φ_lower range: [{phi_lowers_gen.min().item():.6f}, {phi_lowers_gen.max().item():.6f}]')
                        # print(f'  Most persistently failing cells (idx: fail_count):')
                        # for idx, count in zip(most_failed_indices[:5], most_failed_counts[:5]):
                        #     if count > 0:
                        #         print(f'    Cell {idx.item()}: {count.item()} failures')

                    # Focus on worst violators (similar to V loss strategy)
                    # loss_gen = phi_violation[any_failing_mask].max() if num_total_failing > 0 else torch.tensor(0.0)
                    loss_gen = phi_violation.sum()
                    # loss_gen = phi_violation[any_failing_mask].sum()

                    region_losses['generator'] = loss_gen
                    # generator_weight = min(1.0, (epoch - generator_start_epoch) / 1000)
                    total_loss = total_loss + generator_weight * loss_gen
                    t_gv_loss = time.time() - t0
                    # print(f'Generator loss: {loss_gen.item():.6e}')
                    # print(f'  Φ_upper sum: {phi_uppers_gen.sum().item():.6e}, mean: {phi_uppers_gen.mean().item():.6e}')

                    # Print mode switch notification
                    if epoch == crown_bounds_start_epoch:
                        print(f"\n{'='*80}")
                        print(f"SWITCHING TO CROWN BOUNDS at epoch {epoch}")
                        print(f"From now on, using rigorous CROWN bounds for GV training (slower but tighter)")
                        print(f"{'='*80}\n")

                    # Adaptive refinement: refine ALL failing cells periodically
                    REFINE_INTERVAL = 500  # Refine every 50 epochs
                    # REFINE_INTERVAL = 100 if epoch > 99 else 50
                    REFINE_FACTOR = 2  # Split into 2x2 subcells

                    if (epoch + 1) % REFINE_INTERVAL == 0 and num_total_failing > 0 and len(region_cells['generator']) < 1500:
                        print(f"\n[Adaptive Refinement] Refining failing cells at epoch {epoch+1}")
                        print(f"  Before: {len(region_cells['generator'])} cells")
                        region_cells['generator'] = refine_cells_by_mask(
                            region_cells['generator'],
                            any_failing_mask,
                            refinement_factor=REFINE_FACTOR
                        )
                        print(f"  After: {len(region_cells['generator'])} cells")
                        # Flag to rebuild cache after backward pass
                        needs_cache_rebuild = True

                    # DEBUG: Sample Φ values and compare to CROWN bounds
                    if epoch % 50 == 0:
                        with torch.no_grad():
                            # Randomly sample cells from generator region
                            num_cells_to_check = min(100, len(region_cells['generator']))
                            sample_cell_indices = torch.randperm(len(region_cells['generator']))[:num_cells_to_check].tolist()

                            violations_found = False
                            for idx in sample_cell_indices:
                                cell_lower, cell_upper = region_cells['generator'][idx]

                                # Sample random points within this cell
                                num_samples_per_cell = 10
                                # Generate random points: lower + random * (upper - lower)
                                random_offsets = torch.rand(num_samples_per_cell, 2)
                                samples = cell_lower.unsqueeze(0) + random_offsets * (cell_upper - cell_lower).unsqueeze(0)

                                # Evaluate Φ at all sampled points
                                phi_samples = crown_phi_cache.phi_module(samples).squeeze(-1)
                                phi_min = phi_samples.min().item()
                                phi_max = phi_samples.max().item()
                                phi_mean = phi_samples.mean().item()

                                # Get CROWN bounds for this cell
                                phi_lower_crown = phi_lowers_gen[idx].item()
                                phi_upper_crown = phi_uppers_gen[idx].item()

                                # Check if samples violate CROWN bounds
                                lower_violation = phi_min < phi_lower_crown
                                upper_violation = phi_max > phi_upper_crown

                                if lower_violation or upper_violation:
                                    if not violations_found:
                                        print("\n[WARNING] CROWN bound violations detected:")
                                        violations_found = True
                                    print(f"  Cell {idx}: bounds=[{cell_lower.numpy()}, {cell_upper.numpy()}]")
                                    print(f"    Sampled Φ range: [{phi_min:.4f}, {phi_max:.4f}] (mean={phi_mean:.4f})")
                                    print(f"    CROWN bounds:    [{phi_lower_crown:.4f}, {phi_upper_crown:.4f}]")
                                    if lower_violation:
                                        print(f"    ⚠️  Lower violation: {phi_min:.4f} < {phi_lower_crown:.4f} (gap: {phi_lower_crown - phi_min:.4f})")
                                    if upper_violation:
                                        print(f"    ⚠️  Upper violation: {phi_max:.4f} > {phi_upper_crown:.4f} (gap: {phi_max - phi_upper_crown:.4f})")
                        # print()

                elif use_sampling and A is not None and R is not None:
                    # ===== SAMPLING-BASED LOSS (Fast, Approximate) =====
                    t0 = time.time()
                    loss_gen = loss_generator_sampled(
                        model, A, R,
                        x_goal_range, x_unsafe_range, x_range,
                        n_samples=500
                    )
                    t_gv_sample = time.time() - t0

                    region_losses['generator'] = loss_gen
                    total_loss = total_loss + generator_weight * loss_gen

                    # Print mode notification on first use
                    if epoch == generator_start_epoch:
                        print(f"\n{'='*80}")
                        print(f"USING SAMPLING for GV training (fast)")
                        print(f"Will switch to CROWN bounds at epoch {crown_bounds_start_epoch} for final polishing")
                        print(f"{'='*80}\n")

                    if epoch % 100 == 0:
                        print(f'  [Sampling] GV loss: {loss_gen.item():.6f}')

        # Only backward if total_loss has gradients and is non-zero
        has_gradients = total_loss.requires_grad and total_loss.item() > 0
        if has_gradients:
            t0 = time.time()
            total_loss.backward()
            t_backward = time.time() - t0
        else:
            # No new loss computed - skip this epoch's backward pass
            t_backward = 0.0

        # Visualization after backward pass
        # # if epoch == 1000:
        #     visualize_gv_output(
        #         model,
        #         region=x_range,
        #         A_matrix=A_matrix,
        #         R_matrix=R_matrix,
        #         title="Φ(x) - Full State Space",
        #         show_regions=True,
        #         x_init=x_init_range,
        #         x_unsafe=x_unsafe_range,
        #         x_goal=x_goal_range,
        #         training_cells=region_cells['generator'],  # Show generator cells!
        #         show_discretization=True,
        #         phi_module=crown_phi_cache.phi_module if crown_phi_cache is not None else None
        #     )
        #     exit()

        # # Diagnostic: Check if gradients are flowing to V_net
        # if compute_generator_this_epoch and epoch % 10 == 0:
        #     w2_grad = model.output.weight.grad
        #     if w2_grad is not None:
        #         print(f'  [Gradient Check] W2.grad norm: {w2_grad.norm().item():.6e}')
        #         print(f'  [Gradient Check] W2.grad max: {w2_grad.abs().max().item():.6e}')
        #     else:
        #         print(f'  [Gradient Check] W2.grad is None - NO GRADIENTS!')

        # Only step optimizer if we computed gradients
        if has_gradients:
            t0 = time.time()
            optimizer.step()
            t_optimizer = time.time() - t0
        else:
            t_optimizer = 0.0

        # Rebuild CROWN caches after optimizer step if needed (after adaptive refinement)
        if needs_cache_rebuild:
            print(f"  Rebuilding CROWN caches with new cells...")
            all_cells = []
            for name in ['init', 'goal', 'unsafe', 'outside', 'generator']:
                all_cells.extend(region_cells[name])

            # Rebuild input bounds tensors
            input_lowers = torch.stack([cell[0] for cell in all_cells])
            input_uppers = torch.stack([cell[1] for cell in all_cells])

            # Rebuild V CROWN cache
            crown_cache = SymbolicCROWNCache(model, num_cells=len(all_cells), input_dim=2)

            # Rebuild Phi CROWN cache (only for generator region)
            sigma = R[0, 0] if isinstance(R, np.ndarray) else R
            num_generator_cells = len(region_cells['generator'])
            crown_phi_cache = SymbolicCROWNCache_Phi(
                V_net=model,
                A=A,
                sigma=sigma,
                num_cells=num_generator_cells,
                scale_factor=scale_factor,
                input_dim=2,
                device='cpu'
            )

            # Rebuild GV input bounds for generator region
            input_lowers_GV = torch.stack([cell[0] for cell in region_cells['generator']])
            input_uppers_GV = torch.stack([cell[1] for cell in region_cells['generator']])

            # Update failure tally size
            num_generator_cells = len(region_cells['generator'])
            failure_tally = torch.zeros(num_generator_cells, dtype=torch.int32)

            print(f"  Caches rebuilt successfully with {len(all_cells)} total cells")

        # Step 2: Compute bounds AFTER optimizer step for verification (only when checking)
        # This saves ~50% computation time by skipping verification most epochs
        # check_this_epoch = (epoch + 1) % (25 +1) == 0 or epoch == 0
        check_this_epoch = True

        if check_this_epoch:
            model.eval()
            with torch.no_grad():
                if use_manual_softplus:
                    v_lowers, v_uppers = compute_bounds_manual_softplus(model, all_cells)
                else:
                    v_lowers, v_uppers = crown_cache.compute_bounds(input_lowers, input_uppers)

            # Split verification bounds by region
            region_bounds = split_bounds_by_region(v_lowers, v_uppers)

        # Update scheduler less frequently (every 10 epochs is enough)
        if (epoch + 1) % 10 == 0 and has_gradients:
            scheduler.step(total_loss.item())

        # Print progress and check constraints periodically (not every epoch!)
        if check_this_epoch:
            if epoch % 100 == 0:
                loss_str = ", ".join([f"{name}={loss.item():.4f}" for name, loss in region_losses.items()])
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item():.4f} ({loss_str})")
                if epoch % 1000000 == 0:  # Print detailed timing every 10 epochs
                    print(f"  [TIMING] Total: {epoch_time:.3f}s | V_CROWN: {t_v_crown:.3f}s | V_loss: {t_v_loss:.3f}s | GV_CROWN: {t_gv_crown:.3f}s | GV_Sample: {t_gv_sample:.3f}s | GV_loss: {t_gv_loss:.3f}s | Backward: {t_backward:.3f}s | Optimizer: {t_optimizer:.3f}s | Split: {t_split:.3f}s")
                    print(f"  [TIMING] Longest: {max(t_v_crown, t_v_loss, t_gv_crown, t_gv_loss, t_backward, t_optimizer):.3f}s")

            if compute_V:
                # Check constraints
                with torch.no_grad():
                    # Goal: sample
                    # goal_satisfied = (goal_v_samples.min() < BETA_S).item() and (goal_v_samples.min() >= 0).item()
                    goal_satisfied = passed

                    # Others: bounds
                    outside_satisfied = (region_bounds['outside'][0] >= BETA_S).all().item()
                    unsafe_satisfied = (region_bounds['unsafe'][0] >= BETA_RA).all().item()
                    init_satisfied = ((region_bounds['init'][0] >= BETA_S).all().item() and
                                    (region_bounds['init'][1] <= 1.0).all().item())

            if compute_GV:
                # Check if generator constraint is satisfied (if enabled)
                generator_satisfied = True
                # if 'generator' in region_losses:
                #     print(f' Generator in region_losses, Epoch: {epoch+1}')
                if generator_weight > 0 and 'generator' in region_losses:
                    generator_loss_value = region_losses['generator'].item()
                    # With new max-based loss, threshold should be near 0 (not negative!)
                    # generator_satisfied = generator_loss_value == 0.0  # Loss < 0.01 means satisfied
                    generator_satisfied = 1 if num_total_failing == 0 else 0
                    if epoch % 10 == 0:
                        print(f"  Generator loss: {generator_loss_value:.4f}")
                else: 
                    generator_satisfied = False

            # print(f"  Constraints: goal={goal_satisfied}, outside={outside_satisfied}, "
            #     f"unsafe={unsafe_satisfied}, init={init_satisfied}", end="")
            # if generator_weight > 0:
            #     print(f", generator={generator_satisfied} (loss={region_losses.get('generator', torch.tensor(0.0)).item():.4f})")
            # else:
            #     print()
            if compute_V:
                passed_V = goal_satisfied and outside_satisfied and unsafe_satisfied and init_satisfied
            if compute_GV:
                passed_GV = generator_satisfied
            if compute_GV and compute_V:
                passed = passed_V and passed_GV
            elif compute_V:
                passed = passed_V
            elif compute_GV:
                passed = passed_GV

            # if goal_satisfied and outside_satisfied and unsafe_satisfied and init_satisfied: #and generator_satisfied:
            # if generator_satisfied:
            # if (compute_V and passed_V) or (compute_GV and passed_GV):
            if passed:
                test_samples = torch.rand(1000000, 2)
                test_samples[:, 0] = test_samples[:, 0] * (x_goal_range[0, 1] - x_goal_range[0, 0]) + x_goal_range[0, 0]
                test_samples[:, 1] = test_samples[:, 1] * (x_goal_range[1, 1] - x_goal_range[1, 0]) + x_goal_range[1, 0]
                goal_v_samples = model(test_samples).squeeze()

                print(f"\n*** ALL CONSTRAINTS SATISFIED at epoch {epoch+1}! ***\n")
                print(f"Outside of goal bounds: V_min in [{region_bounds['outside'][0].min().item():.4f}, {region_bounds['outside'][0].max().item():.4f}], V_max in [{region_bounds['outside'][1].min().item():.4f}, {region_bounds['outside'][1].min().item():.4f}]")
                print(f"Unsafe region: V_min in [{region_bounds['unsafe'][0].min().item():.4f}, {region_bounds['unsafe'][0].max().item():.4f}], V_max in [{region_bounds['unsafe'][1].min().item():.4f}, {region_bounds['unsafe'][1].max().item():.4f}]")
                print(f"Init region: V_min in [{region_bounds['init'][0].min().item():.4f}, {region_bounds['init'][0].max().item():.4f}], V_max in [{region_bounds['init'][1].min().item():.4f}, {region_bounds['init'][1].max().item():.4f}]")
                print(f"Goal region: V in [{goal_v_samples.min().item():.4f}, {goal_v_samples.max().item():.4f}]")
                if generator_weight > 0:
                    if 'generator' in region_losses:
                        print(f"Generator loss: {region_losses['generator'].item():.4f}")

                # if generator_satisfied is not None:
                if compute_GV:
                    visualize_gv_output(
                        model,
                        region=x_range,
                        A_matrix=A_matrix,
                        R_matrix=R_matrix,
                        title="Φ(x) - Full State Space",
                        show_regions=True,
                        x_init=x_init_range,
                        x_unsafe=x_unsafe_range,
                        x_goal=x_goal_range,
                        phi_lower_range=phi_lowers_gen,
                        phi_upper_range=phi_uppers_gen,
                        training_cells=region_cells['generator'],  # Show generator cells!
                        show_discretization=True,
                        phi_module=crown_phi_cache.phi_module if crown_phi_cache is not None else None
                    )

                visualize_network_output(
                    model,
                    region=x_range,
                    title="V(x) - Full State Space",
                    show_regions=True,
                    x_init=x_init_range,
                    x_unsafe=x_unsafe_range,
                    x_goal=x_goal_range,
                    training_cells=region_cells['outside'],  # Changed from 'outside' to 'generator'
                    show_discretization=True
                )
                # exit()
                break

    end_time = time.time() - start_time
    print(f"Training completed in {end_time:.2f} seconds")

    # visualize_gv_output(
    #     model,
    #     region=x_range,
    #     A_matrix=A_matrix,
    #     R_matrix=R_matrix,
    #     title="Φ(x) - Full State Space",
    #     show_regions=True,
    #     x_init=x_init_range,
    #     x_unsafe=x_unsafe_range,
    #     x_goal=x_goal_range,
    #     phi_lower_range=phi_lowers_gen,
    #     phi_upper_range=phi_uppers_gen,
    #     training_cells=region_cells['generator'],  # Show generator cells!
    #     show_discretization=True,
    #     phi_module=crown_phi_cache.phi_module if crown_phi_cache is not None else None
    # )

    return verified_boxes, verified_bounds

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SIMPLE 3-CONSTRAINT LYAPUNOV TRAINING")
    print("="*80)

    print(f"\nNetwork: {N_INPUTS} -> {N_HIDDEN_1} -> {N_HIDDEN_2} -> {N_OUTPUTS}")
    print(f"Goal region: {x_goal_range[0]} × {x_goal_range[1]}")
    print(f"Full space: {x_range[0]} × {x_range[1]}")

    # Initialize network
    # model = SimpleNN(N_INPUTS, N_HIDDEN_1, N_HIDDEN_2, N_HIDDEN_3, N_OUTPUTS)
    model = SimpleNN(N_INPUTS, N_HIDDEN_1, N_HIDDEN_2, 0, N_OUTPUTS)

    # Initialize with small weights to reduce initial GV loss
    # with torch.no_grad():
    #     for param in model.parameters():
    #         if param.dim() > 1:  # weights
    #             param.data.uniform_(-0.01, 0.01)
    #         else:  # biases
    #             param.data.fill_(0.0)
    #     model.output.bias.fill_(0.01)

    # print(f"\nInitialized with small weights and output bias to 0.5")

    # Define regions_dict: {name: (region, loss_function)}
    regions_dict = {
        'init': (x_init_range, loss_init_bounds),
        'goal': (x_goal_range, None),  # Goal uses sampling, not bounds
        'unsafe': (x_unsafe_range, loss_unsafe_bounds),
        'outside': (x_range, loss_outside_goal_bounds),  # For V constraint: X \ Goal
        'generator': (x_range, None)  # For Φ constraint: X \ (Goal ∪ Unsafe), no V loss
    }

    # Discretization map
    discretization_map = {
        'init': N_DISCRETIZE_INIT,
        'goal': N_DISCRETIZE_GOAL,
        'unsafe': N_DISCRETIZE_UNSAFE,
        'outside': N_DISCRETIZE_OUTSIDE_GOAL,
        'generator': N_DISCRETIZE_GENERATOR
    }

    # Discretize regions
    region_cells = {}
    for name, (region, _) in regions_dict.items():
        n_discretize = discretization_map[name]
        if name == 'outside':
            # Outside = full space \ goal (for V constraint)
            outside_rects = compute_rectangular_partition_outside_goal(x_range, x_goal_range)
            cells = []
            for rect in outside_rects:
                cells.extend(discretize_region(rect, n_discretize))
        elif name == 'generator':
            # Generator = full space \ (goal ∪ unsafe) with radial discretization
            RADIUS_THRESHOLDS = [25.0, 40.0]
            N_SPLITS = [2, 1, 1]
            all_cells = discretize_region_radial(x_range, RADIUS_THRESHOLDS, N_SPLITS, n_subdivide=6)

            # Clip cells to remove overlaps with goal and unsafe regions
            print(f'  Clipping {len(all_cells)} generator cells against goal and unsafe regions...')
            exclusion_regions = [x_goal_range, x_unsafe_range]
            clipped_cells = []
            for cell_lower, cell_upper in all_cells:
                # Convert from torch tensors to numpy for clipping
                lower_np = cell_lower.numpy() if isinstance(cell_lower, torch.Tensor) else cell_lower
                upper_np = cell_upper.numpy() if isinstance(cell_upper, torch.Tensor) else cell_upper

                # Clip this cell against both exclusion regions
                result_cells = clip_cell_against_exclusions(lower_np, upper_np, exclusion_regions)

                # Convert back to torch tensors
                for lower, upper in result_cells:
                    clipped_cells.append((
                        torch.tensor(lower, dtype=torch.float32),
                        torch.tensor(upper, dtype=torch.float32)
                    ))

            cells = clipped_cells
            print(f'  After clipping: {len(cells)} cells (may have increased due to cell splitting)')
            if len(cells) > 0:
                print(f'  First cell bounds in "generator" region: {cells[0][0].numpy()} to {cells[0][1].numpy()}')
        else:
            cells = discretize_region(region, n_discretize)
        region_cells[name] = cells

    print(f"\nDiscretization:")
    for name, cells in region_cells.items():
        print(f"  {name}: {len(cells)} cells")

    print(f"\nRegion definitions:")
    print(f"  'outside': X \\ Goal (for V ≥ β_s constraint)")
    print(f"  'generator': X \\ (Goal ∪ Unsafe) (for Φ ≤ 0 constraint)")

    # Define system dynamics for generator condition
    # TODO: Replace with your actual A and R matrices!
    A_matrix = np.array([
        [-1.5, 1.0],
        [-1.0, -1.5]
        # [0.0, 0.0],
        # [0.0, 0.0]
    ], dtype=np.float32)

    R_matrix = np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ], dtype=np.float32)

    # Generator loss weight (0.0 = disabled, 1.0 = same weight as other losses)
    GENERATOR_WEIGHT = 1.0  # Start small: 0.1-1.0, gradually increase if needed
    GENERATOR_START_EPOCH = 0  # Start adding generator loss after this epoch

    # CROWN verification during training
    ENABLE_CROWN_VERIFICATION = True  # Enable CROWN verification in training loop
    CROWN_VERIFY_INTERVAL = 500  # Run CROWN every N epochs (when constraints satisfied)
    CROWN_REFINE_N = 2 # Each box splits into 2x2 = 4 boxes
    CROWN_MAX_DEPTH = 1  # Quick verification (5 iterations max) for training feedback

    print(f"\nGenerator condition:")
    if GENERATOR_WEIGHT > 0:
        print(f"  ENABLED (weight={GENERATOR_WEIGHT})")
        print(f"  Training V such that Φ(x) ≤ 0")
        print(f"  where Φ(x) = (A*x)·∇V(x) + 0.5·diag(R²)·H(x)")
    else:
        print(f"  DISABLED (set GENERATOR_WEIGHT > 0 to enable)")

    print(f"\nCROWN verification during training:")
    if ENABLE_CROWN_VERIFICATION:
        print(f"  ENABLED (interval={CROWN_VERIFY_INTERVAL} epochs)")
        print(f"  Parameters: refine_N={CROWN_REFINE_N}, max_depth={CROWN_MAX_DEPTH}")
        print(f"  → Training exits early when CROWN verification succeeds")
        print(f"  → Continues training if any boxes need refinement")
    else:
        print(f"  DISABLED (set ENABLE_CROWN_VERIFICATION=True to enable)")

    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # visualize_network_output(
    #   model,
    #   region=x_range,
    #   title="V(x) - Full State Space",
    #   show_regions=True,
    #   x_init=x_init_range,
    #   x_unsafe=x_unsafe_range,
    #   x_goal=x_goal_range,
    #   training_cells=region_cells['outside'],  # Changed from 'outside' to 'generator'
    #   show_discretization=True
    # )

    # exit()

    # Pre-train the network to have a good initial structure
    ENABLE_PRETRAINING = True  # Set to False to skip pre-training
    PRETRAIN_EPOCHS = 500
    PRETRAIN_LR = 0.01

    if ENABLE_PRETRAINING:
        pretrain_structure_aware(
            model,
            x_goal_range,
            x_unsafe_range,
            x_init_range,
            x_range,
            A=A_matrix,
            R=R_matrix,
            scale_factor=SCALE_FACTOR,
            num_epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR
        )

    # GV Training Strategy: Use sampling (fast) first, then CROWN (rigorous) to polish
    CROWN_BOUNDS_START_EPOCH = 0  # Switch to CROWN at 90% of training (polish final 10%)

    verified_boxes, verified_bounds = train_network(
        model, regions_dict, region_cells, NUM_EPOCHS, LEARNING_RATE,
        A=A_matrix, R=R_matrix,
        generator_weight=GENERATOR_WEIGHT,
        generator_start_epoch=GENERATOR_START_EPOCH,
        crown_bounds_start_epoch=CROWN_BOUNDS_START_EPOCH,
        enable_crown_verification=ENABLE_CROWN_VERIFICATION,
        crown_verify_epoch_interval=CROWN_VERIFY_INTERVAL,
        crown_refine_N=CROWN_REFINE_N,
        crown_max_depth=CROWN_MAX_DEPTH,
        scale_factor=SCALE_FACTOR
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': {
            'n_inputs': N_INPUTS,
            'n_hidden_1': N_HIDDEN_1,
            'n_hidden_2': N_HIDDEN_2,
            # 'n_hidden_3': N_HIDDEN_3,
            'n_outputs': N_OUTPUTS
        }
    }, 'trained_simple.pth')
    print("\n✓ Model saved to 'trained_simple.pth'")

    # Visualize network output
    print("\n" + "="*80)
    print("VISUALIZATION & VERIFICATION")
    print("="*80)

    # Verify init region bounds
    print("\n[DEBUG] Verifying init region:")
    print(f"  Init region definition: [{x_init_range[0, 0]}, {x_init_range[0, 1]}] × [{x_init_range[1, 0]}, {x_init_range[1, 1]}]")
    print(f"  Number of init cells: {len(region_cells['init'])}")
    for i, (lower, upper) in enumerate(region_cells['init']):
        print(f"    Cell {i}: [{lower[0]:.1f}, {upper[0]:.1f}] × [{lower[1]:.1f}, {upper[1]:.1f}]")

    # Sample densely from init region and check actual output
    model.eval()
    with torch.no_grad():
        init_samples = torch.rand(1000, 2)
        init_samples[:, 0] = init_samples[:, 0] * (x_init_range[0, 1] - x_init_range[0, 0]) + x_init_range[0, 0]
        init_samples[:, 1] = init_samples[:, 1] * (x_init_range[1, 1] - x_init_range[1, 0]) + x_init_range[1, 0]
        init_outputs = model(init_samples).squeeze()
        print(f"  Actual outputs from 1000 samples: min={init_outputs.min().item():.4f}, max={init_outputs.max().item():.4f}")

        # Also evaluate at the 4 corners
        corners = torch.tensor([
            [x_init_range[0, 0], x_init_range[1, 0]],
            [x_init_range[0, 0], x_init_range[1, 1]],
            [x_init_range[0, 1], x_init_range[1, 0]],
            [x_init_range[0, 1], x_init_range[1, 1]]
        ], dtype=torch.float32)
        corner_outputs = model(corners).squeeze()
        print(f"  Corner outputs: {corner_outputs.numpy()}")
        print(f"  Corner range: [{corner_outputs.min().item():.4f}, {corner_outputs.max().item():.4f}]")
    print()



    # Full space visualization with all regions and discretization
    print("\nVisualizing full state space...")
    visualize_network_output(
        model,
        region=x_range,
        title="V(x) - Full State Space",
        show_regions=True,
        x_init=x_init_range,
        x_unsafe=x_unsafe_range,
        x_goal=x_goal_range,
        training_cells=region_cells['outside'],
        show_discretization=True
    )

    # Individual region visualizations
    print("\nVisualizing individual regions...")

    # Init region
    visualize_network_output(
        model,
        region=x_init_range,
        title=f"V(x) - Init Region (should be: {BETA_S} ≤ V ≤ 1.0)",
        show_regions=False
    )

    # Goal region
    visualize_network_output(
        model,
        region=x_goal_range,
        title=f"V(x) - Goal Region (should have at least one: V < {BETA_S})",
        show_regions=False
    )

    # Unsafe region
    visualize_network_output(
        model,
        region=x_unsafe_range,
        title=f"V(x) - Unsafe Region (should be: V ≥ {BETA_RA})",
        show_regions=False
    )

    # ============================================================================
    # VISUALIZE GV (Φ) FUNCTION WITH CROWN-VERIFIED BOXES
    # ============================================================================
    print("\nVisualizing GV (Φ) function with CROWN-verified boxes...")

    # Check if we have verified boxes from CROWN
    if 'verified_boxes' in locals() and verified_boxes is not None:
        test = True
        # print(f"  Using {len(verified_boxes)} CROWN-verified boxes for visualization")

        # # Create background grid (only outside goal and unsafe)
        # x1_vals = np.linspace(x_range[0, 0], x_range[0, 1], 100)
        # x2_vals = np.linspace(x_range[1, 0], x_range[1, 1], 100)
        # X1, X2 = np.meshgrid(x1_vals, x2_vals)
        # x1_flat = X1.flatten()
        # x2_flat = X2.flatten()

        # # Mask out goal and unsafe regions
        # mask_valid = np.ones(len(x1_flat), dtype=bool)
        # for i in range(len(x1_flat)):
        #     x1_pt, x2_pt = x1_flat[i], x2_flat[i]
        #     # Check if in goal
        #     in_goal = (x_goal_range[0, 0] <= x1_pt <= x_goal_range[0, 1] and
        #               x_goal_range[1, 0] <= x2_pt <= x_goal_range[1, 1])
        #     # Check if in unsafe
        #     in_unsafe = (x_unsafe_range[0, 0] <= x1_pt <= x_unsafe_range[0, 1] and
        #                 x_unsafe_range[1, 0] <= x2_pt <= x_unsafe_range[1, 1])
        #     if in_goal or in_unsafe:
        #         mask_valid[i] = False

        # # Only compute Φ for valid points (outside goal and unsafe)
        # x_grid_valid = torch.tensor(np.stack([x1_flat[mask_valid], x2_flat[mask_valid]], axis=1), dtype=torch.float32)
        # x_grid_valid.requires_grad_(True)

        # model.eval()
        # A_torch = torch.from_numpy(A_matrix).float()
        # fx = x_grid_valid @ A_torch.T
        # f1 = fx[:, 0:1]
        # f2 = fx[:, 1:2]

        # sigma = R_matrix[0, 0]
        # x1 = x_grid_valid[:, 0:1]
        # x2 = x_grid_valid[:, 1:2]
        # g11 = sigma * x1
        # g22 = sigma * x2

        # grad_v, hess_diag = compute_gradient_and_hessian_fast(model, x_grid_valid)
        # dVdx1 = grad_v[:, 0:1]
        # dVdx2 = grad_v[:, 1:2]
        # H11 = hess_diag[:, 0:1]
        # H22 = hess_diag[:, 1:2]

        # drift_term = f1 * dVdx1 + f2 * dVdx2
        # diffusion_term = 0.5 * (g11**2 * H11 + g22**2 * H22)
        # phi_valid = (drift_term + diffusion_term).detach().numpy().flatten()

        # # Create full phi array with NaN for masked regions
        # phi_flat = np.full(len(x1_flat), np.nan)
        # phi_flat[mask_valid] = phi_valid
        # phi = phi_flat.reshape(X1.shape)

        # # Plot
        # plt.figure(figsize=(12, 9))
        # contour = plt.contourf(X1, X2, phi, levels=20, cmap='RdYlGn_r', alpha=0.6)
        # plt.colorbar(contour, label='Φ(x) (background)')
        # plt.contour(X1, X2, phi, levels=[0], colors='gray', linewidths=1, linestyles='--', alpha=0.5)

        # # Overlay CROWN-verified boxes with their bounds
        # from matplotlib.patches import Rectangle
        # from matplotlib.cm import RdYlGn_r
        # import matplotlib.colors as mcolors

        # # Normalize bounds for coloring
        # bounds_array = np.array(verified_bounds)
        # norm = mcolors.Normalize(vmin=-0.1, vmax=0.1)

        # for i, (box, bound) in enumerate(zip(verified_boxes, verified_bounds)):
        #     x_min, x_max = box[0, 0], box[0, 1]
        #     y_min, y_max = box[1, 0], box[1, 1]

        #     # Color based on upper bound (green if < 0, red if > 0)
        #     color = 'green' if bound <= 0 else 'red'
        #     alpha = 0.3 if bound <= 0 else 0.5

        #     rect = Rectangle(
        #         (x_min, y_min),
        #         x_max - x_min,
        #         y_max - y_min,
        #         linewidth=1.5,
        #         edgecolor='black',
        #         facecolor=color,
        #         alpha=alpha
        #     )
        #     plt.gca().add_patch(rect)

        #     # Annotate with bound value (only for boxes with bound > -0.01)
        #     if bound > -0.01:
        #         cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        #         plt.text(cx, cy, f'{bound:.3f}', ha='center', va='center',
        #                 fontsize=7, color='black', weight='bold',
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # # Overlay goal region
        # goal_rect = Rectangle(
        #     (x_goal_range[0, 0], x_goal_range[1, 0]),
        #     x_goal_range[0, 1] - x_goal_range[0, 0],
        #     x_goal_range[1, 1] - x_goal_range[1, 0],
        #     linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.3, label='Goal', linestyle='--'
        # )
        # plt.gca().add_patch(goal_rect)

        # # Overlay unsafe region
        # unsafe_rect = Rectangle(
        #     (x_unsafe_range[0, 0], x_unsafe_range[1, 0]),
        #     x_unsafe_range[0, 1] - x_unsafe_range[0, 0],
        #     x_unsafe_range[1, 1] - x_unsafe_range[1, 0],
        #     linewidth=3, edgecolor='red', facecolor='lightcoral', alpha=0.3, label='Unsafe', linestyle='--'
        # )
        # plt.gca().add_patch(unsafe_rect)

        # plt.xlabel('x1', fontsize=12)
        # plt.ylabel('x2', fontsize=12)
        # plt.title(f"Φ(x) - CROWN-Verified Boxes (n={len(verified_boxes)})\n" +
        #          f"Green boxes: Φ_upper ≤ 0 (verified) | Φ computed only outside Goal ∪ Unsafe",
        #          fontsize=13, fontweight='bold')
        # plt.legend(loc='upper right', fontsize=10)
        # plt.tight_layout()
        # plt.savefig("phi_function_crown_boxes.png", dpi=150)
        # print(f"  → Saved to 'phi_function_crown_boxes.png'")
        # plt.close()

        # # Print statistics
        # print(f"\n  CROWN Box Statistics:")
        # print(f"    Total boxes: {len(verified_boxes)}")
        # print(f"    Φ upper bounds: [{bounds_array.min():.6f}, {bounds_array.max():.6f}]")
        # print(f"    All verified (≤ 0): {np.all(bounds_array <= 0)}")

    else:
        print("  ⚠️  No CROWN-verified boxes available (training may have exited early)")
        print("      Falling back to uniform grid visualization...")

        # Fallback: uniform grid
        x1_vals = np.linspace(x_range[0, 0], x_range[0, 1], 100)
        x2_vals = np.linspace(x_range[1, 0], x_range[1, 1], 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        x1_flat = X1.flatten()
        x2_flat = X2.flatten()
        x_grid = torch.tensor(np.stack([x1_flat, x2_flat], axis=1), dtype=torch.float32)
        x_grid.requires_grad_(True)

        model.eval()
        A_torch = torch.from_numpy(A_matrix).float()
        fx = x_grid @ A_torch.T
        f1 = fx[:, 0:1]
        f2 = fx[:, 1:2]

        sigma = R_matrix[0, 0]
        x1 = x_grid[:, 0:1]
        x2 = x_grid[:, 1:2]
        g11 = sigma * x1
        g22 = sigma * x2

        grad_v, hess_diag = compute_gradient_and_hessian_fast(model, x_grid)
        dVdx1 = grad_v[:, 0:1]
        dVdx2 = grad_v[:, 1:2]
        H11 = hess_diag[:, 0:1]
        H22 = hess_diag[:, 1:2]

        drift_term = f1 * dVdx1 + f2 * dVdx2
        diffusion_term = 0.5 * (g11**2 * H11 + g22**2 * H22)
        phi = (drift_term + diffusion_term).detach().numpy().reshape(X1.shape)

        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X1, X2, phi, levels=20, cmap='RdYlGn_r', vmin=-1, vmax=1)
        plt.colorbar(contour, label='Φ(x)')
        plt.contour(X1, X2, phi, levels=[0], colors='black', linewidths=2)

        from matplotlib.patches import Rectangle
        goal_rect = Rectangle(
            (x_goal_range[0, 0], x_goal_range[1, 0]),
            x_goal_range[0, 1] - x_goal_range[0, 0],
            x_goal_range[1, 1] - x_goal_range[1, 0],
            linewidth=2, edgecolor='blue', facecolor='none', label='Goal'
        )
        plt.gca().add_patch(goal_rect)

        plt.xlabel('x1', fontsize=12)
        plt.ylabel('x2', fontsize=12)
        plt.title(f"Φ(x) - Generator Function (uniform grid)\nRange: [{phi.min():.4f}, {phi.max():.4f}]",
                fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.savefig("phi_function.png", dpi=150)
        print(f"  → Saved to 'phi_function.png'")
        plt.close()
