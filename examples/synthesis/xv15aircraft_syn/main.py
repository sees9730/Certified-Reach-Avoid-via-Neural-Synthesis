from __future__ import annotations

import math
import argparse
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Arc

# -----------------------------------------------------------------------------
# Repo paths (match your Lorentz script)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # repo_root
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
import sys
sys.path.insert(0, str(ROOT))

from src.hyperparameters import Hyperparameters
from src.dynamics import Dynamics
from src.regions import Regions, Region
from src.network import create_V
from src.phi_module import create_GV
from src.discretization import discretize_regions
from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.training_utils import (
    compute_total_loss_bounds,
    compute_loss_goal_bounds,
    evaluate_constraints,
    print_loss_summary,
    print_constraint_summary,
    refine_failing_cells,
    merge_passing_neighbor_cells,
)
from src.save_load_utils import save_eval_bundle, load_eval_bundle, log_loaded_training_epochs, enable_terminal_logging
from src.utils import cleanup_and_setup_directories
from src.visualization import (
    create_summary_plots
)

torch.manual_seed(0)
np.random.seed(0)

DEG = np.pi / 180.0
pi = np.pi
RHO = 1.225
G = 9.81


# =============================================================================
# XV-15 constants + ranges (from your test_dynamics snippet)
# =============================================================================
class XV15Constants:
    MASS: float = 5900.0
    WING_AREA: float = 15.7
    AOA_MAX: float = 16.0 * DEG
    VELOCITY_HORIZONTAL_MIN_SAFE: float = 0.5
    MAX_TILT_ANGLE_RATE: float = 5.0 * DEG


def relu_clamp_min(v: torch.Tensor) -> torch.Tensor:
    return v
    # vmin = float(XV15Constants.VELOCITY_HORIZONTAL_MIN_SAFE)
    # return vmin + F.relu(v - vmin)          # equals v for v>=vmin, equals vmin otherwise


class XV15AeroCoefficients:
    WP_LIFT_LINEAR_COEFF_CRUISE = np.array([0.0849, 0.3482])  # [c1, c0]
    WP_LIFT_LINEAR_COEFF_HOVER  = np.array([0.0646, 0.2709])
    WP_DRAG_COEFF_CRUISE = np.array([0.00042143, 0.0030, 0.0218])  # [a2,a1,a0]
    WP_DRAG_COEFF_HOVER  = np.array([0.000473216, 0.00343, 0.2165])


# =============================================================================
# Differentiable (torch) klinear aero + drift
# =============================================================================
class XV15KLinearAeroTorch(nn.Module):
    """
    Torch aero consistent with source:

    lift_coeff_wing_pylon (kLinear):
      aoa_deg = RAD2DEG * aoa
      tilt_angle_ratio = tilt_angle * 2/pi
      CL_cruise = c1_c * aoa_deg + c0_c
      CL_hover  = c1_h * aoa_deg + c0_h
      CL = CL_cruise*(1-r) + CL_hover*r

    drag_coeff_wing_pylon (quadratic):
      CD_cruise = a2_c*aoa_deg^2 + a1_c*aoa_deg + a0_c
      CD_hover  = a2_h*aoa_deg^2 + a1_h*aoa_deg + a0_h
      CD = CD_cruise*(1-r) + CD_hover*r
    """
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "wp_lift_lin_hover",
            torch.tensor(XV15AeroCoefficients.WP_LIFT_LINEAR_COEFF_HOVER, dtype=torch.float32),
        )
        self.register_buffer(
            "wp_lift_lin_cruise",
            torch.tensor(XV15AeroCoefficients.WP_LIFT_LINEAR_COEFF_CRUISE, dtype=torch.float32),
        )
        self.register_buffer(
            "wp_drag_hover",
            torch.tensor(XV15AeroCoefficients.WP_DRAG_COEFF_HOVER, dtype=torch.float32),
        )
        self.register_buffer(
            "wp_drag_cruise",
            torch.tensor(XV15AeroCoefficients.WP_DRAG_COEFF_CRUISE, dtype=torch.float32),
        )

    def forces(
        self,
        v: torch.Tensor,
        aoa: torch.Tensor,        # alpha in your notation
        tilt_angle: torch.Tensor  # beta in your notation
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # aoa_deg = RAD2DEG * aoa  (RAD2DEG = 1/DEG)
        aoa_deg = aoa / DEG

        # tilt_angle_ratio = tilt_angle * 2/pi
        tilt_ratio = tilt_angle * (2.0 / np.pi)

        # NOTE: for strict consistency with source, do NOT clamp tilt_ratio.
        # If your tilt_angle can leave [0, pi/2], then tilt_ratio leaves [0,1]
        # exactly like the source would.

        # ---- Lift coefficient (kLinear) ----
        c1_c, c0_c = self.wp_lift_lin_cruise[0], self.wp_lift_lin_cruise[1]
        c1_h, c0_h = self.wp_lift_lin_hover[0], self.wp_lift_lin_hover[1]
        CL_cruise = c1_c * aoa_deg + c0_c
        CL_hover  = c1_h * aoa_deg + c0_h
        CL = CL_cruise * (1.0 - tilt_ratio) + CL_hover * tilt_ratio

        # ---- Drag coefficient (quadratic) ----
        # drag_coeff_wing_pylon_cruise = a2*aoa_deg^2 + a1*aoa_deg + a0
        a2_c, a1_c, a0_c = self.wp_drag_cruise[0], self.wp_drag_cruise[1], self.wp_drag_cruise[2]
        a2_h, a1_h, a0_h = self.wp_drag_hover[0], self.wp_drag_hover[1], self.wp_drag_hover[2]

        CD_cruise = a2_c * (aoa_deg ** 2) + a1_c * aoa_deg + a0_c
        CD_hover  = a2_h * (aoa_deg ** 2) + a1_h * aoa_deg + a0_h
        CD = CD_cruise * (1.0 - tilt_ratio) + CD_hover * tilt_ratio

        # ---- Forces ----
        q = 0.5 * RHO * v * v
        L = q * XV15Constants.WING_AREA * CL
        D = q * XV15Constants.WING_AREA * CD
        return L, D


@torch.no_grad()
def find_xv15_equilibrium_for_tilt_min_thrust(
    beta_eq_deg: float,
    aero: torch.nn.Module,  # must provide aero.forces(v, alpha, beta) -> (L,D)
    *,
    v_min: float = 0.5,
    v_max: float = 70.0,
    gamma_min_deg: float = -15.0,
    gamma_max_deg: float = 15.0,
    alpha_min_deg: float = -16.0,
    alpha_max_deg: float = 16.0,
    mass: float = 5900.0,
    thrust_max: float | None = None,  # default: mass*g*1.8
    n_v: int = 220,
    n_gamma: int = 81,
    # bisection settings
    bisect_max_iter: int = 70,
    bisect_tol: float = 1e-10,
    # numerical safety
    cos_min: float = 1e-6,
    # tie-break weights (VERY small; only used when thrust ties numerically)
    tie_gamma_w: float = 1e-6,
    tie_alpha_w: float = 1e-6,
    tie_v_w: float = 1e-9,
    # selection tolerance
    cost_tie_tol: float = 1e-12,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Find an equilibrium (x_eq, u_eq) for the XV-15 drift with fixed tilt angle beta = beta_eq_deg,
    while MINIMIZING thrust T.

    Equilibrium conditions (delta = 0 enforced):
      v_dot     = 0
      gamma_dot = 0
      beta_dot  = delta = 0

    Unknowns we search over:
      v in [v_min, v_max], gamma in [gamma_min, gamma_max]
    For each (v,gamma), solve alpha from:
      (D + m g sin(gamma)) * tan(alpha+beta) + L - m g cos(gamma) = 0
    Then compute thrust from v_dot = 0:
      T = (D + m g sin(gamma)) / cos(alpha+beta)

    Objective:
      minimize T (equivalently minimize T/(m g)).
      Tiny tie-breakers prefer smaller |gamma|, then |alpha|, then smaller v.

    Returns:
      x_eq: torch.Tensor (3,) = [v, gamma, beta]
      u_eq: torch.Tensor (3,) = [T, alpha, 0]
      info: dict diagnostics
    """
    beta = float(beta_eq_deg) * DEG
    gamma_min = float(gamma_min_deg) * DEG
    gamma_max = float(gamma_max_deg) * DEG
    alpha_min = float(alpha_min_deg) * DEG
    alpha_max = float(alpha_max_deg) * DEG

    if thrust_max is None:
        thrust_max = float(mass * G * 1.8)

    # Search grids
    v_grid = np.linspace(v_min, v_max, n_v, dtype=np.float64)
    gamma_grid = np.linspace(gamma_min, gamma_max, n_gamma, dtype=np.float64)

    # --- helpers ---
    def LD(v_val: float, alpha_val: float, beta_val: float) -> tuple[float, float]:
        v_t = torch.tensor([v_val], device=device, dtype=dtype)
        a_t = torch.tensor([alpha_val], device=device, dtype=dtype)
        b_t = torch.tensor([beta_val], device=device, dtype=dtype)
        L_t, D_t = aero.forces(v_t, a_t, b_t)
        return float(L_t.item()), float(D_t.item())

    def equilibrium_scalar_F(alpha_val: float, v_val: float, gamma_val: float, beta_val: float) -> float:
        """
        F(alpha) := (D + m g sin(gamma))*tan(alpha+beta) + L - m g cos(gamma)
        """
        L, D = LD(v_val, alpha_val, beta_val)
        s = math.sin(gamma_val)
        c = math.cos(gamma_val)
        ab = alpha_val + beta_val
        cos_ab = math.cos(ab)
        if abs(cos_ab) < cos_min:
            return float("nan")
        return (D + mass * G * s) * math.tan(ab) + L - mass * G * c

    def solve_alpha_bisection(v_val: float, gamma_val: float, beta_val: float) -> float | None:
        """
        Find alpha in [alpha_min, alpha_max] such that F(alpha)=0 using bisection.
        Bracket is found by coarse sampling then bisection.
        """
        samples = 41
        alphas = np.linspace(alpha_min, alpha_max, samples, dtype=np.float64)

        Fs = np.empty(samples, dtype=np.float64)
        Fs[:] = np.nan
        for i, a in enumerate(alphas):
            f = equilibrium_scalar_F(float(a), v_val, gamma_val, beta_val)
            Fs[i] = f if np.isfinite(f) else np.nan

        # find first adjacent sign change
        for i in range(samples - 1):
            f1, f2 = Fs[i], Fs[i + 1]
            if not (np.isfinite(f1) and np.isfinite(f2)):
                continue
            if f1 == 0.0:
                return float(alphas[i])
            if f1 * f2 < 0.0:
                lo = float(alphas[i])
                hi = float(alphas[i + 1])
                flo = float(f1)
                fhi = float(f2)

                for _ in range(bisect_max_iter):
                    mid = 0.5 * (lo + hi)
                    fmid = equilibrium_scalar_F(mid, v_val, gamma_val, beta_val)
                    if not np.isfinite(fmid):
                        return None
                    if abs(fmid) < bisect_tol or (hi - lo) < bisect_tol:
                        return float(mid)
                    if flo * fmid <= 0.0:
                        hi, fhi = mid, float(fmid)
                    else:
                        lo, flo = mid, float(fmid)
                return float(0.5 * (lo + hi))

        return None

    best = None
    best_cost = float("inf")
    best_dbg = None

    feasible_count = 0
    feasible_v_min = float("inf")
    feasible_v_max = float("-inf")

    for gamma_val in gamma_grid:
        for v_val in v_grid:
            alpha_val = solve_alpha_bisection(float(v_val), float(gamma_val), beta)
            if alpha_val is None:
                continue

            L, D = LD(float(v_val), float(alpha_val), beta)

            ab = float(alpha_val) + beta
            cos_ab = math.cos(ab)
            if abs(cos_ab) < cos_min:
                continue

            # thrust from v_dot = 0
            T_val = (D + mass * G * math.sin(float(gamma_val))) / cos_ab

            # bounds
            if not (0.0 <= T_val <= thrust_max):
                continue
            if not (alpha_min <= alpha_val <= alpha_max):
                continue

            # residual sanity check
            vdot = (T_val * math.cos(ab) - D - mass * G * math.sin(float(gamma_val))) / mass
            gdot = (T_val * math.sin(ab) + L - mass * G * math.cos(float(gamma_val))) / (mass * float(v_val))
            if abs(vdot) > 5e-5 or abs(gdot) > 5e-5:
                continue

            feasible_count += 1
            feasible_v_min = min(feasible_v_min, float(v_val))
            feasible_v_max = max(feasible_v_max, float(v_val))

            # -------------------------
            # OBJECTIVE: minimize thrust
            # -------------------------
            T_norm = float(T_val) / (mass * G)
            cost = T_norm

            # tiny tie-breakers (only matter when thrust is equal within numerical tolerance)
            cost += tie_gamma_w * (abs(float(gamma_val)) / (15.0 * DEG))
            cost += tie_alpha_w * (abs(float(alpha_val)) / (15.0 * DEG))
            cost += tie_v_w * ((float(v_val) - v_min) / max(1e-9, (v_max - v_min)))

            better = (cost < best_cost - cost_tie_tol)

            # Extra explicit tie-break if costs are extremely close:
            tie_better = False
            if best is not None and abs(cost - best_cost) <= cost_tie_tol:
                v_best, g_best, _, T_best, a_best = best
                # primary tie-break: smaller thrust (in case tie weights changed it slightly)
                if T_val < T_best - 1e-9:
                    tie_better = True
                elif abs(T_val - T_best) <= 1e-9:
                    # then smaller |gamma|, then smaller |alpha|, then smaller v
                    if abs(float(gamma_val)) < abs(g_best) - 1e-12:
                        tie_better = True
                    elif abs(abs(float(gamma_val)) - abs(g_best)) <= 1e-12 and abs(float(alpha_val)) < abs(a_best) - 1e-12:
                        tie_better = True
                    elif (abs(abs(float(gamma_val)) - abs(g_best)) <= 1e-12 and
                          abs(abs(float(alpha_val)) - abs(a_best)) <= 1e-12 and
                          float(v_val) < v_best - 1e-12):
                        tie_better = True

            if best is None or better or tie_better:
                best_cost = cost
                best = (float(v_val), float(gamma_val), beta, float(T_val), float(alpha_val))
                best_dbg = {
                    "vdot": float(vdot),
                    "gdot": float(gdot),
                    "L": float(L),
                    "D": float(D),
                    "T": float(T_val),
                    "T_over_mg": float(T_norm),
                    "alpha_deg": float(alpha_val / DEG),
                    "gamma_deg": float(gamma_val / DEG),
                    "v": float(v_val),
                    "cost": float(cost),
                }

    info = {
        "found": best is not None,
        "beta_deg": float(beta_eq_deg),
        "search": {"n_v": int(n_v), "n_gamma": int(n_gamma)},
        "bounds": {
            "v": [float(v_min), float(v_max)],
            "gamma_deg": [float(gamma_min_deg), float(gamma_max_deg)],
            "alpha_deg": [float(alpha_min_deg), float(alpha_max_deg)],
            "T": [0.0, float(thrust_max)],
        },
        "feasible": {
            "count": int(feasible_count),
            "v_min": None if feasible_count == 0 else float(feasible_v_min),
            "v_max": None if feasible_count == 0 else float(feasible_v_max),
        },
        "best": best_dbg,
    }

    if best is None:
        return None, None, info

    v_eq, gamma_eq, beta_eq, T_eq, alpha_eq = best
    x_eq = torch.tensor([v_eq, gamma_eq, beta_eq], device=device, dtype=dtype)
    u_eq = torch.tensor([T_eq, alpha_eq, 0.0], device=device, dtype=dtype)  # delta=0
    return x_eq, u_eq, info


class XV15EqMLPControl(nn.Module):
    def __init__(
        self,
        *,
        x_eq: torch.Tensor,   # (3,)
        u_eq: torch.Tensor,   # (3,) [T, alpha, delta]
        T_min: float,
        T_max: float,
        alpha_max: float,
        delta_max: float,
        hidden_dim: int = 32,
        act: str = "tanh",
        k_T: float = 1.0,
        k_alpha: float = 1.0,
        k_delta: float = 1.0,
    ):
        super().__init__()

        # --- MLP with NO biases => MLP(0)=0 exactly ---
        self.fc1 = nn.Linear(3, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 3, bias=False)

        if act == "tanh":
            self.act = nn.Tanh()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("act must be 'tanh' or 'relu'")

        # bounds
        self.register_buffer("T_min", torch.tensor(float(T_min), dtype=torch.float32))
        self.register_buffer("T_max", torch.tensor(float(T_max), dtype=torch.float32))
        self.register_buffer("alpha_max", torch.tensor(float(alpha_max), dtype=torch.float32))
        self.register_buffer("delta_max", torch.tensor(float(delta_max), dtype=torch.float32))

        # squash steepness
        self.register_buffer("k_T", torch.tensor(float(k_T), dtype=torch.float32))
        self.register_buffer("k_alpha", torch.tensor(float(k_alpha), dtype=torch.float32))
        self.register_buffer("k_delta", torch.tensor(float(k_delta), dtype=torch.float32))

        # ---- keep your scaling logic ----
        self.register_buffer(
            "e_scale",
            torch.tensor([1.0 / T_max, 1.0 / alpha_max, 1.0 / delta_max], dtype=torch.float32),
        )
        self.register_buffer(
            "du_scale",
            torch.tensor([T_max, alpha_max, delta_max], dtype=torch.float32),
        )

        # will be set in set_equilibrium()
        self.set_equilibrium(x_eq, u_eq)

    @staticmethod
    def _logit(p: torch.Tensor) -> torch.Tensor:
        # p must be in (0,1)
        return torch.log(p) - torch.log1p(-p)

    @staticmethod
    def _atanh(y: torch.Tensor) -> torch.Tensor:
        # y must be in (-1,1)
        return 0.5 * (torch.log1p(y) - torch.log1p(-y))

    @torch.no_grad()
    def set_equilibrium(self, x_eq: torch.Tensor, u_eq: torch.Tensor):
        x_eq = x_eq.detach().to(dtype=torch.float32).view(3)
        u_eq = u_eq.detach().to(dtype=torch.float32).view(3)

        if hasattr(self, "x_eq"):
            self.x_eq.copy_(x_eq)
            self.u_eq.copy_(u_eq)
        else:
            self.register_buffer("x_eq", x_eq)
            self.register_buffer("u_eq", u_eq)

        # --- compute z_eq so that squash(z_eq) == u_eq exactly ---
        T_eq, a_eq, d_eq = self.u_eq[0], self.u_eq[1], self.u_eq[2]
        T_min, T_max = self.T_min, self.T_max
        amax, dmax = self.alpha_max, self.delta_max

        # must be strictly interior; otherwise exact inverse doesn't exist
        if not (float(T_min) < float(T_eq) < float(T_max)):
            raise ValueError(f"T_eq must satisfy T_min < T_eq < T_max. Got {float(T_eq)}.")
        if not (abs(float(a_eq)) < float(amax)):
            raise ValueError(f"Need |alpha_eq| < alpha_max. Got {float(a_eq)}.")
        if not (abs(float(d_eq)) < float(dmax)):
            raise ValueError(f"Need |delta_eq| < delta_max. Got {float(d_eq)}.")

        spanT = (T_max - T_min)
        pT = (T_eq - T_min) / spanT              # in (0,1)
        zT_eq = self._logit(pT)
        za_eq = self._atanh(a_eq / amax)         # in R
        zd_eq = self._atanh(d_eq / dmax)

        z_eq = torch.stack([zT_eq, za_eq, zd_eq]).to(dtype=torch.float32)

        if hasattr(self, "z_eq"):
            self.z_eq.copy_(z_eq)
        else:
            self.register_buffer("z_eq", z_eq)

        # --- slope matching at equilibrium ---
        # We choose dz = du_phys / slope_eq so that near eq:
        #   u ≈ u_eq + du_phys   (first-order identity)
        kT = float(self.k_T)
        ka = float(self.k_alpha)
        kd = float(self.k_delta)

        # thrust slope: dT/dz at eq
        p = float(pT)
        slope_T = float(spanT) * kT * (p * (1.0 - p))

        # angle slopes: d(alpha)/dz at eq
        a_norm = float(a_eq / amax)
        d_norm = float(d_eq / dmax)
        slope_a = float(amax) * ka * (1.0 - a_norm * a_norm)
        slope_d = float(dmax) * kd * (1.0 - d_norm * d_norm)

        if slope_T <= 0.0 or slope_a <= 0.0 or slope_d <= 0.0:
            raise ValueError("Equilibrium too close to bounds; slope becomes ~0 and matching is ill-conditioned.")

        inv_slope = torch.tensor([1.0 / slope_T, 1.0 / slope_a, 1.0 / slope_d], dtype=torch.float32)
        if hasattr(self, "inv_slope_eq"):
            self.inv_slope_eq.copy_(inv_slope)
        else:
            self.register_buffer("inv_slope_eq", inv_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) normalized error
        e = x - self.x_eq.unsqueeze(0)                 # (N,3)
        e_scaled = e * self.e_scale.unsqueeze(0)       # (N,3)

        # 2) normalized control
        du_hat = self.fc2(self.act(self.fc1(e_scaled)))  # (N,3), du_hat(0)=0 exactly

        # 3) physical offset (your logic)
        du_phys = du_hat * self.du_scale.unsqueeze(0)    # (N,3)

        # 4) convert physical offset -> latent offset (slope-matched)
        dz = du_phys * self.inv_slope_eq.unsqueeze(0)    # (N,3)

        z = self.z_eq.unsqueeze(0) + dz                  # (N,3)

        # 5) squash (bounded, smooth, no clamp)
        T = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.k_T * z[:, 0])
        alpha = self.alpha_max * torch.tanh(self.k_alpha * z[:, 1])
        delta = self.delta_max * torch.tanh(self.k_delta * z[:, 2])

        return torch.stack([T, alpha, delta], dim=1)

    @torch.no_grad()
    def verify_u_at_equilibrium(
        self,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        verbose: bool = True,
    ) -> bool:
        """
        Verify numerically that u(x_eq) == u_eq (within tolerance).
        """
        x = self.x_eq.view(1, 3)
        u = self.forward(x)                # (1,3)
        ueq = self.u_eq.view(1, 3)

        ok = torch.allclose(u, ueq, atol=atol, rtol=rtol)

        if verbose:
            err = (u - ueq).squeeze(0)
            print("u(x_eq)   =", u.squeeze(0).detach().cpu().numpy())
            print("u_eq      =", ueq.squeeze(0).detach().cpu().numpy())
            print("abs error =", err.abs().detach().cpu().numpy())
            print(f"allclose={bool(ok)} (atol={atol}, rtol={rtol})")

        return bool(ok)


class ConstantControl(nn.Module):
    """u(x) == u_eq for all x. Returns shape (N,3)."""
    def __init__(self, u_eq: torch.Tensor):
        super().__init__()
        u_eq = torch.as_tensor(u_eq, dtype=torch.float32).reshape(3)
        self.register_buffer("u_eq", u_eq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3) -> (N,3)
        return self.u_eq.unsqueeze(0).expand(x.shape[0], -1)


class ClosedLoopDrift(nn.Module):
    """
    Closed-loop XV-15 drift: x_dot = f(x, u(x)), where
      x = [v, gamma, beta]
      u(x) = [T, alpha, delta]

    IMPORTANT: This is NOT additive like Lorenz.
    """
    def __init__(
        self,
        aero: XV15KLinearAeroTorch,
        controller: nn.Module,
        *,
        v_min_safe: float = float(XV15Constants.VELOCITY_HORIZONTAL_MIN_SAFE),
    ):
        super().__init__()
        self.aero = aero
        self.controller = controller
        self.v_min_safe = float(v_min_safe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3) [v,gamma,beta]
        v = x[:, 0]
        gamma = x[:, 1]
        beta = x[:, 2]

        # controller output u(x) should be (N,3) = [T, alpha, delta]
        u = self.controller(x)
        T = u[:, 0]
        alpha = u[:, 1]
        delta = u[:, 2]

        # v_safe = torch.clamp(v, min=self.v_min_safe)
        v_safe = relu_clamp_min(v)

        L, D = self.aero.forces(v_safe, alpha, beta)

        m = float(XV15Constants.MASS)

        v_dot = (T*torch.cos((alpha+beta)) - D - m*G*torch.sin(gamma)) / m
        gamma_dot = (T*torch.sin((alpha+beta)) + L - m*G*torch.cos(gamma)) / (m * v_safe)
        beta_dot = delta

        return torch.stack([v_dot, gamma_dot, beta_dot], dim=1)


def animate_xv15_aircraft_state_control(
    *,
    f_cl_module,               # ClosedLoopDrift(aero, u_nn): x_dot = f(x)
    # u_nn,                      # controller: u(x) = [T, alpha, delta]
    g_fn=None,
    init_range: np.ndarray,    # (3,2) in (v,gamma,beta)
    goal_range: np.ndarray,    # (3,2) in (v,gamma,beta)
    full_range: np.ndarray,    # (3,2) in (v,gamma,beta)
    unsafe_boxes: np.ndarray,  # (K*3,2) or (K,3,2) or (3,2) in (v,gamma,beta)
    device: str = "cpu",
    dt: float = 0.02,
    T: float = 12.0,
    seed: int = 0,
    v_safe_min: float = 0.5,
    # visualization
    plane_len: float | None = None,
    show_angle_arcs: bool = True,
    save_path: str | None = None,   # e.g. "outputs/xv15_anim.mp4" (needs ffmpeg)
    show: bool = True,
    controller_label: str | None = None,
):
    """
    Layout (3 columns, left column has 2 rows):
      LEFT-TOP: aircraft animation in x-z (NO init/goal/unsafe boxes here)
      LEFT-BOT: 3D state trajectory (v, gamma, beta) with init/goal boxes + unsafe union boxes
      MIDDLE:   3 rows state vs time (v, gamma, beta) + shaded bands for init/goal/unsafe when limited
      RIGHT:    3 rows control vs time (T, alpha, delta)

    Notes:
      - state is (v,gamma,beta). position reconstructed by integrating:
          xdot = v cos(gamma), zdot = v sin(gamma)
      - unsafe is treated as UNION of axis-aligned boxes in state space.
    """

    rng = np.random.default_rng(seed)

    init_range = np.asarray(init_range, dtype=np.float32)
    goal_range = np.asarray(goal_range, dtype=np.float32)
    full_range = np.asarray(full_range, dtype=np.float32)

    def _as_union_boxes(x_unsafe: np.ndarray) -> np.ndarray:
        x = np.asarray(x_unsafe, dtype=np.float32)
        if x.ndim == 2:
            if x.shape == (3, 2):
                return x[None, ...]
            if x.shape[1] == 2 and (x.shape[0] % 3 == 0):
                K = x.shape[0] // 3
                return x.reshape(K, 3, 2)
            raise ValueError(f"unsafe_boxes 2D must be (3,2) or (K*3,2), got {x.shape}")
        if x.ndim == 3:
            if x.shape[1:] != (3, 2):
                raise ValueError(f"unsafe_boxes 3D must be (K,3,2), got {x.shape}")
            return x
        raise ValueError(f"unsafe_boxes must be (3,2), (K*3,2), or (K,3,2), got {x.shape}")

    unsafeK = _as_union_boxes(unsafe_boxes)  # (K,3,2)

    def _sample_in_box(box_3x2: np.ndarray, N: int) -> np.ndarray:
        low = box_3x2[:, 0]
        high = box_3x2[:, 1]
        return rng.uniform(low, high, size=(N, 3)).astype(np.float32)

    # -----------------------------------------
    # Rollout (main trajectory)
    # -----------------------------------------
    x0 = np.array([
        rng.uniform(init_range[0, 0], init_range[0, 1]),
        rng.uniform(init_range[1, 0], init_range[1, 1]),
        rng.uniform(init_range[2, 0], init_range[2, 1]),
    ], dtype=np.float32)

    N = int(T / dt) + 1
    t = np.linspace(0.0, T, N, dtype=np.float32)

    X = np.zeros((N, 3), dtype=np.float32)   # [v,gamma,beta]
    U = np.zeros((N, 3), dtype=np.float32)   # [T,alpha,delta]
    P = np.zeros((N, 2), dtype=np.float32)   # [x,z]

    X[0] = x0
    P[0] = np.array([0.0, 0.0], dtype=np.float32)

    f_cl_module.eval()
    # u_nn.eval()

    with torch.no_grad():
        for k in range(N - 1):
            xk_t = torch.tensor(X[k:k + 1], dtype=torch.float32, device=device)  # (1,3)

            uk = f_cl_module.controller(xk_t).detach().cpu().numpy().reshape(3)
            xdot = f_cl_module(xk_t).detach().cpu().numpy().reshape(3)

            xnext = X[k] + dt * xdot

            # keep v positive-ish for plotting/position integration
            # xnext[0] = max(float(v_safe_min), float(xnext[0]))
            # clamp to full range for sanity
            # xnext = np.minimum(np.maximum(xnext, full_range[:, 0]), full_range[:, 1])
            if g_fn is None:
                # Deterministic Euler
                xnext = X[k] + dt * xdot
            else:
                # Euler–Maruyama: x_{k+1} = x_k + f(x_k) dt + g(x_k) dW
                # with dW ~ sqrt(dt) * N(0, I)
                gk = g_fn(xk_t).detach().cpu().numpy().reshape(3)  # (3,) diagonal coeffs
                dW = (np.sqrt(dt) * rng.standard_normal(3)).astype(np.float32)  # (3,)
                xnext = X[k] + dt * xdot + gk * dW

            # integrate position from current (v, gamma)
            v_k = float(X[k, 0])
            g_k = float(X[k, 1])
            P[k + 1] = P[k] + dt * np.array([v_k * np.cos(g_k), v_k * np.sin(g_k)], dtype=np.float32)

            X[k + 1] = xnext
            U[k] = uk
        U[-1] = U[-2]

    # -----------------------------------------
    # Helpers for state-space boxes + bands
    # -----------------------------------------
    def _is_tighter(rng_1d: np.ndarray, full_1d: np.ndarray, eps: float = 1e-9) -> bool:
        return (rng_1d[0] > full_1d[0] + eps) or (rng_1d[1] < full_1d[1] - eps)

    def _add_band(ax, lo, hi, color, label=None, alpha=0.12):
        ax.axhspan(lo, hi, color=color, alpha=alpha, label=label, zorder=0)

    def _add_state_bands(ax, dim: int, ax_label: str):
        """
        Draw shaded init/goal bands (single box) and unsafe bands (union boxes)
        ONLY if that dimension is actually limited relative to full_range.
        """
        # init
        if _is_tighter(init_range[dim], full_range[dim]):
            _add_band(ax, init_range[dim, 0], init_range[dim, 1], color="green", label=f"init ({ax_label})")

        # goal
        if _is_tighter(goal_range[dim], full_range[dim]):
            _add_band(ax, goal_range[dim, 0], goal_range[dim, 1], color="blue", label=f"goal ({ax_label})")

        # unsafe union
        any_unsafe = False
        for k in range(unsafeK.shape[0]):
            lo, hi = unsafeK[k, dim, 0], unsafeK[k, dim, 1]
            if _is_tighter(unsafeK[k, dim], full_range[dim]):
                _add_band(ax, lo, hi, color="red", label=("unsafe" if not any_unsafe else None), alpha=0.10)
                any_unsafe = True

    def _collect_band_extents_for_dim(d: int):
        lows = []
        highs = []

        if _is_tighter(init_range[d], full_range[d]):
            lows.append(float(init_range[d, 0])); highs.append(float(init_range[d, 1]))
        if _is_tighter(goal_range[d], full_range[d]):
            lows.append(float(goal_range[d, 0])); highs.append(float(goal_range[d, 1]))

        for k in range(unsafeK.shape[0]):
            if _is_tighter(unsafeK[k, d], full_range[d]):
                lows.append(float(unsafeK[k, d, 0])); highs.append(float(unsafeK[k, d, 1]))

        if len(lows) == 0:
            return None
        return (min(lows), max(highs))

    def _set_ylim_with_bands(ax, y_data, band_extents):
        y = np.asarray(y_data, dtype=np.float32)
        y0 = float(np.min(y))
        y1 = float(np.max(y))
        if band_extents is not None:
            b0, b1 = band_extents
            y0 = min(y0, b0)
            y1 = max(y1, b1)
        if np.isclose(y0, y1):
            y0 -= 1.0
            y1 += 1.0
        pad = 0.10 * (y1 - y0)
        ax.set_ylim(y0 - pad, y1 + pad)

    def _set_ylim(ax, y):
        y = np.asarray(y, dtype=np.float32)
        y0 = float(np.min(y))
        y1 = float(np.max(y))
        if np.isclose(y0, y1):
            y0 -= 1.0
            y1 += 1.0
        pad = 0.10 * (y1 - y0)
        ax.set_ylim(y0 - pad, y1 + pad)

    # -----------------------------------------
    # 3D box drawing helpers
    # -----------------------------------------
    def _draw_box3d(ax3d, box3x2, *, lw=1.5, color="k", alpha=1.0):
        """
        Draw a 3D axis-aligned box as wireframe in (v, gamma, beta).
        box3x2: [[vmin,vmax],[gmin,gmax],[bmin,bmax]]
        """
        v0, v1 = float(box3x2[0, 0]), float(box3x2[0, 1])
        g0, g1 = float(box3x2[1, 0]), float(box3x2[1, 1])
        b0, b1 = float(box3x2[2, 0]), float(box3x2[2, 1])

        corners = np.array([
            [v0, g0, b0],
            [v1, g0, b0],
            [v1, g1, b0],
            [v0, g1, b0],
            [v0, g0, b1],
            [v1, g0, b1],
            [v1, g1, b1],
            [v0, g1, b1],
        ], dtype=np.float32)

        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom
            (4,5),(5,6),(6,7),(7,4),  # top
            (0,4),(1,5),(2,6),(3,7),  # verticals
        ]
        for (i, j) in edges:
            ax3d.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                lw=lw, color=color, alpha=alpha
            )

    # -----------------------------------------
    # Figure layout: 6 rows x 3 cols
    # Left column uses rows [0:3] for aircraft, [3:6] for 3D state
    # -----------------------------------------
    fig = plt.figure(figsize=(18, 9))
    title = "XV-15 rollout: state + control + geometry"
    if controller_label:
        title += f"  |  Controller: {controller_label}"

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92)  # leave room for the suptitle

    gs = fig.add_gridspec(
        6, 3,
        width_ratios=[1.55, 1.0, 1.0],
        height_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.30,
        hspace=0.55,
    )

    ax_air = fig.add_subplot(gs[0:3, 0])
    ax_3d  = fig.add_subplot(gs[3:6, 0], projection="3d")

    ax_sv = fig.add_subplot(gs[0:2, 1])
    ax_sg = fig.add_subplot(gs[2:4, 1], sharex=ax_sv)
    ax_sb = fig.add_subplot(gs[4:6, 1], sharex=ax_sv)

    ax_uT = fig.add_subplot(gs[0:2, 2])
    ax_ua = fig.add_subplot(gs[2:4, 2], sharex=ax_uT)
    ax_ud = fig.add_subplot(gs[4:6, 2], sharex=ax_uT)

    # -----------------------------------------
    # LEFT-TOP: aircraft (x-z)
    # -----------------------------------------
    ax_air.set_title("Aircraft (x–z)")
    ax_air.set_xlabel("x")
    ax_air.set_ylabel("z")
    ax_air.set_aspect("equal", adjustable="box")
    ax_air.grid(True, alpha=0.3)

    # bounds from trajectory only
    all_x = [P[:, 0].min(), P[:, 0].max()]
    all_z = [P[:, 1].min(), P[:, 1].max()]
    pad = 0.10 * max(1e-3, float(max(np.max(all_x) - np.min(all_x), np.max(all_z) - np.min(all_z))))
    xmin, xmax = float(np.min(all_x) - pad), float(np.max(all_x) + pad)
    zmin, zmax = float(np.min(all_z) - pad), float(np.max(all_z) + pad)
    ax_air.set_xlim(xmin, xmax)
    ax_air.set_ylim(zmin, zmax)

    travel = float(np.max(np.linalg.norm(P - P[0], axis=1)))
    if plane_len is None:
        plane_len = max(0.6, 0.03 * max(travel, 10.0))
    plane_w = 0.4 * plane_len

    def _plane_poly(center_xy: np.ndarray, theta_body: float) -> np.ndarray:
        pts = np.array([
            [ plane_len, 0.0],
            [-0.5 * plane_len,  0.6 * plane_w],
            [-0.5 * plane_len, -0.6 * plane_w],
        ], dtype=np.float32)
        c, s = np.cos(theta_body), np.sin(theta_body)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return pts @ R.T + center_xy[None, :]

    plane_patch = Polygon(_plane_poly(P[0], 0.0), closed=True, alpha=0.65)
    ax_air.add_patch(plane_patch)

    traj_line, = ax_air.plot([], [], lw=2)
    pt_line,   = ax_air.plot([], [], marker="o")

    v_line, = ax_air.plot([], [], lw=2)  # velocity direction
    t_line, = ax_air.plot([], [], lw=2)  # thrust direction
    l_line, = ax_air.plot([], [], lw=2)  # lift direction (approx)
    d_line, = ax_air.plot([], [], lw=2)  # drag direction (approx)

    arc_gamma = Arc((0, 0), width=1, height=1, angle=0, theta1=0, theta2=0, lw=2)
    arc_alpha = Arc((0, 0), width=1, height=1, angle=0, theta1=0, theta2=0, lw=2)
    arc_beta  = Arc((0, 0), width=1, height=1, angle=0, theta1=0, theta2=0, lw=2)
    if show_angle_arcs:
        ax_air.add_patch(arc_gamma)
        ax_air.add_patch(arc_alpha)
        ax_air.add_patch(arc_beta)

    txt_air = ax_air.text(0.02, 0.98, "", transform=ax_air.transAxes, va="top")

    def _set_arc(arc: Arc, center: np.ndarray, radius: float, th1_deg: float, th2_deg: float):
        arc.center = (float(center[0]), float(center[1]))
        arc.width = 2.0 * radius
        arc.height = 2.0 * radius
        arc.theta1 = th1_deg
        arc.theta2 = th2_deg

    def _seg(p, theta, s):
        q = p + s * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        return np.array([p[0], q[0]]), np.array([p[1], q[1]])

    # -----------------------------------------
    # LEFT-BOT: 3D state (v, gamma, beta) + boxes
    # -----------------------------------------
    ax_3d.set_title("State trajectory (v, γ, β) + init/goal/unsafe boxes")
    ax_3d.set_xlabel("v")
    ax_3d.set_ylabel("gamma")
    ax_3d.set_zlabel("beta")

    # Draw full range lightly (optional)
    _draw_box3d(ax_3d, full_range, lw=1.0, color="0.5", alpha=0.35)

    # init/goal single boxes
    _draw_box3d(ax_3d, init_range, lw=2.0, color="green", alpha=0.9)
    _draw_box3d(ax_3d, goal_range, lw=2.0, color="blue", alpha=0.9)

    # unsafe union boxes
    for k in range(unsafeK.shape[0]):
        _draw_box3d(ax_3d, unsafeK[k], lw=1.8, color="red", alpha=0.75)

    # 3D trajectory artists (updated during animation)
    line3d, = ax_3d.plot([], [], [], lw=2)
    pt3d,   = ax_3d.plot([], [], [], marker="o")

    # set 3D view limits based on full range
    ax_3d.set_xlim(float(full_range[0, 0]), float(full_range[0, 1]))
    ax_3d.set_ylim(float(full_range[1, 0]), float(full_range[1, 1]))
    ax_3d.set_zlim(float(full_range[2, 0]), float(full_range[2, 1]))
    ax_3d.view_init(elev=22, azim=-55)

    # -----------------------------------------
    # MIDDLE: states vs time (with bands)
    # -----------------------------------------
    ax_sv.set_title("States vs time")
    ax_sv.set_ylabel("v")
    ax_sg.set_ylabel("gamma")
    ax_sb.set_ylabel("beta")
    ax_sb.set_xlabel("t [s]")
    for ax in (ax_sv, ax_sg, ax_sb):
        ax.grid(True, alpha=0.3)

    _add_state_bands(ax_sv, 0, "v")
    _add_state_bands(ax_sg, 1, "gamma")
    _add_state_bands(ax_sb, 2, "beta")

    # band-aware y-limits
    _set_ylim_with_bands(ax_sv, X[:, 0], _collect_band_extents_for_dim(0))
    _set_ylim_with_bands(ax_sg, X[:, 1], _collect_band_extents_for_dim(1))
    _set_ylim_with_bands(ax_sb, X[:, 2], _collect_band_extents_for_dim(2))

    ax_sv.set_xlim(0, float(T))

    # Make legend only once (top state axis), avoiding duplicates
    handles, labels = ax_sv.get_legend_handles_labels()
    if len(handles) > 0:
        ax_sv.legend(loc="upper right", framealpha=0.85)

    # -----------------------------------------
    # RIGHT: controls vs time
    # -----------------------------------------
    ax_uT.set_title("Controls vs time")
    ax_uT.set_ylabel("T")
    ax_ua.set_ylabel("alpha")
    ax_ud.set_ylabel("delta")
    ax_ud.set_xlabel("t [s]")
    for ax in (ax_uT, ax_ua, ax_ud):
        ax.grid(True, alpha=0.3)

    _set_ylim(ax_uT, U[:, 0])
    _set_ylim(ax_ua, U[:, 1])
    _set_ylim(ax_ud, U[:, 2])
    ax_uT.set_xlim(0, float(T))

    # -----------------------------------------
    # Time-series lines (3 + 3)
    # -----------------------------------------
    lv, = ax_sv.plot([], [], lw=2)
    lg, = ax_sg.plot([], [], lw=2)
    lb, = ax_sb.plot([], [], lw=2)

    lT, = ax_uT.plot([], [], lw=2)
    la, = ax_ua.plot([], [], lw=2)
    ld, = ax_ud.plot([], [], lw=2)

    # -----------------------------------------
    # Animation init/update
    # -----------------------------------------
    def init_anim():
        traj_line.set_data([], [])
        pt_line.set_data([], [])
        v_line.set_data([], [])
        t_line.set_data([], [])
        l_line.set_data([], [])
        d_line.set_data([], [])
        txt_air.set_text("")

        lv.set_data([], [])
        lg.set_data([], [])
        lb.set_data([], [])
        lT.set_data([], [])
        la.set_data([], [])
        ld.set_data([], [])

        line3d.set_data([], [])
        line3d.set_3d_properties([])
        pt3d.set_data([], [])
        pt3d.set_3d_properties([])

        return (
            plane_patch, traj_line, pt_line, v_line, t_line, l_line, d_line,
            arc_gamma, arc_alpha, arc_beta, txt_air,
            lv, lg, lb, lT, la, ld,
            line3d, pt3d
        )

    def update(i: int):
        i = int(i)
        xs = X[:i + 1]
        us = U[:i + 1]
        ps = P[:i + 1]

        v = float(xs[-1, 0])
        gamma = float(xs[-1, 1])
        beta = float(xs[-1, 2])
        Tcmd = float(us[-1, 0])
        alpha = float(us[-1, 1])

        # headings for glyph (rough)
        th_v = gamma
        th_body = gamma + alpha
        th_T = gamma + alpha + beta

        plane_patch.set_xy(_plane_poly(ps[-1], th_body))

        traj_line.set_data(ps[:, 0], ps[:, 1])
        pt_line.set_data([ps[-1, 0]], [ps[-1, 1]])

        # vector lengths (purely visual scaling)
        s_v = 0.15 * plane_len + 0.02 * v
        s_T = 0.15 * plane_len + 0.00002 * abs(Tcmd)
        s_L = 0.9 * plane_len
        s_D = 0.9 * plane_len

        xv, zv = _seg(ps[-1], th_v, s_v)
        xT, zT = _seg(ps[-1], th_T, s_T)
        xL, zL = _seg(ps[-1], th_v + 0.5 * np.pi, s_L)
        xD, zD = _seg(ps[-1], th_v + np.pi, s_D)

        v_line.set_data(xv, zv)
        t_line.set_data(xT, zT)
        l_line.set_data(xL, zL)
        d_line.set_data(xD, zD)

        if show_angle_arcs:
            center = ps[-1]
            r1, r2, r3 = 0.9 * plane_len, 1.2 * plane_len, 1.5 * plane_len

            def arc_deg(a1, a2):
                a1d = np.degrees(a1)
                a2d = np.degrees(a2)
                if a2d < a1d:
                    a2d += 360.0
                return float(a1d), float(a2d)

            g1, g2 = arc_deg(0.0, th_v)
            a1, a2 = arc_deg(th_v, th_body)
            b1, b2 = arc_deg(th_body, th_T)
            _set_arc(arc_gamma, center, r1, g1, g2)
            _set_arc(arc_alpha, center, r2, a1, a2)
            _set_arc(arc_beta,  center, r3, b1, b2)

        # middle: states vs time
        lv.set_data(t[:i + 1], xs[:, 0])
        lg.set_data(t[:i + 1], xs[:, 1])
        lb.set_data(t[:i + 1], xs[:, 2])

        # right: controls vs time
        lT.set_data(t[:i + 1], us[:, 0])
        la.set_data(t[:i + 1], us[:, 1])
        ld.set_data(t[:i + 1], us[:, 2])

        # 3D: state trajectory
        line3d.set_data(xs[:, 0], xs[:, 1])
        line3d.set_3d_properties(xs[:, 2])
        pt3d.set_data([xs[-1, 0]], [xs[-1, 1]])
        pt3d.set_3d_properties([xs[-1, 2]])

        txt_air.set_text(
            f"t={t[i]:.2f}s\n"
            f"v={v:.2f}, gamma={gamma:.3f}, beta={beta:.3f}\n"
            f"T={Tcmd:.2f}, alpha={alpha:.3f}"
        )

        return (
            plane_patch, traj_line, pt_line, v_line, t_line, l_line, d_line,
            arc_gamma, arc_alpha, arc_beta, txt_air,
            lv, lg, lb, lT, la, ld,
            line3d, pt3d
        )

    frame_skip = 20          # show 1 out of every 5 steps  (5x faster)
    frames = range(0, N, frame_skip)
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init_anim,
        interval=30,   # can keep this, or lower it too
        blit=False,
    )

    if save_path is not None:
        ani.save(save_path, dpi=150)

    if show:
        plt.show()

    return {"t": t, "X": X, "U": U, "P": P, "x0": x0}


def mc_reach_avoid(
    *,
    f_cl_module,              # torch module: xdot = f(x), x shape (1,3)
    g_fn=None,                # torch callable: g(x) -> (3,) or (1,3); if None => deterministic
    init_range: np.ndarray,   # (3,2)
    goal_range: np.ndarray,   # (3,2)
    full_range: np.ndarray,   # (3,2)
    unsafe_boxes: np.ndarray, # (K*3,2) or (K,3,2) or (3,2)
    N_trials: int = 200,
    dt: float = 0.02,
    T: float = 12.0,
    seed: int = 0,
    device: str = "cpu",
):
    """Monte-Carlo reach-avoid probability over [0,T]. Uses ONLY numpy RNG (seeded)."""

    rng = np.random.default_rng(seed)

    init_range = np.asarray(init_range, np.float32)
    goal_range = np.asarray(goal_range, np.float32)
    full_range = np.asarray(full_range, np.float32)
    ub = np.asarray(unsafe_boxes, np.float32)

    # unsafe -> (K,3,2)
    if ub.ndim == 2:
        ub = ub[None, ...] if ub.shape == (3, 2) else ub.reshape(-1, 3, 2)

    def in_box(x, box3x2):
        return bool(np.all(x >= box3x2[:, 0]) and np.all(x <= box3x2[:, 1]))

    def in_unsafe(x):
        for k in range(ub.shape[0]):
            if in_box(x, ub[k]):
                return True
        return False

    N_steps = int(T / dt) + 1
    sqrt_dt = float(np.sqrt(dt))

    f_cl_module.eval()

    successes = 0
    with torch.no_grad():
        for _ in range(N_trials):
            # sample x0 in init
            x = np.array([rng.uniform(*init_range[d]) for d in range(3)], dtype=np.float32)

            failed = False
            for _k in range(N_steps - 1):
                # fail if outside full or unsafe
                if (not in_box(x, full_range)) or in_unsafe(x):
                    failed = True
                    break
                # succeed if in goal
                if in_box(x, goal_range):
                    break

                xt = torch.tensor(x[None, :], dtype=torch.float32, device=device)
                xdot = f_cl_module(xt).detach().cpu().numpy().reshape(3).astype(np.float32)

                if g_fn is None:
                    x = x + dt * xdot
                else:
                    gk = g_fn(xt).detach().cpu().numpy().reshape(3).astype(np.float32)
                    dW = (sqrt_dt * rng.standard_normal(3)).astype(np.float32)
                    x = x + dt * xdot + gk * dW

            if (not failed) and in_box(x, goal_range):
                successes += 1

    return {
        "p_reach_avoid": successes / float(N_trials),
        "successes": int(successes),
        "N_trials": int(N_trials),
    }


def pretrain_network_samples(
    model,
    x_goal_range,
    x_unsafe_range,
    x_init_range,
    x_range,
    params,
    GV_net=None,
    num_epochs=1000,
    lr=0.01,
    device='cpu',
    control_net=None,
    n_each: int = 400,   # samples per region per epoch
    lambda_w = 1.0,
    save_v_path=None,            # NEW
    save_control_path=None,      # NEW
):
    """
    Pre-train V network using sampled points to match constraint structure.

    Supports x_unsafe_range as:
      1) (D,2) single box
      2) (K,D,2) union of K boxes
      3) (2*D,2) produced by np.vstack((box1, box2, ...))  <-- your case

    Losses:
      - full-range: enforce v(x) >= 0
      - init-range: enforce v(x) <= 1
      - unsafe-range: enforce v(x) >= pretrain_unsafe_target
      - optional phi loss on x_others: enforce phi(x) <= 0
    """
    print("\n" + "="*80)
    print("PRE-TRAINING: Constraint-Structured Initialization (Sample-Based)")
    print("="*80)

    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    if GV_net is not None:
        print("  GV (Φ) pre-training ENABLED (using provided GV_net)")
    else:
        print("  GV (Φ) pre-training DISABLED (no GV_net provided)")

    best_loss = float('inf')
    best_model_state = None
    best_control_state = None

    # Convert ranges to torch once: each is (D,2)
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t    = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    init_t    = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
    low  = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low
    v_eq = model.output_offset.detach()

    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        """box: (D,2) -> samples: (N,D) uniform in box."""
        b_low = box[:, 0]
        b_high = box[:, 1]
        return torch.rand(N, D, device=device) * (b_high - b_low) + b_low

    def _in_box(x_batch: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """x_batch: (N,D), box: (D,2) -> mask: (N,)"""
        return ((x_batch >= box[:, 0]) & (x_batch <= box[:, 1])).all(dim=1)

    # -----------------------------
    # Unsafe region: allow union of boxes
    # -----------------------------
    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
        """
        Returns unsafe_boxes as torch.Tensor of shape (K,D,2).
        Accepts:
          - (D,2)
          - (K,D,2)
          - (K*D,2) from np.vstack((box1, box2, ...))
        """
        t = torch.as_tensor(x_unsafe, dtype=torch.float32, device=device)

        if t.dim() == 2:
            if t.shape == (D, 2):
                return t.unsqueeze(0)  # (1,D,2)
            if t.shape[1] == 2 and (t.shape[0] % D == 0):
                K = int(t.shape[0] // D)
                return t.view(K, D, 2)  # (K,D,2)  <-- handles vstack case
            raise ValueError(f"x_unsafe_range 2D must be (D,2) or (K*D,2); got {tuple(t.shape)}")

        if t.dim() == 3:
            if t.shape[1:] != (D, 2):
                raise ValueError(f"x_unsafe_range 3D must be (K,D,2) with D={D}; got {tuple(t.shape)}")
            return t

        raise ValueError(f"x_unsafe_range must be (D,2), (K,D,2), or (K*D,2); got {tuple(t.shape)}")

    unsafe_boxes = _unsafe_to_boxes(x_unsafe_range)  # (K,D,2)
    K_unsafe = int(unsafe_boxes.shape[0])

    def _in_unsafe_union(x_batch: torch.Tensor) -> torch.Tensor:
        """mask True if x is inside ANY unsafe box."""
        mask = torch.zeros(x_batch.shape[0], dtype=torch.bool, device=device)
        for k in range(K_unsafe):
            mask |= _in_box(x_batch, unsafe_boxes[k])
        return mask

    def _sample_in_unsafe_union(N: int) -> torch.Tensor:
        """
        Sample N points from EACH of the K unsafe boxes (torch only).

        Returns:
        x: (K_unsafe * N, D)

        Notes:
        - If K_unsafe == 1, this is just N samples from that box.
        - Shuffles so the batch is not grouped by box.
        """
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")

        if K_unsafe == 1:
            return _sample_in_box(unsafe_boxes[0], N)

        xs = []
        for k in range(K_unsafe):
            xs.append(_sample_in_box(unsafe_boxes[k], N))  # (N, D) per box

        x = torch.cat(xs, dim=0)  # (K_unsafe * N, D)
        x = x[torch.randperm(x.shape[0], device=device)]  # shuffle
        return x
    
    def _l2_weight_penalty(model: torch.nn.Module, exclude_bias: bool = True) -> torch.Tensor:
        reg = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_bias and (p.dim() == 1 or name.endswith("bias")):
                continue
            reg = reg + (p ** 2).sum()
        return reg

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # 1) full-range samples -> enforce v(x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # 2) init-range samples -> enforce v(x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # 3) unsafe-range samples -> enforce v(x) >= pretrain_unsafe_target
        x_unsafe = _sample_in_unsafe_union(int(n_each/6))
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.pretrain_unsafe_target - v_unsafe).sum()

        # 4) goal-range samples -> enforce v(x) <= 1.0
        x_goal = _sample_in_box(goal_t, n_each)
        v_goal = model(x_goal).squeeze(-1)
        v_loss_inside_goal = F.relu(0.0 - v_goal).sum()
        # miv_v_goal = torch.min(v_goal)

        # 5) samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= 0.0
        x_others_list = []
        need = n_each
        max_tries = 20
        tries = 0
        while need > 0 and tries < max_tries:
            tries += 1
            x_cand = torch.rand(max(need * 4, 32), D, device=device) * span + low
            cand_in_goal = _in_box(x_cand, goal_t)
            cand_in_unsafe = _in_unsafe_union(x_cand)
            keep = ~(cand_in_goal | cand_in_unsafe)
            x_keep = x_cand[keep]
            if x_keep.shape[0] > 0:
                take = min(need, x_keep.shape[0])
                x_others_list.append(x_keep[:take])
                need -= take

        if len(x_others_list) == 0:
            x_others = x_full.detach()
        else:
            x_others = torch.cat(x_others_list, dim=0)

        v_others = model(x_others).squeeze(-1)
        v_loss_others = F.relu(v_eq - v_others).sum()

        # Total V loss (v_loss_inside_goal and v_loss_others are not used anymore)
        loss_v = (v_loss_full
            + v_loss_init
            + v_loss_unsafe
            + v_loss_inside_goal
            + v_loss_others
        )

        # Phi loss on SAME x_others -> enforce phi(x) <= 0
        loss_phi = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_phi = x_others.detach().clone().requires_grad_(True)
            phi_output = GV_net(x_phi).squeeze(-1)
            loss_phi = F.relu(phi_output).sum()

        total_loss = loss_v + loss_phi

        # Add regularization for V network
        reg_w = _l2_weight_penalty(model, exclude_bias=True)
        reg_w_control = _l2_weight_penalty(control_net, exclude_bias=True)
        # print(reg_w)
        total_loss = total_loss + lambda_w * (reg_w)

        # Track best
        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        # Logging
        if epoch % 100 == 0:
            if GV_net is not None:
                print(
                    f"  Epoch [{epoch}/{num_epochs}]: "
                    f"V={loss_v.item():.3f} "
                    f"(full={v_loss_full.item():.3f}, init={v_loss_init.item():.3f}, "
                    f"unsafe={v_loss_unsafe.item():.3f}, "
                    f"goal={v_loss_inside_goal.item():.3f}), {v_loss_others.item():.3f}"
                    f"Φ={loss_phi.item():.3e}, Reg={reg_w.item():.3f}, {reg_w_control.item():.3f}, Total={total_loss.item():.3e}"
                )
                print(model(model.input_offset))
            else:
                print(
                    f"  Epoch [{epoch}/{num_epochs}]: "
                    f"V={loss_v.item():.6f} "
                    f"(full={v_loss_full.item():.3f}, init={v_loss_init.item():.3f}, "
                    f"unsafe={v_loss_unsafe.item():.3f}, "
                    f"goal={v_loss_inside_goal.item():.3f})"
                )

        # Optimize/update networks
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\n  Best loss: {best_loss:.6f}")

        # NEW: save best pretrained weights (state_dict)
        if save_v_path is not None:
            torch.save(best_model_state, save_v_path)
            print(f"  Saved pretrained V_net to: {save_v_path}")
        if (control_net is not None) and (best_control_state is not None) and (save_control_path is not None):
            torch.save(best_control_state, save_control_path)
            print(f"  Saved pretrained Controller_net to: {save_control_path}")

    print("="*80)
    networks_trained = ["V"]
    if GV_net is not None:
        networks_trained.append("GV (Φ)")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")
    print("="*80 + "\n")


def train_network_bounds(
    V_net,
    GV_net,
    region_cells: dict,
    regions: Regions,
    params: Hyperparameters,
    device: str = 'cpu',
    visualize_interval: int = 5000,
    control_net: nn.Module = None
):
    """
    Train the value network using CROWN bounds.

    Args:
        V_net: Value network
        GV_net: Generator network
        region_cells: Dictionary of discretized cells
        regions: Regions object
        params: Hyperparameters
        device: Device for training
        visualize_interval: Interval for visualization (0 to disable)
        control_net: if this is provided, then we do [control synthesis]
    """
    print("\n" + "="*80)
    print("BOUND-BASED TRAINING (using CROWN)")
    print("="*80)

    # Move models to device
    V_net = V_net.to(device)
    v_eq = V_net.output_offset.detach()

    def _log_V_zero_at_offset(V_net, epoch: int, device: str, every: int = 100, atol: float = 1e-6):
        if epoch % every != 0:
            return
        V_net.eval()
        with torch.no_grad():
            x0 = V_net.input_offset.to(device=device, dtype=next(V_net.parameters()).dtype)
            y0 = V_net(x0).item()  # (out,) or scalar-ish
            print(f"[Check] epoch={epoch:06d}  {y0:.3e} ")
        V_net.train()

    # Collect ALL cells in the same order as original (init, goal, unsafe, outside, generator)
    print("\nCollecting all cells for V network...")
    all_cells_V = []
    cell_counts_V = {}
    region_order_V = ['init', 'goal', 'unsafe', 'outside']

    for name in region_order_V:
        cells = region_cells[name]
        all_cells_V.extend(cells)
        cell_counts_V[name] = len(cells)
        print(f"  {name}: {len(cells)} cells")

    total_cells_V = len(all_cells_V)
    print(f"  Total V cells: {total_cells_V}")

    # Prepare ALL input bounds at once (matching original)
    print("\nPreparing concatenated input bounds for V network...")
    if total_cells_V > 0:
        input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)
    else:
        input_lowers_all = torch.empty(0, params.network.n_inputs, device=device)
        input_uppers_all = torch.empty(0, params.network.n_inputs, device=device)

    # Create ONE big CROWN cache for ALL V cells (matching original!)
    print(f"\nInitializing CROWN cache for ALL {total_cells_V} V cells...")
    crown_cache_all = SymbolicCROWNCache(
        model=V_net,
        num_cells=total_cells_V,
        input_dim=params.network.n_inputs,
        device=device
    )

    # Create CROWN cache for generator (Phi) - separate cache
    crown_cache_phi = None
    input_lowers_gen = None
    input_uppers_gen = None
    if len(region_cells['generator']) > 0 and params.training.generator_weight > 0:
        print(f"\nCreating CROWN cache for 'generator' (Phi)...")
        print(f"  generator: {len(region_cells['generator'])} cells")
        crown_cache_phi = SymbolicCROWNCache_Phi(
            phi_module=GV_net,
            num_cells=len(region_cells['generator']),
            input_dim=params.network.n_inputs,
            device=device
        )
        # Prepare generator input bounds
        input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

    learnable_beta_s = None
    beta_s_value = params.constraints.beta_s
    print(f"\nUsing CONSTANT beta_s = {beta_s_value}")
    opt_params = list(V_net.parameters())
    # [control synthesis]
    if control_net is not None:
        opt_params += list(control_net.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=params.training.learning_rate)

    # Scheduler (matching testing_simple3.py)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,   # every 2000 epochs
        gamma=0.95         # multiply lr by 0.5
    )

    # Training loop
    loss_history = []
    refinement_epochs = {'outside': [], 'generator': []}
    start_time = time.time()
    final_beta_s = params.constraints.beta_s 
    # Compute total loss from bounds
    loss_kwargs = {
        'beta_ra': params.constraints.beta_ra,
        'device': params.training.device,
        'compute_V': params.compute_V,
        'compute_GV': params.compute_GV,
    }

    for epoch in range(params.training.num_epochs):
        V_net.train()

        optimizer.zero_grad()
        if params.compute_V:
            # Compute bounds for ALL V cells at once (matching original!)
            if total_cells_V > 0:
                v_lowers_all, v_uppers_all = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all = torch.tensor([], device=device)
                v_uppers_all = torch.tensor([], device=device)

            # Split bounds by region (matching original's split_bounds_by_region)
            bounds = {}
            cell_idx = 0
            for name in region_order_V:
                num_cells = cell_counts_V[name]
                if num_cells > 0:
                    bounds[name] = (
                        v_lowers_all[cell_idx:cell_idx + num_cells],
                        v_uppers_all[cell_idx:cell_idx + num_cells]
                    )
                    cell_idx += num_cells
                else:
                    bounds[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        if params.compute_GV: 
            # Compute generator bounds if enabled
            needs_cache_rebuild = False
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None):
                phi_uppers = crown_cache_phi.compute_bounds(input_lowers_gen, input_uppers_gen)
                current_gen_weight = params.training.generator_weight

                # Track failing cells for adaptive refinement
                phi_upper_failing_mask = phi_uppers > 0.0
                num_total_failing = phi_upper_failing_mask.sum().item()
                
            else:
                phi_uppers = torch.tensor([], device=device)
                current_gen_weight = 0.0
                num_total_failing = 0

        # Get current beta_s value (learnable or constant)
        if learnable_beta_s is not None:
            current_beta_s = learnable_beta_s.value
        else:
            current_beta_s = beta_s_value

        # Update loss kwargs with current bounds (reuse pre-allocated dict)
        if params.compute_V:
            loss_kwargs['beta_ra'] = params.constraints.beta_ra
            loss_kwargs['V_goal_lower'] = bounds['goal'][0]
            loss_kwargs['V_unsafe_lower'] = bounds['unsafe'][0]
            loss_kwargs['V_init_upper'] = bounds['init'][1]
            loss_kwargs['V_outside_lower'] = bounds['outside'][0]

        if params.compute_GV:
            loss_kwargs['Phi_upper'] = phi_uppers
            loss_kwargs['generator_weight'] = current_gen_weight

        total_loss, loss_dict, _ = compute_total_loss_bounds(**loss_kwargs)

        # Backward pass
        total_loss.backward()

        # Recompute bounds after optimizer step for verification
        V_net.eval()
        with torch.no_grad():
            if total_cells_V > 0:
                v_lowers_all_updated, v_uppers_all_updated = crown_cache_all.compute_bounds(input_lowers_all, input_uppers_all)
            else:
                v_lowers_all_updated = torch.tensor([], device=device)
                v_uppers_all_updated = torch.tensor([], device=device)

            # Split updated bounds by region
            bounds_updated = {}
            cell_idx = 0
            for name in region_order_V:
                num_cells = cell_counts_V[name]
                if num_cells > 0:
                    bounds_updated[name] = (
                        v_lowers_all_updated[cell_idx:cell_idx + num_cells],
                        v_uppers_all_updated[cell_idx:cell_idx + num_cells]
                    )
                    cell_idx += num_cells
                else:
                    bounds_updated[name] = (torch.tensor([], device=device), torch.tensor([], device=device))

        # Get current beta_s value for constraint checks
        beta_s_check = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s

        # Adaptive refinement for V outside region cells
        needs_cache_rebuild = False
        if params.compute_V and len(bounds_updated['outside'][0]) > 0:
            # Track failing cells in outside region
            outside_failing_mask = bounds_updated['outside'][0] <= 0.0
            num_outside_failing = outside_failing_mask.sum().item()

            # Ensure bounds match current cell count
            if len(outside_failing_mask) == len(region_cells['outside']) and num_outside_failing > 0:
                REFINE_INTERVAL = 200
                REFINE_FACTOR = 2
                MAX_CELLS = 40000

                # if epoch > 2500:
                #     REFINE_INTERVAL = 50

                if ((epoch + 1) % REFINE_INTERVAL == 0 and
                    len(region_cells['outside']) < MAX_CELLS):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['outside'],
                        outside_failing_mask,
                        REFINE_FACTOR,
                        scores=-bounds_updated['outside'][0]
                    )
                    region_cells['outside'] = new_cells
                    print(f"[Refine-Outside] Epoch {epoch+1}: {num_outside_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['outside'].append(epoch + 1)

            if((epoch + 1) % 501 == 0):
                outside_failing_mask_relax = bounds_updated['outside'][0] <= 8.0 + v_eq
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['outside'],
                    outside_failing_mask_relax,
                    max_passes=4,
                    max_merges=None,   # cap work; set None for full greedy
                    seed=0,
                    eps=1e-6,
                )
                region_cells['outside'] = merged_cells
                print(f"[Merge-Outside] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Adaptive refinement for generator cells (before optimizer step)
        if params.compute_GV:
            if (epoch >= params.training.generator_start_epoch and
                params.training.generator_weight > 0 and
                crown_cache_phi is not None and
                num_total_failing > 0):

                # Refinement parameters (matching testing_simple3.py)
                REFINE_INTERVAL = 500  # Refine every 100k epochs
                REFINE_FACTOR = 2  # Split into 2x2 subcells
                MAX_CELLS = 100000  # Don't refine if we already have too many cells
                N_TO_REFINE = 100

                # # Adjust interval for later epochs
                # if epoch > 2500:
                #     REFINE_INTERVAL = 250

                # Check if it's time to refine
                if ((epoch + 1) % REFINE_INTERVAL == 0 and
                    len(region_cells['generator']) < MAX_CELLS):
                    new_cells, num_refined = refine_failing_cells(
                        region_cells['generator'],
                        phi_upper_failing_mask,
                        REFINE_FACTOR,
                        N_to_refine=N_TO_REFINE,
                        scores=phi_uppers,
                    )
                    region_cells['generator'] = new_cells
                    print(f"[Refine-Generator] Epoch {epoch+1}: {num_total_failing} failing → refined {num_refined} cells → {len(new_cells)} total")
                    needs_cache_rebuild = True
                    refinement_epochs['generator'].append(epoch + 1)
            
            if((epoch + 1) % 501 == 0):
                phi_upper_failing_mask_relax = phi_uppers > -500.0
                merged_cells, num_merges = merge_passing_neighbor_cells(
                    region_cells['generator'],
                    phi_upper_failing_mask_relax,
                    max_passes=8,
                    max_merges=None,   # cap work; set None for full greedy
                    seed=0,
                    eps=1e-6,
                )
                region_cells['generator'] = merged_cells
                print(f"[Merge-Generator] Epoch {epoch+1}: merged {num_merges} pairs → {len(merged_cells)} total")
                needs_cache_rebuild = True

        # Logging
        if epoch % 10 == 0 or epoch == params.training.num_epochs - 1 or epoch == 0:
            # Add beta_s to loss dict for logging
            if learnable_beta_s is not None:
                beta_s_log = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s
                print(f"Epoch [{epoch}/{params.training.num_epochs}]: Loss={total_loss.item():.4f}, β_s={beta_s_log:.4f}")
            print_loss_summary(epoch, loss_dict, compute_V=params.compute_V, compute_GV=params.compute_GV)
            # if control_net is not None:
            #     for name, param in control_net.named_parameters():
            #         if param.requires_grad:
            #             print(f" [Controller Params] {name} = {param.data}")
            loss_dict['epoch'] = epoch
            if learnable_beta_s is not None:
                loss_dict['beta_s'] = beta_s_log
            loss_history.append(loss_dict.copy())

        # Early stopping check
        with torch.no_grad():
            # Check V constraints
            all_satisfied = True
            if params.compute_V:
                show = (epoch % 10 == 0)
                _, goal_satisfied = compute_loss_goal_bounds(bounds_updated["goal"][0])
                unsafe_satisfied = (bounds_updated['unsafe'][0].min() >= params.constraints.beta_ra)
                init_satisfied = (bounds_updated['init'][1].max() <= 1.0)
                outside_satisfied = (bounds_updated['outside'][0].min() >= 0.0)
                all_satisfied = all_satisfied and goal_satisfied and unsafe_satisfied and init_satisfied and outside_satisfied

            # Check GV constraints
            if params.compute_GV:
                generator_satisfied = (phi_uppers.max() < 0.0)
                all_satisfied = all_satisfied and generator_satisfied

            # Early stop if all active constraints are satisfied
            if all_satisfied:
                print("\n" + "="*80)
                print("ALL CONSTRAINTS SATISFIED - EARLY STOPPING!")
                print("="*80)
                print(f"Training converged at epoch {epoch}")

                # Print relevant losses
                loss_parts = []
                if params.compute_V:
                    loss_parts.append(f"Goal={loss_dict['goal']:.4f}, Unsafe={loss_dict['unsafe']:.4f}, Init={loss_dict['init']:.4f}, Outside={loss_dict['outside']:.4f}")
                if params.compute_GV:
                    loss_parts.append(f"Gen={loss_dict['generator']:.4f}")
                print(f"Final losses: {', '.join(loss_parts)}")

                if control_net is not None:
                    for name, param in control_net.named_parameters():
                        if param.requires_grad:
                            print(f" [Controller Params] {name} = {param.data}")

                final_beta_s = bounds_updated['outside'][0].min()
                print("final beta_s: {:.4f}".format(final_beta_s))

                break

        # Detailed evaluation and visualization
        if (epoch % 500 == 0) or epoch == params.training.num_epochs - 1:
            print(f"\nEpoch {epoch} - Detailed Evaluation:")
            # For evaluation, we can just create temporary caches (not in the hot path)
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                crown_cache_all=crown_cache_all,
                crown_cache_phi=crown_cache_phi,
                input_bounds_all=(input_lowers_all, input_uppers_all),
                cell_counts_V=cell_counts_V,
                input_bounds_gen=(input_lowers_gen, input_uppers_gen) if input_lowers_gen is not None else None,
                beta_ra=params.constraints.beta_ra,
                device=device,
            )
            print_constraint_summary(results, prefix="  ")

            # Print bound statistics
            if params.compute_V:
                if len(bounds['goal'][0]) > 0:
                    print(f"  Goal bounds: V ∈ [{bounds['goal'][0].min().item():.3f}, {bounds['goal'][1].max().item():.3f}]")
                if len(bounds['unsafe'][0]) > 0:
                    print(f"  Unsafe bounds: V ∈ [{bounds['unsafe'][0].min().item():.3f}, {bounds['unsafe'][1].max().item():.3f}]")
            if params.compute_GV:
                if len(phi_uppers) > 0:
                    print(f"  Generator bounds: Φ ∈ [{phi_uppers.min().item():.6e}, {phi_uppers.max().item():.6e}]")
                    failing_cells = (phi_uppers > 0).sum().item()
                    print(f"  Generator failing cells: {failing_cells}/{len(phi_uppers)} ({100*failing_cells/len(phi_uppers):.1f}%)")
            print()
            # Visualize progress
        
        # _log_V_zero_at_offset(V_net, epoch, device, every=10, atol=1e-6)  # before step
        # Optimizer step
        optimizer.step()
        scheduler.step()
        _log_V_zero_at_offset(V_net, epoch, device, every=10, atol=1e-6)  # before step

        # Rebuild CROWN caches after optimizer step if needed (after adaptive refinement)
        # if params.compute_GV:
        if True:
            if needs_cache_rebuild:
                # with torch.no_grad():
                print(f"  Rebuilding CROWN caches with new generator cells...")

                # Rebuild V cache with all cells (in order: init, goal, unsafe, outside)
                # Note: generator cells are NOT included in V cache
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])

                total_cells_V = len(all_cells_V)
                input_lowers_all, input_uppers_all = prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

                # Rebuild V CROWN cache
                print(f"    Rebuilding V cache with {total_cells_V} cells...")
                crown_cache_all = SymbolicCROWNCache(
                    model=V_net,
                    num_cells=total_cells_V,
                    input_dim=params.network.n_inputs,
                    device=device
                )

                # Rebuild Phi CROWN cache (only for generator region)
                num_generator_cells = len(region_cells['generator'])
                print(f"    Rebuilding Phi cache with {num_generator_cells} cells...")
                crown_cache_phi = SymbolicCROWNCache_Phi(
                    phi_module=GV_net,
                    num_cells=num_generator_cells,
                    input_dim=params.network.n_inputs,
                    device=device
                )

                # Rebuild generator input bounds
                input_lowers_gen, input_uppers_gen = prepare_cell_bounds(region_cells['generator'], device, input_dim=params.network.n_inputs)

                # Update cell counts after refinement
                for name in region_order_V:
                    cell_counts_V[name] = len(region_cells[name])
                
                # Rebuild concatenated input bounds for V
                all_cells_V = []
                for name in region_order_V:
                    all_cells_V.extend(region_cells[name])
                    input_lowers_all, input_uppers_all =  prepare_cell_bounds(all_cells_V, device, input_dim=params.network.n_inputs)

                print(f"  Caches rebuilt successfully!")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Return final beta_s value along with loss history
    # final_beta_s = current_beta_s.item() if isinstance(current_beta_s, torch.Tensor) else current_beta_s
    # if learnable_beta_s is not None:
    #     print(f"\nFinal learned β_s = {final_beta_s:.4f}")

    return loss_history, final_beta_s, refinement_epochs


def _diag_ggt_from_g(g_out: torch.Tensor, D: int) -> torch.Tensor:
    """
    Convert g(x) output to diag(GG^T) in shape (N,D), (D,) or scalar.

    Supports:
      - (N,D)      : diagonal diffusion vector
      - (N,D,m)    : general diffusion factors
      - (N,D,D)    : full matrix per sample
      - (D,D)      : constant matrix
      - (D,)       : constant diagonal vector
      - scalar     : isotropic
    """
    if isinstance(g_out, (float, int)):
        return float(g_out) ** 2

    if g_out.dim() == 0:
        return float(g_out.item()) ** 2

    if g_out.dim() == 1:
        if g_out.numel() != D:
            raise ValueError(f"g_out is (D,) but numel={g_out.numel()} != D={D}")
        return g_out.square()  # (D,)

    if g_out.dim() == 2:
        # (D,D) constant matrix or (N,D) diagonal vector
        if g_out.shape == (D, D):
            return g_out.square().sum(dim=1)  # (D,)
        if g_out.shape[1] != D:
            raise ValueError(f"g_out is (N,D) but second dim={g_out.shape[1]} != D={D}")
        return g_out.square()  # (N,D)

    if g_out.dim() == 3:
        # (N,D,m) or (N,D,D): diag(GG^T) = sum_k G_{i,k}^2
        if g_out.shape[1] != D:
            raise ValueError(f"g_out is (N,D,*) but dim1={g_out.shape[1]} != D={D}")
        return g_out.square().sum(dim=2)  # (N,D)

    raise ValueError(f"Unsupported g_out shape: {tuple(g_out.shape)}")


def check_gv_matches_autograd_full_range(
    V_net,
    GV_net,
    dynamics,
    full_range,                       # np.ndarray or torch.Tensor with shape (D,2)
    *,
    num_points: int = 512,            # how many samples in full_range
    seed: int = 0,
    batch_size: int = 128,            # autograd Hessian diag is expensive; batch it
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    report_topk: int = 10,
):
    """
    Compare GV_net(x) against an autograd-computed generator on MANY samples drawn uniformly
    from full_range.

    Prints aggregate error stats and worst offenders (top-k).
    """
    torch.manual_seed(seed)

    # --- prepare bounds ---
    if not isinstance(full_range, torch.Tensor):
        full_range_t = torch.tensor(full_range, device=device, dtype=dtype)
    else:
        full_range_t = full_range.to(device=device, dtype=dtype)

    assert full_range_t.dim() == 2 and full_range_t.shape[1] == 2, "full_range must be (D,2)"
    D = int(full_range_t.shape[0])
    lo = full_range_t[:, 0]
    hi = full_range_t[:, 1]

    # --- sample uniformly in full_range ---
    x_all = lo.unsqueeze(0) + (hi - lo).unsqueeze(0) * torch.rand(num_points, D, device=device, dtype=dtype)

    # --- get f,g once ---
    f = dynamics.get_f()
    g = dynamics.get_g()

    # --- storage for errors ---
    errs = []
    gv_fast_list = []
    gv_auto_list = []

    # --- loop in batches (avoid huge graphs) ---
    num_batches = (num_points + batch_size - 1) // batch_size

    for bi in range(num_batches):
        s = bi * batch_size
        e = min(num_points, (bi + 1) * batch_size)
        x = x_all[s:e]  # (B,D)
        B = x.shape[0]

        # ----- fast GV (no grad) -----
        with torch.no_grad():
            gv_fast = GV_net(x)
            if gv_fast.dim() == 2 and gv_fast.shape[1] == 1:
                gv_fast = gv_fast[:, 0]
            else:
                gv_fast = gv_fast.view(-1)
            gv_fast = gv_fast.detach()

        # ----- autograd GV -----
        x_req = x.clone().detach().requires_grad_(True)

        V_out = V_net(x_req)
        if V_out.dim() == 2 and V_out.shape[1] == 1:
            Vs = V_out[:, 0]          # (B,)
        elif V_out.dim() == 1:
            Vs = V_out                # (B,)
        else:
            raise ValueError(f"Expected scalar V output, got shape {tuple(V_out.shape)}")

        # grad: (B,D)
        gradV = torch.autograd.grad(Vs.sum(), x_req, create_graph=True)[0]

        # Hessian diagonal: (B,D)
        Hdiag_cols = []
        for i in range(D):
            gi = gradV[:, i]
            dgi = torch.autograd.grad(gi.sum(), x_req, create_graph=True)[0][:, i]
            Hdiag_cols.append(dgi)
        Hdiag = torch.stack(Hdiag_cols, dim=1)  # (B,D)

        # f(x): expected (B,D)
        if callable(f):
            fx = f(x_req)
            if isinstance(fx, np.ndarray):
                fx = torch.from_numpy(fx).to(device=x_req.device, dtype=x_req.dtype)
        else:
            fx = x_req @ f.to(x_req).T

        if fx.shape != x_req.shape:
            raise ValueError(f"f(x) must be (B,D)={tuple(x_req.shape)}; got {tuple(fx.shape)}")

        # diag(GG^T): (B,D)
        if g is None:
            gdiag = torch.zeros_like(x_req)
        elif callable(g):
            gout = g(x_req)
            if isinstance(gout, np.ndarray):
                gout = torch.from_numpy(gout).to(device=x_req.device, dtype=x_req.dtype)

            gdiag_raw = _diag_ggt_from_g(gout, D)

            if isinstance(gdiag_raw, (float, int)):
                gdiag = torch.full_like(x_req, float(gdiag_raw))
            elif isinstance(gdiag_raw, torch.Tensor) and gdiag_raw.dim() == 1:
                # (D,) -> broadcast to (B,D)
                gdiag = gdiag_raw.to(device=x_req.device, dtype=x_req.dtype).view(1, D).expand_as(x_req)
            else:
                # (B,D)
                gdiag = gdiag_raw.to(device=x_req.device, dtype=x_req.dtype)
        else:
            gout = g.to(x_req)
            gdiag_raw = _diag_ggt_from_g(gout, D)
            if isinstance(gdiag_raw, (float, int)):
                gdiag = torch.full_like(x_req, float(gdiag_raw))
            elif isinstance(gdiag_raw, torch.Tensor) and gdiag_raw.dim() == 1:
                gdiag = gdiag_raw.view(1, D).expand_as(x_req)
            else:
                gdiag = gdiag_raw

        gv_auto = ((fx * gradV) + 0.5 * (gdiag * Hdiag)).sum(dim=1)  # (B,)
        gv_auto = gv_auto.detach()

        # ----- compare -----
        err = (gv_fast - gv_auto).abs()  # (B,)

        errs.append(err.cpu())
        gv_fast_list.append(gv_fast.cpu())
        gv_auto_list.append(gv_auto.cpu())

        # free graph ASAP
        del x_req, Vs, V_out, gradV, Hdiag, fx, gdiag, gv_auto

    errs = torch.cat(errs, dim=0)                 # (N,)
    gv_fast_all = torch.cat(gv_fast_list, dim=0)  # (N,)
    gv_auto_all = torch.cat(gv_auto_list, dim=0)  # (N,)

    # --- stats ---
    max_err = float(errs.max())
    mean_err = float(errs.mean())
    med_err = float(errs.median())
    p95 = float(errs.kthvalue(int(0.95 * (errs.numel() - 1)) + 1).values)

    print("=== GV vs Autograd check over full_range ===")
    print(f"samples: {num_points}, batch_size: {batch_size}, D: {D}")
    print(f"max |diff|  = {max_err}")
    print(f"mean|diff|  = {mean_err}")
    print(f"median|diff|= {med_err}")
    print(f"p95 |diff|  = {p95}")

    # --- worst offenders ---
    k = min(report_topk, errs.numel())
    top_err, top_idx = torch.topk(errs, k=k, largest=True)
    print(f"\nTop-{k} worst errors:")
    for rank in range(k):
        i = int(top_idx[rank].item())
        print(
            f"[{rank:02d}] idx={i:05d}  |diff|={float(top_err[rank]):.6e}  "
            f"gv_fast={float(gv_fast_all[i]):.6e}  gv_auto={float(gv_auto_all[i]):.6e}  "
            f"x={x_all[i].detach().cpu().numpy()}"
        )

    return {
        "errs": errs,
        "gv_fast": gv_fast_all,
        "gv_auto": gv_auto_all,
        "x": x_all.detach().cpu(),
        "stats": {"max": max_err, "mean": mean_err, "median": med_err, "p95": p95},
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    args = parser.parse_args()

    print("=" * 80)
    print("XV-15 CERTIFICATE-BASED CONTROL SYNTHESIS (V + Phi)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1) Hyperparameters (clone your Lorentz defaults, then adjust)
    # -------------------------------------------------------------------------
    params = Hyperparameters.default()

    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64

    # Scaling: pick something roughly comparable to variable ranges
    # v ~ [20,100] (span 80), gamma ~ [-0.26,0.26], beta ~ [0,1.57]
    # scale choices affect training conditioning; tune if needed.
    params.network.input_scale = [100.0, 20*DEG, 90*DEG]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 30000
    params.training.learnable_scale = False
    params.training.learnable_input_scale = False
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0
    params.training.learnable_beta_s = False
    params.constraints.beta_s = 0.00
    params.constraints.beta_ra = 5.0

    params.compute_V = True
    params.compute_GV = True

    device = params.training.device

    # -------------------------------------------------------------------------
    # 3) Regions (init, goal, unsafe union, full)
    # -------------------------------------------------------------------------
    full_range = np.array([
        [0.5, 100.0], # airspeed
        [-20.0 * DEG, 20.0 * DEG], # flight path angle
        [0.0 * DEG, 90.0 * DEG], # tilt angle
    ], dtype=np.float32)

    init_range = np.array([
        [28.0, 32.0],
        [8.5 * DEG,  10.5 * DEG],
        [58.0 * DEG, 62.0 * DEG],
    ], dtype=np.float32)

    goal_range = np.array([
        [65.0, 85.0],
        [-2.0 * DEG, 10.0 * DEG],
        [25.0 * DEG, 35.0 * DEG],
    ], dtype=np.float32)

    unsafe_up = np.array([
        full_range[0, :],
        [full_range[1,1]-1*DEG, full_range[1,1]],
        full_range[2, :],
    ], dtype=np.float32)

    unsafe_dn = np.array([
        full_range[0, :],
        [full_range[1,0], full_range[1,0]+1*DEG],
        full_range[2, :],
    ], dtype=np.float32)

    unsafe_min_vel = np.array([
        [full_range[0,0], full_range[0,0]+0.5],
        full_range[1, :],
        full_range[2, :],
    ], dtype=np.float32)

    unsafe_max_vel = np.array([
        [full_range[0,1]-0.5, full_range[0,1]],
        full_range[1, :],
        full_range[2, :],
    ], dtype=np.float32)

    unsafe_min_beta = np.array([
        full_range[0, :],
        full_range[1, :],
        [full_range[2,0], full_range[2,0]+1*DEG],
    ], dtype=np.float32)

    unsafe_max_beta = np.array([
        full_range[0, :],
        full_range[1, :],
        [full_range[2,1]-1*DEG, full_range[2,1]],
    ], dtype=np.float32)

    unsafe_range = np.vstack((unsafe_up, unsafe_dn, unsafe_min_beta, unsafe_max_beta,
                              unsafe_min_vel, unsafe_max_vel
                             ))
    init = Region(init_range)
    goal = Region(goal_range)
    full = Region(full_range)

    unsafe_up_reg = Region(unsafe_up)
    unsafe_dn_reg = Region(unsafe_dn)
    unsafe_min_beta_reg = Region(unsafe_min_beta)
    unsafe_max_beta_reg = Region(unsafe_max_beta)
    unsafe_min_vel_reg = Region(unsafe_min_vel)
    unsafe_max_vel_reg = Region(unsafe_max_vel)
    # unsafe_max_vel_dn_reg = Region(unsafe_max_vel_dn)

    unsafe = Region.union(unsafe_up_reg, unsafe_dn_reg, 
                          unsafe_min_beta_reg, unsafe_max_beta_reg,
                          unsafe_min_vel_reg, unsafe_max_vel_reg)
    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # Discretization: start modest; refine happens during training
    params.discretization.n_goal = 32
    params.discretization.n_outside_goal = 8
    params.discretization.n_generator = 4
    params.discretization.n_unsafe = 16
    params.discretization.n_init = 28

    # -------------------------------------------------------------------------
    # 2) Dynamics (closed-loop, torch, differentiable)
    # -------------------------------------------------------------------------
    # Create diffusion coefficients as a persistent tensor to avoid TracerWarnings
    g_coeffs = torch.tensor([0.5, 0.1 * DEG, 0.1 * DEG], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term g(x) that is constant: [1.0, 1.0, 1.0] for every x.

        If x has shape (D,), returns (D,).
        If x has shape (N, D), returns (N, D) with each row [1.0, 1.0, 1.0].
        """
        base = g_coeffs.to(device=x.device, dtype=x.dtype)

        if x.dim() == 1:
            # x is shape (D,)
            return base
        elif x.dim() == 2:
            # x is shape (N, D)
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)  # (N, D)
        else:
            raise ValueError(f"g(x) expects x of shape (D,) or (N, D), got {tuple(x.shape)}")

    aero = XV15KLinearAeroTorch().to(device)
    x_eq, u_eq, info = find_xv15_equilibrium_for_tilt_min_thrust(
        beta_eq_deg=30.0,
        aero=aero,
        v_min=0.5,
        v_max=80,
        gamma_min_deg=0.0,
        gamma_max_deg=15.0,
        device="cpu",
    )
    print(info)
    print("x_eq (m/s, Deg, Deg)=", x_eq[0], x_eq[1] / DEG, x_eq[2] / DEG)  # [v, gamma, beta]
    print("u_eq (N, Deg, Deg/s)=", u_eq[0], u_eq[1] / DEG, u_eq[2] / DEG)  # [T, alpha, 0]

    u_nn = XV15EqMLPControl(
        x_eq=x_eq,          # (3,) torch tensor
        u_eq=u_eq,          # (3,) torch tensor
        T_min=XV15Constants.MASS * 9.81 * 0.1,
        T_max=XV15Constants.MASS * 9.81 * 1.8,
        alpha_max=XV15Constants.AOA_MAX,
        delta_max=XV15Constants.MAX_TILT_ANGLE_RATE,
        hidden_dim=64,
        act="tanh",
    ).to(x_eq.device)
    u_nn.verify_u_at_equilibrium()

    f_cl_module = ClosedLoopDrift(aero=aero, controller=u_nn).to(device)
    dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=3)

    # -------------------------------------------------------------------------
    # 4) Networks (V and GV)
    # -------------------------------------------------------------------------
    input_offset = [x_eq[0], x_eq[1], x_eq[2]]
    output_offset = np.float32(0.1)
    V_net = create_V(params.network, input_offset=input_offset, output_offset=output_offset)
    V_net.verify_zero_at_offset(atol=1e-6, rtol=1e-6)

    GV_net = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=params.network,
        input_offset=input_offset,
        verify=False
    )

    res = check_gv_matches_autograd_full_range(
        V_net, GV_net, dynamics,
        full_range=full_range,
        num_points=1024,
        batch_size=64,
        device=device,
    )

    # -------------------------------------------------------------------------
    # 5) Discretize regions
    # -------------------------------------------------------------------------
    region_cells = discretize_regions(
        regions,
        params.discretization,
        use_radial_generator=False,
    )

    # -------------------------------------------------------------------------
    # 6) Pretrain + train (or load)
    # -------------------------------------------------------------------------
    params.constraints.pretrain_goal_target = params.constraints.beta_s
    params.constraints.pretrain_unsafe_target = params.constraints.beta_ra
    params.constraints.pretrain_init_target = 1.0
    params.constraints.pretrain_phi_target = 0.0
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")
        print(f"\n{dynamics}")
        print(f"\nUsing device: {device}")

        ENABLE_PRETRAINING = True
        PRETRAIN_EPOCHS = 15000
        PRETRAIN_LR = 0.01

        if ENABLE_PRETRAINING:
            pretrain_network_samples(
                model=V_net,
                x_goal_range=goal_range,
                x_unsafe_range=unsafe_range,     # pass first unsafe piece
                x_init_range=init_range,
                x_range=full_range,
                params=params,
                GV_net=GV_net,
                num_epochs=PRETRAIN_EPOCHS,
                lr=PRETRAIN_LR,
                device=device,
                control_net=u_nn,
                lambda_w=1e-3,
                n_each=1200,
                save_v_path= OUTPUT_DIR / "V_pretrained.pth",
                save_control_path= OUTPUT_DIR / "controller_pretrained.pth"
            )
            print("Pretraining completed.\n")

        V_net.load_state_dict(torch.load(OUTPUT_DIR / "V_pretrained.pth", map_location=device))
        u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
        
        loss_history, final_beta_s, refinement_epochs = train_network_bounds(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            device=device,
            visualize_interval=1000,
            control_net=u_nn,  # CONTROL SYNTHESIS
        )

        print("\n" + "=" * 80)
        print("FINAL EVALUATION")
        print("=" * 80)

        results = evaluate_constraints(
            V_net, GV_net, region_cells,
            beta_ra=params.constraints.beta_ra,
            device=device,
        )
        print_constraint_summary(results)

        print("\n" + "=" * 80)
        print("CREATING FINAL VISUALIZATIONS")
        print("=" * 80)

        create_summary_plots(
            V_net=V_net,
            GV_net=GV_net,
            regions=regions,
            region_cells=region_cells,
            beta_s=final_beta_s,
            beta_ra=params.constraints.beta_ra,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
            output_dir="results"
        )

        print("\n" + "=" * 80)
        print("SAVING EVAL BUNDLE")
        print("=" * 80)

        save_eval_bundle(
            OUTPUT_DIR,
            V_net=V_net,
            GV_net=GV_net,
            control_net=u_nn,
            params=params,
            regions=regions,
            region_cells=region_cells,
            final_beta_s=final_beta_s,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
        )

    else:
        print("\n" + "=" * 80)
        print("LOADING SAVED BUNDLE (skip training)")
        print("=" * 80)

        bundle = load_eval_bundle(bundle_path, map_location="cpu")

        params = Hyperparameters.from_dict(bundle["hyperparameters"])
        device = params.training.device

        # rebuild region_cells (already cpu tensors)
        region_cells = bundle["region_cells"]

        # rebuild networks & load
        V_net.load_state_dict(bundle["V_state_dict"])

        # load controller
        if bundle["control_state_dict"] is not None:
            print("load u_nn")
            u_nn.load_state_dict(bundle["control_state_dict"])

        # rebuild dynamics and GV
        u_nn.verify_u_at_equilibrium()
        f_cl_module = ClosedLoopDrift(aero=aero, controller=u_nn).to(device)

        # create a constant u_eq open-loop control for comparison
        u_eq_device = u_eq.to(device=device, dtype=torch.float32)  # (3,)
        u_const = ConstantControl(u_eq_device).to(device)
        f_cl_module_no_control = ClosedLoopDrift(aero=aero, controller=u_const).to(device)

        # create the controller from pretraining
        u_nn_pretrain = XV15EqMLPControl(
            x_eq=x_eq,          # (3,) torch tensor
            u_eq=u_eq,          # (3,) torch tensor
            T_min=XV15Constants.MASS * 9.81 * 0.1,
            T_max=XV15Constants.MASS * 9.81 * 1.8,
            alpha_max=XV15Constants.AOA_MAX,
            delta_max=XV15Constants.MAX_TILT_ANGLE_RATE,
            hidden_dim=64,
            act="tanh",
        ).to(x_eq.device)
        u_nn_pretrain.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
        u_nn_pretrain.verify_u_at_equilibrium()
        f_cl_module_pretrain = ClosedLoopDrift(aero=aero, controller=u_nn_pretrain).to(device)

        # animation
        # open-loop constant u_eq animation
        animate_xv15_aircraft_state_control(
            f_cl_module=f_cl_module_no_control, g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
            device=device,
            dt=0.02, T=120.0,
            seed=0, save_path=None,
            show=True,
            controller_label="open-loop"
        )
        # pretrain animation
        animate_xv15_aircraft_state_control(
            f_cl_module=f_cl_module_pretrain, g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
            device=device,
            dt=0.02, T=120.0,
            seed=0, save_path=None,
            show=True,
            controller_label="pre-train"
        )
        # control-synthesis animation
        animate_xv15_aircraft_state_control(
            f_cl_module=f_cl_module, g_fn=g,
            init_range=init_range, goal_range=goal_range,
            full_range=full_range, unsafe_boxes=unsafe_range,   # can be (K*3,2) from vstack
            device=device,
            dt=0.02, T=120.0,
            seed=0, save_path=None,
            show=True,
            controller_label="certified-synthesis"
        )

        print("\n" + "=" * 80)
        print("Monte Carlo results")
        print("=" * 80)

        # monte-carlo (open-loop)
        mc = mc_reach_avoid(
            f_cl_module=f_cl_module_no_control,
            g_fn=g,  # or None
            init_range=init_range,
            goal_range=goal_range,
            full_range=full_range,
            unsafe_boxes=unsafe_range,
            N_trials=100,
            dt=0.02,
            T=120.0,
            seed=0,
            device=device,
        )
        print("Open Loop Control")
        print(mc)

        # monte-carlo (pretrain)
        mc = mc_reach_avoid(
            f_cl_module=f_cl_module_pretrain,
            g_fn=g,  # or None
            init_range=init_range,
            goal_range=goal_range,
            full_range=full_range,
            unsafe_boxes=unsafe_range,
            N_trials=100,
            dt=0.02,
            T=120.0,
            seed=0,
            device=device,
        )
        print("Pretrain Control")
        print(mc)

        # monte-carlo (synthesis)
        mc = mc_reach_avoid(
            f_cl_module=f_cl_module,
            g_fn=g,  # or None
            init_range=init_range,
            goal_range=goal_range,
            full_range=full_range,
            unsafe_boxes=unsafe_range,
            N_trials=100,
            dt=0.02,
            T=120.0,
            seed=0,
            device=device,
        )
        print("Synthesized Control")
        print(mc)

        print("\n" + "=" * 80)
        print("Recreating Plots")
        print("=" * 80)

        # re-render plots
        dynamics = Dynamics.dynamics(f=f_cl_module, g=g, state_dim=3)
        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network,
            input_offset=input_offset,
            verify=False
        ).to(device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"])

        # move cells to device
        region_cells = {
            k: [(lo.to(device), hi.to(device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }

        final_beta_s = bundle["final_beta_s"]
        loss_history = bundle["loss_history"]
        refinement_epochs = bundle["refinement_epochs"]
        results = None
        if results is None:
            print("No saved results found in bundle, recomputing evaluation...")
            results = evaluate_constraints(
                V_net, GV_net, region_cells,
                beta_ra=params.constraints.beta_ra,
                device=device,
            )

        print("\n" + "=" * 80)
        print("FINAL EVALUATION (LOADED)")
        print("=" * 80)
        print_constraint_summary(results)

        print("\n" + "=" * 80)
        print("CREATING FINAL VISUALIZATIONS (LOADED)")
        print("=" * 80)
        log_loaded_training_epochs(loss_history)

        create_summary_plots(
            V_net=V_net,
            GV_net=GV_net,
            regions=regions,
            region_cells=region_cells,
            beta_s=final_beta_s,
            beta_ra=params.constraints.beta_ra,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
            output_dir="results"
        )


if __name__ == "__main__":
    main()