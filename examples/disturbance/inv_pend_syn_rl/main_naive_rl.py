"""
REINFORCE — Inverted Pendulum (Naive RL Baseline)
==================================================

Simplest possible policy-gradient training for reach-avoid comparison.

Algorithm : REINFORCE with mean-return baseline  [Williams 1992]
Reward    : sparse — +R_SUCCESS on goal, −R_FAIL on unsafe/exit, 0 otherwise
Policy    : Gaussian, μ = InvertControlNN(x),  σ = exp(log_std)  (learnable)
Update    : maximise  E[ Σ_t  G_t · log π(a_t|s_t) ]
            where G_t = Σ_{k≥t} γ^{k−t} r_k  (Monte-Carlo return)
            baseline = mean(G_0) across the batch to reduce variance

No value network, no GAE, no clipping, no entropy bonus.

Output
------
  outputs/naive_rl_controller.pth
      — raw state_dict of WrapperConterlNN
      — load in plot.py with IS_PRETRAINED=True  (same as rl_controller.pth)

Usage
-----
    python main_naive_rl.py
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN

# =========================================================================
# Physical constants  (must match inv_pend_syn/main.py)
# =========================================================================
pi         = np.pi
g_NOM      = 9.81
L_NOM      = 0.5
b_NOM      = 0.1
m_NOM      = 0.15
M_TORQUE   = 6.0
sigma      = 0.2
M_mLsquare = M_TORQUE / (m_NOM * L_NOM**2)

PARAM_RANGES = {
    "g": (9.81, 9.81),
    "L": (0.40, 0.60),
    "b": (0.10, 0.10),
    "m": (0.15, 0.15),
}

X1_MIN, X1_MAX = -2*pi,  2*pi
X2_MIN, X2_MAX = -20.0,  20.0

INIT_BOX = [ 3*pi/4,   5*pi/4,  -1.0,   1.0]
GOAL_BOX = [-0.4*pi,   0.4*pi,  -4.0,   4.0]
UNSAFE_1 = [-2*pi,    -3*pi/2, -20.0, -10.0]
UNSAFE_2 = [ 3*pi/2,   2*pi,   10.0,   20.0]

# =========================================================================
# Hyperparameters
# =========================================================================
DT        = 0.005
T_MAX     = 8.0
MAX_STEPS = int(T_MAX / DT)   # 1600

R_SUCCESS =  10.0
R_FAIL    = -10.0

LR          = 3e-4
GAMMA       = 0.99
N_COLLECT   = 16     # episodes per gradient update
N_UPDATES   = 2000
PRINT_EVERY = 50


# =========================================================================
# Environment helpers
# =========================================================================
def _in_box(x1: float, x2: float, box: list) -> bool:
    return box[0] <= x1 <= box[1] and box[2] <= x2 <= box[3]


def _sample_params(rng: np.random.Generator) -> tuple:
    return (
        rng.uniform(*PARAM_RANGES["g"]),
        rng.uniform(*PARAM_RANGES["L"]),
        rng.uniform(*PARAM_RANGES["b"]),
        rng.uniform(*PARAM_RANGES["m"]),
    )


def _step_dynamics(x1: float, x2: float, u_raw: float,
                   g: float, L: float, b: float, m: float,
                   rng: np.random.Generator) -> tuple:
    u2  = M_mLsquare * float(np.clip(u_raw, -1.0, 1.0))
    dW  = np.sqrt(DT) * rng.standard_normal()
    dx1 = x2
    dx2 = (g / L) * np.sin(x1) - (b / (m * L**2)) * x2 + u2
    return float(x1 + dx1 * DT), float(x2 + dx2 * DT + sigma * dW)


# =========================================================================
# Policy network  (actor only — no critic)
# =========================================================================
class Policy(nn.Module):
    """Gaussian policy: μ = InvertControlNN(x), σ = exp(log_std)."""

    def __init__(self):
        super().__init__()
        self.actor   = InvertControlNN(input_dim=2, hidden_dim=8, output_dim=1)
        self.log_std = nn.Parameter(torch.tensor([0.5]))

    def act(self, x: torch.Tensor):
        mu       = self.actor(x)
        std      = self.log_std.clamp(-3.0, 0.5).exp()
        dist     = Normal(mu, std.expand_as(mu))
        a_raw    = dist.rsample()
        a        = torch.tanh(a_raw)
        log_prob = dist.log_prob(a_raw) - torch.log1p(-a.pow(2) + 1e-6)
        return a, log_prob.sum(-1)


# =========================================================================
# Episode rollout
# =========================================================================
def rollout_episode(policy: Policy, rng: np.random.Generator, device):
    """
    Collect one full episode under the current policy.

    Returns
    -------
    log_probs : (T,) tensor  — log π(a_t | s_t)
    rewards   : list[float]  — sparse reward at each step
    outcome   : str          — 'success' | 'fail' | 'timeout'
    """
    x1 = rng.uniform(*INIT_BOX[:2])
    x2 = rng.uniform(*INIT_BOX[2:])
    g, L, b, m = _sample_params(rng)

    log_probs = []
    rewards   = []
    outcome   = "timeout"

    for _ in range(MAX_STEPS):
        xt = torch.tensor([[x1, x2]], dtype=torch.float32, device=device)
        a, lp = policy.act(xt)
        u = float(a.squeeze().cpu())

        x1n, x2n = _step_dynamics(x1, x2, u, g, L, b, m, rng)
        out_of_domain = not (X1_MIN <= x1n <= X1_MAX) or not (X2_MIN <= x2n <= X2_MAX)

        if out_of_domain:
            rewards.append(R_FAIL);    log_probs.append(lp)
            outcome = "fail";          break
        elif _in_box(x1n, x2n, GOAL_BOX):
            rewards.append(R_SUCCESS); log_probs.append(lp)
            outcome = "success";       break
        elif _in_box(x1n, x2n, UNSAFE_1) or _in_box(x1n, x2n, UNSAFE_2):
            rewards.append(R_FAIL);    log_probs.append(lp)
            outcome = "fail";          break
        else:
            rewards.append(0.0);       log_probs.append(lp)

        x1 = float(np.clip(x1n, X1_MIN, X1_MAX))
        x2 = float(np.clip(x2n, X2_MIN, X2_MAX))

    return torch.stack(log_probs), rewards, outcome


# =========================================================================
# Discounted returns
# =========================================================================
def discounted_returns(rewards: list) -> torch.Tensor:
    """G_t = Σ_{k≥t} γ^{k−t} r_k  (Monte-Carlo, no bootstrapping)."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


# =========================================================================
# Main training loop
# =========================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("REINFORCE (naive) — Inverted Pendulum")
    print(f"  L ~ Uniform{PARAM_RANGES['L']},  sigma = {sigma}")
    print(f"  Reward: sparse (+{R_SUCCESS}/−{abs(R_FAIL)}/0)")
    print(f"  Device : {device}")
    print(f"  Updates: {N_UPDATES},  episodes/update: {N_COLLECT}")
    print("=" * 60)

    rng    = np.random.default_rng(seed=42)
    policy = Policy().to(device)
    optim  = torch.optim.Adam(policy.parameters(), lr=LR)

    best_sr      = -1.0
    best_weights = None
    t0           = time.time()

    for upd in range(1, N_UPDATES + 1):
        all_log_probs = []
        all_returns   = []
        outcomes      = {"success": 0, "fail": 0, "timeout": 0}

        # ---- collect batch of episodes -----------------------------------
        for _ in range(N_COLLECT):
            lps, rews, outcome = rollout_episode(policy, rng, device)
            Gt = discounted_returns(rews).to(device)
            all_log_probs.append(lps)
            all_returns.append(Gt)
            outcomes[outcome] += 1

        # ---- REINFORCE update with mean-return baseline ------------------
        lp_cat  = torch.cat(all_log_probs)          # (total_steps,)
        ret_cat = torch.cat(all_returns)             # (total_steps,)
        baseline = ret_cat.mean()
        loss     = -((ret_cat - baseline) * lp_cat).mean()

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optim.step()

        # ---- bookkeeping -------------------------------------------------
        sr = outcomes["success"] / N_COLLECT
        if sr > best_sr:
            best_sr      = sr
            best_weights = {k: v.clone().cpu() for k, v in policy.state_dict().items()}

        if upd % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            print(f"Update {upd:5d}/{N_UPDATES} | "
                  f"SR={sr:.2f}  "
                  f"S={outcomes['success']:2d}/"
                  f"F={outcomes['fail']:2d}/"
                  f"T={outcomes['timeout']:2d} | "
                  f"BestSR={best_sr:.2f} | "
                  f"Elapsed={elapsed:.0f}s")

    print(f"\nTraining complete.  Best SR = {best_sr:.3f}")

    # ---- restore best policy and save ------------------------------------
    if best_weights is not None:
        policy.load_state_dict(best_weights)
    policy.cpu().eval()

    u_nn      = WrapperConterlNN(policy.actor)
    save_path = OUTPUT_DIR / "naive_rl_controller.pth"
    torch.save(u_nn.state_dict(), save_path)
    print(f"Saved: {save_path}")
    print("  → load in plot.py with IS_PRETRAINED=True")


if __name__ == "__main__":
    main()
