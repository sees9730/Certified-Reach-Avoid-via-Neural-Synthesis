"""
PPO-based RL Controller Training — Inverted Pendulum (Additive Box Disturbance)
================================================================================

Trains a controller using Proximal Policy Optimization (PPO)
[Schulman et al., 2017: https://arxiv.org/abs/1707.06347]
in the same environment as inv_pend_syn_disturb_baseline/main.py.

Environment
-----------
  State  : (x1, x2) = (angle, angular velocity)
  Domain : [-2π, 2π] × [-20, 20]
  Disturbance: w(t) ~ Uniform([-1,1]^2)  additive box disturbance (matches baseline)
  SDE    : dx1 = (x2 + w1) dt
            dx2 = ((g/L)·sin(x1) − b·x2/(m·L²) + u_applied + w2) dt + σ dW
  Init   : X_init = [3π/4, 5π/4] × [−1, 1]   (pendulum near bottom)
  Goal   : X_goal = [−0.4π, 0.4π] × [−4, 4]   (pendulum upright)
  Unsafe : ([−2π, −3π/2] × [−20, −10]) ∪ ([3π/2, 2π] × [10, 20])

Reward
------
  Terminal: +R_SUCCESS if next state ∈ goal; −R_FAIL if next state ∈ unsafe / domain-exit
  Running : C_SHAPE · (γ·cos(x1_next) − cos(x1_curr))   (potential-based shaping)

Controller architecture
-----------------------
  Actor   : InvertControlNN(input_dim=2, hidden_dim=8, output_dim=1)
            — identical to inv_pend_syn_disturb_baseline/main.py
  Critic  : separate MLP with 2×64 hidden layers (not exported)

Output
------
  outputs/rl_controller.pth
      — raw state_dict of WrapperConterlNN
      — load in plot.py with IS_PRETRAINED=True

Usage
-----
    python main.py
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN

# =========================================================================
# Physical constants  (must match inv_pend_syn_disturb_baseline/main.py)
# =========================================================================
pi          = np.pi
g_NOM       = 9.81
L_NOM       = 0.5
b_NOM       = 0.1
m_NOM       = 0.15
M_TORQUE    = 6.0
sigma       = 0.2
M_mLsquare  = M_TORQUE / (m_NOM * L_NOM**2)   # ≈ 160  (torque scale factor)

# Additive box disturbance bounds  w(t) ∈ [-DRIFT_UNC, DRIFT_UNC]^2
DRIFT_UNC   = np.array([1.0, 1.0], dtype=float)

# =========================================================================
# State-space  (must match inv_pend_syn_disturb_baseline/main.py)
# =========================================================================
X1_MIN, X1_MAX = -2*pi,  2*pi
X2_MIN, X2_MAX = -20.0,  20.0

# Spec regions  [x1_min, x1_max, x2_min, x2_max]
INIT_BOX  = [3*pi/4,   5*pi/4,  -1.0,   1.0]
GOAL_BOX  = [-0.4*pi,  0.4*pi,  -4.0,   4.0]
UNSAFE_1  = [-2*pi,   -3*pi/2, -20.0, -10.0]
UNSAFE_2  = [ 3*pi/2,  2*pi,   10.0,   20.0]

# =========================================================================
# Reward + environment hyperparameters
# =========================================================================
DT        = 0.005
T_MAX     = 8.0
MAX_STEPS = int(T_MAX / DT)  # 1600

R_SUCCESS  =  10.0    # terminal: reach goal
R_FAIL     = -10.0    # terminal: hit unsafe/domain-exit
C_SHAPE    =  2.0     # potential shaping coeff: γ·cos(x1_next) − cos(x1_curr)
                      # guides swing-up without changing the optimal policy

# =========================================================================
# PPO hyperparameters
# =========================================================================
LR          = 3e-4
GAMMA       = 0.99
GAE_LAM     = 0.95
CLIP_EPS    = 0.2
ENT_COEF    = 0.01
VF_COEF     = 0.5
MAX_GRAD    = 0.5
PPO_EPOCHS  = 4
BATCH_SIZE  = 64

N_COLLECT   = 16      # episodes collected per PPO update
N_UPDATES   = 2000    # total PPO updates
PRINT_EVERY = 50      # print status every N updates


# =========================================================================
# Environment helpers
# =========================================================================
def _in_box(x1: float, x2: float, box: list) -> bool:
    return box[0] <= x1 <= box[1] and box[2] <= x2 <= box[3]


def _step_dynamics(x1: float, x2: float, u_raw: float,
                   rng: np.random.Generator) -> tuple:
    """
    One Euler-Maruyama step with additive box disturbance.

    Dynamics (matches inv_pend_syn_disturb_baseline / plot.py):
        dx1 = (x2 + w1) dt
        dx2 = ((g/L)·sin(x1) − b·x2/(m·L²) + u2 + w2) dt + σ dW

    where w = [w1, w2] ~ Uniform(-DRIFT_UNC, DRIFT_UNC).
    Returns unclipped next state.
    """
    u2  = M_mLsquare * float(np.clip(u_raw, -1.0, 1.0))
    w   = rng.uniform(-DRIFT_UNC, DRIFT_UNC)            # w ∈ [-1,1]^2
    dW  = np.sqrt(DT) * rng.standard_normal()

    dx1 = x2 + w[0]
    dx2 = (g_NOM / L_NOM) * np.sin(x1) - (b_NOM / (m_NOM * L_NOM**2)) * x2 + u2 + w[1]

    x1n = x1 + dx1 * DT
    x2n = x2 + dx2 * DT + sigma * dW
    return float(x1n), float(x2n)


# =========================================================================
# Networks
# =========================================================================
class ActorCritic(nn.Module):
    """
    Actor  : InvertControlNN(2, 8, 1)  — same architecture as inv_pend_syn_disturb_baseline
    Critic : separate 2×64 MLP with normalised inputs (not exported)
    """

    def __init__(self):
        super().__init__()
        self.actor   = InvertControlNN(input_dim=2, hidden_dim=8, output_dim=1)
        self.log_std = nn.Parameter(torch.tensor([0.5]))  # start with higher exploration
        self.critic  = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Scale state to ≈ [−1, 1] for critic stability."""
        scale = x.new_tensor([2 * pi, 20.0])
        return x / scale

    def act(self, x: torch.Tensor, deterministic: bool = False):
        """
        Returns
        -------
        a        : action ∈ (−1, 1) after tanh squash
        log_prob : log π(a|x), scalar per sample
        entropy  : H[π(·|x)], scalar per sample
        """
        mu  = self.actor(x)
        # Clamp log_std to keep std in [exp(-3), exp(0.5)] ≈ [0.05, 1.65]
        std = self.log_std.clamp(-3.0, 0.5).exp()
        if deterministic:
            return mu, None, None
        dist     = Normal(mu, std.expand_as(mu))
        a_raw    = dist.rsample()
        a        = torch.tanh(a_raw)
        log_prob = dist.log_prob(a_raw) - torch.log1p(-a.pow(2) + 1e-6)
        return a, log_prob.sum(-1), dist.entropy().sum(-1)

    def evaluate(self, x: torch.Tensor, a: torch.Tensor):
        """Recompute log_prob and entropy for stored (x, a) during PPO update."""
        mu   = self.actor(x)
        std  = self.log_std.clamp(-3.0, 0.5).exp()
        dist = Normal(mu, std.expand_as(mu))
        a_c  = a.clamp(-1 + 1e-6, 1 - 1e-6)
        a_raw = torch.atanh(a_c)
        lp   = dist.log_prob(a_raw) - torch.log1p(-a_c.pow(2) + 1e-6)
        return lp.sum(-1), dist.entropy().sum(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self._norm(x)).squeeze(-1)


# =========================================================================
# Episode rollout
# =========================================================================
def rollout_episode(policy: ActorCritic, rng: np.random.Generator, device):
    """
    Collect one episode using the current policy.

    Returns
    -------
    S       : (T, 2)  states
    A       : (T, 1)  actions (u_raw)
    R       : (T,)    rewards
    LP      : (T,)    log-probs at collection time
    V       : (T,)    values at collection time
    last_v  : float   bootstrap value (0 if done, V(x_last) if timeout)
    outcome : str     'success' | 'fail' | 'timeout'
    """
    x1 = rng.uniform(*INIT_BOX[:2])
    x2 = rng.uniform(*INIT_BOX[2:])

    states, acts, rews, lps, vals = [], [], [], [], []
    outcome = "timeout"
    last_v  = 0.0

    for _ in range(MAX_STEPS):
        xt = torch.tensor([[x1, x2]], dtype=torch.float32, device=device)
        with torch.no_grad():
            a, lp, _ = policy.act(xt)
            v        = policy.value(xt)

        u = float(a.squeeze().cpu())
        x1n, x2n = _step_dynamics(x1, x2, u, rng)

        # Domain-exit check (excessive rotation / angular speed) → fail
        out_of_domain = not (X1_MIN <= x1n <= X1_MAX) or not (X2_MIN <= x2n <= X2_MAX)

        # Potential-based shaping: GAMMA * cos(x1_next) − cos(x1_curr)
        # Rewards progress toward upright (x1=0) on every step
        shaping = C_SHAPE * (GAMMA * np.cos(x1n) - np.cos(x1))

        # Reward based on next state
        if out_of_domain:
            r, done, outcome = R_FAIL, True, "fail"
        elif _in_box(x1n, x2n, GOAL_BOX):
            r, done, outcome = R_SUCCESS, True, "success"
        elif _in_box(x1n, x2n, UNSAFE_1) or _in_box(x1n, x2n, UNSAFE_2):
            r, done, outcome = R_FAIL, True, "fail"
        else:
            r    = shaping
            done = False

        states.append([x1, x2])
        acts.append([u])
        rews.append(r)
        lps.append(float(lp.squeeze().cpu()))
        vals.append(float(v.cpu()))

        # Clip only when episode continues (keeps trajectory in domain)
        x1, x2 = float(np.clip(x1n, X1_MIN, X1_MAX)), float(np.clip(x2n, X2_MIN, X2_MAX))

        if done:
            last_v = 0.0
            break
    else:
        # Timeout: bootstrap value of last state
        xt = torch.tensor([[x1, x2]], dtype=torch.float32, device=device)
        with torch.no_grad():
            last_v = float(policy.value(xt).cpu())

    S  = torch.tensor(states, dtype=torch.float32, device=device)
    A  = torch.tensor(acts,   dtype=torch.float32, device=device)
    R  = torch.tensor(rews,   dtype=torch.float32, device=device)
    LP = torch.tensor(lps,    dtype=torch.float32, device=device)
    V  = torch.tensor(vals,   dtype=torch.float32, device=device)
    return S, A, R, LP, V, last_v, outcome


# =========================================================================
# GAE advantage estimation  (per episode)
# =========================================================================
def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                last_v: float) -> tuple:
    """Generalised Advantage Estimation [Schulman et al., 2015]."""
    N      = rewards.shape[0]
    adv    = rewards.new_zeros(N)
    gae    = 0.0
    next_v = last_v

    for t in reversed(range(N)):
        delta  = rewards[t] + GAMMA * next_v - values[t]
        gae    = delta + GAMMA * GAE_LAM * gae
        adv[t] = gae
        next_v = float(values[t])

    return adv, adv + values


# =========================================================================
# PPO update  (clipped surrogate objective)
# =========================================================================
def ppo_update(policy: ActorCritic, optimizer: torch.optim.Optimizer,
               S: torch.Tensor, A: torch.Tensor,
               Ret: torch.Tensor, Adv: torch.Tensor,
               LP_old: torch.Tensor) -> None:
    """Standard PPO clipped policy + value + entropy update."""
    N = S.shape[0]
    # Global advantage normalisation (avoids NaN from size-1 mini-batches)
    Adv = (Adv - Adv.mean()) / (Adv.std(correction=0) + 1e-8)

    idx = torch.randperm(N, device=S.device)

    for _ in range(PPO_EPOCHS):
        for start in range(0, N, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]

            lp_new, ent = policy.evaluate(S[b], A[b])
            ratio = (lp_new - LP_old[b]).exp().clamp(0.0, 10.0)

            pg_loss = -torch.min(
                Adv[b] * ratio,
                Adv[b] * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS),
            ).mean()

            vf_loss = F.mse_loss(policy.value(S[b]), Ret[b])
            loss    = pg_loss + VF_COEF * vf_loss - ENT_COEF * ent.mean()

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
            optimizer.step()


# =========================================================================
# Main training loop
# =========================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("PPO Training — Inverted Pendulum (Additive Box Disturbance)")
    print(f"  w(t) ~ Uniform([-{DRIFT_UNC[0]},{DRIFT_UNC[0]}]^2),  sigma = {sigma}")
    print(f"  Device : {device}")
    print(f"  Updates: {N_UPDATES},  episodes/update: {N_COLLECT}")
    print("=" * 65)

    rng    = np.random.default_rng(seed=42)
    policy = ActorCritic().to(device)
    optim  = torch.optim.Adam(policy.parameters(), lr=LR)

    best_sr      = -1.0
    best_weights = None
    t0           = time.time()

    for upd in range(1, N_UPDATES + 1):
        # ---- collect episodes ----------------------------------------
        Ss, As, Rets, Advs, LPs = [], [], [], [], []
        outcomes = {"success": 0, "fail": 0, "timeout": 0}

        for _ in range(N_COLLECT):
            S, A, R, LP, V, lv, outcome = rollout_episode(policy, rng, device)
            adv, ret = compute_gae(R, V, lv)
            Ss.append(S);   As.append(A)
            Rets.append(ret); Advs.append(adv); LPs.append(LP)
            outcomes[outcome] += 1

        S_all   = torch.cat(Ss)
        A_all   = torch.cat(As)
        Ret_all = torch.cat(Rets)
        Adv_all = torch.cat(Advs)
        LP_all  = torch.cat(LPs)

        # ---- PPO update -----------------------------------------------
        ppo_update(policy, optim, S_all, A_all, Ret_all, Adv_all, LP_all)

        # ---- bookkeeping ----------------------------------------------
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

    # ---- restore best policy -----------------------------------------
    if best_weights is not None:
        policy.load_state_dict(best_weights)
    policy.cpu().eval()

    # ---- save as WrapperConterlNN state_dict (IS_PRETRAINED=True) ----
    u_nn      = WrapperConterlNN(policy.actor)
    save_path = OUTPUT_DIR / "rl_controller.pth"
    torch.save(u_nn.state_dict(), save_path)
    print(f"Saved: {save_path}")
    print("  → load in plot.py with IS_PRETRAINED=True")


if __name__ == "__main__":
    main()
