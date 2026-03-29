"""
PPO-based RL Controller Training — Two-Room Temperature Control
===============================================================

Trains a controller using Proximal Policy Optimization (PPO)
[Schulman et al., 2017: https://arxiv.org/abs/1707.06347]
in the same environment as room_temperature_disturb_baseline/main.py.

Environment
-----------
  State  : (x1, x2) = temperatures [°C] of two adjacent rooms
  Domain : [10, 30]^2
  SDE    : dx = (A@x + B@u + E*T_e − k_rad*(x+273.15)^4) dt + σ·I dW
  T_e    : ambient temperature, sampled Uniform(18, 22) each step
  Init   : X_init = [17, 23]^2   (both rooms at moderate temperature)
  Goal   : X_goal = [19, 21]^2   (comfort zone)
  Safe   : X_safe = [10.5, 29.5]^2; unsafe = outside safe set

Control
-------
  u ∈ [−U_MAX, U_MAX]^2,   U_MAX = 20.0   (HVAC inputs, independent per room)

Controller architecture
-----------------------
  Actor  : RoomTempControlNN(input_dim=2, hidden_dim=32, output_dim=2, u_max=20)
           — identical to room_temperature_disturb_baseline/main.py
  Critic : separate MLP with 2×64 hidden layers (not exported)

Output
------
  outputs/rl_controller.pth
      — raw state_dict of RoomTempControlNN (fc1.*, fc2.*, u_max not saved as param)
      — load in plot.py as raw policy-network weights (IS_PRETRAINED=True pattern)

Usage (from this directory):
    python main.py
"""

import sys
import time
import math
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

from src.control_network import RoomTempControlNN

# =========================================================================
# Physical constants  (must match room_temperature_disturb/test.py)
# =========================================================================
ALPHA    = 5e-2     # inter-room heat transfer coefficient  [1/s]
ALPHA_E  = 5e-3     # room-to-exterior heat loss coefficient [1/s]
BETA     = 0.1      # heater effectiveness                   [°C/(s·u)]
T_E_MIN  = 18.0     # ambient temperature range: lower bound [°C]
T_E_MAX  = 22.0     # ambient temperature range: upper bound [°C]
SIGMA    = 0.05     # diffusion intensity                    [°C/sqrt(s)]
KELVIN   = 273.15   # Kelvin offset
K_RAD    = 1e-11    # radiative cooling gain                 [1/(s·K^4)]
U_MAX    = 20.0     # max heater input per room

# System matrices
A_DRIFT = np.array([
    [-(ALPHA + ALPHA_E),  ALPHA             ],
    [ ALPHA,             -(ALPHA + ALPHA_E)  ],
], dtype=float)

B_INPUT = np.array([
    [BETA, 0.0 ],
    [0.0,  BETA],
], dtype=float)

E_PARAM = np.array([ALPHA_E, ALPHA_E], dtype=float)   # multiplies T_e
G_DIFF  = SIGMA * np.eye(2, dtype=float)               # diffusion matrix

# =========================================================================
# State-space  (must match room_temperature_disturb/test.py)
# =========================================================================
X_MIN, X_MAX = 10.0, 30.0       # domain per room [°C]
X0_LO, X0_HI = 17.0, 23.0      # initial set per room
XG_LO, XG_HI = 19.0, 21.0      # goal (comfort zone) per room
XS_LO, XS_HI = 10.5, 29.5      # safe set per room

X_GOAL_CENTER = np.array([0.5 * (XG_LO + XG_HI)] * 2, dtype=float)   # [20, 20]

# =========================================================================
# Reward hyperparameters
# =========================================================================
DT        = 0.005
T_MAX     = 20.0
MAX_STEPS = int(T_MAX / DT)    # 4000

R_SUCCESS =  10.0   # terminal: reach goal
R_FAIL    = -10.0   # terminal: leave safe set / domain exit
C_SHAPE   =  2.0    # potential shaping coeff

# =========================================================================
# PPO hyperparameters
# =========================================================================
LR         = 3e-4
GAMMA      = 0.99
GAE_LAM    = 0.95
CLIP_EPS   = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
MAX_GRAD   = 0.5
PPO_EPOCHS = 4
BATCH_SIZE = 64

N_COLLECT   = 16    # episodes per PPO update
N_UPDATES   = 2000  # total PPO updates
PRINT_EVERY = 50


# =========================================================================
# Environment helpers
# =========================================================================
def _in_box_2d(x: np.ndarray, lo: float, hi: float) -> bool:
    return bool(np.all(x >= lo) and np.all(x <= hi))


def _step_dynamics(x: np.ndarray, u: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """
    One Euler-Maruyama step.

    Dynamics (matches room_temperature_disturb/test.py):
        dx = (A@x + B@u + E*T_e − k_rad*(x+273.15)^4) dt + G_DIFF @ dW

    T_e ~ Uniform(T_E_MIN, T_E_MAX) per step.
    Returns next state (unclipped).
    """
    T_e     = rng.uniform(T_E_MIN, T_E_MAX)
    x_K     = x + KELVIN
    radiat  = -K_RAD * (x_K ** 4)
    f       = A_DRIFT @ x + B_INPUT @ u + E_PARAM * T_e + radiat
    dW      = rng.standard_normal(2) * math.sqrt(DT)
    return x + f * DT + G_DIFF @ dW


# =========================================================================
# Networks
# =========================================================================
class ActorCritic(nn.Module):
    """
    Actor  : same linear structure as RoomTempControlNN(2, 32, 2, U_MAX)
             — fc1: Linear(2→32), fc2: Linear(32→2); tanh activations
    Critic : separate 2×64 MLP (not exported)

    The actor stores fc1/fc2 directly so we can:
      - Get the pre-tanh output mu_raw for the Normal distribution
      - Export weights directly into RoomTempControlNN for saving
    """

    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(2, 32)
        self.fc2     = nn.Linear(32, 2)
        self.log_std = nn.Parameter(torch.tensor([0.5, 0.5]))   # per action dim
        self.critic  = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise temperatures from [10, 30] to [−1, 1] for the critic."""
        return (x - 20.0) / 10.0

    def _mu_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-tanh logit output, shape (B, 2)."""
        return self.fc2(torch.tanh(self.fc1(x)))

    def act(self, x: torch.Tensor, deterministic: bool = False):
        """
        Returns
        -------
        a        : action ∈ (−U_MAX, U_MAX)^2 after tanh squash
        log_prob : log π(a|x), scalar per sample
        entropy  : H[π(·|x)], scalar per sample
        """
        mu_raw = self._mu_raw(x)
        if deterministic:
            return U_MAX * torch.tanh(mu_raw), None, None
        std = self.log_std.clamp(-3.0, 0.5).exp()
        dist    = Normal(mu_raw, std.expand_as(mu_raw))
        a_raw   = dist.rsample()                           # (B, 2), unbounded
        a       = U_MAX * torch.tanh(a_raw)               # (B, 2), in (−U_MAX, U_MAX)
        # Log-prob with tanh-squash Jacobian correction
        lp = dist.log_prob(a_raw) \
             - torch.log1p(-(a / U_MAX).pow(2) + 1e-6) \
             - math.log(U_MAX)
        return a, lp.sum(-1), dist.entropy().sum(-1)

    def evaluate(self, x: torch.Tensor, a: torch.Tensor):
        """Recompute log_prob and entropy for stored (x, a) during PPO update."""
        mu_raw = self._mu_raw(x)
        std    = self.log_std.clamp(-3.0, 0.5).exp()
        dist   = Normal(mu_raw, std.expand_as(mu_raw))
        a_c    = (a / U_MAX).clamp(-1 + 1e-6, 1 - 1e-6)
        a_raw  = torch.atanh(a_c)
        lp = dist.log_prob(a_raw) \
             - torch.log1p(-a_c.pow(2) + 1e-6) \
             - math.log(U_MAX)
        return lp.sum(-1), dist.entropy().sum(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self._norm(x)).squeeze(-1)

    def to_policy_net(self) -> RoomTempControlNN:
        """Export actor weights into a standalone RoomTempControlNN."""
        policy_net = RoomTempControlNN(input_dim=2, hidden_dim=32, output_dim=2, u_max=U_MAX)
        policy_net.fc1.weight.data.copy_(self.fc1.weight.data)
        policy_net.fc1.bias.data.copy_(self.fc1.bias.data)
        policy_net.fc2.weight.data.copy_(self.fc2.weight.data)
        policy_net.fc2.bias.data.copy_(self.fc2.bias.data)
        return policy_net


# =========================================================================
# Episode rollout
# =========================================================================
def rollout_episode(policy: ActorCritic, rng: np.random.Generator, device):
    """
    Collect one episode using the current policy.

    Returns
    -------
    S       : (T, 2)  states
    A       : (T, 2)  actions
    R       : (T,)    rewards
    LP      : (T,)    log-probs at collection time
    V       : (T,)    values at collection time
    last_v  : float   bootstrap value
    outcome : str     'success' | 'fail' | 'timeout'
    """
    x = rng.uniform(X0_LO, X0_HI, size=2)

    states, acts, rews, lps, vals = [], [], [], [], []
    outcome = "timeout"
    last_v  = 0.0

    for _ in range(MAX_STEPS):
        xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, lp, _ = policy.act(xt)
            v        = policy.value(xt)

        u  = a.squeeze(0).cpu().numpy()
        xn = _step_dynamics(x, u, rng)

        # Domain exit: temperature outside [X_MIN, X_MAX]^2 → fail
        out_of_domain = not _in_box_2d(xn, X_MIN, X_MAX)

        # Potential-based shaping: γ·φ(x_next) − φ(x_curr)
        # φ(x) = −||x − x_goal||^2  (rewards reducing distance to goal)
        phi_cur  = -float(np.sum((x  - X_GOAL_CENTER) ** 2))
        phi_next = -float(np.sum((xn - X_GOAL_CENTER) ** 2))
        shaping  = C_SHAPE * (GAMMA * phi_next - phi_cur)

        if out_of_domain:
            r, done, outcome = R_FAIL, True, "fail"
        elif _in_box_2d(xn, XG_LO, XG_HI):
            r, done, outcome = R_SUCCESS, True, "success"
        elif not _in_box_2d(xn, XS_LO, XS_HI):
            r, done, outcome = R_FAIL, True, "fail"
        else:
            r    = shaping
            done = False

        states.append(x.copy())
        acts.append(u.copy())
        rews.append(r)
        lps.append(float(lp.squeeze().cpu()))
        vals.append(float(v.cpu()))

        # Clip when continuing (keeps trajectory in domain)
        x = np.clip(xn, X_MIN, X_MAX)

        if done:
            last_v = 0.0
            break
    else:
        xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            last_v = float(policy.value(xt).cpu())

    S  = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    A  = torch.tensor(np.array(acts),   dtype=torch.float32, device=device)
    R  = torch.tensor(rews,             dtype=torch.float32, device=device)
    LP = torch.tensor(lps,              dtype=torch.float32, device=device)
    V  = torch.tensor(vals,             dtype=torch.float32, device=device)
    return S, A, R, LP, V, last_v, outcome


# =========================================================================
# GAE advantage estimation
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
# PPO update
# =========================================================================
def ppo_update(policy: ActorCritic, optimizer: torch.optim.Optimizer,
               S: torch.Tensor, A: torch.Tensor,
               Ret: torch.Tensor, Adv: torch.Tensor,
               LP_old: torch.Tensor) -> None:
    """Standard PPO clipped policy + value + entropy update."""
    N   = S.shape[0]
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
    print("PPO Training — Two-Room Temperature Control")
    print(f"  T_e ~ Uniform({T_E_MIN}, {T_E_MAX}),  σ = {SIGMA}")
    print(f"  Init: [{X0_LO},{X0_HI}]^2  Goal: [{XG_LO},{XG_HI}]^2  Safe: [{XS_LO},{XS_HI}]^2")
    print(f"  DT={DT}, T_MAX={T_MAX}s ({MAX_STEPS} steps/ep)")
    print(f"  Device: {device},  Updates: {N_UPDATES},  episodes/update: {N_COLLECT}")
    print("=" * 65)

    rng    = np.random.default_rng(seed=42)
    policy = ActorCritic().to(device)
    optim  = torch.optim.Adam(policy.parameters(), lr=LR)

    best_sr      = -1.0
    best_weights = None
    t0           = time.time()

    for upd in range(1, N_UPDATES + 1):
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

        ppo_update(policy, optim, S_all, A_all, Ret_all, Adv_all, LP_all)

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

    # Restore best policy
    if best_weights is not None:
        policy.load_state_dict(best_weights)
    policy.cpu().eval()

    # Export as RoomTempControlNN state_dict (raw tensor dict)
    # plot.py loads this via policy_net.load_state_dict(control_state, strict=True)
    policy_net = policy.to_policy_net()
    save_path  = OUTPUT_DIR / "rl_controller.pth"
    torch.save(policy_net.state_dict(), save_path)
    print(f"Saved: {save_path}")
    print("  → load in plot.py as: --controller rl=path/to/rl_controller.pth")


if __name__ == "__main__":
    main()
