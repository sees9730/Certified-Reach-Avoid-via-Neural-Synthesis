"""
Off-the-shelf RL Controller Training — Inverted Pendulum (Additive Box Disturbance)
====================================================================================

Companion to `rl_sb3_ppo/main.py` and `rl_sb3_sac/main.py`, using a third
off-the-shelf algorithm: Stable-Baselines3's `DDPG` (Deep Deterministic Policy
Gradient).

DDPG is off-policy (replay buffer, like SAC) but learns a *deterministic*
policy with a single Q-critic, rather than SAC's stochastic squashed-Gaussian
policy with twin critics and entropy regularization. Exploration comes from
explicit action noise added on top of the deterministic action during
training, not from a learned stochastic policy. It's the direct precursor to
TD3/SAC and a useful "vanilla deterministic-policy" baseline to round out the
comparison: PPO (on-policy, stochastic), SAC (off-policy, stochastic,
entropy-regularized), DDPG (off-policy, deterministic, noise-driven
exploration).

Environment
-----------
  State  : (x1, x2) = (angle, angular velocity)
  Domain : [-2pi, 2pi] x [-20, 20]                (shared via config.json)
  Disturbance: d(t) = [0, d2(t)], d2(t) ~ Uniform[-DRIFT_MAG, DRIFT_MAG] (torque channel only)
  SDE    : dx1 = x2 dt
           dx2 = ((g/L)*sin(x1) - b*x2/(m*L^2) + u_applied + d2) dt + sigma dW
  Init   : X_init = [3pi/4, 5pi/4] x [-1, 1]      (pendulum near bottom)
  Goal   : X_goal (shared via config.json)
  Unsafe : union of shared config-defined unsafe regions

Reward
------
  Terminal: +R_SUCCESS if next state in goal; -R_FAIL if next state in unsafe / domain-exit
  Running : potential-based shaping + small time penalty (matches rl_ppo_finetune)

Why distillation
-----------------
`run_mc.py` loads every controller as a `WrapperConterlNN(InvertControlNN(...))`
state_dict — a single tanh-hidden-layer network. Stable-Baselines3's DDPG actor
is structured differently (separate `mu` network with its own layer sizes) even
though it also ends in a Tanh, so its raw weights aren't a direct match. Instead,
after DDPG training converges, the SB3 policy is distilled (behavior-cloned)
into an `InvertControlNN` via supervised regression on (state, deterministic
SB3 action) pairs — identical methodology to `rl_sb3_ppo/main.py` and
`rl_sb3_sac/main.py`, so all three off-the-shelf baselines are exported and
compared the same way.

Controller architecture (exported)
-----------------------------------
  InvertControlNN(input_dim=2, hidden_dim=CONTROLLER_HIDDEN_DIM, output_dim=1)
  — identical to neural_certified/main.py and the other rl_* baselines.

Output (all under seed<TRAIN_SEED>/outputs/, e.g. seed0/outputs/)
------
  terminal_log.txt
      -- saved terminal output from the training run
  ddpg_offtheshelf.zip
      -- raw Stable-Baselines3 DDPG checkpoint (for reference / re-distillation)
  rl_controller.pth
      -- raw state_dict of WrapperConterlNN(distilled InvertControlNN)
      -- load in run_mc.py / plot.py with IS_PRETRAINED=True
  rl_training_history.csv
      -- periodic deterministic-eval metrics logged during DDPG training
  rl_training_history.pdf
      -- training-history figure generated from the saved CSV metrics
  rl_training_summary.json
      -- hyperparameters + final SB3-vs-distilled evaluation summary

Usage
-----
    python main.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN
from src.inv_pend_adversarial_config import (
    load_controller_hidden_dim, load_dynamics_params, load_region_arrays,
)
from src.save_load_utils import enable_terminal_logging

with (EXAMPLE_ROOT / "config.json").open("r", encoding="utf-8") as f:
    EXAMPLE_CONFIG = json.load(f)


# ============================================================================
# Shared configuration  (must match neural_certified/main.py, rl_ppo_finetune/main.py)
# ============================================================================
pi = np.pi

REGIONS = load_region_arrays(EXAMPLE_CONFIG)
CONTROLLER_HIDDEN_DIM = load_controller_hidden_dim(EXAMPLE_CONFIG)
DRIFT_MAG = float(EXAMPLE_CONFIG["drift_mag"])

FULL_RANGE = REGIONS["full_range"]
INIT_RANGE = REGIONS["init_range"]
GOAL_RANGE = REGIONS["goal_range"]
UNSAFE_RANGES = REGIONS["unsafe_ranges"]

X1_MIN, X1_MAX = map(float, FULL_RANGE[0])
X2_MIN, X2_MAX = map(float, FULL_RANGE[1])

INIT_BOX = [
    float(INIT_RANGE[0, 0]),
    float(INIT_RANGE[0, 1]),
    float(INIT_RANGE[1, 0]),
    float(INIT_RANGE[1, 1]),
]
GOAL_BOX = [
    float(GOAL_RANGE[0, 0]),
    float(GOAL_RANGE[0, 1]),
    float(GOAL_RANGE[1, 0]),
    float(GOAL_RANGE[1, 1]),
]
UNSAFE_BOXES = [
    [float(box[0, 0]), float(box[0, 1]), float(box[1, 0]), float(box[1, 1])]
    for box in UNSAFE_RANGES
]


# ============================================================================
# Physical constants  (single source of truth: config.json "dynamics")
# ============================================================================
DYNAMICS = load_dynamics_params(EXAMPLE_CONFIG)
G_NOM = DYNAMICS["g"]
L_NOM = DYNAMICS["L"]
B_NOM = DYNAMICS["b"]
M_NOM = DYNAMICS["m"]
M_TORQUE = DYNAMICS["M_torque"]
SIGMA = DYNAMICS["sigma"]
TORQUE_SCALE = M_TORQUE / (M_NOM * L_NOM ** 2)


# ============================================================================
# Environment and reward settings  (matches rl_ppo_finetune/main.py)
# ============================================================================
DT = 0.01
T_MAX = 30.0
MAX_STEPS = int(T_MAX / DT)

R_SUCCESS = 1.0
R_FAIL = -1.0
STEP_PENALTY = -0.002
POTENTIAL_ANGLE_WEIGHT = 0.20
POTENTIAL_VEL_WEIGHT = 0.05
GAMMA = 0.99  # discount, also used for potential-based reward shaping


# ============================================================================
# DDPG (Stable-Baselines3) hyperparameters
# ============================================================================
TRAIN_SEED = 3
N_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000

LR = 1e-3
BUFFER_SIZE = 300_000
LEARNING_STARTS = 5_000
BATCH_SIZE = 256
TAU = 0.02
TRAIN_FREQ = (N_ENVS, "step")   # collect N_ENVS steps between gradient updates
GRADIENT_STEPS = N_ENVS         # one gradient step per env step collected
ACTION_NOISE_SIGMA = 0.2        # std of Gaussian exploration noise (action in [-1, 1])


# ============================================================================
# Deterministic-evaluation / early-stopping settings
# ============================================================================
EVAL_EVERY_STEPS = 20_000
EVAL_EPISODES = 64
FINAL_EVAL_EPISODES = 300

SUCCESS_MA_WINDOW = 10
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.005
EARLY_STOPPING_WARMUP_EVALS = 10


# ============================================================================
# Distillation settings  (SB3 policy -> InvertControlNN, for run_mc.py export)
# ============================================================================
N_DISTILL_RANDOM = 20_000       # domain-covering random states
N_DISTILL_TRAJ_EPISODES = 200   # on-policy trajectories (state coverage near rollouts)
DISTILL_EPOCHS = 300
DISTILL_LR = 1e-3
DISTILL_BATCH_SIZE = 512


def configure_reproducibility(seed: int) -> None:
    """Configure deterministic behavior for repeated runs on the same stack."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _in_box(x1: float, x2: float, box: list[float]) -> bool:
    return box[0] <= x1 <= box[1] and box[2] <= x2 <= box[3]


def _in_unsafe(x1: float, x2: float) -> bool:
    return any(_in_box(x1, x2, box) for box in UNSAFE_BOXES)


# ============================================================================
# Gymnasium environment
# ============================================================================
import gymnasium as gym
from gymnasium import spaces


class InvPendReachAvoidEnv(gym.Env):
    """
    Inverted-pendulum SDE reach-avoid task, wrapped as a Gymnasium environment
    so it can be trained with an off-the-shelf RL package (Stable-Baselines3).

    Action a in [-1, 1] is the raw control u_raw; the applied torque is
    TORQUE_SCALE * clip(a, -1, 1), matching WrapperConterlNN's scaling so the
    distilled InvertControlNN is a drop-in match for the SB3-trained policy.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([X1_MIN, X2_MIN], dtype=np.float32),
            high=np.array([X1_MAX, X2_MAX], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(2, dtype=np.float32)
        self._step_count = 0

    def _potential(self, state: np.ndarray) -> float:
        x1, x2 = float(state[0]), float(state[1])
        vel_term = min(abs(x2) / max(abs(X2_MIN), abs(X2_MAX)), 1.0)
        return POTENTIAL_ANGLE_WEIGHT * np.cos(x1) - POTENTIAL_VEL_WEIGHT * vel_term

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        x1 = self.np_random.uniform(INIT_BOX[0], INIT_BOX[1])
        x2 = self.np_random.uniform(INIT_BOX[2], INIT_BOX[3])
        self.state = np.array([x1, x2], dtype=np.float32)
        self._step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        x1, x2 = float(self.state[0]), float(self.state[1])
        action_raw = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        torque = TORQUE_SCALE * action_raw

        # Physical disturbance: unknown external torque acting on x2 only
        # (x1 = theta has no direct force input).
        disturbance = np.array(
            [0.0, self.np_random.uniform(-DRIFT_MAG, DRIFT_MAG)], dtype=np.float32,
        )
        brownian_increment = np.sqrt(DT) * self.np_random.standard_normal()

        dx1 = x2 + disturbance[0]
        dx2 = (
            (G_NOM / L_NOM) * np.sin(x1)
            - (B_NOM / (M_NOM * L_NOM ** 2)) * x2
            + torque
            + disturbance[1]
        )
        x1n = x1 + dx1 * DT
        x2n = x2 + dx2 * DT + SIGMA * brownian_increment
        next_state = np.array([x1n, x2n], dtype=np.float32)

        out_of_domain = not (X1_MIN <= x1n <= X1_MAX) or not (X2_MIN <= x2n <= X2_MAX)
        self._step_count += 1

        terminated = False
        truncated = False
        outcome = "timeout"

        if out_of_domain:
            reward, terminated, outcome = R_FAIL, True, "fail"
        elif _in_box(x1n, x2n, GOAL_BOX):
            reward, terminated, outcome = R_SUCCESS, True, "success"
        elif _in_unsafe(x1n, x2n):
            reward, terminated, outcome = R_FAIL, True, "fail"
        else:
            shaping = GAMMA * self._potential(next_state) - self._potential(self.state)
            reward = STEP_PENALTY + shaping

        if not terminated:
            next_state = np.clip(
                next_state, [X1_MIN, X2_MIN], [X1_MAX, X2_MAX]
            ).astype(np.float32)
            if self._step_count >= MAX_STEPS:
                truncated = True

        self.state = next_state
        info = {"outcome": outcome if (terminated or truncated) else "running"}
        return self.state.copy(), float(reward), terminated, truncated, info


# ============================================================================
# Shared deterministic-rollout evaluation  (works for any action_fn: obs -> action)
# ============================================================================
def evaluate_action_fn(action_fn, n_episodes: int, seed_base: int) -> dict[str, float]:
    env = InvPendReachAvoidEnv()
    outcomes = {"success": 0, "fail": 0, "timeout": 0}
    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        info: dict = {}
        while not done:
            action = np.asarray(action_fn(obs), dtype=np.float32).reshape(1)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
        returns.append(ep_return)
        lengths.append(ep_length)

    n = n_episodes
    return {
        "success_rate": outcomes["success"] / n,
        "fail_rate": outcomes["fail"] / n,
        "timeout_rate": outcomes["timeout"] / n,
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def print_eval_summary(label: str, metrics: dict[str, float]) -> None:
    w = 62
    print(f"\n{'=' * w}")
    print(f"  {label}")
    print(f"{'=' * w}")
    print(f"  success_rate   = {metrics['success_rate']:.4f}")
    print(f"  fail_rate      = {metrics['fail_rate']:.4f}")
    print(f"  timeout_rate   = {metrics['timeout_rate']:.4f}")
    print(f"  mean_return    = {metrics['mean_return']:.4f}")
    print(f"  mean_ep_length = {metrics['mean_episode_length']:.1f}")


# ============================================================================
# DDPG training callback: periodic deterministic eval, best-checkpoint tracking,
# and success-plateau early stopping (mirrors rl_ppo_finetune's stopping rule).
# ============================================================================
from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_every_steps: int,
        eval_episodes: int,
        ma_window: int,
        patience: int,
        min_delta: float,
        warmup_evals: int,
        seed_base: int = 200_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_every_steps = eval_every_steps
        self.eval_episodes = eval_episodes
        self.ma_window = ma_window
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_evals = warmup_evals
        self.seed_base = seed_base

        self.history: list[dict[str, float]] = []
        self.validation_window: deque[float] = deque(maxlen=ma_window)
        self.best_eval_success = -float("inf")
        self.best_weights: dict[str, torch.Tensor] | None = None
        self.best_timesteps = 0
        self.plateau_evals = 0
        self.best_eval_ma = -float("inf")
        self.stop_timesteps: int | None = None
        self._last_eval_step = 0
        self._t0 = time.time()

    def _on_training_start(self) -> None:
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_every_steps:
            return True
        self._last_eval_step = self.num_timesteps

        action_fn = lambda obs: self.model.predict(obs, deterministic=True)[0]
        metrics = evaluate_action_fn(action_fn, self.eval_episodes, self.seed_base)
        self.validation_window.append(metrics["success_rate"])
        eval_ma = float(np.mean(self.validation_window))

        if metrics["success_rate"] > self.best_eval_success + 1e-12:
            self.best_eval_success = metrics["success_rate"]
            self.best_timesteps = self.num_timesteps
            self.best_weights = {
                k: v.detach().cpu().clone() for k, v in self.model.policy.state_dict().items()
            }

        if len(self.validation_window) == self.ma_window:
            if eval_ma > self.best_eval_ma + self.min_delta:
                self.best_eval_ma = eval_ma
                self.plateau_evals = 0
            else:
                self.plateau_evals += 1
        elif eval_ma > self.best_eval_ma:
            self.best_eval_ma = eval_ma

        elapsed = time.time() - self._t0
        self.history.append({
            "timesteps": self.num_timesteps,
            "eval_success_rate": metrics["success_rate"],
            "eval_fail_rate": metrics["fail_rate"],
            "eval_timeout_rate": metrics["timeout_rate"],
            "eval_mean_return": metrics["mean_return"],
            "eval_mean_episode_length": metrics["mean_episode_length"],
            "eval_success_ma": eval_ma,
            "best_eval_success_rate": self.best_eval_success,
            "plateau_evals": self.plateau_evals,
            "elapsed_sec": elapsed,
        })

        print(
            f"t={self.num_timesteps:>9d} | EvalSR={metrics['success_rate']:.2f} "
            f"EvalMA={eval_ma:.2f} BestSR={self.best_eval_success:.2f} "
            f"Fail={metrics['fail_rate']:.2f} Timeout={metrics['timeout_rate']:.2f} "
            f"Elapsed={elapsed:.0f}s"
        )

        evals_seen = len(self.history)
        if (
            len(self.validation_window) == self.ma_window
            and evals_seen >= self.warmup_evals
            and self.plateau_evals >= self.patience
        ):
            self.stop_timesteps = self.num_timesteps
            print(
                "\nEarly stopping: validation success plateaued "
                f"(window={self.ma_window}, patience={self.patience}, "
                f"min_delta={self.min_delta})."
            )
            return False
        return True


# ============================================================================
# Distillation: SB3 policy -> InvertControlNN (for run_mc.py / WrapperConterlNN)
# ============================================================================
def build_distillation_dataset(model, n_random: int, n_traj_episodes: int, seed: int):
    """Sample (state, deterministic SB3 action) pairs covering the full domain
    plus states actually visited by the trained policy."""
    rng = np.random.default_rng(seed)
    random_states = np.stack([
        rng.uniform(X1_MIN, X1_MAX, size=n_random),
        rng.uniform(X2_MIN, X2_MAX, size=n_random),
    ], axis=1).astype(np.float32)

    traj_states: list[np.ndarray] = []
    env = InvPendReachAvoidEnv()
    for ep in range(n_traj_episodes):
        obs, _ = env.reset(seed=seed + 500_000 + ep)
        done = False
        while not done:
            traj_states.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    traj_states_arr = np.asarray(traj_states, dtype=np.float32)

    states = np.concatenate([random_states, traj_states_arr], axis=0)

    actions_chunks = []
    for i in range(0, len(states), 4096):
        chunk = states[i:i + 4096]
        a_chunk, _ = model.predict(chunk, deterministic=True)
        actions_chunks.append(a_chunk)
    actions = np.concatenate(actions_chunks, axis=0).astype(np.float32)

    return (
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.float32),
    )


def train_distilled_controller(
    states: torch.Tensor,
    actions: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> tuple[InvertControlNN, list[float]]:
    net = InvertControlNN(input_dim=2, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    states = states.to(device)
    actions = actions.to(device)
    n = states.shape[0]
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        idx = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            pred = net(states[b])
            loss = F.mse_loss(pred, actions[b])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * b.numel()
        epoch_loss /= n
        loss_history.append(epoch_loss)

        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(f"  [distill] epoch {epoch:4d}/{epochs}  mse={epoch_loss:.6f}")

    net.cpu().eval()
    return net, loss_history


def make_distilled_action_fn(control_net: InvertControlNN):
    def action_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = control_net(xt).squeeze(0).numpy()
        return a
    return action_fn


# ============================================================================
# Logging helpers  (match the CSV / JSON / PDF conventions of the other rl_* baselines)
# ============================================================================
def save_training_history(output_dir: Path, history: list[dict[str, float]]) -> Path:
    history_path = output_dir / "rl_training_history.csv"
    if not history:
        return history_path

    fieldnames = list(history[0].keys())
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return history_path


def save_training_summary(output_dir: Path, summary: dict[str, object]) -> Path:
    summary_path = output_dir / "rl_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def plot_training_history(output_dir: Path, history: list[dict[str, float]]) -> Path | None:
    if not history:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    timesteps = [row["timesteps"] for row in history]
    eval_sr = [row["eval_success_rate"] for row in history]
    eval_ma = [row["eval_success_ma"] for row in history]
    mean_return = [row["eval_mean_return"] for row in history]

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    axes[0].plot(timesteps, eval_sr, label="Eval success rate", linewidth=1.5, alpha=0.6)
    axes[0].plot(timesteps, eval_ma, label="Moving average", linewidth=2.0)
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(timesteps, mean_return, linewidth=2.0, color="tab:green")
    axes[1].set_ylabel("Eval Mean Return")
    axes[1].set_xlabel("Timesteps")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("SB3 DDPG Training History (Off-the-shelf)")
    fig.tight_layout()

    plot_path = output_dir / "rl_training_history.pdf"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    active_output_dir = HERE / f"seed{TRAIN_SEED}" / "outputs"
    active_output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = enable_terminal_logging(active_output_dir / "terminal_log.txt", append=False)
    configure_reproducibility(TRAIN_SEED)

    from stable_baselines3 import DDPG
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.noise import NormalActionNoise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("Off-the-shelf DDPG Training (Stable-Baselines3) - Inverted Pendulum SDE")
    print(f"  Disturbance : d=[0, d2], d2 ~ Uniform([-{DRIFT_MAG:.1f}, {DRIFT_MAG:.1f}])")
    print(f"  Diffusion   : sigma = {SIGMA}")
    print(f"  Device      : {device}")
    print(f"  Seed        : {TRAIN_SEED}")
    print(f"  Envs        : {N_ENVS}")
    print(f"  Total steps : {TOTAL_TIMESTEPS}")
    print(f"  Eval        : every {EVAL_EVERY_STEPS} steps on {EVAL_EPISODES} episodes")
    print("=" * 72)

    def make_env():
        return InvPendReachAvoidEnv()

    vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=TRAIN_SEED)

    n_actions = vec_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=ACTION_NOISE_SIGMA * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh),
        learning_rate=LR,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        action_noise=action_noise,
        seed=TRAIN_SEED,
        device=device,
        verbose=0,
    )

    callback = SuccessRateEvalCallback(
        eval_every_steps=EVAL_EVERY_STEPS,
        eval_episodes=EVAL_EPISODES,
        ma_window=SUCCESS_MA_WINDOW,
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        warmup_evals=EARLY_STOPPING_WARMUP_EVALS,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    total_training_time = time.time() - t0

    print(f"\nTraining complete. Best validation SR = {callback.best_eval_success:.3f} "
          f"at t={callback.best_timesteps}.")
    print(f"Training completed in {total_training_time:.2f} seconds "
          f"({total_training_time / 60.0:.2f} minutes)")

    if callback.best_weights is not None:
        model.policy.load_state_dict(callback.best_weights)
    model.policy.set_training_mode(False)

    sb3_model_path = active_output_dir / "ddpg_offtheshelf.zip"
    model.save(sb3_model_path)
    print(f"Saved SB3 checkpoint: {sb3_model_path}")

    # ---- distill the SB3 policy into an InvertControlNN for run_mc.py -------
    print("\nBuilding distillation dataset from trained SB3 policy ...")
    states, actions = build_distillation_dataset(
        model, N_DISTILL_RANDOM, N_DISTILL_TRAJ_EPISODES, seed=TRAIN_SEED,
    )
    print(f"  dataset size: {states.shape[0]} state-action pairs")

    print(f"Distilling into InvertControlNN(hidden_dim={CONTROLLER_HIDDEN_DIM}) ...")
    distilled_actor, distill_loss_history = train_distilled_controller(
        states, actions, CONTROLLER_HIDDEN_DIM, DISTILL_EPOCHS, DISTILL_LR,
        DISTILL_BATCH_SIZE, device,
    )

    print(f"\nEvaluating SB3 policy vs. distilled controller "
          f"(deterministic, {FINAL_EVAL_EPISODES} episodes) ...")
    sb3_action_fn = lambda obs: model.predict(obs, deterministic=True)[0]
    distilled_action_fn = make_distilled_action_fn(distilled_actor)
    sb3_metrics = evaluate_action_fn(sb3_action_fn, FINAL_EVAL_EPISODES, seed_base=999_000)
    distilled_metrics = evaluate_action_fn(distilled_action_fn, FINAL_EVAL_EPISODES, seed_base=999_000)
    print_eval_summary("SB3 DDPG policy (pre-distillation)", sb3_metrics)
    print_eval_summary("Distilled InvertControlNN (exported)", distilled_metrics)

    # ---- save as WrapperConterlNN state_dict (IS_PRETRAINED=True) -----------
    wrapper = WrapperConterlNN(distilled_actor)
    controller_path = active_output_dir / "rl_controller.pth"
    torch.save(wrapper.state_dict(), controller_path)
    print(f"\nSaved controller: {controller_path}")
    print("  -> load in run_mc.py / plot.py with IS_PRETRAINED=True")

    history_path = save_training_history(active_output_dir, callback.history)
    summary_path = save_training_summary(active_output_dir, {
        "algorithm": "stable_baselines3.DDPG",
        "best_eval_success_rate": float(callback.best_eval_success),
        "best_timesteps": int(callback.best_timesteps),
        "total_training_time_sec": float(total_training_time),
        "seed": int(TRAIN_SEED),
        "n_envs": int(N_ENVS),
        "total_timesteps": int(TOTAL_TIMESTEPS),
        "dt": float(DT),
        "t_max": float(T_MAX),
        "sigma": float(SIGMA),
        "drift_mag": float(DRIFT_MAG),
        "controller_hidden_dim": int(CONTROLLER_HIDDEN_DIM),
        "lr": float(LR),
        "gamma": float(GAMMA),
        "buffer_size": int(BUFFER_SIZE),
        "learning_starts": int(LEARNING_STARTS),
        "batch_size": int(BATCH_SIZE),
        "tau": float(TAU),
        "train_freq": list(TRAIN_FREQ),
        "gradient_steps": int(GRADIENT_STEPS),
        "action_noise_sigma": float(ACTION_NOISE_SIGMA),
        "eval_every_steps": int(EVAL_EVERY_STEPS),
        "eval_episodes": int(EVAL_EPISODES),
        "success_ma_window": int(SUCCESS_MA_WINDOW),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "early_stopping_min_delta": float(EARLY_STOPPING_MIN_DELTA),
        "early_stopping_warmup_evals": int(EARLY_STOPPING_WARMUP_EVALS),
        "stopped_early": bool(callback.stop_timesteps is not None),
        "stop_timesteps": None if callback.stop_timesteps is None else int(callback.stop_timesteps),
        "distillation": {
            "n_random_states": int(N_DISTILL_RANDOM),
            "n_traj_episodes": int(N_DISTILL_TRAJ_EPISODES),
            "n_state_action_pairs": int(states.shape[0]),
            "epochs": int(DISTILL_EPOCHS),
            "lr": float(DISTILL_LR),
            "final_mse": float(distill_loss_history[-1]) if distill_loss_history else None,
        },
        "final_eval_episodes": int(FINAL_EVAL_EPISODES),
        "sb3_eval_metrics": sb3_metrics,
        "distilled_eval_metrics": distilled_metrics,
        "history_csv": history_path.name,
        "controller_path": controller_path.name,
        "sb3_model_path": sb3_model_path.name,
    })
    plot_path = plot_training_history(active_output_dir, callback.history)

    print(f"Saved training history CSV: {history_path}")
    print(f"Saved training summary JSON: {summary_path}")
    if plot_path is not None:
        print(f"Saved training history plot: {plot_path}")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_handle.close()


if __name__ == "__main__":
    main()
