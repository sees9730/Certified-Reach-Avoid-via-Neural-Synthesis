"""Off-the-shelf SB3 PPO followed by XV-15 controller distillation.

Mirrors inv_pend_adversarial/rl_sb3_ppo/main.py: standard MlpPolicy with
[64, 64] tanh layers, the same PPO defaults, periodic success-rate validation,
best-weight restoration, moving-average plateau stopping, and supervised MSE
behavior cloning from domain samples plus deterministic teacher trajectories.

The exported student is XV15EqMLPControl, exactly as in neural_certified*:
bias-free 3 -> hidden_dim -> 3, tanh hidden activation, nominal trim anchor,
state-error normalization, and bounded physical outputs. The teacher is an
unconstrained standard SB3 MLP. Student and teacher are evaluated separately.

XV-15 adaptations: three physical state/action channels, the shared aircraft
SDE and regions, independent uniform density/mass draws, geometric goal-distance
shaping, and normalized action-space MSE so thrust does not dominate the two
angular controls. Terminal rewards and running-only shaping follow the pendulum.
No certificate is computed by this empirical RL baseline.

Run from the repository root:
    ./.venv/bin/python -u examples/xv15_uncertain/rl_sb3_ppo/main.py --seed 0

Outputs under seed<seed>/outputs/: ppo_offtheshelf.zip (selected teacher),
ppo_last.zip, rl_controller.pth (distilled XV15EqMLPControl for run_mc.py),
run_config.json, terminal_log.txt, rl_training_history.csv/.pdf, and
rl_training_summary.json (teacher/student evaluation and distillation MSE).
Training starts from scratch and overwrites these filenames. --help lists
optional overrides. Dependencies: torch, numpy, scipy, gymnasium,
stable-baselines3, matplotlib.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
except ImportError as exc:
    raise ImportError("This baseline requires gymnasium and stable-baselines3.") from exc

from examples.xv15_uncertain.model import (
    DEG, DiagonalDiffusion, XV15Aero, XV15EqMLPControl,
    find_goal_equilibrium, load_config, load_region_arrays,
)
from examples.xv15_uncertain.uncertain_density import load_density_interval
from examples.xv15_uncertain.uncertain_parameters import load_mass_interval
from src.save_load_utils import enable_terminal_logging

# Editable defaults, following the pendulum PPO baseline's training budget.
TRAIN_SEED = 0
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8
N_STEPS = 1024
BATCH_SIZE = 256
N_EPOCHS = 10
LR = 3e-4
GAMMA = 0.99
GAE_LAM = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
DT = 0.01
T_MAX = 30.0
R_SUCCESS = 1.0
R_FAIL = -1.0
STEP_PENALTY = -0.002
POTENTIAL_WEIGHT = 0.20
EVAL_EVERY_STEPS = 20_000
EVAL_EPISODES = 64
FINAL_EVAL_EPISODES = 300
SUCCESS_MA_WINDOW = 10
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.005
EARLY_STOPPING_WARMUP_EVALS = 10


# Same distillation budget and optimizer as the pendulum reference.
N_DISTILL_RANDOM = 20_000
N_DISTILL_TRAJ_EPISODES = 200
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


def make_controller(config):
    """Use the same constructor, trim, and input scales as neural_certified*."""
    aero = XV15Aero(config)
    x_eq, u_eq = find_goal_equilibrium(config, aero)
    return XV15EqMLPControl(config, x_eq, u_eq, [100.0, 20.0 * DEG, 90.0 * DEG])


def action_bounds(config):
    c = config["control"]
    weight = config["dynamics"]["mass"] * config["dynamics"]["gravity"]
    t_min, t_max = np.asarray(c["thrust_over_weight"]) * weight
    a_max, d_max = c["alpha_max_deg"] * DEG, c["delta_max_deg_per_second"] * DEG
    return (np.asarray([t_min, -a_max, -d_max], dtype=np.float32),
            np.asarray([t_max, a_max, d_max], dtype=np.float32))


def inside(state, box):
    return bool(((state >= box[:, 0]) & (state <= box[:, 1])).all())


class XV15ReachAvoidEnv(gym.Env):
    """Shared physical model with safety-first events at integration boundaries."""
    metadata = {"render_modes": []}

    def __init__(self, config, *, dt=DT, t_max=T_MAX, gamma=GAMMA,
                 parameter_mode="uniform", uniform_refresh="step"):
        super().__init__()
        if not all(math.isfinite(v) and v > 0 for v in (dt, t_max)):
            raise ValueError("dt and t_max must be finite and positive")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if parameter_mode not in ("uniform", "nominal") or uniform_refresh not in ("step", "episode"):
            raise ValueError("Invalid parameter mode or refresh setting")
        self.arrays = load_region_arrays(config)
        self.aero = XV15Aero(config).eval()
        self.sigma = DiagonalDiffusion(config).sigma.numpy().copy()
        self.density_interval = load_density_interval(config)
        self.mass_interval = load_mass_interval(config)
        self.action_low, self.action_high = action_bounds(config)
        self.dt, self.t_max, self.gamma = dt, t_max, gamma
        self.parameter_mode, self.uniform_refresh = parameter_mode, uniform_refresh
        self.goal_center = self.arrays["goal_range"].mean(axis=1)
        self.input_scale = np.asarray([100.0, 20.0 * DEG, 90.0 * DEG], dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        # Raw physical observations; terminal states may lie outside full_range.
        self.observation_space = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.state = self.goal_center.copy()
        self.elapsed = 0.0
        self._done = True

    def physical_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(3)
        if not np.isfinite(action).all():
            raise ValueError("Action must contain three finite values")
        return self.action_low + (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * (self.action_high - self.action_low)

    def _sample_parameters(self):
        if self.parameter_mode == "nominal":
            return self.aero.density, self.aero.mass
        return (self.np_random.uniform(*self.density_interval),
                self.np_random.uniform(*self.mass_interval))

    def _potential(self, state):
        return -POTENTIAL_WEIGHT * float(np.square((state - self.goal_center) / self.input_scale).sum())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        box = self.arrays["init_range"]
        self.state = self.np_random.uniform(box[:, 0], box[:, 1]).astype(np.float32)
        self.elapsed, self._done = 0.0, False
        self.density, self.mass = self._sample_parameters()
        return self.state.copy(), {"outcome": "running"}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a terminated environment")
        physical = self.physical_action(action)
        if self.uniform_refresh == "step":
            self.density, self.mass = self._sample_parameters()
        step_dt = min(self.dt, self.t_max - self.elapsed)
        with torch.no_grad():
            drift = self.aero(torch.from_numpy(self.state[None]),
                              torch.from_numpy(physical[None]),
                              density=self.density, mass=self.mass)[0].numpy()
        next_state = (self.state + step_dt * drift + self.sigma * math.sqrt(step_dt)
                      * self.np_random.standard_normal(3)).astype(np.float32)
        self.elapsed += step_dt
        reason, outcome = "running", "running"
        if not np.isfinite(next_state).all():
            reason, outcome = "nonfinite", "fail"
        elif not inside(next_state, self.arrays["full_range"]):
            reason, outcome = "domain_exit", "fail"
        elif any(inside(next_state, box) for box in self.arrays["unsafe_ranges"]):
            reason, outcome = "unsafe", "fail"
        elif inside(next_state, self.arrays["goal_range"]):
            reason, outcome = "goal", "success"
        terminated = outcome != "running"
        truncated = not terminated and self.elapsed >= self.t_max - 1e-12
        if truncated:
            reason, outcome = "time_limit", "timeout"
        # Match the pendulum: fixed terminal rewards; potential shaping is
        # applied only to running steps (including a time-limit truncation).
        if terminated:
            reward = R_SUCCESS if outcome == "success" else R_FAIL
        else:
            reward = STEP_PENALTY + self.gamma * self._potential(next_state) - self._potential(self.state)
        if reason != "nonfinite":
            self.state = next_state
        self._done = terminated or truncated
        info = dict(outcome=outcome, reason=reason, time=self.elapsed,
                    density=self.density, mass=self.mass)
        return self.state.copy(), float(reward), terminated, truncated, info


def evaluate_action_fn(action_fn, config, env_kwargs, n_episodes, seed_base):
    env = XV15ReachAvoidEnv(config, **env_kwargs)
    counts = dict(success=0, fail=0, timeout=0)
    returns, lengths = [], []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed_base + ep)
            total, length = 0.0, 0
            while True:
                obs, reward, terminated, truncated, info = env.step(action_fn(obs))
                total += reward
                length += 1
                if terminated or truncated:
                    break
            counts[info["outcome"]] += 1
            returns.append(total)
            lengths.append(length)
    finally:
        env.close()
    return {**{f"{name}_rate": count / n_episodes for name, count in counts.items()},
            "mean_return": float(np.mean(returns)), "mean_episode_length": float(np.mean(lengths))}


class SuccessRateEvalCallback(BaseCallback):
    def __init__(
        self,
        config,
        env_kwargs,
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
        self.config = config
        self.env_kwargs = env_kwargs
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
        metrics = evaluate_action_fn(action_fn, self.config, self.env_kwargs, self.eval_episodes, self.seed_base)
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
            self.patience > 0
            and len(self.validation_window) == self.ma_window
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


def build_distillation_dataset(model, config, env_kwargs, n_random: int, n_traj_episodes: int, seed: int):
    """Sample (state, deterministic SB3 action) pairs covering the full domain
    plus states actually visited by the trained policy."""
    rng = np.random.default_rng(seed)
    full = load_region_arrays(config)["full_range"]
    random_states = rng.uniform(full[:, 0], full[:, 1], size=(n_random, 3)).astype(np.float32)

    traj_states: list[np.ndarray] = []
    env = XV15ReachAvoidEnv(config, **env_kwargs)
    for ep in range(n_traj_episodes):
        obs, _ = env.reset(seed=seed + 500_000 + ep)
        done = False
        while not done:
            traj_states.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    traj_states_arr = np.asarray(traj_states, dtype=np.float32).reshape(-1, 3)

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
    config,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> tuple[XV15EqMLPControl, list[float]]:
    net = make_controller(config).to(device)
    low, high = [torch.as_tensor(bound, device=device) for bound in action_bounds(config)]
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
            # Compare in the teacher's [-1, 1] action units. This balances
            # thrust (newtons) against alpha (radians) and delta (rad/s).
            pred = 2.0 * (net(states[b]) - low) / (high - low) - 1.0
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


def make_distilled_action_fn(control_net: XV15EqMLPControl, config):
    low, high = action_bounds(config)
    def action_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = control_net(xt).squeeze(0).numpy()
        return 2.0 * (a - low) / (high - low) - 1.0
    return action_fn


def save_training_history(output_dir, history):
    if history:
        with (output_dir / "rl_training_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def plot_training_history(output_dir, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    steps = [row["timesteps"] for row in history]
    for name, label in (("eval_success_rate", "Validation success"), ("eval_success_ma", "Moving average")):
        axes[0].plot(steps, [row[name] for row in history], label=label)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].legend()
    axes[1].plot(steps, [row["eval_mean_return"] for row in history])
    axes[1].set_ylabel("Mean return")
    axes[1].set_xlabel("Timesteps")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("XV-15 SB3 PPO teacher training")
    fig.tight_layout()
    fig.savefig(output_dir / "rl_training_history.pdf", bbox_inches="tight")
    plt.close(fig)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--config", type=Path, default=EXAMPLE_ROOT / "config.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--total-timesteps", type=positive_int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=positive_int, default=N_ENVS)
    parser.add_argument("--n-steps", type=positive_int, default=N_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--n-epochs", type=positive_int, default=N_EPOCHS)
    parser.add_argument("--lr", type=positive_float, default=LR)
    parser.add_argument("--gamma", type=positive_float, default=GAMMA)
    parser.add_argument("--dt", type=positive_float, default=DT)
    parser.add_argument("--t-max", type=positive_float, default=T_MAX)
    parser.add_argument("--parameter-mode", choices=("uniform", "nominal"), default="uniform")
    parser.add_argument("--uniform-refresh", choices=("step", "episode"), default="step")
    parser.add_argument("--eval-every-steps", type=positive_int, default=EVAL_EVERY_STEPS)
    parser.add_argument("--eval-episodes", type=positive_int, default=EVAL_EPISODES)
    parser.add_argument("--final-eval-episodes", type=positive_int, default=FINAL_EVAL_EPISODES)
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help="Validation plateaus before stopping; 0 disables early stopping")
    parser.add_argument("--distill-random", type=positive_int, default=N_DISTILL_RANDOM)
    parser.add_argument("--distill-traj-episodes", type=positive_int, default=N_DISTILL_TRAJ_EPISODES)
    parser.add_argument("--distill-epochs", type=positive_int, default=DISTILL_EPOCHS)
    parser.add_argument("--distill-lr", type=positive_float, default=DISTILL_LR)
    parser.add_argument("--distill-batch-size", type=positive_int, default=DISTILL_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: seed<seed>/outputs beside this script")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32-1")
    if args.gamma > 1:
        parser.error("--gamma must be in (0, 1]")
    if args.batch_size < 2 or args.n_steps * args.n_envs < 2:
        parser.error("batch size and total rollout size must each be at least 2")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be nonnegative")
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    load_region_arrays(config)
    load_density_interval(config)
    load_mass_interval(config)
    configure_reproducibility(args.seed)
    output_dir = args.output_dir or HERE / f"seed{args.seed}" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(dt=args.dt, t_max=args.t_max, gamma=args.gamma,
                      parameter_mode=args.parameter_mode, uniform_refresh=args.uniform_refresh)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["output_dir"] = str(output_dir.resolve())
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(mode="sb3_ppo_then_distillation", example=config, settings=settings,
                       ppo=dict(gae_lambda=GAE_LAM, clip_range=CLIP_EPS, ent_coef=ENT_COEF,
                                vf_coef=VF_COEF, max_grad_norm=MAX_GRAD_NORM, net_arch=[64, 64], activation="Tanh"),
                       reward=dict(success=R_SUCCESS, fail=R_FAIL, step=STEP_PENALTY,
                                   potential_weight=POTENTIAL_WEIGHT)), stream, indent=2)
    stdout, stderr = sys.stdout, sys.stderr
    log_handle = enable_terminal_logging(output_dir / "terminal_log.txt", append=False)
    vec_env = None
    try:
        print("XV-15 SB3 PPO: standard MlpPolicy, followed by controller distillation")
        print(f"Seed: {args.seed}; device: {args.device}; outputs: {output_dir.resolve()}")
        print(f"Unobserved density/mass: {args.parameter_mode}, refresh={args.uniform_refresh}; Brownian noise enabled")
        print(f"Teacher: 3 -> 64 -> 64 -> 3; student: XV15EqMLPControl(hidden_dim={config['controller_hidden_dim']})")
        vec_env = make_vec_env(lambda: XV15ReachAvoidEnv(config, **env_kwargs),
                               n_envs=args.n_envs, seed=args.seed)
        model = PPO("MlpPolicy", vec_env,
                    policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh),
                    n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                    learning_rate=args.lr, gamma=args.gamma, gae_lambda=GAE_LAM,
                    clip_range=CLIP_EPS, ent_coef=ENT_COEF, vf_coef=VF_COEF,
                    max_grad_norm=MAX_GRAD_NORM, seed=args.seed, device=args.device, verbose=0)
        callback = SuccessRateEvalCallback(
            config, env_kwargs, eval_every_steps=args.eval_every_steps,
            eval_episodes=args.eval_episodes, ma_window=SUCCESS_MA_WINDOW,
            patience=args.early_stopping_patience, min_delta=EARLY_STOPPING_MIN_DELTA,
            warmup_evals=EARLY_STOPPING_WARMUP_EVALS,
        )
        started = time.perf_counter()
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
        training_seconds = time.perf_counter() - started
        model.save(output_dir / "ppo_last.zip")
        if callback.best_weights is not None:
            model.policy.load_state_dict(callback.best_weights)
        model.policy.set_training_mode(False)
        model.save(output_dir / "ppo_offtheshelf.zip")
        save_training_history(output_dir, callback.history)

        print("Building distillation dataset from the selected SB3 teacher ...")
        states, actions = build_distillation_dataset(
            model, config, env_kwargs, args.distill_random, args.distill_traj_episodes, args.seed,
        )
        print(f"Dataset: {len(states)} state-action pairs")
        print(f"Distilling into XV15EqMLPControl(hidden_dim={config['controller_hidden_dim']}) ...")
        controller, distill_loss_history = train_distilled_controller(
            states, actions, config, args.distill_epochs, args.distill_lr,
            args.distill_batch_size, model.device,
        )
        state = {key: value.detach().cpu().clone() for key, value in controller.state_dict().items()}
        # Reuse run_mc.py's existing XV15EqMLPControl checkpoint loader.
        # No learned certificate is present in this RL checkpoint.
        torch.save(dict(format="xv15_eq_mlp_ppo_distilled_v1", V_state_dict=None, control_state_dict=state),
                   output_dir / "rl_controller.pth")
        print(f"Evaluating teacher and distilled controller on {args.final_eval_episodes} episodes ...")
        sb3_metrics = evaluate_action_fn(lambda obs: model.predict(obs, deterministic=True)[0],
                                         config, env_kwargs, args.final_eval_episodes, 999_000)
        distilled_metrics = evaluate_action_fn(make_distilled_action_fn(controller, config),
                                               config, env_kwargs, args.final_eval_episodes, 999_000)
        summary = dict(algorithm="stable_baselines3.PPO", settings=settings,
                       controller_class="XV15EqMLPControl", controller_hidden_dim=int(config["controller_hidden_dim"]),
                       best_eval_success_rate=(callback.best_eval_success if callback.history else None),
                       best_timesteps=callback.best_timesteps,
                       actual_timesteps=model.num_timesteps, stopped_early=callback.stop_timesteps is not None,
                       stop_timesteps=callback.stop_timesteps,
                       total_training_time_sec=training_seconds, sb3_eval_metrics=sb3_metrics,
                       distilled_eval_metrics=distilled_metrics,
                       distillation=dict(n_random_states=args.distill_random,
                                         n_traj_episodes=args.distill_traj_episodes,
                                         n_state_action_pairs=len(states), epochs=args.distill_epochs,
                                         lr=args.distill_lr, batch_size=args.distill_batch_size,
                                         final_mse=distill_loss_history[-1]),
                       controller_path="rl_controller.pth", sb3_model_path="ppo_offtheshelf.zip",
                       history_csv="rl_training_history.csv")
        with (output_dir / "rl_training_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        print(f"PPO completed in {training_seconds:.1f}s; best validation SR={summary['best_eval_success_rate']}")
        print("Final SB3 metrics:", json.dumps(sb3_metrics))
        print("Final distilled-controller metrics:", json.dumps(distilled_metrics))
        print(f"Saved controller: {output_dir / 'rl_controller.pth'}")
        if not args.no_plots and callback.history:
            plot_training_history(output_dir, callback.history)
    finally:
        if vec_env is not None:
            vec_env.close()
        sys.stdout, sys.stderr = stdout, stderr
        log_handle.close()


if __name__ == "__main__":
    main()
