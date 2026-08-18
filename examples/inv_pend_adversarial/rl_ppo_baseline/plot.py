"""
Plot RL training history for the inverted-pendulum PPO baseline.

Reads metrics from outputs/rl_training_history.csv and regenerates a
publication-friendly summary figure.

Usage
-----
    python plot.py
    python plot.py --csv outputs/rl_training_history.csv --out outputs/rl_training_history.pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
DEFAULT_CSV = OUTPUT_DIR / "rl_training_history.csv"
DEFAULT_OUT = OUTPUT_DIR / "rl_training_history.pdf"


def load_history(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find training history CSV: {csv_path}")

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "update": int(row["update"]),
                "success_rate": float(row["success_rate"]),
                "fail_rate": float(row["fail_rate"]),
                "timeout_rate": float(row["timeout_rate"]),
                "mean_return": float(row["mean_return"]),
                "mean_episode_length": float(row["mean_episode_length"]),
                "best_success_rate": float(row["best_success_rate"]),
                "policy_std": float(row["policy_std"]),
                "elapsed_sec": float(row["elapsed_sec"]),
            })
    if not rows:
        raise ValueError(f"Training history CSV is empty: {csv_path}")
    return rows


def make_plot(history: list[dict], out_path: Path) -> None:
    updates = [row["update"] for row in history]
    success_rate = [row["success_rate"] for row in history]
    best_success_rate = [row["best_success_rate"] for row in history]
    fail_rate = [row["fail_rate"] for row in history]
    timeout_rate = [row["timeout_rate"] for row in history]
    mean_return = [row["mean_return"] for row in history]
    mean_episode_length = [row["mean_episode_length"] for row in history]
    policy_std = [row["policy_std"] for row in history]

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11.0), sharex=True)

    axes[0].plot(updates, success_rate, label="Success rate")
    axes[0].plot(updates, best_success_rate, label="Best success rate", linestyle="--")
    axes[0].plot(updates, fail_rate, label="Fail rate", linestyle=":")
    axes[0].plot(updates, timeout_rate, label="Timeout rate", linestyle="-.")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, loc="best")

    axes[1].plot(updates, mean_return, color="tab:green")
    axes[1].set_ylabel("Mean Return")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(updates, mean_episode_length, color="tab:orange")
    axes[2].set_ylabel("Mean Ep. Len")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(updates, policy_std, color="tab:red")
    axes[3].set_ylabel("Policy Std")
    axes[3].set_xlabel("PPO Update")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("RL Training History")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RL PPO training history from CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to rl_training_history.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output path for the plot PDF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_history(args.csv)
    make_plot(history, args.out)
    print(f"Saved RL training history plot to: {args.out}")


if __name__ == "__main__":
    main()
