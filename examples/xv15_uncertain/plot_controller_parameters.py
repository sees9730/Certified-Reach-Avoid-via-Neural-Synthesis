"""Visualize saved XV15EqMLPControl weights as square-cell heatmaps.

Run from any directory; defaults compare the three seed0 controllers.
Differences use saved neuron order, with no permutation alignment. Independently
trained networks may implement similar controls with different weights.
"""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULTS = (
    HERE / "neural_certified_uncertain_param/seed0/outputs",
    HERE / "rl_sb3_ppo/seed0/outputs",
    HERE / "rl_sb3_ppo_finetuned_certified_uncertain_param/seed0/outputs",
)
LABELS = ("Certified", "PPO", "PPO certified & fine-tuned")
WEIGHTS = ("fc1.weight", "fc2.weight")


def load_state(source):
    """Prefer final bundles over pretraining snapshots in output directories."""
    source = Path(source)
    if source.is_dir():
        source = next((source / name for name in
                       ("eval_bundle.pth", "rl_controller.pth", "controller_pretrained.pth")
                       if (source / name).is_file()), None)
        if source is None:
            raise ValueError("Output directory contains no controller checkpoint")
    # Local evaluation bundles also contain NumPy arrays and training metadata.
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload.get("control_state_dict", payload)
    if not isinstance(state, dict) or not all(name in state for name in WEIGHTS):
        raise ValueError(f"Expected an XV15EqMLPControl checkpoint: {source}")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
           for value in state.values()):
        raise ValueError(f"Invalid or nonfinite controller tensors: {source}")
    if any(name.endswith(".bias") for name in state):
        raise ValueError(f"Expected a bias-free XV15EqMLPControl: {source}")
    if (state[WEIGHTS[0]].ndim != 2 or state[WEIGHTS[1]].ndim != 2
            or state[WEIGHTS[0]].shape[1] != 3 or state[WEIGHTS[1]].shape[0] != 3
            or state[WEIGHTS[0]].shape[0] != state[WEIGHTS[1]].shape[1]):
        raise ValueError(f"Invalid controller layer shapes: {source}")
    return source.resolve(), {key: value.detach().double().numpy() for key, value in state.items()}


def save_figure(fig, output):
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180, facecolor="white")
    plt.close(fig)


def plot_weights(states, name, output):
    """Show each input/output's hidden-neuron weights as equal square tiles."""
    input_layer = name == WEIGHTS[0]
    channels = ("Airspeed (v)", "Flight-path angle (gamma)", "Rotor tilt (beta)") if input_layer else (
        "Thrust", "Angle of attack", "Tilt rate")
    hidden = states[0][name].shape[0 if input_layer else 1]
    side = int(np.ceil(np.sqrt(hidden)))
    limit = max(float(np.abs(state[name]).max()) for state in states) or 1.0
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5))
    row_images = []
    for row, channel in enumerate(channels):
        if not input_layer:
            limit = max(float(np.abs(state[name][row, :]).max()) for state in states) or 1.0
        for col, (state, label) in enumerate(zip(states, LABELS)):
            ax = axes[row, col]
            weights = state[name][:, row] if input_layer else state[name][row, :]
            tiles = np.full(side * side, np.nan)
            tiles[:hidden] = weights
            im = ax.imshow(tiles.reshape(side, side), cmap="RdBu_r", vmin=-limit, vmax=limit,
                           aspect="equal", interpolation="nearest")
            # Boundaries make one square per weight explicit.
            ax.set_xticks(np.arange(side + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(side + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xticks(range(side))
            ax.set_yticks(range(side), np.arange(side) * side)
            ax.tick_params(length=0, labelsize=8, pad=4)
            if row == 0:
                ax.set_title(label, fontsize=13, fontweight="bold", pad=12)
            if col == 0:
                ax.set_ylabel(channel + "\nNeuron row start", fontsize=11, labelpad=8)
            if row == 2:
                ax.set_xlabel("Neuron column offset", fontsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
        row_images.append(im)
    layer_title = "Input-layer weights" if input_layer else (
        "Output-layer weights")
    fig.suptitle(layer_title, fontsize=17, fontweight="bold", y=0.965)
    fig.subplots_adjust(left=0.14, right=0.87, bottom=0.13, top=0.89, wspace=0.18, hspace=0.17)
    if input_layer:
        color_ax = fig.add_axes([0.9, 0.25, 0.018, 0.5])
        fig.colorbar(im, cax=color_ax, label="Saved weight · blue = negative · red = positive")
    else:
        fig.canvas.draw()
        for row, (channel, row_im) in enumerate(zip(channels, row_images)):
            position = axes[row, -1].get_position()
            color_ax = fig.add_axes([0.9, position.y0, 0.018, position.height])
            fig.colorbar(row_im, cax=color_ax, label=f"{channel} weight")
    scale_note = ("All panels share the same color scale." if input_layer else
                  "Each output row has its own scale, shared across controllers. Blue = negative; red = positive.")
    # fig.text(0.14, 0.035,
    #          f"One square = one weight. Each grid contains {hidden} hidden neurons, in saved order.\n"
    #          "Hidden-neuron index = row start + column offset (0-based).\n" + scale_note,
    #          fontsize=9, color="#526170")
    save_figure(fig, output)


def compare(sources, output):
    loaded = [load_state(source) for source in sources]
    paths, states = zip(*loaded)
    for state in states[1:]:
        if set(state) != set(states[0]) or any(state[k].shape != states[0][k].shape for k in state):
            raise ValueError("Controller tensor names and shapes must match for direct comparison")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, buffer_rows = [], []
    for first, second in combinations(range(3), 2):
        for name in (*WEIGHTS, "all_weights"):
            a = (np.concatenate([states[first][k].ravel() for k in WEIGHTS])
                 if name == "all_weights" else states[first][name].ravel())
            b = (np.concatenate([states[second][k].ravel() for k in WEIGHTS])
                 if name == "all_weights" else states[second][name].ravel())
            diff = b - a
            norm = float(np.linalg.norm(a))
            denominator = norm * float(np.linalg.norm(b))
            rows.append(dict(first=LABELS[first], second=LABELS[second], tensor=name,
                             count=int(a.size), changed=int(np.count_nonzero(diff)),
                             mean_abs=float(np.abs(diff).mean()), max_abs=float(np.abs(diff).max()),
                             rms=float(np.sqrt(np.mean(diff ** 2))), l2=float(np.linalg.norm(diff)),
                             relative_l2=float(np.linalg.norm(diff) / norm) if norm else None,
                             cosine=float(np.dot(a, b) / denominator) if denominator else None))
        for name in states[first]:
            if name not in WEIGHTS:
                a, b = states[first][name], states[second][name]
                buffer_rows.append(dict(first=LABELS[first], second=LABELS[second], tensor=name,
                                        equal=bool(np.array_equal(a, b)),
                                        first_value=a.tolist(), second_value=b.tolist(),
                                        max_abs=float(np.abs(b - a).max())))
    plot_weights(states, WEIGHTS[0], output / "controller_input_weights")
    plot_weights(states, WEIGHTS[1], output / "controller_output_weights")
    with (output / "parameter_differences.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(checkpoints=dict(zip(LABELS, map(str, paths))), metrics=rows,
                   fixed_buffers=buffer_rows,
                   interpretation="Differences use saved neuron order, without permutation alignment. "
                   "Relative L2 uses the first controller as reference. Fixed normalization, trim, "
                   "and limit buffers are reported separately from learned weights. "
                   "Weight distance alone does not measure control behavior or certification.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for row in rows:
        if row["tensor"] == "all_weights":
            print(f"{row['second']} minus {row['first']}: "
                  f"{row['changed']}/{row['count']} weights changed, "
                  f"RMS={row['rms']:.6g}, max |difference|={row['max_abs']:.6g}")
    print(f"Plots and summaries saved to {output.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in zip(("certified", "ppo", "finetuned"), DEFAULTS):
        parser.add_argument(f"--{flag}", type=Path, default=default,
                            help="Controller checkpoint file or output directory")
    parser.add_argument("--output-dir", type=Path, default=HERE / "controller_parameter_comparison")
    args = parser.parse_args(argv)
    compare((args.certified, args.ppo, args.finetuned), args.output_dir)


if __name__ == "__main__":
    main()
