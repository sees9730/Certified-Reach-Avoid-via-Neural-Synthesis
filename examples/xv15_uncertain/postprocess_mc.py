"""Rebuild XV-15 Monte Carlo tables and figures from mc_cache.pth only."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from examples.xv15_uncertain.run_mc import OUTPUT_DIR, aggregate_results, write_reports


SUMMARY_GROUPS = ("zero", "uniform", "adversarial")


def collect_boxplot_samples(runs, rows=None):
    """Pool trajectory samples after the same attack selection as the heatmap.

    The third group includes EVERY remaining density/mass pair, including
    zero-uniform and uniform-zero. Each trajectory from each selected run has
    equal weight. Time uses successes only; effort retains all finite outcomes.
    """
    rows = aggregate_results(runs) if rows is None else rows
    selected = {(row["controller"], seed, row["density_mode"], row["mass_mode"]): attack
                for row in rows for seed, attack in row["selected_attacks"].items()}
    labels = list(dict.fromkeys(row["controller"] for row in rows))
    samples = {label: {group: {metric: [] for metric in ("time", "energy")}
                       for group in SUMMARY_GROUPS} for label in labels}
    for run in runs:
        key = (run["label"], run["training_seed"], run["density_mode"], run["mass_mode"])
        if run["attack"] != selected[key]:
            continue
        pair = (run["density_mode"], run["mass_mode"])
        group = ("zero" if pair == ("zero", "zero") else
                 "uniform" if pair == ("uniform", "uniform") else "adversarial")
        result = run["result"]
        success = np.asarray([outcome == "success" for outcome in result["outcomes"]])
        times = np.asarray(result["stop_times"], dtype=float)
        effort = np.asarray(result["normalized_effort"], dtype=float)
        target = samples[run["label"]][group]
        target["time"].append(times[success & np.isfinite(times)])
        target["energy"].append(effort[np.isfinite(effort)])
    return {label: {group: {metric: np.concatenate(parts) if parts else np.empty(0)
                            for metric, parts in metrics.items()}
                   for group, metrics in groups.items()}
            for label, groups in samples.items()}


def plot_results(cache, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate_results(cache["runs"])
    labels = list(dict.fromkeys(row["controller"] for row in rows))
    density_modes = cache["metadata"]["density_modes"]
    mass_modes = cache["metadata"]["mass_modes"]
    colors = {label: plt.get_cmap("tab10")(i % 10) for i, label in enumerate(labels)}
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4), squeeze=False,
                             constrained_layout=True)
    for ax, label in zip(axes[0], labels):
        matrix = np.full((len(density_modes), len(mass_modes)), np.nan)
        for row in rows:
            if row["controller"] == label:
                matrix[density_modes.index(row["density_mode"]), mass_modes.index(row["mass_mode"])] = row["p_success"]
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1%}", ha="center", va="center", color="black" if value > .6 else "white")
        ax.set_xticks(range(len(mass_modes)), mass_modes, rotation=25)
        ax.set_yticks(range(len(density_modes)), density_modes)
        ax.set(xlabel="Mass realization", ylabel="Density realization", title=label)
    fig.colorbar(im, ax=list(axes[0]), label="Mean success rate across training seeds", shrink=.8)
    fig.savefig(output_dir / "success_rate_summary.pdf")
    plt.close(fig)

    samples = collect_boxplot_samples(cache["runs"], rows)
    width = .8 / len(labels)
    for metric, filename, title, ylabel in (
        ("time", "time_summary.pdf", "Time to goal — successful trajectories", "Hitting time (s)"),
        ("energy", "energy_summary.pdf", "Energy summary — all outcomes", "Normalized control effort (not physical energy)"),
    ):
        fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(labels)), 5.5), constrained_layout=True)
        for j, label in enumerate(labels):
            positions = np.arange(len(SUMMARY_GROUPS)) + (j - (len(labels) - 1) / 2) * width
            for position, group in zip(positions, SUMMARY_GROUPS):
                values = samples[label][group][metric]
                if not len(values):
                    ax.text(position, .03, "n/a", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", rotation=90, fontsize=8, color=colors[label])
                    continue
                ax.boxplot(
                    [values], positions=[position], widths=width * .8, patch_artist=True,
                    manage_ticks=False, whis=1.5,
                    boxprops=dict(facecolor=colors[label], edgecolor=colors[label], alpha=.7),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(color=colors[label]), capprops=dict(color=colors[label]),
                    flierprops=dict(marker=".", markersize=2, markerfacecolor=colors[label],
                                    markeredgecolor=colors[label], alpha=.3),
                )
        ax.set_xticks(range(len(SUMMARY_GROUPS)), SUMMARY_GROUPS)
        ax.set_xlim(-.6, len(SUMMARY_GROUPS) - .4)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
        ax.legend(handles=[Patch(facecolor=colors[label], alpha=.7, label=label) for label in labels],
                  loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=min(3, len(labels)), fontsize="small")
        fig.savefig(output_dir / filename)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=OUTPUT_DIR / "mc_cache.pth")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache.get("format") != "xv15_mc_v1":
        parser.error("Unsupported XV-15 Monte Carlo cache format")
    output_dir = args.output_dir or args.cache.parent
    write_reports(cache, output_dir)
    if not args.no_plots:
        plot_results(cache, output_dir)


if __name__ == "__main__":
    main()
