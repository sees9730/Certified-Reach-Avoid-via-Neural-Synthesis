"""Rebuild XV-15 Monte Carlo tables and figures from mc_cache.pth only."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from examples.xv15_uncertain.model import DEG
from examples.xv15_uncertain.run_mc import OUTPUT_DIR, aggregate_results, write_reports


def plot_results(cache, output_dir, max_paths=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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

    scenarios = list(dict.fromkeys((r["density_mode"], r["mass_mode"]) for r in rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(scenarios) * 1.2), 7), sharex=True,
                             constrained_layout=True)
    width = .8 / len(labels)
    for j, label in enumerate(labels):
        selected = {(r["density_mode"], r["mass_mode"]): r for r in rows if r["controller"] == label}
        locations = np.arange(len(scenarios)) + (j - (len(labels) - 1) / 2) * width
        for ax, metric in zip(axes, ("hit_time_s_mean", "normalized_effort_mean")):
            values = [selected.get(key, {}).get(metric) for key in scenarios]
            ax.bar(locations, [np.nan if v is None else v for v in values], width=width,
                   label=label, color=colors[label])
    axes[0].set_ylabel("Mean successful hitting time (s)")
    axes[1].set_ylabel("Mean normalized control effort\n(all outcomes; not physical energy)")
    axes[0].legend()
    axes[1].set_xticks(range(len(scenarios)), [f"{d}\n× {m}" for d, m in scenarios], rotation=25)
    axes[1].set_xlabel("Density × mass realization; worst tested attack per seed")
    fig.savefig(output_dir / "time_and_effort.pdf")
    plt.close(fig)

    for density_mode, mass_mode in scenarios:
        selected_paths = {}
        for label in labels:
            candidates = [(run, path) for run in cache["runs"] if run["label"] == label
                          and run["density_mode"] == density_mode and run["mass_mode"] == mass_mode
                          for path in run["result"]["paths"]]
            selected_paths[label] = candidates[:max_paths]
        if not any(selected_paths.values()):
            continue
        destination = output_dir / f"density_{density_mode}__mass_{mass_mode}"
        destination.mkdir(parents=True, exist_ok=True)
        for filename, names, data_key, times_key, scales in (
            ("trajectories.pdf", ("Airspeed (m/s)", "Flight-path angle (deg)", "Rotor tilt (deg)"),
             "states", "times", (1, 1 / DEG, 1 / DEG)),
            ("controls.pdf", ("Thrust (kN)", "Angle of attack (deg)", "Tilt rate (deg/s)"),
             "controls", "control_times", (.001, 1 / DEG, 1 / DEG)),
            ("parameters.pdf", ("Density (kg/m³)", "Mass (kg)"),
             "parameters", "control_times", (1, 1)),
        ):
            fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.5 * len(names)), sharex=True,
                                     constrained_layout=True)
            for label, paths in selected_paths.items():
                for run, path in paths:
                    data = path[data_key].numpy()
                    times = path[times_key].numpy()
                    style = {"success": "-", "fail": "--", "timeout": ":"}[path["outcome"]]
                    for i, ax in enumerate(axes):
                        ax.plot(times, data[:, i] * scales[i], color=colors[label], linestyle=style, alpha=.65)
            for i, (ax, name) in enumerate(zip(axes, names)):
                ax.set_ylabel(name)
                ax.grid(alpha=.2)
                if data_key == "states":
                    goal = cache["metadata"]["config"]["regions_mps_deg_deg"]["goal_range"][i]
                    ax.axhspan(*goal, color="green", alpha=.08)
            handles = [Line2D([0], [0], color=colors[label], label=label) for label in labels]
            axes[0].legend(handles=handles, fontsize="small")
            axes[0].set_title(f"Density: {density_mode}; mass: {mass_mode}\n"
                              "Sample paths: success — solid; failure — dashed; timeout — dotted")
            axes[-1].set_xlabel("Time (s)")
            fig.savefig(destination / filename)
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=OUTPUT_DIR / "mc_cache.pth")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-paths", type=int, default=5, help="Maximum stored traces plotted per controller and scenario")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.max_paths < 0:
        parser.error("--max-paths must be nonnegative")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache.get("format") != "xv15_mc_v1":
        parser.error("Unsupported XV-15 Monte Carlo cache format")
    output_dir = args.output_dir or args.cache.parent
    write_reports(cache, output_dir)
    if not args.no_plots:
        plot_results(cache, output_dir, max_paths=args.max_paths)


if __name__ == "__main__":
    main()
