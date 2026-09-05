"""
Postprocess MC Validation Comparison
============================================

Re-renders every plot and the success-rate summary table that `run_mc.py`
produces, using the `mc_cache.pth` it saved -- without re-running the
(slow) Monte Carlo rollouts.

Run `python run_mc.py` first to generate `run_mc_results/mc_cache.pth`.

Output (PDF):
  run_mc_results/<mode>/fig1_phase_trajectories.pdf
  run_mc_results/<mode>/fig2_energy_vs_thit.pdf
  run_mc_results/<mode>/fig5_u_raw_trajectories.pdf
  run_mc_results/fig6_success_rate_summary.pdf
  run_mc_results/fig7_failure_trajectories.pdf
  run_mc_results/fig8_energy_distribution.pdf

Figures are saved beside the cache unless --output-dir is specified.

Usage (from this directory):
    python postprocess_mc.py [--cache PATH] [--figures 6 7 8] [--output-dir PATH] [--verbose]
"""

import argparse
from pathlib import Path

from run_mc import (
    MC_CACHE_PATH,
    EVAL_MODES,
    ADVERSARIAL_MODES,
    build_controller_styles,
    load_mc_cache,
    plot_phase_trajectories,
    plot_energy_vs_thit,
    plot_energy_distribution,
    plot_u_raw_trajectories,
    plot_failure_trajectories,
    plot_success_rate_summary,
    pool_viz_across_modes,
    print_summary,
    print_success_rate_table,
    vprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=MC_CACHE_PATH,
        help=f"Path to the mc_cache.pth saved by run_mc.py (default: {MC_CACHE_PATH})",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-controller/per-mode detail (default: only the final summary table).",
    )
    parser.add_argument(
        "--max-fail-trajectories", type=int, default=None, metavar="N",
        help="Cap the number of failure trajectories drawn per controller in "
             "fig7_failure_trajectories.pdf (default: draw every pooled failure).",
    )
    parser.add_argument(
        "--figures", type=int, nargs="+", choices=(1, 2, 5, 6, 7, 8), default=None,
        help="Figures to regenerate (default: recorded selection, or 1 2 5 6 7 for older caches).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for regenerated figures (default: the cache's directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = args.verbose
    cache = load_mc_cache(args.cache)

    controller_labels = cache["controller_labels"]
    results          = cache["results"]
    viz_paths        = cache["viz_paths"]
    viz_success      = cache.get("viz_success")
    success_table    = cache["success_table"]
    per_seed_success = cache["per_seed_success"]
    dt               = cache["dt"]
    meta             = cache.get("meta", {})
    figures = args.figures if args.figures is not None else meta.get("figures", [1, 2, 5, 6, 7])
    output_dir = args.output_dir if args.output_dir is not None else args.cache.parent

    vprint(f"Loaded MC cache: {args.cache}", verbose=verbose)
    if meta:
        vprint(f"  recorded at  : {meta.get('timestamp', '?')}", verbose=verbose)
        vprint(f"  n_mc         : {meta.get('n_mc', '?')}", verbose=verbose)
        vprint(f"  t_max        : {meta.get('t_max', '?')}", verbose=verbose)
        vprint(f"  mc_seed      : {meta.get('mc_seed', '?')}", verbose=verbose)
        vprint(f"  eval modes   : {', '.join(meta.get('eval_modes', EVAL_MODES))}", verbose=verbose)
        vprint(f"  seed counts  : {meta.get('seed_counts', '?')}", verbose=verbose)
    vprint(f"  controllers  : {', '.join(controller_labels)}", verbose=verbose)

    styles = build_controller_styles(controller_labels)

    for mode in EVAL_MODES:
        if mode not in results:
            vprint(f"[skip] mode '{mode}' not present in cache", verbose=verbose)
            continue

        if verbose:
            for label, _res, stats in results[mode]:
                print_summary(f"{label} [{mode}]", stats)

        mode_dir = output_dir / mode
        if 1 in figures:
            plot_phase_trajectories(results[mode], viz_paths[mode], styles, save_dir=mode_dir, verbose=verbose)
        if 2 in figures:
            plot_energy_vs_thit(results[mode], styles, save_dir=mode_dir, verbose=verbose)
        if 5 in figures:
            plot_u_raw_trajectories(results[mode], dt, styles, save_dir=mode_dir, verbose=verbose)

    if 7 in figures:
        if viz_success is not None:
            pooled_viz_paths, pooled_viz_success = pool_viz_across_modes(viz_paths, viz_success, list(EVAL_MODES))
            adv_viz_paths, adv_viz_success = pool_viz_across_modes(viz_paths, viz_success, list(ADVERSARIAL_MODES))
            plot_failure_trajectories(controller_labels, pooled_viz_paths, pooled_viz_success,
                                     adv_viz_paths, adv_viz_success, styles,
                                     save_dir=output_dir, verbose=verbose,
                                     max_per_controller=args.max_fail_trajectories)
        else:
            vprint("[skip] fig7: cache predates viz_success; rerun run_mc.py to regenerate.", verbose=verbose)
    if 6 in figures:
        plot_success_rate_summary(controller_labels, success_table, per_seed_success, styles,
                                  save_dir=output_dir, verbose=verbose)
    if 8 in figures:
        plot_energy_distribution(results, styles, save_dir=output_dir, verbose=verbose)

    print_success_rate_table(controller_labels, success_table)


if __name__ == "__main__":
    main()
