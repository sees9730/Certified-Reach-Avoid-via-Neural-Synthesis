"""Temporal 2D Geometric Brownian Motion verification."""

import argparse
import statistics as stats
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from workflow import (
    build_regions,
    configure_default_params,
    create_gv_net,
    create_v_net_for_training,
    make_gbm_dynamics,
    run_load_path,
    run_training_path,
    set_global_reproducibility,
)


def main(benchmark_mode: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, choices=[0, 1],
                        help="1: train + save bundle, 0: load bundle + eval/plot")
    parser.add_argument("--benchmark", type=int, default=0, choices=[0, 1],
                        help="1: run benchmark (2 runs), 0: normal run")
    args = parser.parse_args()
    if benchmark_mode:
        args.benchmark = 1

    print("=" * 20)
    print("Geometric Brownian Motion Verification")
    print("=" * 20)

    time_horizon = 10.0
    params = configure_default_params(time_horizon)
    set_global_reproducibility(int(params.training.random_seed))
    print(f"[reproducibility] seed={params.training.random_seed}, deterministic_algorithms=True")

    regions, init_range, goal_range, full_range = build_regions(time_horizon, params)

    device = params.training.device
    dynamics = make_gbm_dynamics()
    V_net = create_v_net_for_training(params)
    GV_net = create_gv_net(V_net, dynamics, params)

    bundle_path = OUTPUT_DIR / "eval_bundle.pth"

    if args.train == 1:
        return run_training_path(
            output_dir=OUTPUT_DIR,
            params=params,
            regions=regions,
            init_range=init_range,
            goal_range=goal_range,
            full_range=full_range,
            V_net=V_net,
            GV_net=GV_net,
            device=device,
        )

    run_load_path(output_dir=OUTPUT_DIR, bundle_path=bundle_path)
    return None


if __name__ == '__main__':
    is_benchmark = '--benchmark=1' in sys.argv or ('--benchmark' in sys.argv and '1' in sys.argv)

    if is_benchmark:
        n_runs = 2
        times = []
        cells = []
        print("=" * 20)
        print(f"Benchmark mode: {n_runs} runs")
        print("=" * 20)

        for i in range(n_runs):
            print(f"\n*** Run {i+1}/{n_runs} ***")
            torch.manual_seed(i)
            training_time = main(benchmark_mode=True)
            if training_time is not None:
                times.append(training_time)

            bundle_path = OUTPUT_DIR / 'eval_bundle.pth'
            if bundle_path.exists():
                bundle = torch.load(bundle_path, map_location='cpu')
                cell_counts = {
                    'init': len(bundle['region_cells']['init']),
                    'goal': len(bundle['region_cells']['goal']),
                    'unsafe': len(bundle['region_cells']['unsafe']),
                    'outside': len(bundle['region_cells']['outside']),
                    'generator': len(bundle['region_cells']['generator']),
                }
                cell_counts['total_v'] = cell_counts['init'] + cell_counts['goal'] + cell_counts['unsafe'] + cell_counts['outside']
                cell_counts['total'] = cell_counts['total_v'] + cell_counts['generator']
                cells.append(cell_counts)

        print("\n" + "=" * 20)
        print("Benchmark results")
        print("=" * 20)
        avg_time = stats.mean(times)
        std_time = stats.stdev(times)
        print(f"Training time (pretrain+train): {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"  Individual times: {[f'{t:.2f}s' for t in times]}")

        if cells:
            categories = ['init', 'goal', 'unsafe', 'outside', 'generator', 'total_v', 'total']
            print("\nCell counts:")
            for cat in categories:
                values = [c[cat] for c in cells]
                avg = stats.mean(values)
                std = stats.stdev(values)
                print(f"  {cat:12s}: {avg:.0f} ± {std:.0f}")
    else:
        main()
