"""2D inverted-pendulum synthesis entrypoint."""

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
    create_v_net_for_training,
    create_gv_net,
    make_controller,
    make_invpend_dynamics,
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
    print("2D Inverted Pendulum Synthesis")
    print("=" * 20)

    params = configure_default_params()
    set_global_reproducibility(int(params.training.random_seed))
    print(f"[reproducibility] seed={params.training.random_seed}, deterministic_algorithms=True")

    regions, init_range, goal_range, full_range = build_regions()
    device = params.training.device
    control_net = make_controller(device)
    dynamics = make_invpend_dynamics(control_net)
    V_net = create_v_net_for_training(params, goal_range, use_v_offset=True)
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
            control_net=control_net,
            device=device,
        )

    run_load_path(output_dir=OUTPUT_DIR, bundle_path=bundle_path, regions=regions)
    return None


if __name__ == "__main__":
    is_benchmark = "--benchmark=1" in sys.argv or ("--benchmark" in sys.argv and "1" in sys.argv)

    if is_benchmark:
        n_runs = 2
        times, cells = [], []

        print("=" * 20)
        print(f"Benchmark mode: {n_runs} runs")
        print("=" * 20)

        for i in range(n_runs):
            print(f"\n*** Run {i + 1}/{n_runs} ***")
            torch.manual_seed(i)
            training_time = main(benchmark_mode=True)
            if training_time is not None:
                times.append(training_time)

            bundle_path = OUTPUT_DIR / "eval_bundle.pth"
            if bundle_path.exists():
                bundle = torch.load(bundle_path, map_location="cpu")
                count = {k: len(bundle["region_cells"][k]) for k in ["init", "goal", "unsafe", "outside", "generator"]}
                count["total_v"] = count["init"] + count["goal"] + count["unsafe"] + count["outside"]
                count["total"] = count["total_v"] + count["generator"]
                cells.append(count)

        print("\n" + "=" * 20)
        print("Benchmark results")
        print("=" * 20)
        print(f"Training time (pretrain+train): {stats.mean(times):.2f}s ± {stats.stdev(times):.2f}s")
        print(f"  Individual times: {[f'{t:.2f}s' for t in times]}")

        if cells:
            print("\nCell counts:")
            for cat in ["init", "goal", "unsafe", "outside", "generator", "total_v", "total"]:
                values = [c[cat] for c in cells]
                print(f"  {cat:12s}: {stats.mean(values):.0f} ± {stats.stdev(values):.0f}")
    else:
        main()
