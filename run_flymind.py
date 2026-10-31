"""
FlyMind Main Launcher CLI
Insect-Inspired Visual Flight Intelligence Research Platform.

Usage:
    python run_flymind.py --dashboard [--port 8090]
    python run_flymind.py --benchmark [--seeds 5] [--env forest|gates|looming]
    python run_flymind.py --demo
"""
import argparse
import sys
import time
import numpy as np

from flymind.evaluation.benchmarks.scenario_runner import BenchmarkRunner, ControllerType
from flymind.evaluation.reports.result_compiler import ResultCompiler
from flymind.visualization.web_dashboard import start_dashboard


def run_benchmark(num_seeds: int = 5, env_name: str = "forest") -> None:
    print(f"\n=======================================================")
    print(f" FLYMIND BENCHMARK HARNESS: {num_seeds} SEEDS IN '{env_name.upper()}'")
    print(f"=======================================================\n")

    runner = BenchmarkRunner(dt=0.02, max_episode_time_s=10.0)
    all_results = []

    controllers = [
        ControllerType.FLYMIND_FULL,
        ControllerType.LPLC2_ONLY,
        ControllerType.WTA_FLOW_ONLY,
        ControllerType.CANNY_BASELINE,
        ControllerType.DEPTHNET_BASELINE,
    ]

    for ctrl in controllers:
        print(f"--> Evaluating Controller: {ctrl.value} ...", flush=True)
        for s in range(num_seeds):
            res = runner.run_episode(controller_type=ctrl, environment_name=env_name, seed=42 + s)
            all_results.append(res)
            print(".", end="", flush=True)
        print(" Done.")

    compiler = ResultCompiler(all_results)
    table_md = compiler.compile_summary_table()

    print("\nBenchmark Summary Results:\n")
    print(table_md)
    print("\n")


def run_demo() -> None:
    print("\nRunning single FlyMindBio demonstration flight episode...")
    runner = BenchmarkRunner(dt=0.02, max_episode_time_s=12.0)
    res = runner.run_episode(controller_type=ControllerType.FLYMIND_FULL, environment_name="forest", seed=42)
    print(f"Episode Completed:")
    print(f"  Controller:     {res.controller_name}")
    print(f"  Scenario:       {res.scenario_name}")
    print(f"  Duration:       {res.duration_s:.2f} s")
    print(f"  Path Length:    {res.path_length_m:.2f} m")
    print(f"  Mean Speed:     {res.mean_speed_m_s:.2f} m/s")
    print(f"  Min Clearance:  {res.min_clearance_m:.2f} m")
    print(f"  Goal Reached:   {res.reached_goal}")
    print(f"  Collided:       {res.has_collided}")
    print(f"  Saccade Events: {res.saccade_count}")
    print(f"  Step Latency:   {res.mean_computation_time_ms:.2f} ms\n")


def main():
    parser = argparse.ArgumentParser(description="FlyMind: Insect-Inspired Flight Intelligence Research Platform")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive Web Research Dashboard")
    parser.add_argument("--port", type=int, default=8090, help="Web dashboard port (default: 8090)")
    parser.add_argument("--benchmark", action="store_true", help="Run automated statistical benchmark suite")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per scenario (default: 5)")
    parser.add_argument("--env", type=str, default="forest", choices=["forest", "gates", "looming"], help="Environment")
    parser.add_argument("--demo", action="store_true", help="Run quick single demo episode")

    args = parser.parse_args()

    if args.dashboard:
        print(f"\n=======================================================")
        print(f" Starting FlyMind Web Research Dashboard on port {args.port}...")
        print(f" Open browser at: http://localhost:{args.port}/")
        print(f"=======================================================\n")
        server = start_dashboard(port=args.port)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")
            server.shutdown()

    elif args.benchmark:
        run_benchmark(num_seeds=args.seeds, env_name=args.env)

    else:
        # Default action: run demo
        run_demo()


if __name__ == "__main__":
    main()
