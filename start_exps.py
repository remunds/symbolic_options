#!/usr/bin/env python3
"""
Run multiple experiment configs across a list of GPUs.

Usage:
    python start_exps.py --gpus=0,1,2,3

Each config is dispatched to the next free GPU. The script keeps all GPUs busy
until every config has finished.
"""

import argparse
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# ── Define all runs ──────────────────────────────────────────────────────────

@dataclass
class RunSpec:
    alg: str          # config name (e.g. "pqn_jaxtari_k1_noisy0")
    script: str = "main.py"  # "main.py" or "src/symbolic_options/ppo_jaxtari.py"

RUNS = [
    # ── Seaquest noisy0 (trained WITH noise) ──
    # RunSpec("pqn_jaxtari_sea1_noisy0_scaling"),
    # RunSpec("pqn_jaxtari_sea2_hier_baseline_noisy0_scaling"),
    # RunSpec("pqn_jaxtari_sea2_hier_noisy0_scaling"),
    # RunSpec("pqn_jaxtari_sea3_hier_llm_noisy0_scaling"),
    # RunSpec("pqn_jaxtari_sea4_hier_comb_noisy0_scaling"),
    # RunSpec("ppo_jaxtari_seaquest_noisy0_scaling", script="src/symbolic_options/ppo_jaxtari.py"),

    # # ── Seaquest noisy1 (noisy eval only) ──
    # RunSpec("pqn_jaxtari_sea1_noisy1_scaling"),
    # RunSpec("pqn_jaxtari_sea2_hier_baseline_noisy1_scaling"),
    # RunSpec("pqn_jaxtari_sea2_hier_noisy1_scaling"),
    # RunSpec("pqn_jaxtari_sea3_hier_llm_noisy1_scaling"),
    # RunSpec("pqn_jaxtari_sea4_hier_comb_noisy1_scaling"),
    # RunSpec("ppo_jaxtari_seaquest_noisy1_scaling", script="src/symbolic_options/ppo_jaxtari.py"),

    # # ── Kangaroo noisy0 (trained WITH noise) ──
    # RunSpec("pqn_jaxtari_k1_noisy0_scaling"),
    # RunSpec("pqn_jaxtari_k2_hier_baseline_noisy0_scaling"),
    # RunSpec("pqn_jaxtari_k2_hier_noisy0_scaling"),
    # RunSpec("pqn_jaxtari_k3_hier_llm_noisy0_scaling"),
    # RunSpec("pqn_jaxtari_k4_hier_comb_noisy0_scaling"),
    # RunSpec("ppo_jaxtari_kangaroo_noisy0_scaling", script="src/symbolic_options/ppo_jaxtari.py"),

    # ── Kangaroo noisy1 (noisy eval only) ──
    # RunSpec("pqn_jaxtari_k1_noisy1_scaling"),
    # RunSpec("pqn_jaxtari_k2_hier_baseline_noisy1_scaling"),
    # RunSpec("pqn_jaxtari_k2_hier_noisy1_scaling"),
    # RunSpec("pqn_jaxtari_k3_hier_llm_noisy1_scaling"),
    RunSpec("pqn_jaxtari_k4_hier_comb_noisy1_scaling"),
    # RunSpec("ppo_jaxtari_kangaroo_noisy1_scaling", script="src/symbolic_options/ppo_jaxtari.py"),
]

WORKERS_PER_GPU = 1


def worker(gpu_id: str, task_queue: queue.Queue, extra_args: list):
    """Continuously fetch tasks from the queue and run them on the assigned GPU."""
    while not task_queue.empty():
        try:
            spec = task_queue.get_nowait()
        except queue.Empty:
            break

        print(f"[GPU {gpu_id}] Starting {spec.alg} (script: {spec.script})...")

        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id

        cmd = [
            "uv", "run", spec.script,
            f"+alg={spec.alg}",
        ] + extra_args

        try:
            subprocess.run(cmd, env=env_vars, check=True)
            print(f"[GPU {gpu_id}] Finished {spec.alg} (exit code 0)")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] FAILED {spec.alg} (exit code {e.returncode})")
        finally:
            task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Dispatch experiment configs across GPUs.")
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated list of GPU indices, e.g. --gpus=0,1,2,3",
    )
    args, extra_args = parser.parse_known_args()
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if not gpu_list:
        print("Error: No GPUs specified.")
        return

    # ── Populate task queue ──
    task_queue = queue.Queue()
    for spec in RUNS:
        task_queue.put(spec)

    total_runs = len(RUNS)
    total_workers = len(gpu_list) * WORKERS_PER_GPU
    print(f"Dispatching {total_runs} runs across {len(gpu_list)} GPUs: {gpu_list}")
    print(f"Extra args: {' '.join(extra_args) if extra_args else 'None'}")

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        for gpu_id in gpu_list:
            for _ in range(WORKERS_PER_GPU):
                executor.submit(worker, gpu_id, task_queue, extra_args)

    task_queue.join()
    print(f"\n{'='*60}")
    print(f"All {total_runs} runs completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
