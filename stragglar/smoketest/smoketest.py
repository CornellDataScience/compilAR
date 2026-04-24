#!/usr/bin/env python3
"""
smoketest.py — Identify the straggling GPU in a batch compute run.

Usage:
    python smoketest.py [--gpus 0 1 2 3] [--iters 15] [--seed 42]

Workflow:
    1. run_batch_compute()  — randomized GEMM on all GPUs in parallel
    2. identify_straggler() — find which GPU finished last and by how much
    3. Log exact finish times, straggler verdict, and gap to second-to-last
"""

import argparse
import datetime
import socket

import torch

from stragglar.smoketest.batch_compute import run_batch_compute
from stragglar.smoketest.detect_straggler import identify_straggler


def print_results(result, hostname: str) -> None:
    width = 72
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*width}")
    print(f" SMOKETEST RESULTS — {hostname}  [{now}]")
    print(f"{'='*width}")
    print(
        f" {'GPU':>4}  {'Name':<30}  {'Size':>10}  "
        f"{'Total(s)':>9}  {'Finish +ms':>11}  {'Order':>6}"
    )
    print(f" {'-'*4}  {'-'*30}  {'-'*10}  {'-'*9}  {'-'*11}  {'-'*6}")

    for rank, gid in enumerate(result.finish_order, start=1):
        gpu_name = "unknown"
        try:
            gpu_name = torch.cuda.get_device_properties(gid).name[:30]
        except Exception:
            pass

        size_str = f"{result.matrix_sizes[gid]}²"
        marker   = "  <-- STRAGGLER" if gid == result.straggler_gpu else ""

        print(
            f" {gid:>4}  {gpu_name:<30}  {size_str:>10}  "
            f"{result.total_times[gid]:9.3f}  "
            f"{result.end_times_rel[gid]*1000:+11.1f}  "
            f"#{rank:>4}"
            f"{marker}"
        )

    print(f"\n Finish order (earliest → latest): {result.finish_order}")
    print(f"\n Straggler GPU      : {result.straggler_gpu}")
    print(
        f" Gap to 2nd-to-last : "
        f"{result.delta_to_second*1000:.3f} ms  "
        f"({result.delta_to_second:.6f} s)"
    )

    print(f"\n VERDICT: {result.verdict}")
    print(f"{'='*width}\n")


def main():
    parser = argparse.ArgumentParser(description="GPU straggler smoke test")
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=None,
        help="GPU indices to test (default: all visible GPUs)",
    )
    parser.add_argument(
        "--iters", type=int, default=15,
        help="Number of timed GEMM iterations per GPU (default: 15)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible matrix size assignment",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA devices found.")
        return 1

    gpu_ids  = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
    hostname = socket.gethostname()

    print(f"\n[smoketest] Host: {hostname}")
    print(f"[smoketest] PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}")
    for gid in gpu_ids:
        props = torch.cuda.get_device_properties(gid)
        print(f"[smoketest]   GPU {gid}: {props.name}")

    # Step 1: run batch compute with randomized workloads
    timing = run_batch_compute(
        gpu_ids   = gpu_ids,
        num_iters = args.iters,
        seed      = args.seed,
    )

    # Step 2: identify straggler from finish timestamps
    result = identify_straggler(timing)

    # Step 3: log results
    print_results(result, hostname)

    # Machine-parseable line for launch.sh to grep (unambiguous vs. exit code).
    print(f"STRAGGLER_GPU={result.straggler_gpu}")

    return result.straggler_gpu


if __name__ == "__main__":
    raise SystemExit(main())
