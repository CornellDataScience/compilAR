#!/usr/bin/env python3
"""
gpu_matmul_bench.py — Per-GPU matrix multiplication benchmark for SLURM clusters.

This script is launched once per node by `srun`.  It:
  1. Detects every CUDA GPU visible on this node (via CUDA_VISIBLE_DEVICES).
  2. Runs a large matrix multiplication (GEMM) on each GPU independently.
  3. Includes a warm-up phase so JIT / CUDA context setup is excluded.
  4. Reports per-GPU timing AND a per-node summary.
  5. Adds inter-GPU (network-ish) timing: an all-reduce across all local GPUs.

Output is printed to stdout which SLURM captures in gpu_bench_<jobid>.out.
"""

import os
import socket
import time

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MATRIX_SIZE = 8192          # N×N matrix (large enough to saturate GPU cores)
NUM_WARMUP  = 5             # warm-up iterations (not timed)
NUM_ITERS   = 20            # timed iterations
DTYPE       = torch.float32 # use float32 for broad GPU compatibility


def benchmark_single_gpu(device_index: int) -> dict:
    """Run GEMM benchmark on one GPU and return timing info."""
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    # Query GPU identity
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name
    gpu_uuid = None
    try:
        # nvidia-smi UUID via torch isn't always available; fall back gracefully
        gpu_uuid = torch.cuda.get_device_properties(device).name
    except Exception:
        pass

    # Allocate matrices
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=DTYPE, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=DTYPE, device=device)

    # Warm-up
    for _ in range(NUM_WARMUP):
        _ = torch.mm(A, B)
    torch.cuda.synchronize(device)

    # Timed iterations
    times = []
    for _ in range(NUM_ITERS):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = torch.mm(A, B)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Compute TFLOPS:  2 * N^3 FLOPs for matrix multiply
    flops = 2.0 * (MATRIX_SIZE ** 3)
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    tflops_avg = (flops / avg_time) / 1e12

    return {
        "device_index": device_index,
        "gpu_name": gpu_name,
        "avg_s": avg_time,
        "min_s": min_time,
        "max_s": max_time,
        "std_s": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5,
        "tflops": tflops_avg,
        "all_times": times,
    }


def benchmark_allreduce(num_gpus: int, size: int = MATRIX_SIZE) -> float | None:
    """
    Simulate cross-GPU communication by doing a manual reduce across GPUs.
    Returns average round-trip time in seconds, or None if only 1 GPU.
    """
    if num_gpus < 2:
        return None

    # Create a tensor on each GPU
    tensors = []
    for i in range(num_gpus):
        dev = torch.device(f"cuda:{i}")
        tensors.append(torch.randn(size, size, dtype=DTYPE, device=dev))

    # Warm-up
    for _ in range(3):
        for i in range(1, num_gpus):
            tensors[0].add_(tensors[i].to(tensors[0].device))
        torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(10):
        # Re-init to avoid accumulation issues
        for i in range(num_gpus):
            tensors[i].fill_(1.0)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        # Reduce all tensors onto GPU 0
        for i in range(1, num_gpus):
            tensors[0].add_(tensors[i].to(tensors[0].device))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times)


def main():
    hostname = socket.gethostname()
    num_gpus = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")

    print(f"\n{'='*70}")
    print(f" Node: {hostname}")
    print(f" CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f" GPUs detected: {num_gpus}")
    print(f" Matrix size: {MATRIX_SIZE}×{MATRIX_SIZE}  |  dtype: {DTYPE}")
    print(f" Warm-up iters: {NUM_WARMUP}  |  Timed iters: {NUM_ITERS}")
    print(f" PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}")
    print(f"{'='*70}")

    if num_gpus == 0:
        print(" ERROR: No GPUs visible on this node!")
        return

    # --- Per-GPU benchmark ---------------------------------------------------
    results = []
    for i in range(num_gpus):
        print(f"\n  Benchmarking GPU {i} ...", flush=True)
        r = benchmark_single_gpu(i)
        results.append(r)
        print(f"    {r['gpu_name']}")
        print(f"    Avg: {r['avg_s']*1000:8.2f} ms  |  "
              f"Min: {r['min_s']*1000:8.2f} ms  |  "
              f"Max: {r['max_s']*1000:8.2f} ms  |  "
              f"Std: {r['std_s']*1000:8.2f} ms  |  "
              f"TFLOPS: {r['tflops']:6.2f}")

    # --- Cross-GPU communication benchmark -----------------------------------
    print(f"\n{'-'*70}")
    if num_gpus >= 2:
        print(f"  Cross-GPU reduce benchmark ({num_gpus} GPUs, "
              f"{MATRIX_SIZE}×{MATRIX_SIZE} tensor) ...")
        allreduce_time = benchmark_allreduce(num_gpus)
        if allreduce_time is not None:
            bw = (MATRIX_SIZE * MATRIX_SIZE * 4 * (num_gpus - 1)) / allreduce_time / 1e9
            print(f"    Avg all-reduce time : {allreduce_time*1000:8.2f} ms")
            print(f"    Effective bandwidth : {bw:8.2f} GB/s")
    else:
        print("  (Skipping cross-GPU benchmark — only 1 GPU on this node)")

    # --- Summary table -------------------------------------------------------
    print(f"\n{'='*70}")
    print(f" SUMMARY — {hostname}")
    print(f"{'='*70}")
    print(f" {'GPU':>4}  {'Name':<40}  {'Avg(ms)':>9}  {'Min(ms)':>9}  {'TFLOPS':>8}")
    print(f" {'-'*4}  {'-'*40}  {'-'*9}  {'-'*9}  {'-'*8}")

    sorted_results = sorted(results, key=lambda r: r["avg_s"])
    for r in sorted_results:
        print(f" {r['device_index']:>4}  {r['gpu_name']:<40}  "
              f"{r['avg_s']*1000:9.2f}  {r['min_s']*1000:9.2f}  "
              f"{r['tflops']:8.2f}")

    fastest = sorted_results[0]
    slowest = sorted_results[-1]
    print(f"\n  FASTEST: GPU {fastest['device_index']} "
          f"({fastest['gpu_name']}) — {fastest['avg_s']*1000:.2f} ms avg")
    print(f"  SLOWEST: GPU {slowest['device_index']} "
          f"({slowest['gpu_name']}) — {slowest['avg_s']*1000:.2f} ms avg")

    if num_gpus > 1:
        spread = ((slowest["avg_s"] - fastest["avg_s"]) / fastest["avg_s"]) * 100
        print(f"  SPREAD : {spread:.1f}% slower (slowest vs fastest)")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
