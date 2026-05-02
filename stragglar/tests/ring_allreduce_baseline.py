#!/usr/bin/env python3
"""
ring_allreduce_baseline.py — Ring AllReduce timing baseline using PyTorch NCCL.

Launches 4 processes (one per GPU) via torch.multiprocessing.spawn.
GPU (world_size - 1) sleeps for SLEEP_MS before joining the all_reduce,
simulating the same straggler scenario as the StragglAR binary.

Output CSV (rank 0 only):
    ring,<buffer_bytes>,<iteration>,<sleep_ms>,<runtime_ms>,<BW(GB/s)>

Usage:
    python3 ring_allreduce_baseline.py <buffer_bytes> <num_iters> <sleep_ms>

Arguments:
    buffer_bytes  total buffer size in bytes (must be divisible by 4)
    num_iters     number of timed iterations (one warmup pass is added automatically)
    sleep_ms      straggler delay in ms for rank (world_size-1); 0 = no straggler

Example:
    python3 ring_allreduce_baseline.py 50331648 20 50
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

NUM_GPUS  = 4
NUM_WARMUP = 3
STRAGGLER_RANK = NUM_GPUS - 1   # rank 3 = GPU 3, matching the StragglAR binary

MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")
MASTER_PORT = os.environ.get("MASTER_PORT", "12345")


def _worker(rank: int, world_size: int, buffer_bytes: int, num_iters: int, sleep_ms: float):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    dist.init_process_group(
        backend   = "nccl",
        rank      = rank,
        world_size = world_size,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    n_floats = buffer_bytes // 4
    buf = torch.ones(n_floats, dtype=torch.float32, device=device)

    # Warmup — ensure NCCL and CUDA context are fully initialized before timing
    for _ in range(NUM_WARMUP):
        dist.barrier()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    buf.fill_(1.0)

    if rank == 0:
        print("ring,buffer_size_bytes,iteration,delay,runtime_ms,BW(GB/s)", flush=True)

    for it in range(1, num_iters + 1):
        buf.fill_(1.0)
        torch.cuda.synchronize(device)

        # All ranks sync so the straggler's sleep is included in timing for
        # every rank — matching how the StragglAR binary uses CUDA events.
        dist.barrier()
        t0 = time.perf_counter()

        # Straggler delay: rank (world_size-1) sleeps before calling all_reduce,
        # blocking the ring until it participates.
        if rank == STRAGGLER_RANK and sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0

        # Effective bandwidth: (2 * (N-1) / N) * bytes / time  [ring formula]
        bw = (2.0 * (world_size - 1) / world_size * buffer_bytes) / (t1 - t0) / 1e9

        if rank == 0:
            print(
                f"ring,{buffer_bytes},{it},{sleep_ms:.3f},{ms:.3f},{bw:.3f}",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <buffer_bytes> <num_iters> <sleep_ms>", file=sys.stderr)
        sys.exit(1)

    buffer_bytes = int(sys.argv[1])
    num_iters    = int(sys.argv[2])
    sleep_ms     = float(sys.argv[3])

    if buffer_bytes % 4 != 0:
        print(f"Error: buffer_bytes must be divisible by 4 (got {buffer_bytes})", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("Error: No CUDA devices found.", file=sys.stderr)
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    if n_gpus < NUM_GPUS:
        print(f"Error: need {NUM_GPUS} GPUs, found {n_gpus}", file=sys.stderr)
        sys.exit(1)

    mp.spawn(
        _worker,
        args    = (NUM_GPUS, buffer_bytes, num_iters, sleep_ms),
        nprocs  = NUM_GPUS,
        join    = True,
    )


if __name__ == "__main__":
    main()
