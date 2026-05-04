#!/usr/bin/env python3
"""
ring_allreduce_baseline.py — Ring AllReduce timing baseline, MPI worker.

Must be launched via mpirun -n 4 with MASTER_ADDR, MASTER_PORT, and
NCCL_SOCKET_IFNAME in the environment (use mpirun -x flags).

Each of the 4 MPI ranks handles one GPU:
    local_rank = OMPI_COMM_WORLD_LOCAL_RANK  (0 or 1 per node)
    rank       = OMPI_COMM_WORLD_RANK        (0–3 globally)

Rank (world_size-1) = rank 3 sleeps for SLEEP_MS before calling all_reduce,
reproducing the same straggler scenario as the StragglAR binary.

Output CSV (rank 0 only):
    ring,<buffer_bytes>,<iteration>,<sleep_ms>,<runtime_ms>,<BW(GB/s)>

Usage (via run_tests.sh / mpirun):
    mpirun -n 4 --hostfile hostfile \\
        -x MASTER_ADDR=compute1 -x MASTER_PORT=12346 \\
        -x NCCL_SOCKET_IFNAME=enp5s0 \\
        python3 ring_allreduce_baseline.py <buffer_bytes> <num_iters> <sleep_ms>
"""

import os
import sys
import time

import torch
import torch.distributed as dist

NUM_WARMUP = 3

rank       = int(os.environ.get("OMPI_COMM_WORLD_RANK",       os.environ.get("RANK",       "0")))
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE",       os.environ.get("WORLD_SIZE", "4")))

os.environ["RANK"]       = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)

STRAGGLER_RANK = world_size - 1   # rank 3 = compute4 GPU 1


def main():
    if len(sys.argv) != 4:
        if rank == 0:
            print(
                f"Usage: mpirun -n {world_size} ... {sys.argv[0]}"
                " <buffer_bytes> <num_iters> <sleep_ms>",
                file=sys.stderr,
            )
        sys.exit(1)

    buffer_bytes = int(sys.argv[1])
    num_iters    = int(sys.argv[2])
    sleep_ms     = float(sys.argv[3])

    if buffer_bytes % 4 != 0:
        if rank == 0:
            print("Error: buffer_bytes must be divisible by 4", file=sys.stderr)
        sys.exit(1)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    n_floats = buffer_bytes // 4
    buf = torch.ones(n_floats, dtype=torch.float32, device=device)

    # Warmup: initialize NCCL communication paths before timing
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

        # All ranks enter the timed window together; the straggler's sleep is
        # fully included in every rank's measurement.
        dist.barrier()
        t0 = time.perf_counter()

        if rank == STRAGGLER_RANK and sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

        t1  = time.perf_counter()
        ms  = (t1 - t0) * 1000.0
        bw  = (2.0 * (world_size - 1) / world_size * buffer_bytes) / (t1 - t0) / 1e9

        if rank == 0:
            print(f"ring,{buffer_bytes},{it},{sleep_ms:.3f},{ms:.3f},{bw:.3f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
