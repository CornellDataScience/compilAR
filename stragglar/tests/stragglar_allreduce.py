#!/usr/bin/env python3
"""
stragglar_allreduce.py — StragglAR algorithm timing benchmark, MPI worker.

Implements the StragglAR overlap strategy in Python + PyTorch NCCL:

  1. All ranks sync and start the timed window.
  2. Rank (world_size-1) sleeps for sleep_ms  [the straggler].
  3. Ranks 0..world_size-2 run AllReduce on a healthy sub-group,
     overlapping with the straggler's sleep.
  4. A barrier waits for the straggler to wake up.
  5. A merge step broadcasts the straggler's contribution to all ranks.

Critical path: max(sleep, healthy_allreduce) + merge
vs Ring:        sleep + full_allreduce

Must be launched via mpirun -n 4 with MASTER_ADDR, MASTER_PORT set
(mpirun -x flags).  Same MPI worker pattern as ring_allreduce_baseline.py.

Output CSV (rank 0 only):
    stragglar,<buffer_bytes>,<iter>,<sleep_ms>,<runtime_ms>,<BW(GB/s)>

Usage (via run_tests.sh / mpirun):
    mpirun -n 4 --hostfile hostfile \\
        -x MASTER_ADDR=compute1 -x MASTER_PORT=12347 \\
        -x NCCL_SOCKET_IFNAME=enp5s0 \\
        python3 stragglar_allreduce.py <buffer_bytes> <num_iters> <sleep_ms>
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

    # Sub-group containing only the healthy ranks (all except the straggler)
    healthy_ranks = list(range(world_size - 1))   # [0, 1, 2]
    healthy_group = dist.new_group(ranks=healthy_ranks)

    n_floats = buffer_bytes // 4
    buf = torch.ones(n_floats, dtype=torch.float32, device=device)

    # ---------------------------------------------------------------------------
    # Warmup: initialise all NCCL communicators (full + sub-group) before timing
    # ---------------------------------------------------------------------------
    for _ in range(NUM_WARMUP):
        dist.barrier()
        if rank != STRAGGLER_RANK:
            tmp = buf.clone()
            dist.all_reduce(tmp, group=healthy_group)
        dist.barrier()
        strag_w = buf.clone() if rank == STRAGGLER_RANK else torch.zeros_like(buf)
        dist.all_reduce(strag_w)
        torch.cuda.synchronize(device)

    if rank == 0:
        print("stragglar,buffer_size_bytes,iteration,delay,runtime_ms,BW(GB/s)", flush=True)

    # ---------------------------------------------------------------------------
    # Benchmark loop
    # ---------------------------------------------------------------------------
    for it in range(1, num_iters + 1):
        buf.fill_(1.0)
        torch.cuda.synchronize(device)

        # ── Phase 1 (timed start) ────────────────────────────────────────────
        dist.barrier()
        t0 = time.perf_counter()

        # Concurrent: straggler sleeps while healthy ranks do a 3-rank AllReduce.
        # If sleep_ms > allreduce_3rank_time, healthy ranks finish first and then
        # wait at the barrier below — this is the overlap that gives speedup.
        if rank == STRAGGLER_RANK:
            time.sleep(sleep_ms / 1000.0)
        else:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=healthy_group)
            # buf now holds sum(rank 0, 1, 2) on each of ranks 0-2

        # ── Phase 2 (straggler joins) ────────────────────────────────────────
        dist.barrier()

        # ── Phase 3 (merge) ─────────────────────────────────────────────────
        # Broadcast the straggler's original data to all ranks and add it to
        # the partial sum held by the healthy ranks.
        strag = buf.clone() if rank == STRAGGLER_RANK else torch.zeros_like(buf)
        dist.all_reduce(strag, op=dist.ReduceOp.SUM)   # all ranks receive rank-3 data
        buf.add_(strag)
        # After merge: ranks 0-2 hold sum(0,1,2) + rank3 = sum(all 4 ranks) ✓

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000.0
        bw = (2.0 * (world_size - 1) / world_size * buffer_bytes) / (t1 - t0) / 1e9

        if rank == 0:
            print(f"stragglar,{buffer_bytes},{it},{sleep_ms:.3f},{ms:.3f},{bw:.3f}",
                  flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
