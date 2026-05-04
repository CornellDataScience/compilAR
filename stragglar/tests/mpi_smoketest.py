#!/usr/bin/env python3
"""
mpi_smoketest.py — MPI-launched straggler detection trial runner.

Must be launched via mpirun -n 4 with the following env vars set
(use mpirun -x flag to forward them to all ranks):

    MASTER_ADDR        hostname / IP of rank-0 node (e.g. compute1)
    MASTER_PORT        free TCP port for NCCL rendezvous (e.g. 12345)
    N_TRIALS           number of GEMM trials (default: 1000)
    ITERS_PER_TRIAL    timed GEMM iterations per trial (default: 5)
    NCCL_SOCKET_IFNAME network interface for NCCL (e.g. enp5s0)

Each rank runs on one GPU:
    local_rank = OMPI_COMM_WORLD_LOCAL_RANK  (0 or 1 on each node)
    rank       = OMPI_COMM_WORLD_RANK        (0–3 across both nodes)

All 4 ranks perform the same GEMM (same matrix size per trial, from trial
seed), synchronized by an NCCL barrier before timing.  Rank 0 collects all
timings via all_gather and identifies the straggler.

Output (stdout, rank 0 only — one line per trial):
    STRAGGLER_REPORT=<trial_0indexed>:<straggler_rank>:<delta_ms>

Progress lines go to stderr every 100 trials.
"""

import os
import sys
import time
import socket

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Rank / topology
# ---------------------------------------------------------------------------
rank       = int(os.environ.get("OMPI_COMM_WORLD_RANK",       os.environ.get("RANK",       "0")))
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE",       os.environ.get("WORLD_SIZE", "4")))

# torch.distributed env:// reads RANK / WORLD_SIZE from the environment
os.environ["RANK"]       = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)
# MASTER_ADDR and MASTER_PORT forwarded by mpirun -x flags

N_TRIALS        = int(os.environ.get("N_TRIALS",        1000))
ITERS_PER_TRIAL = int(os.environ.get("ITERS_PER_TRIAL", 5))
NUM_WARMUP      = 3

# ---------------------------------------------------------------------------
# CUDA / distributed init
# ---------------------------------------------------------------------------
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# ---------------------------------------------------------------------------
# Warmup: flush NCCL and the CUDA execution engine
# ---------------------------------------------------------------------------
for _ in range(NUM_WARMUP):
    dist.barrier()
    _w = torch.randn(512, 512, device=device) @ torch.randn(512, 512, device=device)
    torch.cuda.synchronize(device)
del _w

start_wall = time.perf_counter()

# ---------------------------------------------------------------------------
# Trial loop
# ---------------------------------------------------------------------------
for trial in range(N_TRIALS):
    # All ranks derive the same matrix size from the global trial seed so the
    # per-rank timings are directly comparable.
    g = torch.Generator()
    g.manual_seed(trial)
    n = int(torch.randint(4096, 10241, (1,), generator=g).item())

    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    torch.cuda.synchronize(device)

    # Barrier: all 4 GPUs start the timed window together
    dist.barrier()
    t0 = time.perf_counter()

    for _ in range(ITERS_PER_TRIAL):
        c = torch.mm(a, b)
    torch.cuda.synchronize(device)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / ITERS_PER_TRIAL

    # Gather all timings to every rank (costs ~32 bytes over NCCL — negligible)
    t_tensor = torch.tensor([elapsed_ms], dtype=torch.float64, device=device)
    all_t    = [torch.zeros(1, dtype=torch.float64, device=device)
                for _ in range(world_size)]
    dist.all_gather(all_t, t_tensor)

    if rank == 0:
        times       = [x.item() for x in all_t]
        ranked_desc = sorted(enumerate(times), key=lambda p: p[1], reverse=True)
        straggler_r = ranked_desc[0][0]
        delta_ms    = ranked_desc[0][1] - ranked_desc[1][1]
        print(f"STRAGGLER_REPORT={trial}:{straggler_r}:{delta_ms:.3f}", flush=True)

        if (trial + 1) % 100 == 0 or trial == N_TRIALS - 1:
            elapsed = time.perf_counter() - start_wall
            rate    = (trial + 1) / max(elapsed, 1e-9)
            eta     = int((N_TRIALS - trial - 1) / max(rate, 1e-9))
            print(
                f"  [{trial+1:4d} / {N_TRIALS:4d}]  {rate:.2f} trials/s  ETA ~{eta}s",
                file=sys.stderr, flush=True,
            )

dist.destroy_process_group()
