"""
batch_compute.py — Launch randomized matrix multiply on all GPUs concurrently.

Each GPU receives a randomly chosen matrix size, so compute times naturally
vary across the batch — the slowest GPU emerges as the straggler organically.

Returns per-GPU results including exact finish timestamps for downstream
straggler detection.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Matrix sizes to sample from — one is chosen randomly and applied uniformly
# to all GPUs, matching how ML training distributes equal-sized batches.
# At float32: 4096² ≈ 64M elems (~256 MB), 10240² ≈ 100M elems (~400 MB).
MATRIX_SIZE_CHOICES = [4096, 5120, 6144, 7168, 8192, 9216, 10240]
NUM_WARMUP          = 3     # iterations excluded from timing
NUM_ITERS           = 15    # timed iterations per GPU
DTYPE               = torch.float32


@dataclass
class GPUWorkResult:
    gpu_id:      int
    matrix_size: int
    num_iters:   int
    iter_times:  list[float]          # per-iteration wall-clock times (seconds)
    end_time:    float                # time.perf_counter() when GPU finished
    total_time:  float                # sum of iter_times (seconds)
    gpu_name:    str = field(default="")


def _run_on_gpu(
    gpu_id: int,
    matrix_size: int,
    num_warmup: int,
    num_iters: int,
) -> GPUWorkResult:
    """
    Run GEMM on a single GPU with the given parameters.
    Called from a thread — CUDA releases the GIL so threads run in parallel.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_properties(device).name

    # Each GPU gets independently seeded random data — same size, different values.
    # This mirrors data-parallel ML training: uniform workload, distinct mini-batches.
    gen = torch.Generator(device=device)
    gen.manual_seed(torch.initial_seed() + gpu_id)
    A = torch.randn(matrix_size, matrix_size, dtype=DTYPE, device=device, generator=gen)
    B = torch.randn(matrix_size, matrix_size, dtype=DTYPE, device=device, generator=gen)

    for _ in range(num_warmup):
        torch.mm(A, B)
    torch.cuda.synchronize(device)

    iter_times: list[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        torch.mm(A, B)
        torch.cuda.synchronize(device)
        iter_times.append(time.perf_counter() - t0)

    end_time = time.perf_counter()

    return GPUWorkResult(
        gpu_id      = gpu_id,
        matrix_size = matrix_size,
        num_iters   = num_iters,
        iter_times  = iter_times,
        end_time    = end_time,
        total_time  = sum(iter_times),
        gpu_name    = gpu_name,
    )


def run_batch_compute(
    gpu_ids: list[int] | None = None,
    matrix_size_choices: list[int] = MATRIX_SIZE_CHOICES,
    num_warmup: int = NUM_WARMUP,
    num_iters: int = NUM_ITERS,
    seed: int | None = None,
) -> dict[int, GPUWorkResult]:
    """
    Launch GEMM on all GPUs concurrently with randomized matrix sizes.

    Parameters
    ----------
    gpu_ids : GPU indices to use (default: all visible GPUs)
    matrix_size_choices : pool to randomly draw one matrix size from (same for all GPUs)
    num_warmup : warm-up iterations before timing
    num_iters : number of timed iterations per GPU
    seed : optional RNG seed for reproducible matrix size selection

    Returns
    -------
    dict mapping gpu_id -> GPUWorkResult
    """
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        raise RuntimeError("No CUDA GPUs found.")

    rng = random.Random(seed)
    matrix_size = rng.choice(matrix_size_choices)  # one size, uniform across all GPUs

    print(f"\n[batch_compute] {len(gpu_ids)} GPU(s): {gpu_ids}")
    print(f"[batch_compute] Matrix size (all GPUs): {matrix_size}×{matrix_size}  |  {num_iters} iters")
    print(f"[batch_compute] Each GPU uses independently seeded random data")

    print(f"\n  Launching all GPUs in parallel ...", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {
            pool.submit(_run_on_gpu, gid, matrix_size, num_warmup, num_iters): gid
            for gid in gpu_ids
        }
        results: dict[int, GPUWorkResult] = {}
        for fut in as_completed(futures):
            gid = futures[fut]
            results[gid] = fut.result()

    return results
