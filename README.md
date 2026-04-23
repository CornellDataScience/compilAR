# compilAR

## Overview

compilAR is a compiler and runtime for straggler-aware AllReduce over GPU clusters, inspired by the paper [Efficient AllReduce with Stragglers](https://arxiv.org/pdf/2505.23523) (Devraj et al.). It takes a schedule produced by the StragglAR algorithm as input and emits a complete, standalone CUDA + MPI + NCCL implementation of that schedule for any number of GPUs.

The core problem it solves is that standard AllReduce algorithms (ring, tree, recursive-halving-doubling) stall every healthy rank until the slowest GPU catches up. StragglAR lets the N-1 healthy ranks make progress among themselves while the straggler is still computing, then merges the straggler's contribution with a minimal number of additional communication rounds once it is ready. This yields measurable throughput improvements when one rank is consistently slower than the rest.

## How It Works

### The algorithm

Given N GPUs with one designated straggler (rank N-1):

1. **Reduce-scatter phase (while straggler is delayed):** The N-1 healthy ranks run `ncclReduceScatter` among themselves over N-1 equal chunks of the buffer. After this, rank r holds the partial sum of chunk r across all healthy ranks.

2. **Straggler merge phase (schedule-driven):** The schedule synthesizer produces a sequence of rounds, each containing a batch of pairwise exchanges. Three exchange types are used:
   - **StragglerMatching**: a healthy rank and the straggler both hold a partial sum of the same chunk. They swap into a scratch buffer, then both call `reduce_add` to finalize. After this, both hold the complete N-rank sum for that chunk.
   - **OneWayMatching**: a rank holding a fully-reduced chunk pushes it to a rank that does not. Plain copy, no reduction.
   - **TwoWayMatching**: two ranks each hold a fully-reduced chunk the other lacks. They swap simultaneously. Plain copy in each direction.

3. After the last round, every rank holds every chunk fully reduced.

### The compiler

`compilAR.py` takes a schedule file (from `synthesizer_pow2.py` or `synthesizer_nonpow2.py`) and generates a complete `.cu` source file. It uses `allreduce_multinode.cu.template` as its skeleton and substitutes three things:
- `NUM_RANKS`: number of GPUs, inferred from the schedule
- `kStragglerRank`: straggler rank, inferred from the schedule
- Body of `stragglar_allreduce_helper`: per-round NCCL group blocks, generated from the schedule matchings

Everything else in the template (MPI bootstrap, NCCL communicator init, reduce-scatter sub-communicator, benchmark loop, correctness check, cleanup) is agnostic of the number of GPUs and remains unchanged between schedules.

## Dependencies

**Python:**
- Python 3.12+
- numpy

**CUDA build (on our cluster):**
- CUDA Toolkit 12.x
- NCCL 2.x
- OpenMPI or MPICH
- nvcc with MPI-aware host compiler (`mpicxx`)

```bash
uv sync   # or: pip install numpy
```

## Example Usage

### 1. Generate a schedule

```bash
cd stragglar/schedules
python synthesizer_pow2.py 8 > 8gpusched.txt
```

The most common use case is when N is a power of 2. Pre-generated schedules for N=2, 4, 8 are already in `schedules/`.

### 2. Compile the schedule to CUDA

```bash
cd stragglar
python compilAR.py schedules/8gpusched.txt generated_8gpu.cu
```

On success:
```
Wrote generated_8gpu.cu (N=8, straggler=7)
```

### 3. Build the binary (on the cluster)

```bash
nvcc -ccbin mpicxx -O3 -arch=sm_89 generated_8gpu.cu -lnccl -lmpi -o stragglar_8gpu
```

Replace `sm_89` with your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

### 4. Run

```bash
# No straggler delay (correctness check and baseline throughput)
mpirun -n 8 ./stragglar_8gpu 1073741824 stragglar 10 -1

# With 100ms simulated straggler delay on rank N-1
mpirun -n 8 ./stragglar_8gpu 1073741824 stragglar 10 100.0
```

**Arguments:** `<buffer_bytes> <algorithm> <num_iters> <sleep_ms>`

| Argument | Description |
|---|---|
| `buffer_bytes` | Total AllReduce buffer size in bytes (e.g. `1073741824` = 1 GiB) |
| `algorithm` | Must be `stragglar` |
| `num_iters` | Timed iterations; first is discarded as warmup |
| `sleep_ms` | Milliseconds to delay rank N-1. `-1` skips reduce-scatter and runs the merge schedule only |

**Output** (rank 0):
```
algorithm,buffer_size_bytes,iteration,delay,runtime_ms,BW(GB/s)
stragglar,1073741824,1,100.000,12.345,82.345
```

**Correctness:** every element of every rank's output buffer must equal `6.0f`. Failures print `Rank X, idx Y, val Z`.

### Detecting the real straggler GPU

To identify which physical GPU is the actual straggler before running:

```bash
python -m stragglar.smoketest.smoketest --gpus 0 1 2 3 4 5 6 7 --iters 15
```

This runs randomized GEMM workloads on all GPUs concurrently and reports which finishes last and by how much. The straggler GPU index is then used to configure `LOCAL_RANK` so that MPI rank N-1 binds to that physical GPU at launch time.

## Architecture Notes

### Single-process vs. multi-process model

The files in `reference_code/` and `allreduce_4GPU_rewrite.cu` use a single-process model: one process manages all GPUs via `ncclCommInitAll`. This only works when all GPUs are on one machine and is kept for reference.

`allreduce_multinode.cu` (and all generated files) use a multi-process MPI model: one MPI rank per GPU. Each process initializes its NCCL communicator via `ncclCommInitRank` with a token distributed by `MPI_Bcast`. This scales to multi-node configurations.

### Communicator structure

Two NCCL communicators are maintained per process:
- `comm`: all N ranks; used during the straggler merge schedule
- `subComm`: ranks 0 through N-2 only; used for the reduce-scatter phase while the straggler is delayed, built via `MPI_Comm_split` to exclude rank N-1

### GPU binding

Each MPI process binds to its GPU via the `LOCAL_RANK` environment variable (set by `mpirun`, `torchrun`, or SLURM). Without it, the process falls back to `myRank % cudaDeviceCount`. For accurate straggler behavior, the process assigned rank N-1 must be bound to the physically slow GPU.