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

The exact `nvcc` invocation depends on your GPU hardware and where NCCL is installed. The simplest case, for an example homogeneous cluster with system-installed NCCL:

```bash
nvcc -ccbin mpicxx -O3 -arch=sm_61 generated_8gpu.cu -lnccl -lmpi -o stragglar_8gpu
```

**Adjustments for other environments:**

- **Different GPU model**: replace `sm_61` with the compute capability of your card. Check with:
  ```bash
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader
  ```
- **Mixed-GPU cluster**: compile for all target archs plus a PTX fallback:
  ```bash
  nvcc ... -gencode arch=compute_70,code=sm_70 \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_75,code=compute_75 \
           ...
  ```
- **NCCL installed via conda** (not in system lib paths): add include and library search paths:
  ```bash
  nvcc ... -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lnccl -ccbin=mpicxx ...
  ```
- **`module load` clusters**: typically `module load cuda/12.x nccl openmpi` before nvcc picks up the headers and libs.

### 4. Run

There are two ways to launch the binary depending on whether you want to detect a real straggler or simulate one.

#### Automated straggler detection (recommended)

`launch.sh` combines the smoketester, the rank-to-GPU mapping, and the `mpirun` invocation in one step:

```bash
./stragglar/launch.sh 8 ./stragglar_8gpu 1073741824 stragglar 10 -1
```

Under the hood it:
1. Runs the smoketester on all N GPUs to identify the physically slow one
2. Builds a rank-to-GPU mapping so MPI rank N-1 binds to that GPU
3. Exports the mapping and calls `mpirun` with `rank_wrapper.sh`, which sets `LOCAL_RANK` per process

When using `launch.sh`, pass `-1` for `sleep_ms` since the physically slow GPU will lag on its own and no simulated delay is needed.

#### Manual launch (simulated straggler)

For correctness validation or benchmarking without a real straggler:

```bash
# No straggler delay — purely for correctness checks
mpirun -n 8 ./stragglar_8gpu 1073741824 stragglar 10 -1

# With 100ms simulated delay injected on rank N-1
mpirun -n 8 ./stragglar_8gpu 1073741824 stragglar 10 100.0
```

In this mode, rank N-1 is always the straggler regardless of physical GPU placement, and the delay is injected via `gpu_sleep_kernel`.

**Binary arguments:** `<buffer_bytes> <algorithm> <num_iters> <sleep_ms>`

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

## Architecture Notes

### Single-process vs. multi-process model

The files in `reference_code/` and `allreduce_4GPU_rewrite.cu` use a single-process model: one process manages all GPUs via `ncclCommInitAll`. This only works when all GPUs are on one machine and is kept for reference.

`allreduce_multinode.cu` (and all generated files) use a multi-process MPI model: one MPI rank per GPU. Each process initializes its NCCL communicator via `ncclCommInitRank` with a token distributed by `MPI_Bcast`. This scales to multi-node configurations.

### Communicator structure

Two NCCL communicators are maintained per process:
- `comm`: all N ranks; used during the straggler merge schedule
- `subComm`: ranks 0 through N-2 only; used for the reduce-scatter phase while the straggler is delayed, built via `MPI_Comm_split` to exclude rank N-1

### GPU binding

Each MPI process binds to its GPU via the `LOCAL_RANK` environment variable (set by `mpirun`, `torchrun`, or SLURM). Without it, the process falls back to `myRank % cudaDeviceCount`. For accurate straggler behavior, the process assigned rank N-1 must be bound to the physically slow GPU. `launch.sh` handles this automatically by parsing the smoketester's output and constructing the correct mapping.

## Portability and Known Assumptions

Beyond the build-time flags covered in the usage section, the runtime code and launch harness make a few assumptions worth flagging before you run on a new cluster:

- **`launch.sh` assumes all N GPUs are on one node.** The smoketester is a single Python process that enumerates visible GPUs via `torch.cuda.device_count()`, so it can only rank GPUs on the node it runs on. For multi-node runs you need to either (a) extend the smoketester to aggregate timings from a per-node launcher, or (b) fall back to manual `LOCAL_RANK` assignment with a known straggler GPU.
- **Open MPI is assumed by `launch.sh`.** The `-x RANK_TO_GPU` flag on the final `mpirun` line is Open MPI syntax. For MPICH, replace with `-env RANK_TO_GPU "$RANK_TO_GPU"` or `-envall`. `rank_wrapper.sh` already handles rank detection across Open MPI / MPICH / SLURM.
- **Data type is hardcoded to `float32`.** `constructNCCL` emits `ncclFloat` calls and the template allocates `float*` buffers. To support fp16 / bf16, both the template (`ncclHalf` / `ncclBfloat16`, `__half*` / `__nv_bfloat16*`) and the generator would need to accept a dtype parameter.
- **Buffer size must be divisible by `(N-1) * sizeof(float)`.** The template computes `chunkSize = size / (numRanks - 1)` without checking the remainder. Passing an odd buffer size silently truncates. Stick to powers of 2 in bytes for predictable behavior.
- **Straggler is always rank N-1.** This is baked into the generated code. `launch.sh` maps the physical straggler GPU to rank N-1 at launch time, but if you need the straggler to be a different logical rank (e.g., for testing), you'd need to regenerate the schedule.
- **Correctness check assumes the built-in fill pattern.** The `kExpectedSum = 6.0f` verification only holds when the straggler fills its whole buffer with `3.0f` and each non-straggler fills only its own chunk. If you wire up real input data, remove or replace that check at the bottom of `main()` in the template.
- **Clock-based straggler delay is calibrated to device 0.** `calculate_sleep_cycles` queries `cudaDevAttrClockRate` on device 0, so on clusters with heterogeneous GPU clocks the simulated delay will not match wall-clock ms on other devices. Doesn't affect correctness, just the meaning of `sleep_ms`.

If your setup violates any of these, the code will mostly still run — it just won't do what you expect. The build-time footguns (wrong `-arch`, missing NCCL paths) fail loudly with nvcc errors; the runtime ones (dtype, fill pattern) fail silently.

## Acknowledgements

The StragglAR algorithm and the schedule synthesizer in `stragglar/schedules/` are the work of Devraj et al., [Efficient AllReduce with Stragglers](https://arxiv.org/pdf/2505.23523). This project is not the original algorithm — it is a compiler and launch harness built around their work. The algorithmic contribution (the round-matching formulation, optimality bounds, and synthesis procedure) is theirs; what is new here is the code-generation pipeline, the MPI+NCCL runtime template, and the straggler-aware launch integration.