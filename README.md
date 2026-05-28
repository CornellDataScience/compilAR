# compilAR

## Overview

compilAR is a compiler and runtime for straggler-aware AllReduce over GPU clusters, inspired by the paper [Efficient AllReduce with Stragglers](https://arxiv.org/pdf/2505.23523) (Devraj et al.). It takes a schedule produced by the StragglAR algorithm as input and emits a complete, standalone CUDA + MPI + NCCL implementation of that schedule for any number of GPUs.

The core problem it solves is that standard AllReduce algorithms (ring, tree, recursive-halving-doubling) stall every healthy rank until the slowest GPU catches up. StragglAR lets the N-1 healthy ranks make progress among themselves while the straggler is still computing, then merges the straggler's contribution with a minimal number of additional communication rounds once it is ready.

<img width="1422" height="825" alt="Screenshot 2026-05-28 at 2 12 12 PM" src="https://github.com/user-attachments/assets/0fcacc38-66ab-4a56-94d3-8db9fe4a1cc0" />


## How It Works

### The algorithm

Given N GPUs with one designated straggler (rank N-1):

1. **Reduce-scatter phase:** While the straggler is delayed, the N-1 healthy ranks run `ncclReduceScatter` among themselves over N-1 equal chunks of the buffer. After this, rank r holds the partial sum of chunk r across all healthy ranks.

2. **Straggler merge phase:** The schedule synthesizer produces a sequence of rounds, each containing a batch of pairwise exchanges. Three exchange types are used:
   - **StragglerMatching**: a healthy rank and the straggler both hold a partial sum of the same chunk. They swap into a scratch buffer, then both call `reduce_add` to finalize.
   - **OneWayMatching**: a rank holding a fully-reduced chunk pushes it to a rank that does not. Plain copy.
   - **TwoWayMatching**: two ranks each hold a fully-reduced chunk the other lacks. They swap simultaneously.

3. After the last round, every rank holds every chunk fully reduced.

### The compiler

`compilAR.py` takes a schedule file (from `synthesizer_pow2.py` or `synthesizer_nonpow2.py`) and generates a complete `.cu` source file. It uses `allreduce_multinode.cu.template` as its skeleton and substitutes:
- `NUM_RANKS`: number of GPUs, inferred from the schedule
- `kStragglerRank`: straggler rank, inferred from the schedule
- Body of `stragglar_allreduce_helper`: per-round NCCL group blocks, generated from the schedule matchings

Everything else in the template (MPI bootstrap, NCCL communicator init, reduce-scatter sub-communicator, benchmark loop, correctness check, cleanup) is agnostic of the number of GPUs.

## Environment

All builds and runs happen inside an Apptainer container defined by `stragglar/compilar.def`. The container ships a coherent CUDA + NCCL + OpenMPI + PyTorch toolchain (NGC `pytorch:23.10-py3`), so the host only needs:

- An NVIDIA driver (≥ 535 for our base image)
- Apptainer ≥ 1.3
- An MPI launcher (host `mpirun` on a TCP cluster, or Slurm `srun` on an HPC cluster)

No host-side CUDA, NCCL, MPI, Python, or PyTorch installation is required.

### Build the SIF (once per cluster)

```bash
cd stragglar
sudo apptainer build compilAR.sif compilar.def
# or, if sudo is unavailable:
apptainer build --fakeroot compilAR.sif compilar.def
```

Place `compilAR.sif` somewhere visible from every node (e.g. NFS-shared `$HOME`).

## Usage

### 1. Generate a schedule

```bash
cd stragglar/schedules
python synthesizer_pow2.py 8 > 8gpusched.txt
```

Pre-generated schedules for N=2, 4, 8 are already in `schedules/`.

### 2. Compile the schedule to CUDA

```bash
cd stragglar
apptainer exec --nv compilAR.sif python3 compilAR.py schedules/8gpusched.txt generated_8gpu.cu
```

### 3. Build the binary

The container provides `stragglar-build`, which produces a fat binary that runs on every shipping NVIDIA GPU from Pascal (sm_61) through Hopper (sm_90), with PTX fallback for newer cards:

```bash
apptainer exec --nv compilAR.sif stragglar-build generated_8gpu.cu stragglar_8gpu
```

For a faster, GPU-specific build (host's GPU only):

```bash
apptainer exec --nv compilAR.sif stragglar-build --native generated_8gpu.cu stragglar_8gpu
```

### 4. Run

#### TCP cluster (host `mpirun` as launcher)

```bash
mpirun -np 8 -H host1:4,host2:4 \
    --mca btl_tcp_if_include <iface> \
    -x NCCL_P2P_DISABLE=1 \
    -x PMIX_MCA_psec=native \
    -x PMIX_MCA_gds=hash \
    apptainer exec --nv compilAR.sif ./stragglar_8gpu 117440512 stragglar 10 100.0
```

The `PMIX_MCA_*` flags reconcile the host PMIx with the container's PMIx; drop them if your host and container OpenMPI versions match. `NCCL_P2P_DISABLE=1` is needed only on consumer (GeForce) GPUs where PCIe P2P is disabled in the driver.

#### Slurm cluster with InfiniBand

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

srun --mpi=pmix \
    apptainer exec --nv \
        --bind /dev/infiniband --bind /sys/class/infiniband \
        compilAR.sif ./stragglar_8gpu 117440512 stragglar 10 100.0
```

Add `-x NCCL_DEBUG=INFO` (TCP) or `--export=ALL,NCCL_DEBUG=INFO` (Slurm) on first run to confirm `Using network IB` (or `Using network Socket` on TCP) and reasonable bandwidth.

#### Automated straggler detection

`launch.sh` runs a smoketester to identify the physically slow GPU and binds it to rank N-1, useful when you don't want to simulate a delay. It currently assumes a single-node, host-toolchain layout — if you use it, run it inside the container:

```bash
apptainer exec --nv compilAR.sif ./stragglar/launch.sh 8 ./stragglar_8gpu 117440512 stragglar 10 -1
```

#### Binary arguments

`<buffer_bytes> <algorithm> <num_iters> <sleep_ms>`

| Argument | Description |
|---|---|
| `buffer_bytes` | Total AllReduce buffer size in bytes. Must satisfy `(buffer_bytes / sizeof(float)) % (N - 1) == 0`; the binary errors out otherwise with suggested valid sizes. The examples above use `117440512` (112 MiB, valid for N=8) and `50331648` (48 MiB, valid for N=4) — both pick a 4 MiB chunk per rank |
| `algorithm` | Must be `stragglar` |
| `num_iters` | Timed iterations; first is discarded as warmup |
| `sleep_ms` | Milliseconds to delay rank N-1. `-1` skips reduce-scatter and runs the merge schedule only |


## Architecture

### Multi-process model

`allreduce_multinode.cu` (and all generated files) use one MPI rank per GPU. Each process initializes its NCCL communicator via `ncclCommInitRank` with a token distributed by `MPI_Bcast`. This scales to multi-node configurations.

The single-process variants in `reference_code/` and `allreduce_4GPU_rewrite.cu` use `ncclCommInitAll` and only work on one host. They are kept for reference.

### Communicator structure

Two NCCL communicators are maintained per process:
- `comm`: all N ranks; used during the straggler merge schedule
- `subComm`: ranks 0 through N-2; used for the reduce-scatter phase while the straggler is delayed (built via `MPI_Comm_split`)

### GPU binding

Each MPI process binds to its GPU via `LOCAL_RANK` (set by `mpirun`, `torchrun`, or Slurm). Without it, the process falls back to `myRank % cudaDeviceCount`. For accurate straggler behavior, rank N-1 must be bound to the physically slow GPU. `launch.sh` does this automatically.

## Known Assumptions

- **`launch.sh` assumes single-node.** The smoketester enumerates GPUs via `torch.cuda.device_count()` on one host. Multi-node runs need either a per-node aggregation step or manual `LOCAL_RANK` assignment.
- **Data type is hardcoded to `float32`.** Supporting fp16 / bf16 requires changes to both the template and the generator.
- **Buffer size must be divisible by `(N-1) * sizeof(float)`.** The binary checks at startup and exits with an error and suggested valid sizes if not. Note that round-power-of-2 sizes (1 MiB, 1 GiB) are *not* valid for non-power-of-2 values of N-1 — for N=4 use multiples of 12 bytes (e.g. 48 MiB, 192 MiB), for N=8 use multiples of 28 bytes (e.g. 112 MiB, 896 MiB). The helper at `stragglar/smoketest/pad_buffers.py N` prints a list of valid sizes for a given N.
- **Straggler is always rank N-1.** Baked into the generated code; `launch.sh` maps the physical straggler GPU to that rank at launch.
- **Correctness check assumes the built-in fill pattern.** `kExpectedSum = 6.0f` only holds when the straggler fills its buffer with the arbitrary `3.0f` and each non-straggler fills its own chunk. Replace this check when wiring real input data.
- **Clock-based straggler delay is calibrated to device 0.** On heterogeneous-clock GPUs, `sleep_ms` won't match wall-clock ms on other devices. Doesn't affect correctness.

## Key Results

We achieved ~1.3x speedup compared to standard Ring AllReduce on our in-house CDS Compute Cluster. This is with a deliberate 50 ms simulated delay to showcase how stragglers are handled, and across multiple nodes with heterogeneous and hardware (RTX 2080 Ti and GTX 1070). We reasonably project that on production or research grade homogeneous hardware on a single host device, we can approach a 2x speedup compared to traditional AllReduce methods. 

<img width="579" height="419" alt="Screenshot 2026-05-28 at 2 38 06 PM" src="https://github.com/user-attachments/assets/27e69058-5555-48c4-a15f-1df60ecd96c6" /> 
<img width="619" height="374" alt="Screenshot 2026-05-28 at 2 40 57 PM" src="https://github.com/user-attachments/assets/ce3e45ae-bf66-4c7b-8886-0a4e9603294e" />


Our straggler detection system also correctly identifies the slowest GPU. In our in-house tests, one of our nodes had older GPUs and so they were always identified as the stragglers at a statistically significant rate. 

<img width="524" height="379" alt="Screenshot 2026-05-28 at 2 40 33 PM" src="https://github.com/user-attachments/assets/b68860d3-c1ab-4e2b-a2ef-051c75a86191" />


## Acknowledgements

The StragglAR algorithm and the schedule synthesizer in `stragglar/schedules/` are the work of Devraj et al., [Efficient AllReduce with Stragglers](https://arxiv.org/pdf/2505.23523). This project is not the original algorithm, but a compiler and launch harness built around it.
