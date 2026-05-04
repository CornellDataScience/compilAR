# StragglAR Test Suite

Tests for straggler detection significance and AllReduce speedup on a real 2-node, 4-GPU cluster.

## Cluster topology

| Node | GPUs | MPI ranks |
|---|---|---|
| `compute1` | 2× RTX 2080 Ti (GPU 0, GPU 1) | 0, 1 |
| `compute4` | 2× GTX 1070   (GPU 0, GPU 1) | 2, 3 |

The nodes communicate over the `enp5s0` interface.  There is no shared filesystem; both nodes must have the project checked out at the same path and the conda environment synced (see below).

## Files

| File | Purpose |
|---|---|
| `run_tests.sh` | Main test script — runs all five phases end-to-end |
| `mpi_smoketest.py` | MPI worker for Phase 1 — each rank times a GEMM on its local GPU |
| `ring_allreduce_baseline.py` | MPI worker for Phase 3 — ring AllReduce timing baseline |
| `hostfile` | OpenMPI hostfile (`compute1 slots=2` / `compute4 slots=2`) |
| `plot_results.py` | Standalone plotting script — reads a results directory and generates graphs |

## Requirements

| Requirement | Notes |
|---|---|
| 2 nodes: `compute1`, `compute4` | Both must be reachable via passwordless SSH |
| 4× NVIDIA GPUs | 2 per node, visible via `nvidia-smi` |
| CUDA driver | Tested with driver ≥ 535 |
| `mpirun` (OpenMPI 4.x) | Must be on PATH on both nodes |
| `python3` with `torch` + NCCL | Tested with PyTorch pip wheel (cu124); **same env on both nodes** |
| `scipy` | For exact p-values — falls back to normal approximation if missing |
| `matplotlib` + `numpy` | For plot generation |
| StragglAR binary (`4gpu`) | Must be compiled and present at the project root on **both** nodes |

### Syncing the conda environment to compute4

```bash
# Run once from compute1 after installing on compute1
rsync -az /home/cadmin/.conda/envs/compilar_env/ compute4:/home/cadmin/.conda/envs/compilar_env/
```

Install optional Python dependencies (then re-sync):

```bash
/home/cadmin/.conda/envs/compilar_env/bin/pip install scipy matplotlib numpy
rsync -az /home/cadmin/.conda/envs/compilar_env/ compute4:/home/cadmin/.conda/envs/compilar_env/
```

## Quick Start

Run from the project root or the `stragglar/tests/` directory:

```bash
cd /path/to/compilAR
./stragglar/tests/run_tests.sh
```

The script uses `/home/cadmin/.conda/envs/compilar_env/bin/python3` on both nodes and auto-detects the `4gpu` binary at the project root.

Expected runtime with default settings: **~45–60 minutes** (1000 smoketest trials plus 10 AllReduce benchmark runs).

---

## What the Tests Do

The test suite runs in five sequential phases, sharing data between them so the 1000 smoketest trials serve both the significance test and the speedup benchmark.

### Phase 1 — Straggler detection trials

Launches `mpi_smoketest.py` across all 4 ranks via a single `mpirun` call.  Each trial:

1. All 4 ranks synchronise with an NCCL barrier.
2. Each rank runs the same randomized matrix multiplication (GEMM) on its local GPU.
3. Timings are collected to rank 0 via `dist.all_gather`.
4. Rank 0 records which rank finished last — the **straggler rank** — and **`delta_ms`** — how many milliseconds behind the second-to-last rank it was.

The GEMM matrix size is the same for all 4 ranks each trial (sampled randomly from 4096² to 10240² using the trial number as a seed).  Timing differences emerge from real hardware: the GTX 1070s on compute4 (ranks 2–3) are slower than the RTX 2080 Tis on compute1 (ranks 0–1).

Each trial emits one line (rank 0 stdout):
```
STRAGGLER_REPORT=<trial_0indexed>:<straggler_rank>:<delta_ms>
```

Progress is printed to stderr every 100 trials with an ETA.

### Phase 2 — Significance analysis

After all trials, the 1000 `(straggler_gpu, delta_ms)` records are analysed with two statistical tests.

**Chi-squared goodness-of-fit test**

Tests whether the distribution of detected stragglers is uniform across all 4 GPUs.

- H₀: each GPU is detected as the straggler with equal probability (25%).
- If chi² > 7.815 (critical value, df = 3, α = 0.05), H₀ is rejected.
- Interpretation: a significant result means one or more GPUs are systematically slower.

**Binomial test on the most-detected GPU**

Tests whether the GPU detected most often is identified significantly more than chance.

- H₀: the most-detected GPU is the straggler at the chance rate (25%).
- Uses an exact binomial test (scipy) or a normal approximation z-test as fallback.
- Interpretation: a significant result (p < 0.05) confirms that one specific GPU is the hardware straggler.

Phase 2 also computes the **delta distribution statistics** (min, p25, median, p75, max, mean) from the 1000 real measurements. These feed directly into Phase 3 — no synthetic delays are used.

### Phase 3 — Speedup benchmark

Uses the five real delay quantiles from Phase 1 as the `sleep_ms` input for the AllReduce benchmarks. At each point, two benchmarks run:

**Ring AllReduce baseline** (`ring_allreduce_baseline.py`)

- Launched via `mpirun -n 4 --hostfile hostfile` (one rank per GPU, 2 per node).
- Each rank reads its `OMPI_COMM_WORLD_LOCAL_RANK` to select its local CUDA device.
- Rank 3 (compute4 GPU 1, matching the StragglAR binary convention) sleeps for `sleep_ms` before calling `torch.distributed.all_reduce`.
- Because all ranks must call `all_reduce` before NCCL can proceed, the other three ranks block waiting for rank 3. Total time ≈ `sleep_ms` + ring allreduce time.
- Outputs CSV: `ring,<buffer_bytes>,<iter>,<sleep_ms>,<runtime_ms>,<BW(GB/s)>`

**StragglAR binary** (`./4gpu` via `mpirun -n 4 --hostfile hostfile`)

- Launched with `CUDA_VISIBLE_DEVICES=0,1` so each node sees only its 2 local GPUs. Rank *r* maps to GPU `r % 2` on its node.
- Rank 3 runs a CUDA sleep kernel for `sleep_ms` before participating.
- While rank 3 sleeps, ranks 0–2 perform a reduce-scatter among themselves, overlapping the straggler delay.
- Outputs the same CSV format per rank; the script takes the median across all 4 ranks per iteration, then averages across iterations.

**Speedup** = `ring_mean_ms / stragglar_mean_ms`

The five benchmark points (min, p25, median, p75, max) show how speedup scales with straggler severity — the larger the delay, the more StragglAR's overlap matters.

### Phase 4 — Final report

Prints a combined summary:

```
========================================================
 FINAL REPORT
========================================================
 Trials run          : 1000
 Most-detected GPU   : GPU 3  (34.2% of trials)

 Speedup summary from real delta measurements:

  Point       Sleep(ms)   Ring(ms)    SAR(ms)    Speedup
  ----------  ----------  ----------  ----------  ---------
  min              1.2      48.234      47.891    1.0072x
  p25              8.7      56.012      50.134    1.1172x
  median          18.4      67.891      53.201    1.2762x
  p75             34.1      84.203      55.812    1.5088x
  max             91.6     140.912      58.234    2.4199x

  Average speedup : 1.4459x  (across all 5 delay points)
  Best run        : 2.4199x  (at max observed straggler delay)
========================================================
```

---

## Results Storage

Every run creates a timestamped directory under `stragglar/tests/results/`:

```
stragglar/tests/results/
└── 2026-05-02_12-30-45/
    ├── straggler_trials.csv    ← Phase 1: one row per trial
    ├── significance.json       ← Phase 2: statistical test results
    ├── speedup_results.csv     ← Phase 3: AllReduce benchmark timings
    ├── summary.txt             ← Phase 4: human-readable final report
    └── plots/                  ← Phase 5: generated graphs
        ├── detection_frequency.png
        ├── delta_distribution.png
        ├── speedup_vs_delay.png
        ├── timing_comparison.png
        └── overview.png
```

### File formats

**`straggler_trials.csv`** — raw output from the 1000 smoketest runs

```
trial,straggler_gpu,delta_ms
1,3,18.432
2,1,4.217
3,3,22.891
...
```

**`significance.json`** — chi-squared and binomial test results

```json
{
  "n_trials": 1000,
  "n_gpus": 4,
  "counts": {"0": 221, "1": 238, "2": 201, "3": 340},
  "per_gpu_avg_delta": {"0": 15.2, "1": 14.8, "2": 13.1, "3": 24.7},
  "top_gpu": 3,
  "chi2": 42.318,
  "chi2_p": 3.82e-09,
  "chi2_significant": true,
  "binomial_p": 1.14e-11,
  "binomial_significant": true,
  "exact_pvalues": true,
  "delta_stats": {"min": 0.8, "p25": 7.3, "p50": 17.1, "p75": 33.4, "max": 94.2, "mean": 21.6, "std": 18.3}
}
```

**`speedup_results.csv`** — AllReduce benchmark at five delay points

```
label,sleep_ms,ring_ms,sar_ms,speedup
min,0.8,47.123,46.891,1.0049
p25,7.3,55.234,50.112,1.1022
median,17.1,65.891,53.442,1.2330
p75,33.4,82.103,55.812,1.4710
max,94.2,143.921,58.234,2.4713
```

The `results/` directory is excluded from git (see `.gitignore`).  To override where results are saved:

```bash
RESULTS_DIR=/scratch/my_run ./stragglar/tests/run_tests.sh
```

---

## Generating Plots

Plots are generated automatically at the end of `run_tests.sh`.  To regenerate them from a saved results directory (e.g. after the run, or on a different machine):

```bash
python3 stragglar/tests/plot_results.py stragglar/tests/results/2026-05-02_12-30-45
```

### Plots produced

| File | What it shows |
|---|---|
| `detection_frequency.png` | Bar chart of how often each GPU was the straggler. Dashed line at the 25% chance level; the most-detected GPU is highlighted in red and marked if statistically significant. |
| `delta_distribution.png` | Histogram of the 1000 real straggler delays (ms) with vertical markers at min / p25 / median / p75 / max. Shows the actual delay distribution on this hardware. |
| `speedup_vs_delay.png` | Line chart of StragglAR speedup over Ring as the straggler delay increases through the five benchmark points. Reference lines at 1.0× (no speedup) and the average speedup. |
| `timing_comparison.png` | Grouped bar chart of absolute Ring vs StragglAR latency (ms) at each of the five delay points. |
| `overview.png` | All four plots combined on a single 2×2 figure — useful for reports and presentations. |

---

## Configuration

All settings can be overridden with environment variables. No changes to the script are needed.

| Variable | Default | Description |
|---|---|---|
| `N_TRIALS` | `1000` | Number of smoketest trials for the significance test |
| `ITERS_PER_TRIAL` | `5` | Timed GEMM iterations per smoketest trial |
| `ALLREDUCE_ITERS` | `20` | Timed AllReduce iterations per benchmark point |
| `BUFFER_BYTES` | `50331648` | AllReduce buffer size (48 MiB) — must be divisible by `(N-1) × 4 = 12` |
| `NCCL_IFNAME` | `enp5s0` | Network interface NCCL uses for inter-node communication |
| `SMOKETEST_PORT` | `12345` | NCCL rendezvous port for Phase 1 |
| `RING_PORT` | `12346` | NCCL rendezvous port for Phase 3 ring baseline |
| `PY` | `/home/cadmin/.conda/envs/compilar_env/bin/python3` | Python binary (must exist on both nodes at this path) |
| `BINARY` | auto | Path to the StragglAR binary; auto-detected at `../../4gpu` |
| `RESULTS_DIR` | `results/<timestamp>` | Where to write all output files and plots |

### Examples

Faster run with fewer trials (useful for smoke-checking the setup):

```bash
N_TRIALS=50 ITERS_PER_TRIAL=3 ./stragglar/tests/run_tests.sh
```

Custom binary path or results directory:

```bash
BINARY=/scratch/my_4gpu RESULTS_DIR=/scratch/run1 ./stragglar/tests/run_tests.sh
```

Larger buffer for bandwidth-bound profiling:

```bash
BUFFER_BYTES=201326592 ./stragglar/tests/run_tests.sh   # 192 MiB
```

Override the network interface (e.g. if compute1 uses a different name):

```bash
NCCL_IFNAME=eth0 ./stragglar/tests/run_tests.sh
```

---

## How Speedup Is Measured

The comparison is a fair apples-to-apples timing of the same collective on the same hardware:

```
Ring AllReduce (with straggler)
─────────────────────────────────────────────────────────
GPU 0  ████░░░░░░░░░░░░░░░░░░ waiting ░░░░░░░░░░░░░░████ ring
GPU 1  ████░░░░░░░░░░░░░░░░░░ waiting ░░░░░░░░░░░░░░████ ring
GPU 2  ████░░░░░░░░░░░░░░░░░░ waiting ░░░░░░░░░░░░░░████ ring
GPU 3  █████████████████████████ sleep ████████████████── ring
        ↑ all ranks blocked until GPU 3 finishes sleeping

StragglAR (with same straggler)
─────────────────────────────────────────────────────────
GPU 0  ████ reduce-scatter ████░░ wait ░░░ merge ████
GPU 1  ████ reduce-scatter ████░░ wait ░░░ merge ████
GPU 2  ████ reduce-scatter ████░░ wait ░░░ merge ████
GPU 3  █████████████████████████ sleep ████ merge ████
        ↑ healthy GPUs communicate during straggler's sleep
```

Both timing measurements include the full duration from the start of the straggler's delay through completion of the collective. The delta values used as `sleep_ms` are the **real delays measured on the hardware** in Phase 1, not artificially chosen numbers.

---

## Interpreting Results

**Significance test**

- If **p < 0.05** on both chi-squared and binomial tests: the hardware has a systematic straggler. One GPU is consistently slower due to a real hardware difference (thermal, memory bandwidth, clock variation).
- If **p ≥ 0.05**: straggler detection appears uniform — all GPUs are roughly equivalent and the detected straggler varies randomly across trials.

**Speedup benchmark**

- Speedup at **median delta**: typical expected improvement on this hardware configuration.
- Speedup at **max delta**: best-case scenario — how much StragglAR helps when the straggler is at its worst.
- **Average speedup** across the five points: a balanced summary across the full observed delay range.
- A speedup > 1.0x at all delay points confirms StragglAR outperforms Ring AllReduce whenever a straggler is present.
