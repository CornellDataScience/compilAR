#!/usr/bin/env bash
# run_tests.sh
#
# Combined straggler significance test + StragglAR speedup benchmark.
#
# Phase 1 — Significance test:
#   Runs smoketest.py N_TRIALS times on the real 4-GPU hardware.
#   Records which GPU is identified as the straggler each trial, plus the
#   straggler delta_ms (how far behind the second-to-last GPU it finished).
#   A chi-squared and binomial test determine whether any GPU is a
#   statistically significant straggler.
#
# Phase 2 — Speedup benchmark:
#   Uses the delta_ms values collected in Phase 1 as the actual straggler
#   delay distribution for this hardware.  Runs ring_allreduce_baseline.py
#   and the StragglAR binary at five representative delay points
#   (min, p25, median, p75, max) derived from the real measurements.
#   Reports per-point speedup, average speedup, and best-run speedup.
#
# Usage:
#   ./run_tests.sh
#
# Configuration (override via env):
#   GPUS              space-separated GPU indices          (default: "0 1 2 3")
#   N_TRIALS          smoketest iterations                 (default: 1000)
#   ITERS_PER_TRIAL   GEMM iters per smoketest trial      (default: 5)
#   ALLREDUCE_ITERS   timed AllReduce iters per benchmark  (default: 20)
#   BUFFER_BYTES      AllReduce buffer size in bytes       (default: 50331648)
#   BINARY            path to StragglAR 4-GPU binary       (auto-detected)
#   RESULTS_DIR       where to save outputs                (default: results/<timestamp>)
#
# Requirements:
#   mpirun (OpenMPI)
#   python3 with torch
#   scipy  (pip install scipy)  — falls back to approximations if missing
#   matplotlib + numpy  (pip install matplotlib numpy)  — for plot_results.py

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GPUS="${GPUS:-0 1 2 3}"
N_GPUS=$(echo $GPUS | wc -w | tr -d ' ')
N_TRIALS="${N_TRIALS:-1000}"
ITERS_PER_TRIAL="${ITERS_PER_TRIAL:-5}"
ALLREDUCE_ITERS="${ALLREDUCE_ITERS:-20}"
BUFFER_BYTES="${BUFFER_BYTES:-50331648}"     # 48 MiB — default in launch.sh

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RING_SCRIPT="$SCRIPT_DIR/ring_allreduce_baseline.py"
PLOT_SCRIPT="$SCRIPT_DIR/plot_results.py"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/$TIMESTAMP}"

if [ -n "${BINARY:-}" ]; then
    BINARY_PATH="$BINARY"
elif [ -x "$PROJECT_ROOT/4gpu" ]; then
    BINARY_PATH="$PROJECT_ROOT/4gpu"
elif [ -x "$PROJECT_ROOT/stragglar/allreduce_multinode" ]; then
    BINARY_PATH="$PROJECT_ROOT/stragglar/allreduce_multinode"
else
    echo "Error: StragglAR binary not found. Set BINARY=/path/to/binary or" >&2
    echo "       compile and place it at $PROJECT_ROOT/4gpu" >&2
    exit 1
fi

PY="$(command -v python3 || command -v python || true)"
[ -z "$PY" ] && { echo "Error: python3 not in PATH" >&2; exit 1; }
MPI="$(command -v mpirun || true)"
[ -z "$MPI" ] && { echo "Error: mpirun not in PATH" >&2; exit 1; }
[ ! -f "$RING_SCRIPT" ] && { echo "Error: $RING_SCRIPT not found" >&2; exit 1; }

# Create results directory
mkdir -p "$RESULTS_DIR"

SMOKETEST_RESULTS="$RESULTS_DIR/straggler_trials.csv"
SPEEDUP_RESULTS="$RESULTS_DIR/speedup_results.csv"
SIGNIFICANCE_JSON="$RESULTS_DIR/significance.json"
SUMMARY_TXT="$RESULTS_DIR/summary.txt"

echo "========================================================"
echo " StragglAR Test Suite"
echo "========================================================"
echo " GPUs             : $GPUS  (${N_GPUS} total)"
echo " Significance runs: $N_TRIALS  (${ITERS_PER_TRIAL} GEMM iters each)"
echo " AllReduce iters  : $ALLREDUCE_ITERS  per benchmark point"
echo " Buffer           : $BUFFER_BYTES bytes  ($((BUFFER_BYTES/1024/1024)) MiB)"
echo " Binary           : $BINARY_PATH"
echo " Results dir      : $RESULTS_DIR"
echo "========================================================"
echo ""

# ===========================================================================
# PHASE 1 — Run N_TRIALS smoketest trials
# ===========================================================================
echo "--- Phase 1: Straggler detection  ($N_TRIALS trials) ---"
echo ""
echo "trial,straggler_gpu,delta_ms" > "$SMOKETEST_RESULTS"

TRIAL=0
FAIL_STREAK=0
START="$(date +%s)"

while [ "$TRIAL" -lt "$N_TRIALS" ]; do
    RAW="$(
        cd "$PROJECT_ROOT" && \
        "$PY" -m stragglar.smoketest.smoketest \
            --gpus $GPUS \
            --iters "$ITERS_PER_TRIAL" \
            --seed  "$TRIAL" \
            2>/dev/null \
        | grep "^STRAGGLER_REPORT=" || true
    )"

    if [ -z "$RAW" ]; then
        FAIL_STREAK=$((FAIL_STREAK + 1))
        [ "$FAIL_STREAK" -ge 5 ] && { echo "ERROR: smoketest failed 5 times in a row." >&2; exit 1; }
        echo "  [trial $((TRIAL+1))] WARNING: no output — retrying" >&2
        continue
    fi

    FAIL_STREAK=0
    REPORT="${RAW#STRAGGLER_REPORT=}"
    GPU_ID="$(echo "$REPORT" | cut -d: -f2)"
    DELTA_MS="$(echo "$REPORT" | cut -d: -f3)"
    echo "$((TRIAL+1)),${GPU_ID},${DELTA_MS}" >> "$SMOKETEST_RESULTS"
    TRIAL=$((TRIAL + 1))

    if [ $((TRIAL % 100)) -eq 0 ] || [ "$TRIAL" -eq "$N_TRIALS" ]; then
        NOW="$(date +%s)"
        ELAPSED=$((NOW - START))
        RATE=$(awk "BEGIN { printf \"%.1f\", $TRIAL / ($ELAPSED + 0.001) }")
        ETA=$(awk "BEGIN { printf \"%.0f\", ($N_TRIALS - $TRIAL) / ($RATE + 0.001) }")
        printf "  [%4d / %4d]  %.1f trials/s  ETA ~%ds\n" "$TRIAL" "$N_TRIALS" "$RATE" "$ETA"
    fi
done

echo ""
ELAPSED=$(( $(date +%s) - START ))
echo "Phase 1 complete in ${ELAPSED}s.  Saved: $SMOKETEST_RESULTS"
echo ""

# ===========================================================================
# PHASE 2 — Significance analysis + extract delta quantiles
# ===========================================================================
echo "--- Phase 2: Significance analysis ---"
echo ""

QUANTILES="$("$PY" - "$SMOKETEST_RESULTS" "$N_GPUS" "$N_TRIALS" "$SIGNIFICANCE_JSON" << 'PYEOF'
import sys, csv, math, statistics, json

results_file    = sys.argv[1]
n_gpus          = int(sys.argv[2])
n_trials        = int(sys.argv[3])
significance_out = sys.argv[4]

counts = {i: 0 for i in range(n_gpus)}
deltas = []
per_gpu_deltas = {i: [] for i in range(n_gpus)}

with open(results_file) as f:
    for row in csv.DictReader(f):
        g = int(row["straggler_gpu"])
        d = float(row["delta_ms"])
        counts[g] += 1
        deltas.append(d)
        per_gpu_deltas[g].append(d)

total     = sum(counts.values())
p_chance  = 1.0 / n_gpus
top_gpu   = max(counts, key=counts.__getitem__)
top_count = counts[top_gpu]

print(f"========================================================", file=sys.stderr)
print(f" Detection Counts  (n={total})", file=sys.stderr)
print(f"========================================================", file=sys.stderr)
print(f" {'GPU':>4}  {'Detections':>12}  {'Rate':>8}  {'Avg delta(ms)':>14}", file=sys.stderr)
print(f" {'-'*4}  {'-'*12}  {'-'*8}  {'-'*14}", file=sys.stderr)
for g in range(n_gpus):
    avg_d = statistics.mean(per_gpu_deltas[g]) if per_gpu_deltas[g] else 0.0
    print(f" {g:>4}  {counts[g]:>12}  {counts[g]/total:>8.1%}  {avg_d:>14.2f}", file=sys.stderr)
print(file=sys.stderr)

expected  = total / n_gpus
chi2      = sum((counts[g] - expected) ** 2 / expected for g in range(n_gpus))
df        = n_gpus - 1
CV_TABLE  = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}
critical  = CV_TABLE.get(df, 7.815)

p_hat = top_count / total
z     = (p_hat - p_chance) / math.sqrt(p_chance * (1 - p_chance) / total)
p_binom_approx = 0.5 * math.erfc(z / math.sqrt(2))

try:
    from scipy.stats import chisquare, binomtest
    _, chi2_p = chisquare([counts[g] for g in range(n_gpus)], [expected]*n_gpus)
    p_binom   = binomtest(top_count, total, p_chance, alternative="greater").pvalue
    exact     = True
except ImportError:
    chi2_p  = None
    p_binom = p_binom_approx
    exact   = False

chi2_sig  = (chi2_p < 0.05) if chi2_p is not None else (chi2 > critical)
binom_sig = p_binom < 0.05

print(f" Chi-squared  chi2={chi2:.3f}  (critical df={df}: {critical})", file=sys.stderr)
if exact:
    print(f"              p={chi2_p:.4e}  {'SIGNIFICANT' if chi2_sig else 'NOT SIGNIFICANT'}", file=sys.stderr)
else:
    print(f"              {'chi2 > critical → SIGNIFICANT' if chi2_sig else 'NOT SIGNIFICANT'}  (install scipy for exact p)", file=sys.stderr)

print(f" Binomial     GPU {top_gpu}: {top_count}/{total} ({p_hat:.1%})", file=sys.stderr)
if exact:
    print(f"              p={p_binom:.4e}  {'SIGNIFICANT' if binom_sig else 'NOT SIGNIFICANT'}", file=sys.stderr)
else:
    print(f"              z={z:.3f}  p≈{p_binom:.4e}  {'SIGNIFICANT' if binom_sig else 'NOT SIGNIFICANT'}", file=sys.stderr)
print(file=sys.stderr)

# Delta quantiles
deltas_sorted = sorted(deltas)

def quantile(s, q):
    idx = q * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])

q_stats = {
    "min":  deltas_sorted[0],
    "p25":  quantile(deltas_sorted, 0.25),
    "p50":  quantile(deltas_sorted, 0.50),
    "p75":  quantile(deltas_sorted, 0.75),
    "max":  deltas_sorted[-1],
    "mean": statistics.mean(deltas_sorted),
    "std":  statistics.stdev(deltas_sorted) if len(deltas_sorted) > 1 else 0.0,
}

# Save significance JSON
sig_data = {
    "n_trials":           total,
    "n_gpus":             n_gpus,
    "counts":             {str(g): counts[g] for g in range(n_gpus)},
    "per_gpu_avg_delta":  {str(g): round(statistics.mean(per_gpu_deltas[g]), 3) if per_gpu_deltas[g] else 0.0 for g in range(n_gpus)},
    "top_gpu":            top_gpu,
    "chi2":               round(chi2, 4),
    "chi2_p":             round(chi2_p, 6) if chi2_p is not None else None,
    "chi2_significant":   chi2_sig,
    "binomial_p":         round(p_binom, 6),
    "binomial_significant": binom_sig,
    "exact_pvalues":      exact,
    "delta_stats":        {k: round(v, 3) for k, v in q_stats.items()},
}
with open(significance_out, "w") as f:
    json.dump(sig_data, f, indent=2)

# Print quantile key=value to stdout for the shell to consume
for k, v in q_stats.items():
    print(f"{k}={v:.3f}")
PYEOF
)"

# Parse quantile values
get_q() { echo "$QUANTILES" | grep "^$1=" | cut -d= -f2; }
D_MIN="$(get_q min)";  D_P25="$(get_q p25)";  D_P50="$(get_q p50)"
D_P75="$(get_q p75)";  D_MAX="$(get_q max)";  D_MEAN="$(get_q mean)"

echo "Straggler delta statistics from $N_TRIALS real trials:"
printf "  min=%.1f ms  p25=%.1f ms  median=%.1f ms  p75=%.1f ms  max=%.1f ms  mean=%.1f ms\n" \
    "$D_MIN" "$D_P25" "$D_P50" "$D_P75" "$D_MAX" "$D_MEAN"
echo "Saved: $SIGNIFICANCE_JSON"
echo ""

# ===========================================================================
# PHASE 3 — Speedup benchmark at five representative delay points
# ===========================================================================
echo "--- Phase 3: AllReduce speedup benchmark ---"
echo ""
echo "  Using real straggler delays measured in Phase 1."
echo "  Benchmarking ring_allreduce vs StragglAR at: min / p25 / median / p75 / max"
echo ""

echo "label,sleep_ms,ring_ms,sar_ms,speedup" > "$SPEEDUP_RESULTS"

run_bench_point() {
    local label="$1"
    local sleep_ms="$2"

    printf "  %-10s  (sleep=%.1f ms)  ring..." "$label" "$sleep_ms"

    RING_CSV="$(
        cd "$PROJECT_ROOT" && \
        "$PY" "$RING_SCRIPT" "$BUFFER_BYTES" "$ALLREDUCE_ITERS" "$sleep_ms" 2>/dev/null
    )"
    RING_MEAN="$(echo "$RING_CSV" | awk -F, 'NR>1 && $1=="ring" {sum+=$5; n++} END {printf "%.3f", sum/n}')"

    printf "  done.  sar..."

    SAR_CSV="$(
        cd "$PROJECT_ROOT" && \
        "$MPI" -n 4 \
            -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
            "$BINARY_PATH" "$BUFFER_BYTES" stragglar "$ALLREDUCE_ITERS" "$sleep_ms" \
            2>/dev/null
    )"
    SAR_MEAN="$(echo "$SAR_CSV" | awk -F, '
        NR>1 && $1=="stragglar" { iter=$3; times[iter] = times[iter] "," $5; n[iter]++ }
        END {
            total=0; count=0
            for (it in times) {
                split(times[it], arr, ",")
                asort(arr)
                mid = int(length(arr)/2)+1
                total += arr[mid]; count++
            }
            printf "%.3f", total/count
        }
    ')"

    SPEEDUP="$(awk "BEGIN { printf \"%.4f\", $RING_MEAN / $SAR_MEAN }")"

    printf "  done.  ring=%.1f ms  sar=%.1f ms  speedup=%sx\n" \
        "$RING_MEAN" "$SAR_MEAN" "$SPEEDUP"

    echo "$label,$sleep_ms,$RING_MEAN,$SAR_MEAN,$SPEEDUP" >> "$SPEEDUP_RESULTS"
}

run_bench_point "min"    "$D_MIN"
run_bench_point "p25"    "$D_P25"
run_bench_point "median" "$D_P50"
run_bench_point "p75"    "$D_P75"
run_bench_point "max"    "$D_MAX"

echo ""
echo "Saved: $SPEEDUP_RESULTS"
echo ""

# ===========================================================================
# PHASE 4 — Final combined report
# ===========================================================================
"$PY" - "$SMOKETEST_RESULTS" "$SPEEDUP_RESULTS" "$SIGNIFICANCE_JSON" "$N_GPUS" "$N_TRIALS" "$SUMMARY_TXT" << 'PYEOF3'
import sys, csv, json, statistics

smoketest_csv   = sys.argv[1]
speedup_csv     = sys.argv[2]
sig_json        = sys.argv[3]
n_gpus          = int(sys.argv[4])
n_trials        = int(sys.argv[5])
summary_out     = sys.argv[6]

with open(sig_json) as f:
    sig = json.load(f)

speedup_rows = []
with open(speedup_csv) as f:
    for row in csv.DictReader(f):
        speedup_rows.append(row)

speedups  = [float(r["speedup"]) for r in speedup_rows]
avg_sp    = statistics.mean(speedups)
best_sp   = max(speedups)
best_row  = speedup_rows[speedup_rows.index(max(speedup_rows, key=lambda r: float(r["speedup"])))]

lines = []
lines.append("========================================================")
lines.append(" FINAL REPORT")
lines.append("========================================================")
lines.append(f" Trials run          : {n_trials}")
lines.append(f" Most-detected GPU   : GPU {sig['top_gpu']}  "
             f"({sig['counts'][str(sig['top_gpu'])]/n_trials:.1%} of trials)")
chi_label = "SIGNIFICANT" if sig["chi2_significant"] else "not significant"
bi_label  = "SIGNIFICANT" if sig["binomial_significant"] else "not significant"
lines.append(f" Chi-squared test    : {chi_label}  "
             f"(chi2={sig['chi2']:.3f}"
             + (f", p={sig['chi2_p']:.2e}" if sig['chi2_p'] is not None else "") + ")")
lines.append(f" Binomial test       : {bi_label}  "
             f"(p={sig['binomial_p']:.2e})")
lines.append("")
lines.append(f" {'Point':<10}  {'Sleep(ms)':>10}  {'Ring(ms)':>10}  {'SAR(ms)':>10}  {'Speedup':>9}")
lines.append(f" {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}")
for r in speedup_rows:
    lines.append(f" {r['label']:<10}  {float(r['sleep_ms']):>10.1f}  "
                 f"{float(r['ring_ms']):>10.3f}  {float(r['sar_ms']):>10.3f}  "
                 f"{float(r['speedup']):>8.4f}x")
lines.append("")
lines.append(f" Average speedup : {avg_sp:.4f}x  (across all 5 delay points)")
lines.append(f" Best run        : {best_sp:.4f}x  "
             f"(at {best_row['label']} delay = {float(best_row['sleep_ms']):.1f} ms)")
lines.append("========================================================")

report = "\n".join(lines)
print(report)
with open(summary_out, "w") as f:
    f.write(report + "\n")
PYEOF3

echo ""
echo "Saved: $SUMMARY_TXT"
echo ""

# ===========================================================================
# PHASE 5 — Generate plots
# ===========================================================================
if [ -f "$PLOT_SCRIPT" ]; then
    echo "--- Phase 5: Generating plots ---"
    echo ""
    "$PY" "$PLOT_SCRIPT" "$RESULTS_DIR" && \
        echo "Plots saved to: $RESULTS_DIR/plots/" || \
        echo "WARNING: plot generation failed (check matplotlib is installed: pip install matplotlib numpy)"
    echo ""
fi

echo "========================================================"
echo " All outputs saved to: $RESULTS_DIR"
echo "   straggler_trials.csv  — raw trial data (Phase 1)"
echo "   significance.json     — statistical test results (Phase 2)"
echo "   speedup_results.csv   — AllReduce benchmark (Phase 3)"
echo "   summary.txt           — human-readable report (Phase 4)"
echo "   plots/                — graphs (Phase 5)"
echo "========================================================"
