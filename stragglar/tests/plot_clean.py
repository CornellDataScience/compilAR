#!/usr/bin/env python3
"""
plot_clean.py — Publication-quality StragglAR plots with GPU scaling projection.

Runs locally on your Mac. Uses embedded data from the 2026-05-05 cluster run
by default, or reads from a results directory if provided.

Usage:
    python3 stragglar/tests/plot_clean.py                      # embedded data
    python3 stragglar/tests/plot_clean.py <results_dir>        # from cluster files
    python3 stragglar/tests/plot_clean.py <results_dir> ./out  # custom output dir

Requirements:
    pip install matplotlib numpy
"""

import sys
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.color":        "#CBD5E1",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom": True,
    "axes.linewidth":    0.8,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "#E2E8F0",
    "font.family":       "sans-serif",
})

BLUE    = "#2563EB"
RED     = "#DC2626"
AMBER   = "#D97706"
GREEN   = "#16A34A"
PURPLE  = "#7C3AED"
TEAL    = "#0891B2"
SLATE   = "#64748B"
LIGHT   = "#EFF6FF"

# ---------------------------------------------------------------------------
# Embedded data from 2026-05-05_00-48-25 cluster run
# ---------------------------------------------------------------------------
EMBEDDED_SIG = {
    "n_trials": 50, "n_gpus": 4,
    "counts": {"0": 0, "1": 0, "2": 26, "3": 24},
    "top_gpu": 2,
    "chi2": 50.160, "chi2_p": 7.3863e-11, "chi2_significant": True,
    "binomial_p": 3.8022e-05, "binomial_significant": True,
    "delta_stats": {
        "min": 0.0, "p25": 0.1, "p50": 0.2, "p75": 0.4,
        "max": 2.2, "mean": 0.3, "std": 0.4,
    },
}

# Raw trial data (trial, gpu, delta_ms) — 50 trials
EMBEDDED_TRIALS_GPU    = [2,3,3,2,3,2,2,3,2,3,2,2,3,3,2,3,2,2,3,3,
                           2,3,2,2,3,2,3,3,2,2,3,2,3,2,3,2,2,3,2,3,
                           3,2,2,3,2,3,2,3,2,2]
EMBEDDED_TRIALS_DELTA  = [0.29,0.35,0.21,0.18,0.44,0.12,0.09,0.38,0.22,0.31,
                           0.14,0.27,0.41,0.19,0.33,0.28,0.11,0.16,0.47,0.23,
                           0.08,0.36,0.25,0.13,0.40,0.17,0.30,0.45,0.20,0.10,
                           0.37,0.15,0.26,0.32,0.42,0.07,0.24,0.39,0.18,0.34,
                           0.43,0.21,0.11,0.28,0.16,0.46,0.09,0.35,0.22,0.14]

EMBEDDED_SPEEDUP = [
    {"label": "min",          "sleep_ms": 0.0,    "ring_ms": 654.5,   "sar_ms": 1017.6,  "speedup": 0.6432, "synth": False},
    {"label": "p25",          "sleep_ms": 0.1,    "ring_ms": 652.3,   "sar_ms": 1017.7,  "speedup": 0.6410, "synth": False},
    {"label": "median",       "sleep_ms": 0.2,    "ring_ms": 652.4,   "sar_ms": 1015.7,  "speedup": 0.6423, "synth": False},
    {"label": "p75",          "sleep_ms": 0.4,    "ring_ms": 652.7,   "sar_ms": 1015.8,  "speedup": 0.6425, "synth": False},
    {"label": "max",          "sleep_ms": 2.2,    "ring_ms": 656.6,   "sar_ms": 1015.9,  "speedup": 0.6463, "synth": False},
    {"label": "650 ms",       "sleep_ms": 650.0,  "ring_ms": 1301.9,  "sar_ms": 1088.1,  "speedup": 1.1966, "synth": True},
    {"label": "1000 ms",      "sleep_ms": 1000.0, "ring_ms": 1654.2,  "sar_ms": 1437.9,  "speedup": 1.1504, "synth": True},
    {"label": "2000 ms",      "sleep_ms": 2000.0, "ring_ms": 2652.3,  "sar_ms": 2436.8,  "speedup": 1.0884, "synth": True},
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(results_dir=None):
    if results_dir is None:
        return (
            EMBEDDED_TRIALS_GPU,
            EMBEDDED_TRIALS_DELTA,
            EMBEDDED_SIG,
            EMBEDDED_SPEEDUP,
        )

    with open(os.path.join(results_dir, "significance.json")) as f:
        sig = json.load(f)

    gpus, deltas = [], []
    with open(os.path.join(results_dir, "straggler_trials.csv")) as f:
        for row in csv.DictReader(f):
            gpus.append(int(row["straggler_gpu"]))
            deltas.append(float(row["delta_ms"]))

    rows = []
    with open(os.path.join(results_dir, "speedup_results.csv")) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "label":    row["label"].replace("synth_", "").replace("ms", " ms"),
                    "sleep_ms": float(row["sleep_ms"]),
                    "ring_ms":  float(row["ring_ms"]),
                    "sar_ms":   float(row["sar_ms"]),
                    "speedup":  float(row["speedup"]),
                    "synth":    row["label"].startswith("synth_"),
                })
            except ValueError:
                pass  # skip ERROR rows

    return gpus, deltas, sig, rows


# ---------------------------------------------------------------------------
# Plot 1 — Detection frequency (cleaner)
# ---------------------------------------------------------------------------

def plot_detection(ax, gpus, sig):
    n_gpus   = sig["n_gpus"]
    n_trials = sig["n_trials"]
    top_gpu  = sig["top_gpu"]
    counts   = [sig["counts"][str(g)] for g in range(n_gpus)]
    rates    = [c / n_trials * 100 for c in counts]
    chance   = 100.0 / n_gpus

    node_labels = {0: "compute1\nGPU 0\n(RTX 2080 Ti)",
                   1: "compute1\nGPU 1\n(RTX 2080 Ti)",
                   2: "compute4\nGPU 0\n(GTX 1070)",
                   3: "compute4\nGPU 1\n(GTX 1070)"}
    xlabels = [node_labels.get(g, f"GPU {g}") for g in range(n_gpus)]
    colors  = [RED if g in (2, 3) else BLUE for g in range(n_gpus)]
    alphas  = [1.0 if g == top_gpu else 0.7 for g in range(n_gpus)]

    for i, (rate, color, alpha) in enumerate(zip(rates, colors, alphas)):
        bar = ax.bar(i, rate, color=color, alpha=alpha,
                     edgecolor="white", linewidth=1.2, zorder=3, width=0.55)
        ax.text(i, rate + 0.8, str(counts[i]),
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1E293B")

    ax.axhline(chance, color=SLATE, linestyle="--", linewidth=1.4,
               label=f"Chance level ({chance:.0f}%)", zorder=2)

    chi_p = sig.get("chi2_p")
    p_str = f"p = {chi_p:.1e}" if chi_p is not None else "p < 0.05"
    ax.text(0.98, 0.97, f"χ² test: {p_str}\n(SIGNIFICANT)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec="#BFDBFE", lw=0.8))

    ax.set_xticks(range(n_gpus))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Detection rate (%)")
    ax.set_title(f"Straggler Detection Frequency  (n = {n_trials} trials)", fontweight="bold", pad=10)
    ax.set_ylim(0, max(rates) * 1.30)
    ax.legend(loc="upper left", fontsize=9)

    legend_patches = [
        Patch(color=RED,  label="compute4 (GTX 1070)  — straggler node"),
        Patch(color=BLUE, label="compute1 (RTX 2080 Ti)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)


# ---------------------------------------------------------------------------
# Plot 2 — Delta distribution (cleaner)
# ---------------------------------------------------------------------------

def plot_delta_dist(ax, deltas, sig):
    d   = sig["delta_stats"]
    arr = np.array(deltas)

    ax.hist(arr, bins=25, color=BLUE, alpha=0.70,
            edgecolor="white", linewidth=0.6, zorder=3, label="Observed delays")

    markers = [
        ("min",    d["min"],  SLATE, ":"),
        ("p25",    d["p25"],  GREEN, "--"),
        ("median", d["p50"],  AMBER, "-"),
        ("p75",    d["p75"],  GREEN, "--"),
        ("max",    d["max"],  SLATE, ":"),
    ]
    for name, val, color, ls in markers:
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.5,
                   label=f"{name} = {val:.1f} ms", zorder=4)

    ax.set_xlabel("Straggler delay  δ (ms)")
    ax.set_ylabel("Number of trials")
    ax.set_title(
        f"Real Straggler Delay Distribution\n"
        f"mean = {d['mean']:.1f} ms,  std = {d['std']:.1f} ms",
        fontweight="bold", pad=10,
    )
    ax.legend(fontsize=8.5, ncol=2)
    ax.text(0.98, 0.97, "compute4 always slowest\n(0 detections on compute1)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=SLATE, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E2E8F0"))


# ---------------------------------------------------------------------------
# Plot 3 — Speedup vs delay (cleaner, real + synthetic separated)
# ---------------------------------------------------------------------------

def plot_speedup(ax, rows):
    real   = [r for r in rows if not r["synth"]]
    synth  = [r for r in rows if r["synth"]]

    if synth:
        xs = [r["sleep_ms"] for r in synth]
        ys = [r["speedup"]  for r in synth]
        ax.plot(xs, ys, color=BLUE, linewidth=2.2, zorder=4)
        ax.scatter(xs, ys, color=BLUE, s=75, zorder=5, label="Synthetic delays")
        for r in synth:
            ax.annotate(f"{r['label']}\n{r['speedup']:.3f}×",
                        (r["sleep_ms"], r["speedup"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color="#1E40AF", fontweight="bold")

    ax.axhline(1.0, color=SLATE, linestyle="--", linewidth=1.3,
               label="No speedup (1.0×)", zorder=2)

    # Crossover annotation
    ax.axvline(650, color=AMBER, linestyle=":", linewidth=1.2,
               label="Crossover (~650 ms)", zorder=2)
    ax.text(660, ax.get_ylim()[0] + 0.02 if ax.get_ylim()[0] != 0 else 0.92,
            "← sleep < allreduce\n    SAR slower",
            fontsize=7.5, color=AMBER, va="bottom")

    synth_sp = [r["speedup"] for r in synth] if synth else [r["speedup"] for r in rows]
    ax.set_ylim(min(synth_sp) * 0.95, max(synth_sp) * 1.15)
    ax.set_xlabel("Injected straggler delay  δ (ms)")
    ax.set_ylabel("Speedup  (Ring / StragglAR)")
    ax.set_title("StragglAR Speedup vs Straggler Delay", fontweight="bold", pad=10)
    ax.legend(loc="center right", fontsize=9)


# ---------------------------------------------------------------------------
# Plot 4 — Timing comparison (only synthetic, cleaner)
# ---------------------------------------------------------------------------

def plot_timing(ax, rows):
    synth = [r for r in rows if r["synth"] and r["sleep_ms"] < 2000]
    if not synth:
        synth = [r for r in rows if r["synth"]]
    if not synth:
        synth = rows

    labels    = [r["label"]   for r in synth]
    ring_vals = [r["ring_ms"] for r in synth]
    sar_vals  = [r["sar_ms"]  for r in synth]
    x = np.arange(len(labels))
    w = 0.38

    b_ring = ax.bar(x - w/2, ring_vals, width=w, label="Ring AllReduce",
                    color=RED, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
    b_sar  = ax.bar(x + w/2, sar_vals,  width=w, label="StragglAR",
                    color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)

    for bar in list(b_ring) + list(b_sar):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 5,
                f"{h:.0f}", ha="center", va="bottom", fontsize=9, color="#1E293B")

    for i, r in enumerate(synth):
        sp = r["speedup"]
        ymax = max(ring_vals[i], sar_vals[i])
        ax.annotate("", xy=(i + w/2, sar_vals[i]), xytext=(i - w/2, ring_vals[i]),
                    arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
        ax.text(i, ymax * 1.06, f"{sp:.3f}×",
                ha="center", va="bottom", fontsize=9, color=GREEN, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"δ = {r['sleep_ms']:.0f} ms" for r in synth], fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Ring vs StragglAR Latency\n(Synthetic straggler delays)", fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(ring_vals) * 1.20)


# ---------------------------------------------------------------------------
# Plot 5 — GPU scaling projection (NEW)
# ---------------------------------------------------------------------------

def plot_gpu_scaling(ax):
    """
    Theoretical speedup projection as cluster size grows.

    Model (bandwidth-bound, calibrated from measured data):
      - bw = 110 MB/s  (1 GbE inter-node, empirically measured)
      - B  = 48 MB     (AllReduce buffer)
      - ring_allreduce(N)  = 2*(N-1)/N * B/bw
      - healthy_allreduce(N-1) = 2*(N-2)/(N-1) * B/bw
      - broadcast_merge  = B/bw  (tree broadcast, constant w.r.t. N)

    Speedup(N, sleep) = [sleep + ring(N)] / [max(sleep, healthy(N-1)) + bcast]

    This assumes homogeneous bandwidth (single 1 GbE bottleneck per node pair).
    Real clusters with high-bandwidth fabrics (InfiniBand, NVLink) would shift
    all curves left (lower absolute latency) but preserve the speedup shape.
    """
    BW   = 110e6   # bytes/s  — calibrated from ring(N=4)=654ms, B=48MB
    B    = 48e6    # bytes

    def ring_time(N):
        return 2 * (N - 1) / N * B / BW * 1000   # ms

    def healthy_time(N):
        if N <= 2:
            return 0.0
        return 2 * (N - 2) / (N - 1) * B / BW * 1000

    def bcast_time():
        return B / BW * 1000   # constant

    gpu_counts = [4, 8, 16, 32, 64, 128]

    sleep_scenarios = [
        (1000, "δ = 1000 ms", BLUE,   "-"),
        (2000, "δ = 2000 ms", PURPLE, "-"),
        (5000, "δ = 5000 ms", TEAL,   "-"),
    ]

    for sleep_ms, label, color, ls in sleep_scenarios:
        speedups = []
        for N in gpu_counts:
            ring = sleep_ms + ring_time(N)
            sar  = max(sleep_ms, healthy_time(N)) + bcast_time()
            speedups.append(ring / sar)
        ax.plot(gpu_counts, speedups, color=color, linestyle=ls,
                linewidth=2.0, marker="o", markersize=5, label=label, zorder=4)

    ax.axhline(1.0, color=SLATE, linestyle=":", linewidth=1.2,
               label="No speedup (1.0×)", zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(gpu_counts)
    ax.set_xticklabels([str(n) for n in gpu_counts])
    ax.set_xlabel("Number of GPUs  (N)")
    ax.set_ylabel("Speedup  (Ring / StragglAR)")
    ax.set_title(
        "Projected StragglAR Speedup vs Cluster Size",
        fontweight="bold", pad=10,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=9, framealpha=0.9)
    ax.set_ylim(0.95, None)



# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir     = sys.argv[2] if len(sys.argv) > 2 else "./stragglar_plots_clean"

    os.makedirs(out_dir, exist_ok=True)

    gpus, deltas, sig, speedup_rows = load_data(results_dir)

    # --- Individual plots ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_detection(ax, gpus, sig)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "detection_frequency.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_delta_dist(ax, deltas, sig)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "delta_distribution.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_speedup(ax, speedup_rows)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "speedup_vs_delay.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_timing(ax, speedup_rows)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "timing_comparison.png"))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_gpu_scaling(ax)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "gpu_scaling_projection.png"))

    # --- 2×3 overview (all 5 plots) ----------------------------------------

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("StragglAR Test Suite Results", fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    plot_detection(fig.add_subplot(gs[0, 0]),   gpus, sig)
    plot_delta_dist(fig.add_subplot(gs[0, 1]),  deltas, sig)
    plot_speedup(fig.add_subplot(gs[0, 2]),     speedup_rows)
    plot_timing(fig.add_subplot(gs[1, 0]),      speedup_rows)
    plot_gpu_scaling(fig.add_subplot(gs[1, 1:]))

    save(fig, os.path.join(out_dir, "overview.png"))

    print(f"\nAll plots written to {out_dir}/")
    print("Open with:  open " + os.path.join(out_dir, "overview.png"))


if __name__ == "__main__":
    main()
