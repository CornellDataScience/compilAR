#!/usr/bin/env python3
"""
plot_results.py — Generate graphs from a run_tests.sh results directory.

Reads the three output files written by run_tests.sh and produces four plots:

  1. detection_frequency.png  — bar chart of how often each GPU was detected
                                 as the straggler across all trials
  2. delta_distribution.png   — histogram of real straggler delay values,
                                 annotated with quartile markers
  3. speedup_vs_delay.png     — StragglAR speedup over Ring as straggler
                                 delay increases across the five benchmark points
  4. timing_comparison.png    — grouped bar chart comparing Ring vs StragglAR
                                 absolute latency at each delay point
  5. overview.png             — all four plots on one figure (2×2 grid)

Usage:
    python3 plot_results.py <results_dir>

Arguments:
    results_dir   path to the timestamped results directory created by run_tests.sh

Requirements:
    matplotlib >= 3.5
    numpy

Example:
    python3 plot_results.py stragglar/tests/results/2026-05-02_12-30-45
"""

import sys
import os
import csv
import json
import math

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — works without a display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required.  pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        11,
})

BLUE  = "#2563EB"
RED   = "#DC2626"
GREEN = "#16A34A"
GREY  = "#9CA3AF"
AMBER = "#D97706"

LABEL_ORDER = ["min", "p25", "median", "p75", "max"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trials(results_dir: str) -> tuple[list[int], list[float]]:
    path = os.path.join(results_dir, "straggler_trials.csv")
    gpus, deltas = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gpus.append(int(row["straggler_gpu"]))
            deltas.append(float(row["delta_ms"]))
    return gpus, deltas


def load_significance(results_dir: str) -> dict:
    path = os.path.join(results_dir, "significance.json")
    with open(path) as f:
        return json.load(f)


def load_speedup(results_dir: str) -> list[dict]:
    path = os.path.join(results_dir, "speedup_results.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "label":    row["label"],
                "sleep_ms": float(row["sleep_ms"]),
                "ring_ms":  float(row["ring_ms"]),
                "sar_ms":   float(row["sar_ms"]),
                "speedup":  float(row["speedup"]),
            })
    # Ensure consistent ordering
    order = {l: i for i, l in enumerate(LABEL_ORDER)}
    rows.sort(key=lambda r: order.get(r["label"], 99))
    return rows


# ---------------------------------------------------------------------------
# Plot 1 — Detection frequency
# ---------------------------------------------------------------------------

def plot_detection_frequency(ax, gpus: list[int], sig: dict):
    n_gpus   = sig["n_gpus"]
    n_trials = sig["n_trials"]
    top_gpu  = sig["top_gpu"]
    counts   = [sig["counts"][str(g)] for g in range(n_gpus)]
    rates    = [c / n_trials for c in counts]
    chance   = 1.0 / n_gpus

    colors = [RED if g == top_gpu else BLUE for g in range(n_gpus)]
    bars = ax.bar(
        [f"GPU {g}" for g in range(n_gpus)],
        [r * 100 for r in rates],
        color=colors, edgecolor="white", linewidth=0.8, zorder=3,
    )

    # Chance-level reference line
    ax.axhline(chance * 100, color=GREY, linestyle="--", linewidth=1.4,
               label=f"Chance ({chance:.0%})", zorder=2)

    # Count labels on top of bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=10, color="#374151",
        )

    # Significance annotation on top GPU
    chi_sig  = sig.get("chi2_significant", False)
    binom_sig = sig.get("binomial_significant", False)
    if chi_sig or binom_sig:
        ax.text(
            top_gpu, rates[top_gpu] * 100 + 2.5,
            "★ straggler",
            ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold",
        )

    ax.set_ylabel("Detection rate (%)")
    ax.set_title(f"Straggler Detection Frequency  (n = {n_trials} trials)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(rates) * 100 * 1.25)


# ---------------------------------------------------------------------------
# Plot 2 — Delta distribution
# ---------------------------------------------------------------------------

def plot_delta_distribution(ax, deltas: list[float], sig: dict):
    d = sig["delta_stats"]
    arr = np.array(deltas)

    n_bins = min(60, max(20, int(math.sqrt(len(arr)))))
    ax.hist(arr, bins=n_bins, color=BLUE, alpha=0.75, edgecolor="white",
            linewidth=0.5, zorder=3, label="trials")

    # Quartile markers
    for label, val, color, ls in [
        ("min",    d["min"],  GREY,  ":"),
        ("p25",    d["p25"],  GREEN, "--"),
        ("median", d["p50"],  AMBER, "-"),
        ("p75",    d["p75"],  GREEN, "--"),
        ("max",    d["max"],  GREY,  ":"),
    ]:
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.4,
                   label=f"{label} = {val:.1f} ms", zorder=4)

    ax.set_xlabel("Straggler delay  δ (ms)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Real Straggler Delay Distribution\n"
        f"mean = {d['mean']:.1f} ms,  std = {d['std']:.1f} ms",
        fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=2)


# ---------------------------------------------------------------------------
# Plot 3 — Speedup vs delay
# ---------------------------------------------------------------------------

def plot_speedup_vs_delay(ax, rows: list[dict]):
    x      = [r["sleep_ms"] for r in rows]
    y      = [r["speedup"]  for r in rows]
    labels = [r["label"]    for r in rows]

    ax.plot(x, y, color=BLUE, linewidth=2, zorder=3)
    ax.scatter(x, y, color=BLUE, s=70, zorder=4)

    # Label each point
    for xi, yi, lbl in zip(x, y, labels):
        ax.annotate(
            f"{lbl}\n{yi:.3f}×",
            (xi, yi),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=8.5, color="#1E40AF",
        )

    # Reference line at 1.0
    ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1.2,
               label="No speedup (1.0×)")

    avg_sp = sum(y) / len(y)
    ax.axhline(avg_sp, color=AMBER, linestyle="-.", linewidth=1.2,
               label=f"Average ({avg_sp:.3f}×)")

    ax.set_xlabel("Straggler delay  δ (ms)")
    ax.set_ylabel("Speedup  (Ring / StragglAR)")
    ax.set_title("StragglAR Speedup vs Straggler Delay", fontweight="bold")
    ax.legend(fontsize=9)

    y_max = max(y) * 1.20
    ax.set_ylim(0.9, y_max)


# ---------------------------------------------------------------------------
# Plot 4 — Timing comparison (grouped bar)
# ---------------------------------------------------------------------------

def plot_timing_comparison(ax, rows: list[dict]):
    labels    = [r["label"]    for r in rows]
    ring_vals = [r["ring_ms"]  for r in rows]
    sar_vals  = [r["sar_ms"]   for r in rows]
    x         = np.arange(len(labels))
    w         = 0.35

    b_ring = ax.bar(x - w / 2, ring_vals, width=w, label="Ring AllReduce",
                    color=RED,  alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
    b_sar  = ax.bar(x + w / 2, sar_vals,  width=w, label="StragglAR",
                    color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels
    for bar in list(b_ring) + list(b_sar):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=8.5, color="#374151",
        )

    # Sleep-ms secondary labels on x-axis
    sleep_labels = [f"{r['label']}\n({r['sleep_ms']:.1f} ms)" for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(sleep_labels, fontsize=9)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Ring vs StragglAR Absolute Latency", fontweight="bold")
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    for fname in ("straggler_trials.csv", "significance.json", "speedup_results.csv"):
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            print(f"Error: {fpath} not found — run run_tests.sh first", file=sys.stderr)
            sys.exit(1)

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    gpus, deltas = load_trials(results_dir)
    sig          = load_significance(results_dir)
    speedup_rows = load_speedup(results_dir)

    # --- Individual plots ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_detection_frequency(ax, gpus, sig)
    save(fig, os.path.join(plots_dir, "detection_frequency.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_delta_distribution(ax, deltas, sig)
    save(fig, os.path.join(plots_dir, "delta_distribution.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_speedup_vs_delay(ax, speedup_rows)
    save(fig, os.path.join(plots_dir, "speedup_vs_delay.png"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_timing_comparison(ax, speedup_rows)
    save(fig, os.path.join(plots_dir, "timing_comparison.png"))

    # --- 2×2 overview -------------------------------------------------------

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("StragglAR Test Suite Results", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    plot_detection_frequency(fig.add_subplot(gs[0, 0]), gpus, sig)
    plot_delta_distribution(fig.add_subplot(gs[0, 1]),  deltas, sig)
    plot_speedup_vs_delay(fig.add_subplot(gs[1, 0]),    speedup_rows)
    plot_timing_comparison(fig.add_subplot(gs[1, 1]),   speedup_rows)

    save(fig, os.path.join(plots_dir, "overview.png"))

    print(f"\nAll plots written to {plots_dir}/")


if __name__ == "__main__":
    main()
