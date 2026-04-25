#!/bin/bash
set -euo pipefail

# launch.sh — Orchestrator for a StragglAR run (single- or multi-node).
#
# Usage: ./launch.sh <N> <binary> [binary_args...]
#   e.g.: ./launch.sh 4 ./stragglar_4gpu 1073741824 stragglar 10 -1
#
# Pipeline:
#   1. Determine the participating nodes (from SLURM env, $HOSTFILE, or localhost)
#   2. Run smoketester once per node via mpirun --map-by ppr:1:node
#   3. Aggregate the per-node STRAGGLER_REPORT lines via pick_global_straggler.py
#   4. Build a per-node rank-to-GPU mapping. The straggler node's mapping swaps
#      local-rank K-1 with the straggler GPU id, so MPI rank N-1 (which lands as
#      local rank K-1 on the last node in the hostfile) binds to the slow GPU.
#   5. Reorder the hostfile so the straggler node is last; launch via mpirun.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <N> <binary> [binary_args...]" >&2
    echo "  e.g.: $0 4 ./stragglar_4gpu 1073741824 stragglar 10 -1" >&2
    exit 1
fi

N="$1"
BINARY="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$SCRIPT_DIR/rank_wrapper.sh"
PICKER="$SCRIPT_DIR/smoketest/pick_global_straggler.py"

if [ ! -x "$WRAPPER" ]; then
    echo "Error: $WRAPPER not found or not executable" >&2
    echo "Run: chmod +x $WRAPPER" >&2
    exit 1
fi
if [ ! -f "$PICKER" ]; then
    echo "Error: $PICKER not found" >&2
    exit 1
fi

# --- Step 1: assemble hostfile -----------------------------------------------
HOSTFILE_TEMP="$(mktemp)"
trap 'rm -f "$HOSTFILE_TEMP" "$HOSTFILE_TEMP.reordered"' EXIT

if [ -n "${SLURM_NODELIST:-}" ]; then
    scontrol show hostnames "$SLURM_NODELIST" > "$HOSTFILE_TEMP"
elif [ -n "${HOSTFILE:-}" ]; then
    cp "$HOSTFILE" "$HOSTFILE_TEMP"
else
    hostname -s > "$HOSTFILE_TEMP"
fi

NUM_NODES=$(wc -l < "$HOSTFILE_TEMP")
NUM_NODES=$((NUM_NODES + 0))  # strip whitespace

if [ "$NUM_NODES" -lt 1 ]; then
    echo "[launch] Error: hostfile is empty" >&2
    exit 1
fi

# Require uniform GPU count per node so the rank-N-1-on-last-node trick holds.
K=$((N / NUM_NODES))
if [ $((K * NUM_NODES)) -ne "$N" ]; then
    echo "[launch] Error: N=$N is not evenly divisible by NUM_NODES=$NUM_NODES" >&2
    echo "[launch] (heterogeneous node sizes are not yet supported)" >&2
    exit 1
fi

echo "[launch] $NUM_NODES node(s), $K GPU(s) per node, $N total ranks" >&2

# --- Step 2: per-node smoketester via mpirun --------------------------------
echo "[launch] Running smoketester on each node..." >&2

# --map-by ppr:1:node = one process per node. Each instance runs the
# smoketester on that node's GPUs and emits STRAGGLER_REPORT=host:gpu:delta.
SMOKE_OUT=$(
    mpirun -n "$NUM_NODES" \
        --hostfile "$HOSTFILE_TEMP" \
        --map-by ppr:1:node \
        python -m stragglar.smoketest.smoketest --iters 15 \
        | tee /dev/stderr
) || {
    echo "[launch] Error: smoketester invocation failed" >&2
    exit 1
}

# --- Step 3: pick the global straggler --------------------------------------
PICK_OUT=$(echo "$SMOKE_OUT" | python "$PICKER") || {
    echo "[launch] Error: pick_global_straggler failed" >&2
    exit 1
}

STRAGGLER_HOSTNAME=$(echo "$PICK_OUT" | grep -E '^STRAGGLER_HOSTNAME=' | head -1 | cut -d= -f2)
STRAGGLER_GPU=$(echo "$PICK_OUT"      | grep -E '^STRAGGLER_GPU=[0-9]+$' | head -1 | cut -d= -f2)

if [ -z "$STRAGGLER_HOSTNAME" ] || [ -z "$STRAGGLER_GPU" ]; then
    echo "[launch] Error: picker did not produce STRAGGLER_HOSTNAME / STRAGGLER_GPU" >&2
    exit 1
fi

if [ "$STRAGGLER_GPU" -ge "$K" ] || [ "$STRAGGLER_GPU" -lt 0 ]; then
    echo "[launch] Error: straggler GPU $STRAGGLER_GPU out of range [0, $K)" >&2
    exit 1
fi

echo "[launch] Global straggler: $STRAGGLER_HOSTNAME, GPU $STRAGGLER_GPU" >&2

# --- Step 4: reorder hostfile so straggler node is last ----------------------
grep -v "^${STRAGGLER_HOSTNAME}$" "$HOSTFILE_TEMP" > "$HOSTFILE_TEMP.reordered"
echo "$STRAGGLER_HOSTNAME" >> "$HOSTFILE_TEMP.reordered"
mv "$HOSTFILE_TEMP.reordered" "$HOSTFILE_TEMP"

# --- Step 5: build per-node rank-to-GPU mapping ------------------------------
# Encoding: pipe-separated entries, each "hostname:csv". Indexed by local rank.
#   non-straggler node  -> identity     0,1,...,K-1
#   straggler node      -> swap index K-1 with straggler_gpu
RANK_TO_GPU=""
while IFS= read -r host; do
    if [ "$host" = "$STRAGGLER_HOSTNAME" ]; then
        nodemap=""
        for ((i=0; i<K; i++)); do
            if [ "$i" -eq $((K - 1)) ]; then
                nodemap="$nodemap,$STRAGGLER_GPU"
            elif [ "$i" -eq "$STRAGGLER_GPU" ]; then
                nodemap="$nodemap,$((K - 1))"
            else
                nodemap="$nodemap,$i"
            fi
        done
    else
        nodemap=""
        for ((i=0; i<K; i++)); do
            nodemap="$nodemap,$i"
        done
    fi
    nodemap="${nodemap#,}"

    if [ -z "$RANK_TO_GPU" ]; then
        RANK_TO_GPU="$host:$nodemap"
    else
        RANK_TO_GPU="$RANK_TO_GPU|$host:$nodemap"
    fi
done < "$HOSTFILE_TEMP"

echo "[launch] Mapping: $RANK_TO_GPU" >&2
export RANK_TO_GPU

# --- Step 6: launch ----------------------------------------------------------
exec mpirun -n "$N" \
    --hostfile "$HOSTFILE_TEMP" \
    --map-by slot \
    -x RANK_TO_GPU \
    "$WRAPPER" "$BINARY" "$@"
