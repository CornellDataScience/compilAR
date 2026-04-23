#!/bin/bash
set -euo pipefail

# Orchestrator for a StragglAR run

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

if [ ! -x "$WRAPPER" ]; then
    echo "Error: $WRAPPER not found or not executable" >&2
    echo "Run: chmod +x $WRAPPER" >&2
    exit 1
fi

# Detect straggler
GPU_LIST="$(seq 0 $((N - 1)))"

echo "[launch] Running smoketester on $N GPUs..." >&2

set +e
python -m stragglar.smoketest.smoketest --gpus $GPU_LIST --iters 15
STRAGGLER_GPU=$?
set -e

if [ "$STRAGGLER_GPU" -ge "$N" ] || [ "$STRAGGLER_GPU" -lt 0 ]; then
    echo "[launch] Error: smoketest returned invalid GPU id: $STRAGGLER_GPU" >&2
    exit 1
fi

echo "[launch] Straggler GPU: $STRAGGLER_GPU" >&2

# Build GPU mapping
MAPPING=""
gpu_idx=0
for rank in $(seq 0 $((N - 2))); do
    # Skip the straggler GPU when assigning to non-straggler ranks
    while [ "$gpu_idx" -eq "$STRAGGLER_GPU" ]; do
        gpu_idx=$((gpu_idx + 1))
    done
    MAPPING="$MAPPING $gpu_idx"
    gpu_idx=$((gpu_idx + 1))
done
MAPPING="$MAPPING $STRAGGLER_GPU"
MAPPING="${MAPPING# }"  # strip leading space

echo "[launch] Rank -> GPU mapping: $MAPPING" >&2
export RANK_TO_GPU="$MAPPING"

# Launch via mpirun
exec mpirun -n "$N" -x RANK_TO_GPU "$WRAPPER" "$BINARY" "$@"
