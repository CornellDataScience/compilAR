#!/bin/bash
###############################################################################
# submit_bench.sh — Wrapper to submit the GPU benchmark with variable resources
#
# Usage:
#   ./submit_bench.sh <GPUS_PER_NODE> <NUM_NODES> [PARTITION]
#
# Examples:
#   ./submit_bench.sh 1 1                    # 1 GPU, 1 node, default partition=gpu
#   ./submit_bench.sh 2 3                    # 2 GPUs/node × 3 nodes = 6 GPUs total
#   ./submit_bench.sh 4 2 default_partition  # use low-priority queue
#
# The script:
#   1. Validates inputs
#   2. Calls sbatch with the right --nodes / --gres / --ntasks-per-node flags
#   3. srun inside the .sub file fans out gpu_matmul_bench.py to every node
###############################################################################

set -euo pipefail

GPUS_PER_NODE=${1:?  "Usage: $0 <GPUS_PER_NODE> <NUM_NODES> [PARTITION]"}
NUM_NODES=${2:?      "Usage: $0 <GPUS_PER_NODE> <NUM_NODES> [PARTITION]"}
PARTITION=${3:-gpu}

# --- Validate ---------------------------------------------------------------
if (( GPUS_PER_NODE < 1 || GPUS_PER_NODE > 8 )); then
    echo "ERROR: GPUS_PER_NODE must be between 1 and 8 (got $GPUS_PER_NODE)"
    exit 1
fi
if (( NUM_NODES < 1 )); then
    echo "ERROR: NUM_NODES must be >= 1 (got $NUM_NODES)"
    exit 1
fi

TOTAL_GPUS=$(( GPUS_PER_NODE * NUM_NODES ))
echo "==> Submitting benchmark: ${GPUS_PER_NODE} GPU(s)/node × ${NUM_NODES} node(s) = ${TOTAL_GPUS} GPU(s) total"
echo "    Partition: ${PARTITION}"

# --- Get the directory this script lives in ---------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Submit -----------------------------------------------------------------
sbatch --requeue \
    --nodes="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --gres="gpu:${GPUS_PER_NODE}" \
    --cpus-per-task=4 \
    --mem=16000 \
    --partition="${PARTITION}" \
    -t 1:00:00 \
    -J "gpu_bench_${GPUS_PER_NODE}gpn_${NUM_NODES}n" \
    -o "gpu_bench_%j.out" \
    -e "gpu_bench_%j.err" \
    "${SCRIPT_DIR}/gpu_bench.sub"

echo "==> Job submitted. Check output with:  cat gpu_bench_<JOBID>.out"
