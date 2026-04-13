#!/bin/bash
###############################################################################
# submit_smoketest.sh — Submit the GPU straggler smoke test to SLURM
#
# Usage:
#   bash submit_smoketest.sh [NUM_GPUS] [-- smoketest.py args]
#
# Examples:
#   bash submit_smoketest.sh          # 4 GPUs (default)
#   bash submit_smoketest.sh 8        # 8 GPUs
#   bash submit_smoketest.sh 4 -- --iters 20 --seed 7
###############################################################################

# =============================================================================
# Preamble — edit these to configure the job
# =============================================================================
NUM_GPUS=${1:-4}                  # number of GPUs to allocate
LOG_DIR="$(dirname "$0")/../logs" # output directory for .out / .err files
# =============================================================================

# Shift past NUM_GPUS arg; remaining args (after --) pass through to smoketest.py
shift 1 2>/dev/null
if [[ "$1" == "--" ]]; then shift; fi
EXTRA_ARGS="$*"

mkdir -p "$LOG_DIR"

echo "Submitting smoketest:"
echo "  GPUs    : $NUM_GPUS"
echo "  Log dir : $LOG_DIR"
[[ -n "$EXTRA_ARGS" ]] && echo "  Args    : $EXTRA_ARGS"

sbatch \
  --gres=gpu:"${NUM_GPUS}" \
  --output="${LOG_DIR}/smoketest_%j.out" \
  --error="${LOG_DIR}/smoketest_%j.err" \
  smoketest.sub ${EXTRA_ARGS}
