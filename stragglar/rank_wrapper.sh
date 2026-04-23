#!/bin/bash
set -euo pipefail

# Per-rank launcher invoked by mpirun
#
# Reads this process's MPI rank from the environment, looks up its assigned
# GPU in $RANK_TO_GPU (set by launch.sh), exports LOCAL_RANK, then execs the
# real binary with all passed args

# Detect the MPI rank. Try Open MPI first, then MPICH, then SLURM.
MY_RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-}}}"

if [ -z "$MY_RANK" ]; then
    echo "rank_wrapper: could not determine MPI rank from environment" >&2
    exit 1
fi

if [ -z "${RANK_TO_GPU:-}" ]; then
    echo "rank_wrapper: RANK_TO_GPU env var is not set (launch via launch.sh)" >&2
    exit 1
fi

# Split space-separated list into a bash array and index by rank
MAPPING=($RANK_TO_GPU)
export LOCAL_RANK="${MAPPING[$MY_RANK]}"

exec "$@"
