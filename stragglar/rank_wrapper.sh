#!/bin/bash
set -euo pipefail

# rank_wrapper.sh — per-rank launcher invoked by mpirun.
#
# Reads $RANK_TO_GPU (set by launch.sh) which encodes a per-node mapping:
#     "host1:0,1,2,3|host2:0,1,3,2"
# Looks up the entry matching this rank's hostname, indexes the CSV by the
# local MPI rank to get the GPU id, exports LOCAL_RANK, and execs the binary.

MY_HOSTNAME="$(hostname -s)"
MY_LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${SLURM_LOCALID:-}}}"

if [ -z "$MY_LOCAL_RANK" ]; then
    echo "rank_wrapper: could not determine local MPI rank from environment" >&2
    exit 1
fi

if [ -z "${RANK_TO_GPU:-}" ]; then
    echo "rank_wrapper: RANK_TO_GPU env var is not set (launch via launch.sh)" >&2
    exit 1
fi

# Find the entry for our hostname in the pipe-separated mapping
NODEMAP=""
IFS='|' read -ra ENTRIES <<< "$RANK_TO_GPU"
for entry in "${ENTRIES[@]}"; do
    host="${entry%%:*}"
    if [ "$host" = "$MY_HOSTNAME" ]; then
        NODEMAP="${entry#*:}"
        break
    fi
done

if [ -z "$NODEMAP" ]; then
    echo "rank_wrapper: no mapping entry for hostname '$MY_HOSTNAME' in RANK_TO_GPU" >&2
    echo "  RANK_TO_GPU=$RANK_TO_GPU" >&2
    exit 1
fi

# Split the per-node CSV and index by local rank
IFS=',' read -ra MAPPING <<< "$NODEMAP"

if [ "$MY_LOCAL_RANK" -ge "${#MAPPING[@]}" ]; then
    echo "rank_wrapper: local rank $MY_LOCAL_RANK out of bounds for node mapping (${#MAPPING[@]} entries)" >&2
    exit 1
fi

export LOCAL_RANK="${MAPPING[$MY_LOCAL_RANK]}"

exec "$@"
