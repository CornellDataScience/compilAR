
import os
import sys
import re

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'allreduce_multinode.cu.template')
MARKERS = ['/*{{NUM_RANKS}}*/', '/*{{STRAGGLER_RANK}}*/', '/*{{HELPER_BODY}}*/']

class StraggleMatching:
    def __init__(self, rank1, rank2):
        self.rank1 = rank1
        self.rank2 = rank2

# this function finds all the matchings in the file,
# split up by which round they occurred in
def findMatchings(lines):
    rounds = []
    for line in lines:
        matchings = []
        for m in re.finditer(r'StragglerMatching:\s*(\d+)\s*<->\s*(\d+),\s*chunk_id:\s*(\d+)', line):
            a, b, chunk = map(int, m.groups())
            matchings.append(('straggler', a, b, chunk))

        for m in re.finditer(r'OneWayMatching:\s*(\d+)\s*->\s*(\d+),\s*chunk_id:\s*(\d+)', line):
            a, b, chunk = map(int, m.groups())
            matchings.append(('oneway', a, b, chunk))

        for m in re.finditer(
            r'TwoWayMatching:\s*(\d+)\s*->\s*(\d+),\s*chunk_id:\s*(\d+)\s*;\s*(\d+)\s*->\s*(\d+),\s*chunk_id:\s*(\d+)',
            line
        ):
            a, b, chunk1, c, d, chunk2 = map(int, m.groups())
            matchings.append(('twoway', a, b, chunk1, c, d, chunk2))

        if matchings:
            rounds.append(matchings)

    return rounds

# this takes the rounds and then creates the actual nccl code
# we will construct a string
def constructNCCL(rounds):
    # preamble typa stuff
    res = ""

    # define nb
    res += "const int nb = (chunkSize + kReduceThreads - 1) / kReduceThreads;\n"
    
    for round in rounds:
        res += "ncclGroupStart(); \n" # each round is grouped by a nccl group start/end

        # in here is the logic to do the nccl send recvs
        stub = ""
        for matching in round:

            pass
            # if one way matching do something
            if (matching[0] == "oneway"):
                res +=  f"if (myRank == {matching[1]}) {{ CHECK_NCCL(ncclSend(d_buffer + {matching[3]} * chunkSize, chunkSize, ncclFloat, {matching[2]}, comm, stream)); }} \n"
                res +=  f"if (myRank == {matching[2]}) {{ CHECK_NCCL(ncclRecv(d_buffer + {matching[3]} * chunkSize, chunkSize, ncclFloat, {matching[1]}, comm, stream)); }} \n"

            # if straggler way do something
            if (matching[0] == "straggler"):
                res += f"""if (myRank == {matching[1]}) {{
                    CHECK_NCCL(ncclSend(d_buffer + {matching[3]} * chunkSize,           chunkSize, ncclFloat, {matching[2]}, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, {matching[2]}, comm, stream));
                }} \n"""
                res += f"""if (myRank == {matching[2]}) {{
                    CHECK_NCCL(ncclSend(d_buffer  + {matching[3]} * chunkSize,           chunkSize, ncclFloat, {matching[1]}, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, {matching[1]}, comm, stream));
                }} \n"""
                stub += f"""if (myRank == {matching[1]} || myRank == {matching[2]}) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + {matching[3]}*chunkSize, d_tempbuf, chunkSize); \n"""

            # if 2 way do something
            if (matching[0] == "twoway"):
                res += f"""if (myRank == {matching[1]}) {{
                    CHECK_NCCL(ncclSend(d_buffer + {matching[3]}*chunkSize, chunkSize, ncclFloat, {matching[2]}, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + {matching[6]}*chunkSize, chunkSize, ncclFloat, {matching[2]}, comm, stream));
                }} \n"""
                res += f"""if (myRank == {matching[2]}) {{
                    CHECK_NCCL(ncclRecv(d_buffer + {matching[3]}*chunkSize, chunkSize, ncclFloat, {matching[1]}, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + {matching[6]}*chunkSize, chunkSize, ncclFloat, {matching[1]}, comm, stream));
                }} \n"""
        res += "ncclGroupEnd();\n"
        res += stub

        # now do the reduce add for the straggler matching

    return res

# infer number of ranks N from the schedule
def infer_n(rounds) -> int:
    max_rank = 0
    for round in rounds:
        for matching in round:
            if matching[0] == 'twoway':
                for i in [1, 2, 4, 5]:
                    max_rank = max(max_rank, matching[i])
            else:
                for i in [1, 2]:
                    max_rank = max(max_rank, matching[i])
    return max_rank + 1

# infer straggler rank: the rank that appears in every StragglerMatching
def infer_straggler(rounds) -> int:
    straggler_set = None
    for round in rounds:
        per_round = set()
        for matching in round:
            if matching[0] == 'straggler':
                per_round.add(matching[1])
                per_round.add(matching[2])
        if not per_round:
            continue  # skip rounds with no straggler matching (otherwise intersection wipes everything)
        if straggler_set is None:
            straggler_set = per_round
        else:
            straggler_set &= per_round
    if straggler_set is None or len(straggler_set) != 1:
        raise ValueError(f"Could not uniquely identify straggler rank; got candidates: {straggler_set}")
    return next(iter(straggler_set))

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <schedule_file> <output.cu>", file=sys.stderr)
        sys.exit(1)

    schedule_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(schedule_path, 'r') as f:
        lines = f.readlines()

    rounds = findMatchings(lines)
    if not rounds:
        raise ValueError(f"No rounds parsed from {schedule_path}")

    n = infer_n(rounds)
    straggler = infer_straggler(rounds)
    assert straggler == n - 1, f"Expected straggler to be rank N-1 = {n-1}; got {straggler}"

    body = constructNCCL(rounds)

    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()

    for marker in MARKERS:
        if marker not in template:
            raise ValueError(f"Template missing marker: {marker}")

    out = (template
           .replace('/*{{NUM_RANKS}}*/', str(n))
           .replace('/*{{STRAGGLER_RANK}}*/', str(straggler))
           .replace('/*{{HELPER_BODY}}*/', body))

    with open(output_path, 'w') as f:
        f.write(out)

    print(f"Wrote {output_path} (N={n}, straggler={straggler})", file=sys.stderr)

if __name__ == "__main__":
    main()
    


