
import regex as re
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
        m = re.search(r'StragglerMatching:\s*(\d+)\s*<->\s*(\d+),\s*chunk_id:\s*(\d+)', line)
        if m:
            a, b, chunk = map(int, m.groups())
            matchings.append(('straggler', a, b, chunk))

        m = re.search(r'OneWayMatching:\s*(\d+)\s*->\s*(\d+),\s*chunk_id:\s*(\d+)', line)
        if m:
            a, b, chunk = map(int, m.groups())
            matchings.append(('oneway', a, b, chunk))
        
        m = re.search(
            r'TwoWayMatching:\s*(\d+)\s*->\s*(\d+),\s*chunk_id:\s*(\d+)\s*;\s*(\d+)\s*->\s*(\d+),\s*chunk_id:\s*(\d+)',
            line
        )
        if m:
            a, b, chunk1, c, d, chunk2 = map(int, m.groups())
            matchings.append(('twoway', a, b, chunk1, c, d, chunk2))
        if (len(matchings) != 0):
            rounds.append(matchings)

    return rounds

# this takes the rounds and then creates the actual nccl code
# we will construct a string
def constructNCCL(rounds):
    # preamble typa stuff
    res = ""
    for round in rounds:
        res += "ncclGroupStart(); \n" # each round is grouped by a nccl group start/end

        # in here is the logic to do the nccl send recvs
        stub = ""
        for matching in round:

            pass
            # if one way matching do something
            if (matching[0] == "oneway"):
                res +=  f"else if (myRank == {matching[1]}) {{ CHECK_NCCL(ncclSend(d_buffer + {matching[3]} * chunkSize, chunkSize, ncclFloat, {matching[2]}, comm, stream)); }} \n"
                res +=  f"else if (myRank == {matching[2]}) {{ CHECK_NCCL(ncclRecv(d_buffer + {matching[3]} * chunkSize, chunkSize, ncclFloat, {matching[1]}, comm, stream)); }} \n"

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


def main():
    with open('schedules/4gpusched.txt', 'r') as f:
        lines = f.readlines()
        res = findMatchings(lines)
        ans = constructNCCL(res)
        print(ans)

if __name__ == "__main__":
    main()
    


