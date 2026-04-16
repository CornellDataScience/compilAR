
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
        for matching in round:
            pass
            # if straggler matching do something
            
            # if one way do something

            # if 2 way do something
        res += "ncclGroupEnd();\n"

        # now do the reduce add for the straggler matching
    return res


def main():
    with open('4gpusched.txt', 'r') as f:
        lines = f.readlines()
        res = findMatchings(lines)
        print(res)

if __name__ == "__main__":
    main()
    


