#!/usr/bin/env python3
"""
Compile a StragglAR schedule file into a CUDA/NCCL stragglar_allreduce_helper function.

Usage:
    python compile_schedule.py <schedule_file> [function_name]

Output is written to stdout. Redirect to a .cu or .h file as needed.

Schedule operation types:
  StragglerMatching: A <-> S, chunk_id: C
      Bidirectional exchange between non-straggler A and straggler S for chunk C.
      Both sides exchange into tempbufs, then reduce_add into their own d_buffers.

  OneWayMatching: A -> B, chunk_id: C
      A sends its fully-reduced chunk C directly to B.
      B receives into d_buffers (no reduce needed).

  TwoWayMatching: A -> B, chunk_id: C1; B -> A, chunk_id: C2
      A and B exchange fully-reduced chunks C1 and C2.
      Both receive directly into d_buffers (no reduce needed).
"""

import re
import sys


def parse_schedule(filename):
    """
    Returns:
        num_gpus  (int)
        straggler (int)  -- GPU index of the straggler
        rounds    (list of lists of operation tuples)

    Operation tuples:
        ('straggler', a, s, c)        StragglerMatching
        ('oneway',    a, b, c)        OneWayMatching
        ('twoway',    a, b, c1, c2)   TwoWayMatching  (a sends c1, b sends c2)
    """
    rounds = []
    current_ops = None
    straggler = None
    num_gpus = 0

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if re.match(r'^Round \d+$', line):
                if current_ops is not None:
                    rounds.append(current_ops)
                current_ops = []
                continue

            m = re.match(r'^GPU (\d+) chunks', line)
            if m:
                num_gpus = max(num_gpus, int(m.group(1)) + 1)
                continue

            # Line with operation list, e.g.: 2 ['StragglerMatching: ...', ...]
            # Normalise curly/smart quotes to ASCII before matching
            line_norm = line.replace('\u2018', "'").replace('\u2019', "'")
            m = re.match(r'^\d+ \[(.+)\]$', line_norm)
            if m and current_ops is not None:
                for op_str in re.findall(r"'([^']+)'", m.group(1)):
                    op_str = op_str.strip()

                    m2 = re.match(
                        r'StragglerMatching: (\d+) <-> (\d+), chunk_id: (\d+)', op_str)
                    if m2:
                        a, s, c = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                        straggler = s
                        current_ops.append(('straggler', a, s, c))
                        continue

                    m2 = re.match(
                        r'OneWayMatching: (\d+) -> (\d+), chunk_id: (\d+)', op_str)
                    if m2:
                        a, b, c = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                        current_ops.append(('oneway', a, b, c))
                        continue

                    m2 = re.match(
                        r'TwoWayMatching: (\d+) -> (\d+), chunk_id: (\d+); '
                        r'(\d+) -> (\d+), chunk_id: (\d+)', op_str)
                    if m2:
                        a, b, c1 = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                        c2 = int(m2.group(6))
                        current_ops.append(('twoway', a, b, c1, c2))
                        continue

                    print(f"WARNING: unrecognised operation: {op_str!r}", file=sys.stderr)

    if current_ops is not None:
        rounds.append(current_ops)

    return num_gpus, straggler, rounds


# ── Code-generation helpers ────────────────────────────────────────────────────

def buf(gpu, chunk):
    """Pointer expression for d_buffers[gpu][chunk]."""
    if chunk == 0:
        return f"d_buffers[{gpu}]"
    return f"d_buffers[{gpu}] + {chunk} * chunkSize"


def tmp(gpu):
    return f"d_tempbufs[{gpu}]"


def generate(num_gpus, straggler, rounds, func_name):
    I = "  "   # one level of indent
    out = []

    out.append(f"void {func_name}(float** d_buffers, float** d_tempbufs, int* devs,")
    out.append(f"    cudaStream_t* streams, ncclComm_t* comms,")
    out.append(f"    cudaEvent_t start, cudaEvent_t stop,")
    out.append(f"    int numRanks, int chunkSize) {{")
    out.append("")
    out.append(f"{I}int numBlocks = (chunkSize + 128 - 1) / 128;")
    out.append("")

    # Synchronise all streams before starting
    out.append(f"{I}for (int r = 0; r < numRanks; ++r) {{")
    out.append(f"{I}  cudaSetDevice(devs[r]);")
    out.append(f"{I}  cudaStreamSynchronize(streams[r]);")
    out.append(f"{I}}}")

    for step_idx, ops in enumerate(rounds):
        out.append("")
        out.append(f"{I}// step {step_idx + 1}")
        out.append(f"{I}ncclGroupStart();")

        straggler_ops = [op for op in ops if op[0] == 'straggler']
        oneway_ops    = [op for op in ops if op[0] == 'oneway']
        twoway_ops    = [op for op in ops if op[0] == 'twoway']

        # StragglerMatching: bilateral exchange → tempbufs, then reduce
        for op in straggler_ops:
            _, a, s, c = op
            out.append(f"{I}ncclSend({buf(s, c)}, chunkSize, ncclFloat, {a}, comms[{s}], streams[{s}]);")
            out.append(f"{I}ncclRecv({tmp(s)}, chunkSize, ncclFloat, {a}, comms[{s}], streams[{s}]);")
            out.append(f"{I}ncclRecv({tmp(a)}, chunkSize, ncclFloat, {s}, comms[{a}], streams[{a}]);")
            out.append(f"{I}ncclSend({buf(a, c)}, chunkSize, ncclFloat, {s}, comms[{a}], streams[{a}]);")

        # OneWayMatching: sender → receiver direct into d_buffers (fully reduced)
        for op in oneway_ops:
            _, a, b, c = op
            out.append(f"{I}ncclSend({buf(a, c)}, chunkSize, ncclFloat, {b}, comms[{a}], streams[{a}]);")
            out.append(f"{I}ncclRecv({buf(b, c)}, chunkSize, ncclFloat, {a}, comms[{b}], streams[{b}]);")

        # TwoWayMatching: a sends c1 to b, b sends c2 to a — both fully reduced
        for op in twoway_ops:
            _, a, b, c1, c2 = op
            out.append(f"{I}ncclSend({buf(a, c1)}, chunkSize, ncclFloat, {b}, comms[{a}], streams[{a}]);")
            out.append(f"{I}ncclRecv({buf(a, c2)}, chunkSize, ncclFloat, {b}, comms[{a}], streams[{a}]);")
            out.append(f"{I}ncclRecv({buf(b, c1)}, chunkSize, ncclFloat, {a}, comms[{b}], streams[{b}]);")
            out.append(f"{I}ncclSend({buf(b, c2)}, chunkSize, ncclFloat, {a}, comms[{b}], streams[{b}]);")

        out.append(f"{I}ncclGroupEnd();")

        # reduce_add only for straggler matches (both sides received into tempbufs)
        for op in straggler_ops:
            _, a, s, c = op
            out.append(f"{I}cudaSetDevice(devs[{a}]);")
            out.append(f"{I}reduce_add<<<numBlocks, 128, 0, streams[{a}]>>>({buf(a, c)}, {tmp(a)}, chunkSize);")
            out.append(f"{I}cudaSetDevice(devs[{s}]);")
            out.append(f"{I}reduce_add<<<numBlocks, 128, 0, streams[{s}]>>>({buf(s, c)}, {tmp(s)}, chunkSize);")

    out.append("")
    out.append(f"{I}cudaSetDevice(devs[0]);")
    out.append(f"{I}cudaEventRecord(stop, 0);")
    out.append(f"{I}cudaEventSynchronize(stop);")
    out.append("}")

    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <schedule_file> [function_name]", file=sys.stderr)
        sys.exit(1)

    filename  = sys.argv[1]
    func_name = sys.argv[2] if len(sys.argv) > 2 else "stragglar_allreduce_helper"

    num_gpus, straggler, rounds = parse_schedule(filename)

    if num_gpus == 0:
        print("ERROR: could not determine GPU count from schedule file.", file=sys.stderr)
        sys.exit(1)

    print(f"// Generated by compile_schedule.py from: {filename}")
    print(f"// numRanks={num_gpus}, straggler=GPU{straggler}, rounds={len(rounds)}")
    print()
    print(generate(num_gpus, straggler, rounds, func_name))


if __name__ == "__main__":
    main()
