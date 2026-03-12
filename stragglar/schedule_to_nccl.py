#!/usr/bin/env python3
"""Convert synthesizer round logs into NCCL C++ helper code.

Expected input is the textual output from synthesizer_pow2.py or
synthesizer_nonpow2.py, for example:

Round 0
1 ['StragglerMatching: 0 <-> 3, chunk_id: 0']
...

The parser ignores GPU state lines and only consumes matching entries.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import List


ROUND_RE = re.compile(r"^\s*Round\s+(\d+)\s*$", re.MULTILINE)
MATCH_RE = re.compile(
    r"TwoWayMatching:\s*(\d+)\s*->\s*(\d+)\s*,\s*chunk_id:\s*(\d+)\s*;\s*(\d+)\s*->\s*(\d+)\s*,\s*chunk_id:\s*(\d+)"
    r"|StragglerMatching:\s*(\d+)\s*<->\s*(\d+)\s*,\s*chunk_id:\s*(\d+)"
    r"|OneWayMatching:\s*(\d+)\s*->\s*(\d+)\s*,\s*chunk_id:\s*(\d+)"
)


@dataclass
class Transfer:
    send: int
    recv: int
    chunk_id: int
    source_kind: str


@dataclass
class RoundSchedule:
    round_id: int
    transfers: List[Transfer]


def _split_round_blocks(text: str) -> List[tuple[int, str]]:
    headers = list(ROUND_RE.finditer(text))
    if not headers:
        return []

    blocks: List[tuple[int, str]] = []
    for i, match in enumerate(headers):
        round_id = int(match.group(1))
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        blocks.append((round_id, text[start:end]))
    return blocks


def parse_schedule(text: str) -> List[RoundSchedule]:
    rounds: List[RoundSchedule] = []
    for round_id, block in _split_round_blocks(text):
        transfers: List[Transfer] = []
        for match in MATCH_RE.finditer(block):
            if match.group(1) is not None:
                # TwoWayMatching: a -> b (chunk c1); b -> a (chunk c2)
                a, b, c1, b2, a2, c2 = map(int, match.group(1, 2, 3, 4, 5, 6))
                if a != a2 or b != b2:
                    raise ValueError(
                        f"Round {round_id}: malformed TwoWayMatching endpoints: "
                        f"{a}->{b} and {b2}->{a2}"
                    )
                transfers.append(Transfer(a, b, c1, "TwoWayMatching"))
                transfers.append(Transfer(b, a, c2, "TwoWayMatching"))
                continue

            if match.group(7) is not None:
                # StragglerMatching: a <-> b (same chunk both ways)
                a, b, c = map(int, match.group(7, 8, 9))
                transfers.append(Transfer(a, b, c, "StragglerMatching"))
                transfers.append(Transfer(b, a, c, "StragglerMatching"))
                continue

            if match.group(10) is not None:
                # OneWayMatching: a -> b (chunk c)
                a, b, c = map(int, match.group(10, 11, 12))
                transfers.append(Transfer(a, b, c, "OneWayMatching"))

        if transfers:
            # Each receiver should get at most one chunk per round.
            seen_recvs = set()
            for t in transfers:
                if t.recv in seen_recvs:
                    raise ValueError(
                        f"Round {round_id}: receiver GPU {t.recv} appears multiple times. "
                        "Cannot map to one temp buffer per rank without extra handling."
                    )
                seen_recvs.add(t.recv)

            rounds.append(RoundSchedule(round_id=round_id, transfers=transfers))

    if not rounds:
        raise ValueError("No round schedule found in input.")
    return rounds


def _buf_ptr(rank: int, chunk_id: int, chunk_var: str) -> str:
    if chunk_id == 0:
        return f"d_buffers[{rank}]"
    return f"d_buffers[{rank}] + ({chunk_id} * {chunk_var})"


def generate_cpp(
    rounds: List[RoundSchedule],
    function_name: str,
    chunk_var: str,
    reduce_threads: int,
    include_sync: bool,
) -> str:
    lines: List[str] = []
    lines.append(
        f"void {function_name}(float** d_buffers, float** d_tempbufs, int* devs, "
        "cudaStream_t* streams, ncclComm_t* comms, int numRanks, int chunkSize) {"
    )

    if include_sync:
        lines.extend(
            [
                "  // Ensure all streams are idle before running generated schedule.",
                "  for (int r = 0; r < numRanks; ++r) {",
                "    cudaSetDevice(devs[r]);",
                "    cudaStreamSynchronize(streams[r]);",
                "  }",
                "",
            ]
        )

    for rs in rounds:
        lines.append(f"  // Round {rs.round_id}")
        lines.append("  ncclGroupStart();")
        for t in rs.transfers:
            send_ptr = _buf_ptr(t.send, t.chunk_id, chunk_var)
            lines.append(
                f"  ncclSend({send_ptr}, {chunk_var}, ncclFloat, {t.recv}, comms[{t.send}], streams[{t.send}]);"
            )
            lines.append(
                f"  ncclRecv(d_tempbufs[{t.recv}], {chunk_var}, ncclFloat, {t.send}, comms[{t.recv}], streams[{t.recv}]);"
            )
        lines.append("  ncclGroupEnd();")

        lines.append("  // Apply local reduction for newly received chunks.")
        for t in rs.transfers:
            dst_ptr = _buf_ptr(t.recv, t.chunk_id, chunk_var)
            lines.append(f"  cudaSetDevice(devs[{t.recv}]);")
            lines.append(
                f"  reduce_add<<<({chunk_var} + {reduce_threads} - 1) / {reduce_threads}, {reduce_threads}, 0, streams[{t.recv}]>>>("
                f"{dst_ptr}, d_tempbufs[{t.recv}], {chunk_var});"
            )
        lines.append("")

    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse synthesizer round logs and emit NCCL helper code."
    )
    parser.add_argument("input", help="Path to text file containing round log output")
    parser.add_argument(
        "-o",
        "--output",
        help="Output C++ file/snippet path (defaults to stdout)",
    )
    parser.add_argument(
        "--function-name",
        default="generated_schedule_allreduce_helper",
        help="Name of generated C++ helper function",
    )
    parser.add_argument(
        "--chunk-var",
        default="chunkSize",
        help="C++ variable expression for chunk element count",
    )
    parser.add_argument(
        "--reduce-threads",
        type=int,
        default=128,
        help="Threads per block for generated reduce_add kernel launch",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not emit initial stream synchronization loop",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    rounds = parse_schedule(text)
    cpp = generate_cpp(
        rounds=rounds,
        function_name=args.function_name,
        chunk_var=args.chunk_var,
        reduce_threads=args.reduce_threads,
        include_sync=not args.no_sync,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(cpp)
    else:
        sys.stdout.write(cpp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
