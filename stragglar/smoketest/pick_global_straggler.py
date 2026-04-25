"""
pick_global_straggler.py — Aggregate per-node smoketest reports into a global pick.

Reads combined stdout from `mpirun ... -m stragglar.smoketest.smoketest` (one
process per node) on stdin, finds every line of the form
    STRAGGLER_REPORT=<hostname>:<gpu>:<delta_ms>
and picks the (hostname, gpu) whose delta_ms is largest — i.e., the node whose
slowest GPU is furthest behind its peers on that node.

Outputs two parseable lines on stdout for launch.sh to consume:
    STRAGGLER_HOSTNAME=<hostname>
    STRAGGLER_GPU=<local_gpu_id>

Exits 0 on success, 1 on any parse / input failure.
"""

import re
import sys


def main() -> int:
    text = sys.stdin.read()

    pattern = re.compile(
        r"^STRAGGLER_REPORT=(?P<host>\S+?):(?P<gpu>\d+):(?P<delta>[\d.]+)\s*$",
        re.MULTILINE,
    )
    reports = [
        (m.group("host"), int(m.group("gpu")), float(m.group("delta")))
        for m in pattern.finditer(text)
    ]

    if not reports:
        print(
            "pick_global_straggler: no STRAGGLER_REPORT= lines found in input",
            file=sys.stderr,
        )
        return 1

    # Pick the node with the largest gap to its second-slowest GPU.
    straggler_host, straggler_gpu, _ = max(reports, key=lambda r: r[2])

    print(f"STRAGGLER_HOSTNAME={straggler_host}")
    print(f"STRAGGLER_GPU={straggler_gpu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
