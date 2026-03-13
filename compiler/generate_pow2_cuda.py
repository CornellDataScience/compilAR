#!/usr/bin/env python3
"""Generate StragglAR CUDA/NCCL code from a `synthesizer_pow2.py` schedule.

Design notes from the StragglAR design doc reflected here:
- Build a round-centric schedule, then convert it to a rank-centric IR.
- Precompute chunk pointer offsets at compile-time (`chunk_id * chunkSize`).
- Emit explicit NCCL P2P send/recv calls grouped per round.
- Emit `reduce_add` only for straggler exchanges that receive into temp buffers.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass
class RankAction:
    send_to: int | None = None
    send_chunk: int | None = None
    recv_from: int | None = None
    recv_chunk: int | None = None
    recv_reduce: bool = False


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("synthesizer_pow2_codegen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_send(action: RankAction, peer: int, chunk: int, round_id: int, rank: int) -> None:
    if action.send_to is not None:
        raise ValueError(
            f"Round {round_id}: rank {rank} already has a send action "
            f"(existing peer={action.send_to}, new peer={peer})."
        )
    action.send_to = peer
    action.send_chunk = chunk


def _set_recv(
    action: RankAction,
    peer: int,
    chunk: int,
    reduce_after_recv: bool,
    round_id: int,
    rank: int,
) -> None:
    if action.recv_from is not None:
        raise ValueError(
            f"Round {round_id}: rank {rank} already has a recv action "
            f"(existing peer={action.recv_from}, new peer={peer})."
        )
    action.recv_from = peer
    action.recv_chunk = chunk
    action.recv_reduce = reduce_after_recv


def _build_rank_centric_ir(schedule: list[list[Any]], world_size: int) -> list[list[RankAction]]:
    rounds: list[list[RankAction]] = [
        [RankAction() for _ in range(world_size)] for _ in range(len(schedule))
    ]

    for round_id, matchings in enumerate(schedule):
        for matching in matchings:
            kind = type(matching).__name__

            if kind == "StragglerMatching":
                a = matching.gpu.rank
                b = matching.straggler_gpu.rank
                chunk = int(matching.chunk_id)

                _set_send(rounds[round_id][a], b, chunk, round_id, a)
                _set_recv(rounds[round_id][a], b, chunk, True, round_id, a)
                _set_send(rounds[round_id][b], a, chunk, round_id, b)
                _set_recv(rounds[round_id][b], a, chunk, True, round_id, b)
                continue

            if kind == "OneWayMatching":
                src = matching.send_gpu.rank
                dst = matching.recv_gpu.rank
                chunk = int(matching.chunk_id)

                _set_send(rounds[round_id][src], dst, chunk, round_id, src)
                _set_recv(rounds[round_id][dst], src, chunk, False, round_id, dst)
                continue

            if kind == "TwoWayMatching":
                a = matching.gpu1.rank
                b = matching.gpu2.rank
                chunk_a_to_b = int(matching.chunk_id1)
                chunk_b_to_a = int(matching.chunk_id2)

                _set_send(rounds[round_id][a], b, chunk_a_to_b, round_id, a)
                _set_recv(rounds[round_id][a], b, chunk_b_to_a, False, round_id, a)
                _set_send(rounds[round_id][b], a, chunk_b_to_a, round_id, b)
                _set_recv(rounds[round_id][b], a, chunk_a_to_b, False, round_id, b)
                continue

            raise ValueError(f"Round {round_id}: unsupported matching kind {kind!r}")

    return rounds


def _buffer_ptr(rank: int, chunk: int | None) -> str:
    if chunk is None or chunk == 0:
        return f"d_buffers[{rank}]"
    return f"d_buffers[{rank}] + ({chunk} * chunkSize)"


def _render_function(
    function_name: str,
    world_size: int,
    round_actions: list[list[RankAction]],
    source_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.")
    lines.append(f"// Source schedule synthesizer: {source_path}")
    lines.append(f"// World size: {world_size}")
    lines.append(f"// Number of rounds: {len(round_actions)}")
    lines.append("")
    lines.append(
        f"void {function_name}(float** d_buffers, float** d_tempbufs, int* devs, "
        "cudaStream_t* streams, ncclComm_t* comms, int numRanks, size_t chunkSize) {"
    )
    lines.append(f"  if (numRanks != {world_size}) {{")
    lines.append(
        f'    printf("Expected numRanks={world_size}, got %d\\n", numRanks);'
    )
    lines.append("    return;")
    lines.append("  }")
    lines.append("")
    lines.append("  // Compile-time offset model from the design doc:")
    lines.append("  //   ptr = base + (chunk_id * chunkSize)")
    lines.append("  int numBlocks = (chunkSize + RED_ADD_THREADS - 1) / RED_ADD_THREADS;")
    lines.append("")

    for round_id, actions in enumerate(round_actions):
        lines.append(f"  // Round {round_id}")
        lines.append("  ncclGroupStart();")

        for rank, action in enumerate(actions):
            if action.send_to is not None:
                lines.append(
                    f"  ncclSend({_buffer_ptr(rank, action.send_chunk)}, chunkSize, ncclFloat, "
                    f"{action.send_to}, comms[{rank}], streams[{rank}]);"
                )
            if action.recv_from is not None:
                recv_ptr = (
                    f"d_tempbufs[{rank}]"
                    if action.recv_reduce
                    else _buffer_ptr(rank, action.recv_chunk)
                )
                lines.append(
                    f"  ncclRecv({recv_ptr}, chunkSize, ncclFloat, "
                    f"{action.recv_from}, comms[{rank}], streams[{rank}]);"
                )

        lines.append("  ncclGroupEnd();")

        for rank, action in enumerate(actions):
            if action.recv_reduce:
                lines.append(f"  cudaSetDevice(devs[{rank}]);")
                lines.append(
                    "  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, "
                    f"streams[{rank}]>>>({_buffer_ptr(rank, action.recv_chunk)}, "
                    f"d_tempbufs[{rank}], chunkSize);"
                )

        lines.append("")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _default_output_path(script_dir: Path, world_size: int) -> Path:
    return script_dir / f"generated_stragglar_{world_size}gpu.cuh"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_synth = script_dir.parent / "stragglar" / "synthesizer_pow2.py"

    parser = argparse.ArgumentParser(
        description="Generate CUDA/NCCL code from synthesizer_pow2 schedule."
    )
    parser.add_argument(
        "--synthesizer",
        type=Path,
        default=default_synth,
        help="Path to synthesizer_pow2.py",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="Number of GPUs (must be a power of 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path for generated CUDA code.",
    )
    parser.add_argument(
        "--function-name",
        type=str,
        default="stragglar_allreduce_generated",
        help="Generated CUDA helper function name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesizer_path = args.synthesizer.resolve()
    if not synthesizer_path.exists():
        raise FileNotFoundError(f"Synthesizer file not found: {synthesizer_path}")

    if not _is_pow2(args.world_size):
        raise ValueError(
            f"--world-size must be a positive power of 2, got {args.world_size}."
        )

    module = _load_module(synthesizer_path)
    if not hasattr(module, "GPU") or not hasattr(module, "construct_schedule"):
        raise AttributeError(
            "Synthesizer module must expose `GPU` and `construct_schedule`."
        )

    gpus = [module.GPU(rank, args.world_size) for rank in range(args.world_size)]
    schedule = module.construct_schedule(gpus, print_result=False)
    round_actions = _build_rank_centric_ir(schedule, args.world_size)

    out_path = (
        args.output.resolve()
        if args.output is not None
        else _default_output_path(Path(__file__).resolve().parent, args.world_size)
    )
    rendered = _render_function(
        args.function_name, args.world_size, round_actions, synthesizer_path
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)

    print(f"Wrote {out_path}")
    print(f"Rounds: {len(schedule)}")


if __name__ == "__main__":
    main()
