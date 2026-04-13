"""
detect_straggler.py — Identify the straggling GPU from batch compute results.

Pure analysis module: no CUDA, no torch imports.

Detection is end-time based: the GPU that finishes last is the straggler.
The key reported metric is the gap between the straggler's finish time and
the second-to-last GPU's finish time.
"""

from dataclasses import dataclass, field


@dataclass
class StragglerResult:
    straggler_gpu:     int             # GPU that finished last
    finish_order:      list[int]       # GPU ids sorted by finish time (earliest first)
    end_times:         dict[int, float]  # gpu_id -> absolute finish timestamp
    end_times_rel:     dict[int, float]  # gpu_id -> time since batch start (seconds)
    delta_to_second:   float           # straggler finish - second-to-last finish (seconds)
    total_times:       dict[int, float]  # gpu_id -> total compute time (seconds)
    matrix_sizes:      dict[int, int]  # gpu_id -> matrix size used
    verdict:           str = field(default="")


def identify_straggler(timing_results: dict) -> StragglerResult:
    """
    Identify the straggling GPU from GPUWorkResult objects.

    Parameters
    ----------
    timing_results : dict mapping gpu_id -> GPUWorkResult
        (as returned by batch_compute.run_batch_compute)

    Returns
    -------
    StragglerResult with straggler_gpu, finish times, and gap to second-to-last
    """
    if not timing_results:
        raise ValueError("timing_results is empty")

    gpu_ids = list(timing_results.keys())

    end_times    = {gid: timing_results[gid].end_time    for gid in gpu_ids}
    total_times  = {gid: timing_results[gid].total_time  for gid in gpu_ids}
    matrix_sizes = {gid: timing_results[gid].matrix_size for gid in gpu_ids}

    # Sort GPUs by finish time: earliest first, straggler last
    finish_order = sorted(gpu_ids, key=end_times.__getitem__)

    # Normalize end times to seconds since the earliest finish
    t_origin = end_times[finish_order[0]]
    end_times_rel = {gid: end_times[gid] - t_origin for gid in gpu_ids}

    straggler_gpu  = finish_order[-1]
    second_to_last = finish_order[-2] if len(finish_order) >= 2 else straggler_gpu

    delta_to_second = end_times[straggler_gpu] - end_times[second_to_last]

    verdict = (
        f"GPU {straggler_gpu} is the straggler — finished "
        f"{delta_to_second*1000:.1f} ms after GPU {second_to_last} "
        f"(matrix size {matrix_sizes[straggler_gpu]}×{matrix_sizes[straggler_gpu]})"
    )

    return StragglerResult(
        straggler_gpu    = straggler_gpu,
        finish_order     = finish_order,
        end_times        = end_times,
        end_times_rel    = end_times_rel,
        delta_to_second  = delta_to_second,
        total_times      = total_times,
        matrix_sizes     = matrix_sizes,
        verdict          = verdict,
    )
