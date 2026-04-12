import subprocess
import re
import sys
import ast


def run_synthesizer(n):
    """Executes the synthesizer and captures its stdout."""
    print(f"[*] Running synthesizer for {n} GPUs...")
    result = subprocess.run(
        ["python", "synthesizer_pow2.py", str(n)], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error running synthesizer:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout


def parse_schedule(output):
    """
    Parses the stdout from the synthesizer into a list of rounds.
    Each round is a list of matchings (strings).
    """
    print("[*] Parsing Round-Centric Global Map...")
    rounds = []
    lines = output.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Round"):
            i += 1
            match_line = lines[i].strip()
            # Extract the list literal from the string e.g., "1 ['...']"
            list_start = match_line.find("[")
            if list_start != -1:
                list_str = match_line[list_start:]
                matchings = ast.literal_eval(list_str)
                rounds.append(matchings)
        i += 1
    return rounds


def generate_stragglar_helper(rounds, n):
    """Translates the Internal Representation into CUDA point-to-point blocks."""
    print(f"[*] Compiling Rank-Centric CUDA instructions for {n} GPUs...")
    code = []
    code.append(
        f"void stragglar_allreduce_helper(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t chunkSize) {{"
    )
    code.append(
        "  int numBlocks = (chunkSize + RED_ADD_THREADS - 1) / RED_ADD_THREADS;\n"
    )

    code.append("  // Synchronize to make sure everything is idle")
    code.append("  for (int r = 0; r < numRanks; ++r) {")
    code.append("    cudaSetDevice(devs[r]);")
    code.append("    cudaStreamSynchronize(streams[r]);")
    code.append("  }\n")

    for step_idx, rnd in enumerate(rounds):
        code.append(f"  // --------- Step {step_idx + 1} ---------")
        code.append("  ncclGroupStart();")

        reductions = []

        for match in rnd:
            if match.startswith("StragglerMatching"):
                m = re.match(
                    r"StragglerMatching: (\d+) <-> (\d+), chunk_id: (\d+)", match
                )
                g1, g2, cid = m.groups()
                offset = f" + {cid} * chunkSize" if cid != "0" else ""

                # Emit P2P Exchanges
                code.append(
                    f"  ncclSend(d_buffers[{g2}]{offset}, chunkSize, ncclFloat, {g1}, comms[{g2}], streams[{g2}]);"
                )
                code.append(
                    f"  ncclRecv(d_tempbufs[{g2}], chunkSize, ncclFloat, {g1}, comms[{g2}], streams[{g2}]);"
                )
                code.append(
                    f"  ncclRecv(d_tempbufs[{g1}], chunkSize, ncclFloat, {g2}, comms[{g1}], streams[{g1}]);"
                )
                code.append(
                    f"  ncclSend(d_buffers[{g1}]{offset}, chunkSize, ncclFloat, {g2}, comms[{g1}], streams[{g1}]);"
                )
                reductions.extend([(g1, cid), (g2, cid)])

            elif match.startswith("OneWayMatching"):
                m = re.match(r"OneWayMatching: (\d+) -> (\d+), chunk_id: (\d+)", match)
                src, dst, cid = m.groups()
                offset = f" + {cid} * chunkSize" if cid != "0" else ""

                code.append(
                    f"  ncclSend(d_buffers[{src}]{offset}, chunkSize, ncclFloat, {dst}, comms[{src}], streams[{src}]);"
                )
                code.append(
                    f"  ncclRecv(d_buffers[{dst}]{offset}, chunkSize, ncclFloat, {src}, comms[{dst}], streams[{dst}]);"
                )

            elif match.startswith("TwoWayMatching"):
                m = re.match(
                    r"TwoWayMatching: (\d+) -> (\d+), chunk_id: (\d+);\s*(\d+) -> (\d+), chunk_id: (\d+)",
                    match,
                )
                s1, d1, c1, s2, d2, c2 = m.groups()
                off1 = f" + {c1} * chunkSize" if c1 != "0" else ""
                off2 = f" + {c2} * chunkSize" if c2 != "0" else ""

                code.append(
                    f"  ncclSend(d_buffers[{s1}]{off1}, chunkSize, ncclFloat, {d1}, comms[{s1}], streams[{s1}]);"
                )
                code.append(
                    f"  ncclRecv(d_buffers[{d1}]{off1}, chunkSize, ncclFloat, {s1}, comms[{d1}], streams[{d1}]);"
                )
                code.append(
                    f"  ncclSend(d_buffers[{s2}]{off2}, chunkSize, ncclFloat, {d2}, comms[{s2}], streams[{s2}]);"
                )
                code.append(
                    f"  ncclRecv(d_buffers[{d2}]{off2}, chunkSize, ncclFloat, {s2}, comms[{d2}], streams[{d2}]);"
                )

        # By grouping all P2P calls in one ncclGroup, NCCL schedules them simultaneously over NVLink
        code.append("  ncclGroupEnd();\n")

        # Emit Reductions
        for g, cid in reductions:
            offset = f" + {cid} * chunkSize" if cid != "0" else ""
            code.append(f"  cudaSetDevice(devs[{g}]);")
            code.append(
                f"  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[{g}]>>>(d_buffers[{g}]{offset}, d_tempbufs[{g}], chunkSize);"
            )
        code.append("")

    code.append("  cudaSetDevice(devs[0]);")
    code.append("  cudaEventRecord(stop, streams[0]);")
    code.append("  cudaEventSynchronize(stop);")
    code.append("}")
    return "\n".join(code)


def generate_cuda_file(n, helper_code):
    """Wraps the generated helper in the necessary standard boilerplate."""
    print(f"[*] Generating stragglar_{n}gpu.cu...")

    # Calculate the expected verification sum for N GPUs initialized to 0, 1, ..., N-1
    expected_sum = (n * (n - 1)) / 2.0

    boilerplate = f"""#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <assert.h>
#include <cstring> 

#define RED_ADD_THREADS 256

#define CHECK_CUDA(cmd) do {{ \\
  cudaError_t e = cmd; \\
  if (e != cudaSuccess) {{ \\
    printf("CUDA error %s:%d: '%s'\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\
    exit(EXIT_FAILURE); \\
  }} \\
}} while (0)

#define CHECK_NCCL(cmd) do {{ \\
  ncclResult_t res = cmd; \\
  if (res != ncclSuccess) {{ \\
    printf("NCCL error %s:%d: '%s'\\n", __FILE__, __LINE__, ncclGetErrorString(res)); \\
    exit(EXIT_FAILURE); \\
  }} \\
}} while (0)

__global__ void reduce_add(float* dst, float* src, int count) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {{
    dst[i] += src[i];
  }}
}}

__global__ void gpu_sleep_kernel(clock_t sleep_cycles) {{
  clock_t start = clock();
  while (clock() - start < sleep_cycles);
}}

__global__ void fill_pattern(float* dst, float v, size_t n) {{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += gridDim.x * blockDim.x)
    dst[i] = v;
}}

clock_t calculate_sleep_cycles(float ms, int* devs) {{
  cudaSetDevice(devs[{n - 1}]);
  int clockRate_kHz;
  cudaDeviceGetAttribute(&clockRate_kHz, cudaDevAttrClockRate, devs[{n - 1}]);
  return static_cast<clock_t>(ms * clockRate_kHz);
}}

// =========================================================================
// AUTO-GENERATED STRAGGLAR HELPER
// =========================================================================
{helper_code}
// =========================================================================

void stragglar_allreduce_delay(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, ncclComm_t* subComms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size, clock_t sleep_cycles) {{
  size_t chunkSize = size / (numRanks - 1);
  cudaSetDevice(devs[0]);
  cudaEventRecord(start, streams[0]);

  cudaSetDevice(devs[{n - 1}]);
  gpu_sleep_kernel<<<1, 1, 0, streams[{n - 1}]>>>(sleep_cycles);

  for (int r = 0; r < numRanks - 1; ++r) {{
    cudaSetDevice(devs[r]);
    cudaStreamSynchronize(streams[r]);
  }}

  ncclGroupStart();
  for (int r = 0; r < numRanks - 1; ++r) {{
    cudaSetDevice(devs[r]);
    ncclReduceScatter(d_buffers[r], d_buffers[r] + (r * chunkSize), chunkSize, ncclFloat, ncclSum, subComms[r], streams[r]);
  }}
  ncclGroupEnd();

  stragglar_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}}

void stragglar_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size) {{
    size_t chunkSize = size / (numRanks - 1);
    cudaSetDevice(devs[0]);
    cudaEventRecord(start, streams[0]);
    stragglar_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}}

int main(int argc, char* argv[]) {{
  const int numRanks = {n};

  if (argc != 5) {{
    fprintf(stderr, "Usage: %s <bufferSize> <algorithm> <numIters> <sleepTimeMs>\\n", argv[0]);
    exit(EXIT_FAILURE);
  }}

  size_t bytes = (size_t)strtoull(argv[1], NULL, 10);
  size_t size = bytes / sizeof(float);
  const char* alg = argv[2];
  int numIters = atoi(argv[3]);
  float sleepTime = atof(argv[4]);

  int nGPUs = 0;
  CHECK_CUDA(cudaGetDeviceCount(&nGPUs));
  if (nGPUs < numRanks) {{
    printf("Need at least %d GPUs\\n", numRanks);
    return -1;
  }}

  int devs[numRanks];
  for (int i = 0; i < numRanks; ++i) devs[i] = i;

  float* d_buffers[numRanks];
  float* d_tempbufs[numRanks];
  cudaStream_t streams[numRanks];
  ncclComm_t comms[numRanks];
  ncclComm_t subComms[numRanks - 1];
  
  clock_t sleep_cycles;
  if (sleepTime >= 0) {{
    sleep_cycles = calculate_sleep_cycles(sleepTime, devs);
    printf("Sleep cycles: %ld\\n", sleep_cycles);
  }}

  cudaSetDevice(devs[0]);
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_NCCL(ncclCommInitAll(comms, numRanks, devs));
  if (sleepTime >= 0) {{
    CHECK_NCCL(ncclCommInitAll(subComms, numRanks - 1, NULL));
  }}

  size_t chunkSize = size / (numRanks - 1);

  for (int i = 0; i < numRanks; ++i) {{
    CHECK_CUDA(cudaSetDevice(devs[i]));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaMallocAsync(&d_buffers[i], size * sizeof(float), streams[i]));
    CHECK_CUDA(cudaMallocAsync(&d_tempbufs[i], chunkSize * sizeof(float), streams[i]));
  }}

  // Warmup 
  for (int iter = 0; iter < 10; ++iter) {{
    ncclGroupStart();
    for(int r = 0; r < numRanks; ++r) {{
        ncclAllReduce(d_buffers[r], d_buffers[r], size, ncclFloat, ncclSum, comms[r], streams[r]);
    }}
    ncclGroupEnd();
  }}

  printf("algorithm,buffer_size_bytes,iteration,delay,runtime_ms,BW(GB/s)\\n");
  for (int iter = 0; iter < numIters + 1; ++iter) {{
    
    for (int i = 0; i < numRanks; ++i) {{
      CHECK_CUDA(cudaSetDevice(devs[i]));
      fill_pattern<<<(size+255)/256, 256, 0, streams[i] >>>(d_buffers[i], float(i), size);
      if (sleepTime < 0 && i < numRanks - 1) {{
        fill_pattern<<< (chunkSize+255)/256, 256, 0, streams[i] >>>(d_buffers[i] + i*chunkSize, float({n - 1}), chunkSize);
      }}
    }}

    for (int i = 0; i < numRanks; ++i) {{
      CHECK_CUDA(cudaSetDevice(devs[i]));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }}

    if (sleepTime >= 0) {{
        stragglar_allreduce_delay(d_buffers, d_tempbufs, devs, streams, comms, subComms, start, stop, numRanks, size, sleep_cycles);
    }} else {{
        stragglar_allreduce(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size);
    }}

    float ms;
    float bw;
    cudaSetDevice(devs[0]);
    cudaEventElapsedTime(&ms, start, stop);
    if (iter == 0) continue;
    if (sleepTime > 0) {{
      bw = (float)size * sizeof(float) / 1024.0 / 1024.0 / 1024.0 * 1000.0 / (ms - sleepTime);
    }} else {{
      bw = (float)size * sizeof(float) / 1024.0 / 1024.0 / 1024.0 * 1000.0 / ms;
    }}
    printf("%s,%zu,%d,%.3f,%.3f,%.3f\\n", alg, (size_t)size * sizeof(float), iter, sleepTime, ms, bw);
  }}
  
  // Verification
  float* hostOut = (float*)malloc(size * sizeof(float));
  for (int r = 0; r < numRanks; ++r) {{
    CHECK_CUDA(cudaSetDevice(devs[r]));
    CHECK_CUDA(cudaMemcpy(hostOut, d_buffers[r], size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {{
      if (hostOut[i] != {expected_sum}) {{
          printf("Mismatch on Rank %d at idx %d! expected %.3f, got %.3f\\n", r, i, {expected_sum}, hostOut[i]);
          break;
      }}
    }}
  }}
  free(hostOut);

  // Cleanup
  for (int i = 0; i < numRanks; ++i) {{
    cudaSetDevice(devs[i]);
    cudaFree(d_buffers[i]);
    cudaFree(d_tempbufs[i]);
    cudaStreamDestroy(streams[i]); 
    ncclCommDestroy(comms[i]);      
    if (sleepTime >= 0 && i < numRanks - 1) {{
      ncclCommDestroy(subComms[i]);
    }}
  }}
  return 0;
}}
"""
    filename = f"stragglar_{n}gpu.cu"
    with open(filename, "w") as f:
        f.write(boilerplate)
    print(f"[+] Successfully wrote {filename}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compiler.py <n>")
        sys.exit(1)

    n = int(sys.argv[1])

    # 1. Run algorithm to get the schedule
    output = run_synthesizer(n)

    # 2. Parse string matching outputs to structured data
    rounds = parse_schedule(output)

    # 3. Compile local scripts/CUDA string
    helper_code = generate_stragglar_helper(rounds, n)

    # 4. Inject into the main CUDA skeleton
    generate_cuda_file(n, helper_code)
