#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <assert.h>
#include <cstring> 

#define NUM_RANKS 4

constexpr int kStragglerRank = 3;
constexpr int kNumRanks = 4;
constexpr int kReduceThreads = 128; 
constexpr int kFillThreads = 256;
// constexpr int kWarmupIters = 10;
constexpr float kExpectedSum = 6.0;


constexpr float kFillValue = 3.0f;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
// inline int CeilDiv(int n , int d) { return (n + d - 1) / d; }


#define CHECK_CUDA(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define CHECK_NCCL(cmd) do { \
  ncclResult_t res = cmd; \
  if (res != ncclSuccess) { \
    printf("NCCL error %s:%d: '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void reduce_add(float* dst, float* src, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] += src[i];
  }
}

__global__ void gpu_sleep_kernel(clock_t sleep_cycles) {
  clock_t start = clock();
  while (clock() - start < sleep_cycles);
}

__global__ void fill_pattern(float* dst, float v, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += gridDim.x * blockDim.x)
    dst[i] = v;
}

// calculate the ms for straggler
clock_t calculate_sleep_cycles(float ms, int* devs) {
  cudaSetDevice(devs[NUM_RANKS - 1]);

  // Query clock rate (in kHz)
  int clockRate_kHz;
  cudaDeviceGetAttribute(&clockRate_kHz, cudaDevAttrClockRate, devs[NUM_RANKS - 1]);

  // Compute number of cycles to sleep
  clock_t sleep_cycles = static_cast<clock_t>(ms * clockRate_kHz);

  return sleep_cycles;
}


void stragglar_allreduce_helper(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, int chunkSize) {
  int numBlocks = (chunkSize + 128 - 1) / 128;

  for (int r = 0; r < numRanks; ++r) {
    cudaSetDevice(devs[r]);
    cudaStreamSynchronize(streams[r]);
  }

  // step 1
  ncclGroupStart();
  ncclSend(d_buffers[3], chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclRecv(d_tempbufs[0], chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclGroupEnd();
  cudaSetDevice(devs[0]);
  float* dstPtr = d_buffers[0];
  reduce_add<<<numBlocks, 128, 0, streams[0]>>>(dstPtr, d_tempbufs[0], chunkSize);
  cudaSetDevice(devs[3]);
  float* dstPtr_straggler = d_buffers[3];
  reduce_add<<<numBlocks, 128, 0, streams[3]>>>(dstPtr_straggler, d_tempbufs[3], chunkSize);

  // step 2
  ncclGroupStart();
  ncclSend(d_buffers[3] + chunkSize, chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_tempbufs[1], chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclSend(d_buffers[1] + chunkSize, chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[2], chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclGroupEnd();
  cudaSetDevice(devs[1]);
  float* dstPtr2 = d_buffers[1] + chunkSize;
  reduce_add<<<numBlocks, 128, 0, streams[1]>>>(dstPtr2, d_tempbufs[1], chunkSize);
  cudaSetDevice(devs[3]);
  dstPtr_straggler = d_buffers[3] + chunkSize;
  reduce_add<<<numBlocks, 128, 0, streams[3]>>>(dstPtr_straggler, d_tempbufs[3], chunkSize);

  // step 3
  ncclGroupStart();
  ncclSend(d_buffers[3] + 2 * chunkSize, chunkSize, ncclFloat, 2, comms[3], streams[3]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 2, comms[3], streams[3]);
  ncclRecv(d_tempbufs[2], chunkSize, ncclFloat, 3, comms[2], streams[2]);
  ncclSend(d_buffers[2] + 2 * chunkSize, chunkSize, ncclFloat, 3, comms[2], streams[2]);
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 1, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + chunkSize, chunkSize, ncclFloat, 1, comms[0], streams[0]);
  ncclRecv(d_buffers[1], chunkSize, ncclFloat, 0, comms[1], streams[1]);
  ncclSend(d_buffers[1] + chunkSize, chunkSize, ncclFloat, 0, comms[1], streams[1]);
  ncclGroupEnd();
  cudaSetDevice(devs[2]);
  float* dstPtr3 = d_buffers[2] + 2 * chunkSize;
  reduce_add<<<numBlocks, 128, 0, streams[2]>>>(dstPtr3, d_tempbufs[2], chunkSize);
  cudaSetDevice(devs[3]);
  dstPtr_straggler = d_buffers[3] + 2 * chunkSize;
  reduce_add<<<numBlocks, 128, 0, streams[3]>>>(dstPtr_straggler, d_tempbufs[3], chunkSize);

  // step 4
  ncclGroupStart();
  ncclSend(d_buffers[3] + 2 * chunkSize, chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_buffers[1] + 2 * chunkSize, chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclRecv(d_buffers[2] + chunkSize, chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclSend(d_buffers[2] + 2 * chunkSize, chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclSend(d_buffers[0] + chunkSize, chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + 2 * chunkSize, chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclGroupEnd();

  cudaSetDevice(devs[0]);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
}

void stragglar_allreduce_delay(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, ncclComm_t* subComms, cudaEvent_t start, cudaEvent_t stop, int numRanks, int size, clock_t sleep_cycles) {
  int chunkSize = size / (numRanks - 1);
  cudaSetDevice(devs[0]);
  cudaEventRecord(start, 0);
  // set stragglar
  cudaSetDevice(devs[NUM_RANKS - 1]);
  gpu_sleep_kernel<<<1, 1, 0, streams[NUM_RANKS - 1]>>>(sleep_cycles); // sleeps the GPU

  // Synchronize to make sure everything is idle
  for (int r = 0; r < numRanks - 1; ++r) {
    cudaSetDevice(devs[r]);
    cudaStreamSynchronize(streams[r]);
  }

  ncclGroupStart();
  for (int r = 0; r < numRanks - 1; ++r) {
    cudaSetDevice(devs[r]);
    ncclReduceScatter(d_buffers[r], d_buffers[r] + (r * chunkSize), chunkSize, ncclFloat, ncclSum, subComms[r], streams[r]);
  }
  ncclGroupEnd();

  stragglar_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}

void stragglar_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, int size) {
    int chunkSize = size / (numRanks - 1);
    cudaSetDevice(devs[0]);
    cudaEventRecord(start, 0);
    stragglar_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <bufferSize> <algorithm> <numIters> <sleepTimeMs>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  // number of bytes in the buffer
  size_t bytes = (size_t)strtoull(argv[1], NULL, 10); 

  // size of the buffer
  size_t size = bytes / sizeof(float); 

  // the algorithm we choose
  const char* alg = argv[2];

  // number of times to run algorithm
  int numIters = atoi(argv[3]);

  // amount of time the straggler sleeps
  float sleepTime = atof(argv[4]);


  // making sure algorithm inputted is valid
  if (strcmp(alg, "stragglar") != 0) {
    fprintf(stderr, "Invalid algorithm: %s\n", alg);
    exit(EXIT_FAILURE);
  }

  // Check GPUs
  int nGPUs = 0;
  CHECK_CUDA(cudaGetDeviceCount(&nGPUs));
  if (nGPUs < NUM_RANKS) {
    printf("Need at least %d GPUs\n", NUM_RANKS);
    return -1;
  }

  // cuda devices
  int devs[NUM_RANKS];

  // [0, 1, 2, 3] for 4 gpus, e.g.
  for (int i = 0; i < numRanks; ++i) devs[i] = i;

  // Allocate device buffers
  float* d_buffers[NUM_RANKS]; // each device has a buffer

  float* d_tempbufs[NUM_RANKS]; // what is a tempbuf? // this is an array of float arrays
  cudaIpcMemHandle_t* sendHandle[NUM_RANKS]; // each gpu has a sendhandle
  cudaIpcMemHandle_t* recvHandle[NUM_RANKS]; // each gpu has a recvhandle
  cudaStream_t streams[NUM_RANKS]; // streams of asynchronous GPU operations in order for each GPU
  
  // communication handles for each GPU
  ncclComm_t comms[NUM_RANKS]; // comms after straggler wakes up
  ncclComm_t subComms[NUM_RANKS - 1]; // comms while the straggler sleeps

  clock_t sleep_cycles; // amount of time the straggler sleeps
  if (sleepTime >= 0) {
    sleep_cycles = calculate_sleep_cycles(sleepTime, devs);
    printf("Sleep cycles: %ld\n", sleep_cycles);
  }

  cudaSetDevice(devs[0]); // first gpu
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start)); // time the entire operation
  CHECK_CUDA(cudaEventCreate(&stop));
  // use the first gpu to time the operation

  CHECK_NCCL(ncclCommInitAll(comms, numRanks, devs));
  if (sleepTime >= 0) {
    CHECK_NCCL(ncclCommInitAll(subComms, numRanks - 1, NULL));
  }
  // initializing the comms
  
  // size of the chunks
  size_t chunkSize;
  chunkSize = size / (numRanks - 1);
  
  
  for (int i = 0; i < numRanks; ++i) {
    CHECK_CUDA(cudaSetDevice(devs[i])); 
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaMallocAsync(&d_buffers[i], size * sizeof(float), streams[i])); // dbuffer is where all the data ends up

    CHECK_CUDA(cudaMallocAsync(&d_tempbufs[i], chunkSize * sizeof(float), streams[i])); // temp is where you receive chunks into
  }

  // TODO add a warmup / method we can call

  printf("algorithm,buffer_size_bytes,iteration,delay,runtime_ms,BW(GB/s)\n");
  for (int iter = 0; iter < numIters + 1; ++iter) {
    // Reset buffers if needed (same init pattern as above)
    for (int i = 0; i < numRanks; ++i) {
      CHECK_CUDA(cudaSetDevice(devs[i]));
    
      // kernel to fill all gpu buffers with the value, but it just does it in parallel
      fill_pattern<<< (chunkSize+255)/256, 256, 0, streams[i] >>>(d_buffers[i] + i*chunkSize, kFillValue, chunkSize); // fill dbuffers[i] + i * chunkSize with 
        // number of blocks, number threads per block, memory allocated per thread
    }

    
    for (int i = 0; i < numRanks; ++i) {
      CHECK_CUDA(cudaSetDevice(devs[i]));
      CHECK_CUDA(cudaStreamSynchronize(streams[i])); // waits until everything in the queue was done
    }

    // Run algorithm 
    if (sleepTime >= 0) {   // identified straggler 
        stragglar_allreduce_delay(d_buffers, d_tempbufs, devs, streams, comms, subComms, start, stop, numRanks, size, sleep_cycles);
    } 
    else {    // no straggler identified
        stragglar_allreduce(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size);
    }

    float ms;
    cudaEventElapsedTime(&ms, start, stop);     // just timing for benchmarking?
    float bw;
    if (iter == 0) continue;
    if (sleepTime > 0) {
      bw = (float)size * sizeof(float) / 1024.0 / 1024.0 / 1024.0 * 1000.0 / (ms - sleepTime); // B: should replace this wtih constant 
    }
    else {
      bw = (float)size * sizeof(float) / 1024.0 / 1024.0 / 1024.0 * 1000.0 / ms; // B: should replace this wtih constant 
    }
    printf("%s,%zu,%d,%.3f,%.3f,%.3f\n",
      alg,
      (size_t)size * sizeof(float),   // bytes, still a size_t
      iter,
      sleepTime,
      ms,
      bw);
  }
  
  // could we set up CUDA managed memory?
  float* hostOut = (float*)malloc(size * sizeof(float));    // copy GPU results to some host CPU; need buffer
  for (int r = 0; r < numRanks; ++r) {
    CHECK_CUDA(cudaSetDevice(devs[r]));
    CHECK_CUDA(cudaMemcpy(hostOut, d_buffers[r],
                          size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
      if (hostOut[i] != kExpectedSum) {     // maybe do val_per_rank * numRanks, where val_per_rank is initialized at beginning

        printf("Rank %d, idx %d, val %.3f\n", r, i, hostOut[i]);
      }
    }
  }
  free(hostOut);

  for (int i = 0; i < numRanks; ++i) {
    cudaSetDevice(devs[i]);
    cudaStreamSynchronize(streams[i]);      // block CPU until every enqued operation in streams[i] is completed on GPU
  }

  // Cleanup
  for (int i = 0; i < numRanks; ++i) {
    cudaSetDevice(devs[i]);
    cudaFree(d_buffers[i]);
    cudaFree(d_tempbufs[i]);
    cudaStreamDestroy(streams[i]);
    ncclCommDestroy(comms[i]);
    if (sleepTime >= 0 && i < numRanks - 1) {
      ncclCommDestroy(subComms[i]);
    }
    printf("Rank %d, done\n", i);
  }

  return 0;
}