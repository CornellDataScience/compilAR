// Multi-node adaptation of allreduce_4GPU_rewrite.cu
// Uses ncclCommInitRank + MPI for cross-host bootstrapping instead of ncclCommInitAll

// One MPI rank per GPU, each process manages only its own GPU
// LOCAL_RANK env var (set by mpirun / torchrun / SLURM) selects the local device index
// Falls back to myRank % cudaDeviceCount if LOCAL_RANK is not set.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <cstring>

#define NUM_RANKS 12

constexpr int kStragglerRank  = 11;
constexpr int kReduceThreads  = 128;
constexpr int kFillThreads    = 256;
constexpr float kExpectedSum  = 6.0f;
constexpr float kFillValue    = 3.0f;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

// Error checking
#define CHECK_CUDA(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
  } \
} while (0)

#define CHECK_NCCL(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
  } \
} while (0)

#define CHECK_MPI(cmd) do { \
  int err = (cmd); \
  if (err != MPI_SUCCESS) { \
    char buf[MPI_MAX_ERROR_STRING]; int len; \
    MPI_Error_string(err, buf, &len); \
    fprintf(stderr, "MPI error %s:%d '%s'\n", __FILE__, __LINE__, buf); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
  } \
} while (0)

// Kernels
__global__ void reduce_add(float* dst, const float* src, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) dst[i] += src[i];
}

__global__ void gpu_sleep_kernel(long long int sleep_cycles) {
  long long int start = clock64();
  while (clock64() - start < sleep_cycles);
}

__global__ void fill_pattern(float* dst, float v, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = v;
}

// Helpers
clock_t calculate_sleep_cycles(float ms) {
  int clockRate_kHz;
  // cudaSetDevice has already been called; device 0 in local context
  cudaDeviceGetAttribute(&clockRate_kHz, cudaDevAttrClockRate, 0);
  return static_cast<clock_t>(ms * clockRate_kHz);
}

// ---------------------------------------------------------------------------
// Per-rank send/recv mapping for the custom StragglAR AllReduce
//
// Each process issues only its own rank's ops;
// NCCL matches sends to recvs across the network automatically.
// ---------------------------------------------------------------------------

void stragglar_allreduce_helper(
    float* d_buffer, float* d_tempbuf,
    int myRank, cudaStream_t stream, ncclComm_t comm,
    cudaEvent_t stop, int chunkSize)
{
  cudaStreamSynchronize(stream);

  const int nb = (chunkSize + kReduceThreads - 1) / kReduceThreads;
ncclGroupStart(); 
if (myRank == 0) {
                    CHECK_NCCL(ncclSend(d_buffer + 0 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 0 * chunkSize,           chunkSize, ncclFloat, 0, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 0, comm, stream));
                } 
ncclGroupEnd();
if (myRank == 0 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 0*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 1) {
                    CHECK_NCCL(ncclSend(d_buffer + 1 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 1 * chunkSize,           chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 0) { CHECK_NCCL(ncclSend(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 4, comm, stream)); } 
if (myRank == 4) { CHECK_NCCL(ncclRecv(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 0, comm, stream)); } 
ncclGroupEnd();
if (myRank == 1 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 1*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 2) {
                    CHECK_NCCL(ncclSend(d_buffer + 2 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 2 * chunkSize,           chunkSize, ncclFloat, 2, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 2, comm, stream));
                } 
if (myRank == 1) { CHECK_NCCL(ncclSend(d_buffer + 1 * chunkSize, chunkSize, ncclFloat, 5, comm, stream)); } 
if (myRank == 5) { CHECK_NCCL(ncclRecv(d_buffer + 1 * chunkSize, chunkSize, ncclFloat, 1, comm, stream)); } 
if (myRank == 0) { CHECK_NCCL(ncclSend(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 10, comm, stream)); } 
if (myRank == 10) { CHECK_NCCL(ncclRecv(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 0, comm, stream)); } 
if (myRank == 4) { CHECK_NCCL(ncclSend(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 9, comm, stream)); } 
if (myRank == 9) { CHECK_NCCL(ncclRecv(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 4, comm, stream)); } 
ncclGroupEnd();
if (myRank == 2 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 2*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 3) {
                    CHECK_NCCL(ncclSend(d_buffer + 3 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 3 * chunkSize,           chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 2) { CHECK_NCCL(ncclSend(d_buffer + 2 * chunkSize, chunkSize, ncclFloat, 6, comm, stream)); } 
if (myRank == 6) { CHECK_NCCL(ncclRecv(d_buffer + 2 * chunkSize, chunkSize, ncclFloat, 2, comm, stream)); } 
if (myRank == 9) { CHECK_NCCL(ncclSend(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 8, comm, stream)); } 
if (myRank == 8) { CHECK_NCCL(ncclRecv(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 9, comm, stream)); } 
if (myRank == 10) { CHECK_NCCL(ncclSend(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 7, comm, stream)); } 
if (myRank == 7) { CHECK_NCCL(ncclRecv(d_buffer + 0 * chunkSize, chunkSize, ncclFloat, 10, comm, stream)); } 
ncclGroupEnd();
if (myRank == 3 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 3*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 4) {
                    CHECK_NCCL(ncclSend(d_buffer + 4 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 4 * chunkSize,           chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 1) {
                    CHECK_NCCL(ncclRecv(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                } 
if (myRank == 2) {
                    CHECK_NCCL(ncclRecv(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 3) {
                    CHECK_NCCL(ncclRecv(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                } 
if (myRank == 0) {
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
if (myRank == 7) {
                    CHECK_NCCL(ncclSend(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclRecv(d_buffer + 0*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
ncclGroupEnd();
if (myRank == 4 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 4*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 5) {
                    CHECK_NCCL(ncclSend(d_buffer + 5 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 5 * chunkSize,           chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 0) {
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                } 
if (myRank == 7) {
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 3) {
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                } 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                } 
if (myRank == 2) {
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 1) {
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
ncclGroupEnd();
if (myRank == 5 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 5*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 6) {
                    CHECK_NCCL(ncclSend(d_buffer + 6 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 6 * chunkSize,           chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 6, comm, stream));
                } 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 1) {
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 0) {
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 3) {
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                } 
if (myRank == 2) {
                    CHECK_NCCL(ncclRecv(d_buffer + 1*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 7) {
                    CHECK_NCCL(ncclRecv(d_buffer + 2*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                } 
ncclGroupEnd();
if (myRank == 6 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 6*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 7) {
                    CHECK_NCCL(ncclSend(d_buffer + 7 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 7 * chunkSize,           chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                } 
if (myRank == 0) {
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 1) {
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 8) {
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 3*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
ncclGroupEnd();
if (myRank == 7 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 7*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 8 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 8 * chunkSize,           chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 1) { CHECK_NCCL(ncclSend(d_buffer + 6 * chunkSize, chunkSize, ncclFloat, 2, comm, stream)); } 
if (myRank == 2) { CHECK_NCCL(ncclRecv(d_buffer + 6 * chunkSize, chunkSize, ncclFloat, 1, comm, stream)); } 
if (myRank == 10) { CHECK_NCCL(ncclSend(d_buffer + 4 * chunkSize, chunkSize, ncclFloat, 3, comm, stream)); } 
if (myRank == 3) { CHECK_NCCL(ncclRecv(d_buffer + 4 * chunkSize, chunkSize, ncclFloat, 10, comm, stream)); } 
if (myRank == 7) {
                    CHECK_NCCL(ncclSend(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclRecv(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                } 
if (myRank == 0) {
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
ncclGroupEnd();
if (myRank == 8 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 8*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 9 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 9 * chunkSize,           chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 1) {
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                } 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                } 
if (myRank == 0) {
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 7) {
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                } 
if (myRank == 2) {
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 3) {
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 4*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
ncclGroupEnd();
if (myRank == 9 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 9*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 10 * chunkSize,           chunkSize, ncclFloat, 11, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 11, comm, stream));
                } 
if (myRank == 11) {
                    CHECK_NCCL(ncclSend(d_buffer  + 10 * chunkSize,           chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclRecv(d_tempbuf,          chunkSize, ncclFloat, 10, comm, stream));
                } 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 1) {
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
if (myRank == 7) {
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 3) {
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 0) {
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                } 
if (myRank == 2) {
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 6*chunkSize, chunkSize, ncclFloat, 0, comm, stream));
                } 
ncclGroupEnd();
if (myRank == 10 || myRank == 11) reduce_add<<<nb, kReduceThreads, 0, stream>>>(d_buffer + 10*chunkSize, d_tempbuf, chunkSize); 
ncclGroupStart(); 
if (myRank == 0) { CHECK_NCCL(ncclSend(d_buffer + 8 * chunkSize, chunkSize, ncclFloat, 1, comm, stream)); } 
if (myRank == 1) { CHECK_NCCL(ncclRecv(d_buffer + 8 * chunkSize, chunkSize, ncclFloat, 0, comm, stream)); } 
if (myRank == 11) { CHECK_NCCL(ncclSend(d_buffer + 10 * chunkSize, chunkSize, ncclFloat, 2, comm, stream)); } 
if (myRank == 2) { CHECK_NCCL(ncclRecv(d_buffer + 10 * chunkSize, chunkSize, ncclFloat, 11, comm, stream)); } 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 7) {
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclSend(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclRecv(d_buffer + 7*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 3) {
                    CHECK_NCCL(ncclRecv(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 5*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                } 
ncclGroupEnd();
ncclGroupStart(); 
if (myRank == 11) { CHECK_NCCL(ncclSend(d_buffer + 10 * chunkSize, chunkSize, ncclFloat, 0, comm, stream)); } 
if (myRank == 0) { CHECK_NCCL(ncclRecv(d_buffer + 10 * chunkSize, chunkSize, ncclFloat, 11, comm, stream)); } 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclRecv(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                } 
if (myRank == 7) {
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 3) {
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 2) {
                    CHECK_NCCL(ncclSend(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclRecv(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 1) {
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
ncclGroupEnd();
ncclGroupStart(); 
if (myRank == 11) { CHECK_NCCL(ncclSend(d_buffer + 9 * chunkSize, chunkSize, ncclFloat, 0, comm, stream)); } 
if (myRank == 0) { CHECK_NCCL(ncclRecv(d_buffer + 9 * chunkSize, chunkSize, ncclFloat, 11, comm, stream)); } 
if (myRank == 7) {
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 4, comm, stream));
                } 
if (myRank == 4) {
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 7, comm, stream));
                } 
if (myRank == 2) {
                    CHECK_NCCL(ncclSend(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 1, comm, stream));
                } 
if (myRank == 1) {
                    CHECK_NCCL(ncclRecv(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 2, comm, stream));
                } 
if (myRank == 6) {
                    CHECK_NCCL(ncclSend(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 5, comm, stream));
                } 
if (myRank == 5) {
                    CHECK_NCCL(ncclRecv(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 6, comm, stream));
                } 
if (myRank == 8) {
                    CHECK_NCCL(ncclSend(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 3, comm, stream));
                } 
if (myRank == 3) {
                    CHECK_NCCL(ncclRecv(d_buffer + 9*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 8, comm, stream));
                } 
if (myRank == 10) {
                    CHECK_NCCL(ncclSend(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                    CHECK_NCCL(ncclRecv(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 9, comm, stream));
                } 
if (myRank == 9) {
                    CHECK_NCCL(ncclRecv(d_buffer + 10*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                    CHECK_NCCL(ncclSend(d_buffer + 8*chunkSize, chunkSize, ncclFloat, 10, comm, stream));
                } 
ncclGroupEnd();
ncclGroupStart(); 
if (myRank == 11) { CHECK_NCCL(ncclSend(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 0, comm, stream)); } 
if (myRank == 0) { CHECK_NCCL(ncclRecv(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 11, comm, stream)); } 
if (myRank == 4) { CHECK_NCCL(ncclSend(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 10, comm, stream)); } 
if (myRank == 10) { CHECK_NCCL(ncclRecv(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 4, comm, stream)); } 
if (myRank == 7) { CHECK_NCCL(ncclSend(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 3, comm, stream)); } 
if (myRank == 3) { CHECK_NCCL(ncclRecv(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 7, comm, stream)); } 
if (myRank == 8) { CHECK_NCCL(ncclSend(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 2, comm, stream)); } 
if (myRank == 2) { CHECK_NCCL(ncclRecv(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 8, comm, stream)); } 
if (myRank == 9) { CHECK_NCCL(ncclSend(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 1, comm, stream)); } 
if (myRank == 1) { CHECK_NCCL(ncclRecv(d_buffer + 7 * chunkSize, chunkSize, ncclFloat, 9, comm, stream)); } 
ncclGroupEnd();


  // Rank 0 records stop after its stream finishes the last round
  if (myRank == 0) {
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
  }
}

// With straggler delay: ranks 0-2 do a reduce-scatter among themselves first,
// then all 4 ranks run the custom allreduce helper.
void stragglar_allreduce_delay(
    float* d_buffer, float* d_tempbuf,
    int myRank, cudaStream_t stream,
    ncclComm_t comm, ncclComm_t subComm,
    cudaEvent_t start, cudaEvent_t stop,
    int numRanks, size_t size, clock_t sleep_cycles)
{
  int chunkSize = (int)(size / (numRanks - 1));

  // Barrier so that all ranks start the clock at the same wall-clock moment
  MPI_Barrier(MPI_COMM_WORLD);
  if (myRank == 0) cudaEventRecord(start, stream);

  // Straggler goes to sleep while non-stragglers begin
  if (myRank == kStragglerRank)
    gpu_sleep_kernel<<<1, 1, 0, stream>>>(sleep_cycles);

  // Ranks 0-2 sync then reduce-scatter among themselves
  if (myRank != kStragglerRank) {
    cudaStreamSynchronize(stream);
    ncclGroupStart();
    // subRank == myRank for ranks 0,1,2 because MPI_Comm_split preserves order
    CHECK_NCCL(ncclReduceScatter(
    d_buffer, d_buffer, // Both must be identical
    chunkSize, ncclFloat, ncclSum, subComm, stream));
    ncclGroupEnd();
  }

  stragglar_allreduce_helper(d_buffer, d_tempbuf, myRank, stream, comm,
                             stop, chunkSize);
}

// Without straggler delay: jump straight to the 4-rank allreduce.
void stragglar_allreduce(
    float* d_buffer, float* d_tempbuf,
    int myRank, cudaStream_t stream, ncclComm_t comm,
    cudaEvent_t start, cudaEvent_t stop,
    int numRanks, size_t size)
{
  int chunkSize = (int)(size / (numRanks - 1));
  MPI_Barrier(MPI_COMM_WORLD);
  if (myRank == 0) cudaEventRecord(start, 0);
  stragglar_allreduce_helper(d_buffer, d_tempbuf, myRank, stream, comm,
                             stop, chunkSize);
}

int main(int argc, char* argv[]) {
  CHECK_MPI(MPI_Init(&argc, &argv));

  int myRank, totalRanks;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &totalRanks));

  if (totalRanks != NUM_RANKS) {
    if (myRank == 0)
      fprintf(stderr, "Must launch exactly %d MPI ranks (got %d)\n", NUM_RANKS, totalRanks);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  if (argc != 5) {
    if (myRank == 0)
      fprintf(stderr, "Usage: %s <bufferBytes> <algorithm> <numIters> <sleepMs>\n", argv[0]);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  size_t bytes   = (size_t)strtoull(argv[1], NULL, 10);
  size_t size    = bytes / sizeof(float);
  const char* alg = argv[2];
  int numIters   = atoi(argv[3]);
  float sleepTime = atof(argv[4]);

  if (strcmp(alg, "stragglar") != 0) {
    if (myRank == 0) fprintf(stderr, "Invalid algorithm: %s\n", alg);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // Bind each process to its local GPU
  int localGpu = 0;
  const char* lr = getenv("LOCAL_RANK");
  if (!lr) lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (lr) {
    localGpu = atoi(lr);
  } else {
    int nLocalGPUs = 1;
    cudaGetDeviceCount(&nLocalGPUs);
    localGpu = myRank % nLocalGPUs;
  }
  CHECK_CUDA(cudaSetDevice(localGpu));
  

  // Bootstrap full NCCL communicator (all ranks)
  ncclUniqueId commId;
  if (myRank == 0) CHECK_NCCL(ncclGetUniqueId(&commId));
  CHECK_MPI(MPI_Bcast(&commId, sizeof(commId), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, totalRanks, commId, myRank));

  // Bootstrap sub-communicator (ranks 0..N-2, excluding straggler)
  // MPI_Comm_split is a collective: every rank must call it
  // Non-straggler ranks get color=1; straggler gets MPI_UNDEFINED (excluded).
  ncclComm_t subComm = nullptr;
  MPI_Comm subMpiComm = MPI_COMM_NULL;

  if (sleepTime >= 0) {
    int color = (myRank != kStragglerRank) ? 1 : MPI_UNDEFINED;
    CHECK_MPI(MPI_Comm_split(MPI_COMM_WORLD, color, myRank, &subMpiComm));

    if (myRank != kStragglerRank) {
      int subRank, subSize;
      MPI_Comm_rank(subMpiComm, &subRank);
      MPI_Comm_size(subMpiComm, &subSize);

      ncclUniqueId subId;
      if (subRank == 0) CHECK_NCCL(ncclGetUniqueId(&subId));
      CHECK_MPI(MPI_Bcast(&subId, sizeof(subId), MPI_BYTE, 0, subMpiComm));
      CHECK_NCCL(ncclCommInitRank(&subComm, subSize, subId, subRank));
    }
    // Rank 3 must wait for ranks 0-2 to finish subComm init before any rank
    // proceeds. Without this, rank 3 races into the benchmark loop while NCCL
    // is still scanning GPU topology on compute4, corrupting rank 3's CUDA context.
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  }

  // CUDA resources
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  size_t chunkSize = size / (totalRanks - 1);

  float* d_buffer;
  float* d_tempbuf;
  CHECK_CUDA(cudaMalloc(&d_buffer,  size      * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_tempbuf, chunkSize * sizeof(float)));

  clock_t sleep_cycles = 0;
  if (sleepTime >= 0 && myRank == kStragglerRank) {
    sleep_cycles = calculate_sleep_cycles(sleepTime);
    printf("Rank %d: sleep_cycles=%ld\n", myRank, sleep_cycles);
  }

  if (myRank == 0)
    printf("algorithm,buffer_size_bytes,iteration,delay,runtime_ms,BW(GB/s)\n");

  // Main benchmark loop
  for (int iter = 0; iter < numIters + 1; ++iter) {
    if (myRank == kStragglerRank) {
      // Straggler fills entire buffer — it contributes to every chunk via swap steps
      fill_pattern<<<(size + kFillThreads - 1) / kFillThreads, kFillThreads, 0, stream>>>(
          d_buffer, kFillValue, size);
    } else {
      // Non-straggler: zero entire buffer, then fill own chunk with kFillValue.
      // The zeroing prevents stale values from the previous iteration's allreduce
      // from accumulating in the ReduceScatter.
      CHECK_CUDA(cudaMemsetAsync(d_buffer, 0, size * sizeof(float), stream));
      fill_pattern<<<(chunkSize + kFillThreads - 1) / kFillThreads, kFillThreads, 0, stream>>>(
          d_buffer + myRank * chunkSize, kFillValue, chunkSize);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (sleepTime >= 0) {
      stragglar_allreduce_delay(d_buffer, d_tempbuf, myRank, stream,
                                comm, subComm, start, stop,
                                totalRanks, size, sleep_cycles);
    } else {
      stragglar_allreduce(d_buffer, d_tempbuf, myRank, stream,
                          comm, start, stop, totalRanks, size);
    }

    if (myRank == 0) {
      if (iter == 0) continue; // skip warmup iteration
      float ms;
      cudaEventElapsedTime(&ms, start, stop);
      float bw;
      if (sleepTime > 0)
        bw = (float)size * sizeof(float) / kBytesPerGiB * 1000.0f / (ms - sleepTime);
      else
        bw = (float)size * sizeof(float) / kBytesPerGiB * 1000.0f / ms;
      printf("%s,%zu,%d,%.3f,%.3f,%.3f\n",
             alg, (size_t)size * sizeof(float), iter, sleepTime, ms, bw);
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(stream)); // Add this
  // Correctness check
  float* hostOut = (float*)malloc(size * sizeof(float));
  CHECK_CUDA(cudaMemcpy(hostOut, d_buffer, size * sizeof(float), cudaMemcpyDeviceToHost));
    // Replace the correctness loop with this:
    size_t chunkSize_check = size / (totalRanks - 1);
    int errs_per_chunk[12] = {0};
    int total_errs = 0;
    float first_bad_val[12] = {0};
    size_t first_bad_idx[12] = {0};
    bool seen[12] = {false};

    for (size_t i = 0; i < size; ++i) {
    if (hostOut[i] != kExpectedSum) {
        size_t c = i / chunkSize_check;
        if (c < 12) {
        errs_per_chunk[c]++;
        if (!seen[c]) {
            seen[c] = true;
            first_bad_val[c] = hostOut[i];
            first_bad_idx[c] = i;
        }
        }
        total_errs++;
    }
    }

    fprintf(stderr, "Rank %d: total_mismatches=%d/%zu\n", myRank, total_errs, size);
    for (int c = 0; c < 12; ++c) {
    if (errs_per_chunk[c] > 0)
        fprintf(stderr, "  Rank %d chunk %d: %d errs, first val=%.3f at idx=%zu\n",
                myRank, c, errs_per_chunk[c], first_bad_val[c], first_bad_idx[c]);
    }
    free(hostOut);

  // Cleanup
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaFree(d_buffer));
  CHECK_CUDA(cudaFree(d_tempbuf));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_NCCL(ncclCommDestroy(comm));
  if (subComm)                        CHECK_NCCL(ncclCommDestroy(subComm));
  if (subMpiComm != MPI_COMM_NULL)    MPI_Comm_free(&subMpiComm);

  printf("Rank %d done\n", myRank);
  MPI_Finalize();
  return 0;
}

