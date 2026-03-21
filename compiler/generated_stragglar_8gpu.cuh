// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.
// Source schedule synthesizer: /Users/aryadatla/Documents/compilAR/stragglar/synthesizer_pow2.py
// World size: 8
// Number of rounds: 9

void stragglar_allreduce_generated_8(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, int numRanks, size_t chunkSize) {
  if (numRanks != 8) {
    printf("Expected numRanks=8, got %d\n", numRanks);
    return;
  }

  // Compile-time offset model from the design doc:
  //   ptr = base + (chunk_id * chunkSize)
  int numBlocks = (chunkSize + RED_ADD_THREADS - 1) / RED_ADD_THREADS;

  // Round 0
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 7, comms[0], streams[0]);
  ncclRecv(d_tempbufs[0], chunkSize, ncclFloat, 7, comms[0], streams[0]);
  ncclSend(d_buffers[7], chunkSize, ncclFloat, 0, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 0, comms[7], streams[7]);
  ncclGroupEnd();
  cudaSetDevice(devs[0]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[0]>>>(d_buffers[0], d_tempbufs[0], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7], d_tempbufs[7], chunkSize);

  // Round 1
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclSend(d_buffers[1] + (1 * chunkSize), chunkSize, ncclFloat, 7, comms[1], streams[1]);
  ncclRecv(d_tempbufs[1], chunkSize, ncclFloat, 7, comms[1], streams[1]);
  ncclRecv(d_buffers[3], chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclSend(d_buffers[7] + (1 * chunkSize), chunkSize, ncclFloat, 1, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 1, comms[7], streams[7]);
  ncclGroupEnd();
  cudaSetDevice(devs[1]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[1]>>>(d_buffers[1] + (1 * chunkSize), d_tempbufs[1], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + (1 * chunkSize), d_tempbufs[7], chunkSize);

  // Round 2
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 5, comms[0], streams[0]);
  ncclSend(d_buffers[1] + (1 * chunkSize), chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclSend(d_buffers[2] + (2 * chunkSize), chunkSize, ncclFloat, 7, comms[2], streams[2]);
  ncclRecv(d_tempbufs[2], chunkSize, ncclFloat, 7, comms[2], streams[2]);
  ncclSend(d_buffers[3], chunkSize, ncclFloat, 6, comms[3], streams[3]);
  ncclRecv(d_buffers[4] + (1 * chunkSize), chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclRecv(d_buffers[5], chunkSize, ncclFloat, 0, comms[5], streams[5]);
  ncclRecv(d_buffers[6], chunkSize, ncclFloat, 3, comms[6], streams[6]);
  ncclSend(d_buffers[7] + (2 * chunkSize), chunkSize, ncclFloat, 2, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 2, comms[7], streams[7]);
  ncclGroupEnd();
  cudaSetDevice(devs[2]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[2]>>>(d_buffers[2] + (2 * chunkSize), d_tempbufs[2], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + (2 * chunkSize), d_tempbufs[7], chunkSize);

  // Round 3
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 4, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + (1 * chunkSize), chunkSize, ncclFloat, 4, comms[0], streams[0]);
  ncclSend(d_buffers[1] + (1 * chunkSize), chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclRecv(d_buffers[1], chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclSend(d_buffers[2] + (2 * chunkSize), chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclRecv(d_buffers[2], chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclSend(d_buffers[3] + (3 * chunkSize), chunkSize, ncclFloat, 7, comms[3], streams[3]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 7, comms[3], streams[3]);
  ncclSend(d_buffers[4] + (1 * chunkSize), chunkSize, ncclFloat, 0, comms[4], streams[4]);
  ncclRecv(d_buffers[4], chunkSize, ncclFloat, 0, comms[4], streams[4]);
  ncclSend(d_buffers[5], chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + (1 * chunkSize), chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclSend(d_buffers[6], chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + (2 * chunkSize), chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclSend(d_buffers[7] + (3 * chunkSize), chunkSize, ncclFloat, 3, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 3, comms[7], streams[7]);
  ncclGroupEnd();
  cudaSetDevice(devs[3]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[3]>>>(d_buffers[3] + (3 * chunkSize), d_tempbufs[3], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + (3 * chunkSize), d_tempbufs[7], chunkSize);

  // Round 4
  ncclGroupStart();
  ncclSend(d_buffers[0] + (1 * chunkSize), chunkSize, ncclFloat, 6, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + (2 * chunkSize), chunkSize, ncclFloat, 6, comms[0], streams[0]);
  ncclSend(d_buffers[1] + (1 * chunkSize), chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + (3 * chunkSize), chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclSend(d_buffers[2] + (2 * chunkSize), chunkSize, ncclFloat, 5, comms[2], streams[2]);
  ncclRecv(d_buffers[2] + (1 * chunkSize), chunkSize, ncclFloat, 5, comms[2], streams[2]);
  ncclSend(d_buffers[3] + (3 * chunkSize), chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + (1 * chunkSize), chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclSend(d_buffers[4] + (4 * chunkSize), chunkSize, ncclFloat, 7, comms[4], streams[4]);
  ncclRecv(d_tempbufs[4], chunkSize, ncclFloat, 7, comms[4], streams[4]);
  ncclSend(d_buffers[5] + (1 * chunkSize), chunkSize, ncclFloat, 2, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + (2 * chunkSize), chunkSize, ncclFloat, 2, comms[5], streams[5]);
  ncclSend(d_buffers[6] + (2 * chunkSize), chunkSize, ncclFloat, 0, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + (1 * chunkSize), chunkSize, ncclFloat, 0, comms[6], streams[6]);
  ncclSend(d_buffers[7] + (4 * chunkSize), chunkSize, ncclFloat, 4, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 4, comms[7], streams[7]);
  ncclGroupEnd();
  cudaSetDevice(devs[4]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[4]>>>(d_buffers[4] + (4 * chunkSize), d_tempbufs[4], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + (4 * chunkSize), d_tempbufs[7], chunkSize);

  // Round 5
  ncclGroupStart();
  ncclSend(d_buffers[0] + (2 * chunkSize), chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + (3 * chunkSize), chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclSend(d_buffers[1] + (3 * chunkSize), chunkSize, ncclFloat, 6, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + (2 * chunkSize), chunkSize, ncclFloat, 6, comms[1], streams[1]);
  ncclSend(d_buffers[2] + (2 * chunkSize), chunkSize, ncclFloat, 4, comms[2], streams[2]);
  ncclRecv(d_buffers[2] + (4 * chunkSize), chunkSize, ncclFloat, 4, comms[2], streams[2]);
  ncclSend(d_buffers[3] + (3 * chunkSize), chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + (2 * chunkSize), chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclSend(d_buffers[4] + (4 * chunkSize), chunkSize, ncclFloat, 2, comms[4], streams[4]);
  ncclRecv(d_buffers[4] + (2 * chunkSize), chunkSize, ncclFloat, 2, comms[4], streams[4]);
  ncclSend(d_buffers[5] + (5 * chunkSize), chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclRecv(d_tempbufs[5], chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclSend(d_buffers[6] + (2 * chunkSize), chunkSize, ncclFloat, 1, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + (3 * chunkSize), chunkSize, ncclFloat, 1, comms[6], streams[6]);
  ncclSend(d_buffers[7] + (5 * chunkSize), chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclGroupEnd();
  cudaSetDevice(devs[5]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[5]>>>(d_buffers[5] + (5 * chunkSize), d_tempbufs[5], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + (5 * chunkSize), d_tempbufs[7], chunkSize);

  // Round 6
  ncclGroupStart();
  ncclSend(d_buffers[0] + (3 * chunkSize), chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + (4 * chunkSize), chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclSend(d_buffers[1] + (3 * chunkSize), chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + (4 * chunkSize), chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclSend(d_buffers[2] + (4 * chunkSize), chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclRecv(d_buffers[2] + (3 * chunkSize), chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclSend(d_buffers[3] + (3 * chunkSize), chunkSize, ncclFloat, 5, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + (5 * chunkSize), chunkSize, ncclFloat, 5, comms[3], streams[3]);
  ncclSend(d_buffers[4] + (4 * chunkSize), chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclRecv(d_buffers[4] + (3 * chunkSize), chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclSend(d_buffers[5] + (5 * chunkSize), chunkSize, ncclFloat, 3, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + (3 * chunkSize), chunkSize, ncclFloat, 3, comms[5], streams[5]);
  ncclSend(d_buffers[6] + (6 * chunkSize), chunkSize, ncclFloat, 7, comms[6], streams[6]);
  ncclRecv(d_tempbufs[6], chunkSize, ncclFloat, 7, comms[6], streams[6]);
  ncclSend(d_buffers[7] + (6 * chunkSize), chunkSize, ncclFloat, 6, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 6, comms[7], streams[7]);
  ncclGroupEnd();
  cudaSetDevice(devs[6]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[6]>>>(d_buffers[6] + (6 * chunkSize), d_tempbufs[6], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + (6 * chunkSize), d_tempbufs[7], chunkSize);

  // Round 7
  ncclGroupStart();
  ncclSend(d_buffers[0] + (4 * chunkSize), chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + (5 * chunkSize), chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclSend(d_buffers[1] + (4 * chunkSize), chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + (5 * chunkSize), chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclSend(d_buffers[2] + (4 * chunkSize), chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclRecv(d_buffers[2] + (6 * chunkSize), chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclSend(d_buffers[3] + (5 * chunkSize), chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + (4 * chunkSize), chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclRecv(d_buffers[4] + (6 * chunkSize), chunkSize, ncclFloat, 7, comms[4], streams[4]);
  ncclSend(d_buffers[5] + (5 * chunkSize), chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + (4 * chunkSize), chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclSend(d_buffers[6] + (6 * chunkSize), chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + (4 * chunkSize), chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclSend(d_buffers[7] + (6 * chunkSize), chunkSize, ncclFloat, 4, comms[7], streams[7]);
  ncclGroupEnd();

  // Round 8
  ncclGroupStart();
  ncclSend(d_buffers[0] + (5 * chunkSize), chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + (6 * chunkSize), chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclSend(d_buffers[1] + (5 * chunkSize), chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + (6 * chunkSize), chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclSend(d_buffers[2] + (6 * chunkSize), chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclRecv(d_buffers[2] + (5 * chunkSize), chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclSend(d_buffers[3] + (5 * chunkSize), chunkSize, ncclFloat, 6, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + (6 * chunkSize), chunkSize, ncclFloat, 6, comms[3], streams[3]);
  ncclSend(d_buffers[4] + (6 * chunkSize), chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclRecv(d_buffers[4] + (5 * chunkSize), chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclRecv(d_buffers[5] + (6 * chunkSize), chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclSend(d_buffers[6] + (6 * chunkSize), chunkSize, ncclFloat, 3, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + (5 * chunkSize), chunkSize, ncclFloat, 3, comms[6], streams[6]);
  ncclSend(d_buffers[7] + (6 * chunkSize), chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclGroupEnd();

}
