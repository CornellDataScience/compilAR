void generated_schedule_allreduce_helper(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, int numRanks, int chunkSize) {
  // Ensure all streams are idle before running generated schedule.
  for (int r = 0; r < numRanks; ++r) {
    cudaSetDevice(devs[r]);
    cudaStreamSynchronize(streams[r]);
  }

  // Round 0
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclSend(d_buffers[3], chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclRecv(d_tempbufs[0], chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclGroupEnd();
  // Apply local reduction for newly received chunks.
  cudaSetDevice(devs[3]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[3]>>>(d_buffers[3], d_tempbufs[3], chunkSize);
  cudaSetDevice(devs[0]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[0]>>>(d_buffers[0], d_tempbufs[0], chunkSize);

  // Round 1
  ncclGroupStart();
  ncclSend(d_buffers[1] + (1 * chunkSize), chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclSend(d_buffers[3] + (1 * chunkSize), chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_tempbufs[1], chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_tempbufs[2], chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclGroupEnd();
  // Apply local reduction for newly received chunks.
  cudaSetDevice(devs[3]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[3]>>>(d_buffers[3] + (1 * chunkSize), d_tempbufs[3], chunkSize);
  cudaSetDevice(devs[1]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[1]>>>(d_buffers[1] + (1 * chunkSize), d_tempbufs[1], chunkSize);
  cudaSetDevice(devs[2]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[2]>>>(d_buffers[2], d_tempbufs[2], chunkSize);

  // Round 2
  ncclGroupStart();
  ncclSend(d_buffers[2] + (2 * chunkSize), chunkSize, ncclFloat, 3, comms[2], streams[2]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 2, comms[3], streams[3]);
  ncclSend(d_buffers[3] + (2 * chunkSize), chunkSize, ncclFloat, 2, comms[3], streams[3]);
  ncclRecv(d_tempbufs[2], chunkSize, ncclFloat, 3, comms[2], streams[2]);
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 1, comms[0], streams[0]);
  ncclRecv(d_tempbufs[1], chunkSize, ncclFloat, 0, comms[1], streams[1]);
  ncclSend(d_buffers[1] + (1 * chunkSize), chunkSize, ncclFloat, 0, comms[1], streams[1]);
  ncclRecv(d_tempbufs[0], chunkSize, ncclFloat, 1, comms[0], streams[0]);
  ncclGroupEnd();
  // Apply local reduction for newly received chunks.
  cudaSetDevice(devs[3]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[3]>>>(d_buffers[3] + (2 * chunkSize), d_tempbufs[3], chunkSize);
  cudaSetDevice(devs[2]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[2]>>>(d_buffers[2] + (2 * chunkSize), d_tempbufs[2], chunkSize);
  cudaSetDevice(devs[1]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[1]>>>(d_buffers[1], d_tempbufs[1], chunkSize);
  cudaSetDevice(devs[0]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[0]>>>(d_buffers[0] + (1 * chunkSize), d_tempbufs[0], chunkSize);

  // Round 3
  ncclGroupStart();
  ncclSend(d_buffers[0] + (1 * chunkSize), chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_tempbufs[2], chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclSend(d_buffers[2] + (2 * chunkSize), chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclRecv(d_tempbufs[0], chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclSend(d_buffers[3] + (2 * chunkSize), chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_tempbufs[1], chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclGroupEnd();
  // Apply local reduction for newly received chunks.
  cudaSetDevice(devs[2]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[2]>>>(d_buffers[2] + (1 * chunkSize), d_tempbufs[2], chunkSize);
  cudaSetDevice(devs[0]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[0]>>>(d_buffers[0] + (2 * chunkSize), d_tempbufs[0], chunkSize);
  cudaSetDevice(devs[1]);
  reduce_add<<<(chunkSize + 128 - 1) / 128, 128, 0, streams[1]>>>(d_buffers[1] + (2 * chunkSize), d_tempbufs[1], chunkSize);

}
