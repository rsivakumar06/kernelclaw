#include <cuda_runtime.h>

#include <cstdio>

constexpr int N = 1 << 24;
constexpr int THREADS_PER_BLOCK = 256;

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                 \
      return 1;                                                              \
    }                                                                        \
  } while (0)

__global__ void reduceInterleaved(const float *input, float *output, int n) {
  extern __shared__ float shared[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  shared[tid] = (i < static_cast<unsigned int>(n)) ? input[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if ((tid % (2 * s)) == 0) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

int main() {
  const size_t input_bytes = static_cast<size_t>(N) * sizeof(float);
  const int max_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const size_t scratch_bytes = static_cast<size_t>(max_blocks) * sizeof(float);

  float *input = nullptr;
  float *scratch_a = nullptr;
  float *scratch_b = nullptr;

  CHECK_CUDA(cudaMallocManaged(&input, input_bytes));
  CHECK_CUDA(cudaMallocManaged(&scratch_a, scratch_bytes));
  CHECK_CUDA(cudaMallocManaged(&scratch_b, scratch_bytes));

  for (int i = 0; i < N; ++i) {
    input[i] = static_cast<float>((i % 13) + 1);
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  const size_t shared_bytes = THREADS_PER_BLOCK * sizeof(float);
  int current_n = N;
  const float *current_input = input;
  float *current_output = scratch_a;

  CHECK_CUDA(cudaEventRecord(start));

  while (current_n > 1) {
    int blocks = (current_n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    reduceInterleaved<<<blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        current_input, current_output, current_n);
    CHECK_CUDA(cudaGetLastError());

    current_n = blocks;
    current_input = current_output;
    current_output = (current_output == scratch_a) ? scratch_b : scratch_a;
  }

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

  float checksum = current_input[0];

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(input));
  CHECK_CUDA(cudaFree(scratch_a));
  CHECK_CUDA(cudaFree(scratch_b));

  std::printf("TIME_US=%.2f\n", milliseconds * 1000.0f);
  std::printf("CHECKSUM=%.1f\n", checksum);

  return 0;
}
