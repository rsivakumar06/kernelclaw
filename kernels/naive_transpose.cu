#include <cuda_runtime.h>

#include <cstdio>

constexpr int N = 4096;
constexpr int BLOCK_DIM = 16;

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                 \
      return 1;                                                              \
    }                                                                        \
  } while (0)

__global__ void naiveTranspose(const float *input, float *output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    output[col * N + row] = input[row * N + col];
  }
}

int main() {
  const size_t element_count = static_cast<size_t>(N) * N;
  const size_t bytes = element_count * sizeof(float);

  float *input = nullptr;
  float *output = nullptr;

  CHECK_CUDA(cudaMallocManaged(&input, bytes));
  CHECK_CUDA(cudaMallocManaged(&output, bytes));

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      input[row * N + col] = static_cast<float>((row + 3 * col) % 17);
      output[row * N + col] = 0.0f;
    }
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  CHECK_CUDA(cudaEventRecord(start));
  naiveTranspose<<<grid, block>>>(input, output);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());

  float milliseconds = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

  double checksum = 0.0;
  for (size_t i = 0; i < element_count; ++i) {
    checksum += static_cast<double>(output[i]);
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(input));
  CHECK_CUDA(cudaFree(output));

  std::printf("TIME_US=%.2f\n", milliseconds * 1000.0f);
  std::printf("CHECKSUM=%.1f\n", checksum);

  return 0;
}
