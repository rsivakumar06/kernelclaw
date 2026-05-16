#include <cuda_runtime.h>
#include <cstdio>
constexpr int N = 1024;
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
__global__ void naiveMatmul(const float *a, const float *b, float *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= N) {
    return;
  }
  float sum = 0.0f;
  for (int k = 0; k < N; ++k) {
    sum += a[row * N + k] * b[k * N + col];
  }
  c[row * N + col] = sum;
}
int main() {
  const size_t element_count = static_cast<size_t>(N) * N;
  const size_t bytes = element_count * sizeof(float);
  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;
  CHECK_CUDA(cudaMallocManaged(&a, bytes));
  CHECK_CUDA(cudaMallocManaged(&b, bytes));
  CHECK_CUDA(cudaMallocManaged(&c, bytes));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      a[i * N + j] = static_cast<float>((i + j) % 7);
      b[i * N + j] = static_cast<float>((i * j) % 5);
      c[i * N + j] = 0.0f;
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
  naiveMatmul<<<grid, block>>>(a, b, c);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());
  float milliseconds = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  double checksum = 0.0;
  for (size_t i = 0; i < element_count; ++i) {
    checksum += static_cast<double>(c[i]);
  }
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(a));
  CHECK_CUDA(cudaFree(b));
  CHECK_CUDA(cudaFree(c));
  std::printf("TIME_US=%.2f\n", milliseconds * 1000.0f);
  std::printf("CHECKSUM=%.1f\n", checksum);
  return 0;
}
