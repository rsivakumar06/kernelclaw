#include <cstdio>
#include <cmath>

__global__ void naive_softmax(float* x, float* out, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += expf(x[i]);
        for (int i = 0; i < n; i++)
            out[i] = expf(x[i]) / sum;
    }
}

int main() {
    int n = 1 << 20;
    float *x, *out;
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = 1.0f;
    naive_softmax<<<1, 1>>>(x, out, n);
    cudaDeviceSynchronize();
    printf("out[0]=%f\n", out[0]);
    cudaFree(x); cudaFree(out);
    return 0;
}
