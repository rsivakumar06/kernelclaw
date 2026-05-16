#include <cstdio>
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    int n = 1 << 20;
    float *a, *b, *c;
    cudaMallocManaged(&a, n*sizeof(float));
    cudaMallocManaged(&b, n*sizeof(float));
    cudaMallocManaged(&c, n*sizeof(float));
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    vec_add<<<(n+255)/256, 256>>>(a, b, c, n);
    cudaDeviceSynchronize();
    printf("c[0]=%f c[n-1]=%f\n", c[0], c[n-1]);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
