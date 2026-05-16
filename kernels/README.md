# Deliberately Naive CUDA Kernels

## naive_matmul.cu

Naive 1024x1024 float matrix multiplication with one thread per output element. It reads directly from global memory for every multiply-add and uses no tiling or shared memory, leaving obvious room for a tiled shared-memory rewrite.

## naive_reduction.cu

Sum reduction over 16M floats using interleaved addressing with `if (tid % (2 * s) == 0)`. This deliberately causes warp divergence and inefficient shared-memory access patterns compared with sequential addressing or warp shuffle reductions.

## naive_transpose.cu

Naive 4096x4096 matrix transpose using only global memory. Reads are coalesced, but writes use `output[col * N + row]`, producing uncoalesced global stores that a padded shared-memory tiled transpose can fix.
