#include <cuda_runtime.h>
#define TILE_DIM 16

__global__ void matmul_tiled_square(
    const float* A, const float* B, float* C, int N)
{
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_DIM + ty;
    const int col = blockIdx.x * TILE_DIM + tx;
    float sum = 0.0f;

    for (int q = 0; q < N / TILE_DIM; ++q) {
        A_s[ty][tx] = A[row * N + q * TILE_DIM + tx];
        B_s[ty][tx] = B[(q * TILE_DIM + ty) * N + col];
        __syncthreads();  // Wait for all tile producers.

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += A_s[ty][k] * B_s[k][tx];
        }
        __syncthreads();  // Wait for all tile consumers.
    }
    C[row * N + col] = sum;
}
