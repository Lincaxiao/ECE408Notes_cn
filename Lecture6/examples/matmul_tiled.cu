#include <cuda_runtime.h>
#define TILE_DIM 16

__global__ void matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_DIM + ty;
    const int col = blockIdx.x * TILE_DIM + tx;
    const int numTiles = (K - 1) / TILE_DIM + 1;
    float sum = 0.0f;

    for (int q = 0; q < numTiles; ++q) {
        const int aCol = q * TILE_DIM + tx;
        const int bRow = q * TILE_DIM + ty;
        A_s[ty][tx] = (row < M && aCol < K)
            ? A[row * K + aCol] : 0.0f;
        B_s[ty][tx] = (bRow < K && col < N)
            ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += A_s[ty][k] * B_s[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
