#pragma once
#include <cuda_runtime.h>

constexpr int TILE_WIDTH = 16;
constexpr int MASK_WIDTH = 5;
constexpr int RADIUS = MASK_WIDTH / 2;
constexpr int INPUT_WIDTH = TILE_WIDTH + MASK_WIDTH - 1;
static_assert(TILE_WIDTH > 0 && MASK_WIDTH % 2 == 1);

__constant__ float Mc[MASK_WIDTH][MASK_WIDTH];

__global__ void convolution2DStrategy2(
    const float* N, float* P, int Height, int Width)
{
    __shared__ float tile[INPUT_WIDTH][INPUT_WIDTH];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_o = int(blockIdx.y) * TILE_WIDTH + ty;
    const int col_o = int(blockIdx.x) * TILE_WIDTH + tx;
    const int row_i = row_o - RADIUS;
    const int col_i = col_o - RADIUS;

    if (row_i >= 0 && row_i < Height &&
        col_i >= 0 && col_i < Width) {
        tile[ty][tx] = N[size_t(row_i) * Width + col_i];
    } else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        float Pvalue = 0.0f;
        for (int i = 0; i < MASK_WIDTH; ++i) {
            for (int j = 0; j < MASK_WIDTH; ++j) {
                Pvalue += Mc[i][j] * tile[ty + i][tx + j];
            }
        }
        if (row_o < Height && col_o < Width) {
            P[size_t(row_o) * Width + col_o] = Pvalue;
        }
    }
}
