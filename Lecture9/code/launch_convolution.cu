#include "convolution2d_strategy2.cuh"

cudaError_t launchConvolution(const float* d_N, float* d_P,
                             const float* h_Mc,
                             int Height, int Width)
{
    if (!d_N || !d_P || !h_Mc || d_N == d_P ||
        Height <= 0 || Width <= 0)
        return cudaErrorInvalidValue;

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return err;
    if (INPUT_WIDTH * INPUT_WIDTH > prop.maxThreadsPerBlock ||
        INPUT_WIDTH > prop.maxThreadsDim[0] ||
        INPUT_WIDTH > prop.maxThreadsDim[1] ||
        sizeof(float) * INPUT_WIDTH * INPUT_WIDTH >
            prop.sharedMemPerBlock)
        return cudaErrorInvalidConfiguration;

    err = cudaMemcpyToSymbol(Mc, h_Mc, sizeof(Mc));
    if (err != cudaSuccess) return err;
    dim3 block(INPUT_WIDTH, INPUT_WIDTH, 1);
    dim3 grid((Width - 1) / TILE_WIDTH + 1,
              (Height - 1) / TILE_WIDTH + 1, 1);
    if (grid.x > unsigned(prop.maxGridSize[0]) ||
        grid.y > unsigned(prop.maxGridSize[1]))
        return cudaErrorInvalidConfiguration;
    convolution2DStrategy2<<<grid, block>>>(d_N, d_P, Height, Width);
    return cudaGetLastError();
}
