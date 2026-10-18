#include <cuda_runtime.h>
#include <cstddef>

__device__ unsigned int atomic_add_cas(
    unsigned int* address, unsigned int value)
{
    unsigned int old = atomicCAS(address, 0u, 0u);
    while (true) {
        unsigned int assumed = old;
        old = atomicCAS(address, assumed, assumed + value);
        if (old == assumed) {
            return assumed;
        }
    }
}

__global__ void histogram_global(
    const unsigned char* input, std::size_t n,
    unsigned int* hist)
{
    std::size_t i = std::size_t(blockIdx.x) * blockDim.x
                  + threadIdx.x;
    std::size_t stride = std::size_t(gridDim.x) * blockDim.x;

    for (; i < n; i += stride) {
        unsigned int bin = input[i];
        atomicAdd(&hist[bin], 1u);
    }
}

cudaError_t launch_histogram_global(
    const unsigned char* input, std::size_t n,
    unsigned int* hist, unsigned int blocks,
    unsigned int threads, cudaStream_t stream)
{
    if (blocks == 0 || threads == 0) {
        return cudaErrorInvalidConfiguration;
    }
    cudaError_t status = cudaMemsetAsync(
        hist, 0, 256 * sizeof(unsigned int), stream);
    if (status != cudaSuccess || n == 0) {
        return status;
    }
    histogram_global<<<blocks, threads, 0, stream>>>(
        input, n, hist);
    return cudaGetLastError();
}

__global__ void histogram_private(
    const unsigned char* input, std::size_t n,
    unsigned int* hist)
{
    __shared__ unsigned int shared_hist[256];

    for (unsigned int b = threadIdx.x; b < 256;
         b += blockDim.x) {
        shared_hist[b] = 0u;
    }
    __syncthreads();

    std::size_t i = std::size_t(blockIdx.x) * blockDim.x
                  + threadIdx.x;
    std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        atomicAdd(&shared_hist[input[i]], 1u);
    }
    __syncthreads();

    for (unsigned int b = threadIdx.x; b < 256;
         b += blockDim.x) {
        unsigned int count = shared_hist[b];
        if (count != 0u) {
            atomicAdd(&hist[b], count);
        }
    }
}
