#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>
#include <limits>

// Contract: blockDim == (BT, 1, 1); every thread calls this helper.
// The final barrier lets the caller safely reuse the same scratch array.
template <int BT, bool ReduceMax>
__device__ __forceinline__ float block_reduce(float value, float* scratch) {
    static_assert(BT > 0 && (BT & (BT - 1)) == 0,
                  "BT must be a positive power of two");
    const unsigned int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();

    for (unsigned int step = BT / 2; step > 0; step >>= 1) {
        if (tid < step) {
            const float other = scratch[tid + step];
            scratch[tid] = ReduceMax ? fmaxf(scratch[tid], other)
                                     : scratch[tid] + other;
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

// One block per row. S already contains the 1/sqrt(d) scaling.
// S and P are disjoint row-major N-by-N device buffers; N >= 1.
// Only S[row, j] with j <= row must be finite; masked entries are not read.
template <int BT>
__global__ void causal_softmax_kernel(const float* __restrict__ S,
                                      float* __restrict__ P, int N) {
    __shared__ float scratch[BT];
    const std::size_t row = blockIdx.x;
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t base = row * n;
    const std::size_t tid = threadIdx.x;

    // Inactive threads contribute the identity for a maximum reduction.
    float local_max = -CUDART_INF_F;
    for (std::size_t j = tid; j <= row; j += BT) {
        local_max = fmaxf(local_max, S[base + j]);
    }
    const float row_max = block_reduce<BT, true>(local_max, scratch);

    // P temporarily stores exponentials, owned by the same thread in both loops.
    // Every row includes its diagonal, so at least one exponential equals one.
    float local_sum = 0.0f;
    for (std::size_t j = tid; j <= row; j += BT) {
        const float value = expf(S[base + j] - row_max);
        P[base + j] = value;
        local_sum += value;
    }
    const float row_sum = block_reduce<BT, false>(local_sum, scratch);

    // The conditional avoids reading an uninitialized masked entry of P.
    for (std::size_t j = tid; j < n; j += BT) {
        P[base + j] = (j <= row) ? P[base + j] / row_sum : 0.0f;
    }
}

// The caller supplies disjoint buffers of at least N*N floats on this device.
// Returns argument/device/launch errors; does not wait for kernel completion.
cudaError_t launch_causal_softmax(const float* S, float* P, int N,
                                 cudaStream_t stream = nullptr) {
    constexpr int BT = 256;
    if (S == nullptr || P == nullptr || S == P || N <= 0)
        return cudaErrorInvalidValue;
    const std::size_t n = static_cast<std::size_t>(N);
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float) / n)
        return cudaErrorInvalidValue;

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) return status;
    if (N > prop.maxGridSize[0] || BT > prop.maxThreadsPerBlock ||
        BT > prop.maxThreadsDim[0] ||
        BT * sizeof(float) > prop.sharedMemPerBlock)
        return cudaErrorInvalidConfiguration;

    causal_softmax_kernel<BT><<<static_cast<unsigned int>(N), BT, 0, stream>>>(
        S, P, N);
    return cudaGetLastError();
}
