#include <cuda_runtime.h>
#include "tensor_index.h"

__global__ void unroll_kernel(
    int B, int C, int H, int W, int K,
    const float* X, float* U) {
  const int Hout = H - K + 1;
  const int Wout = W - K + 1;
  const std::size_t D = static_cast<std::size_t>(C) * K * K;
  const std::size_t L = static_cast<std::size_t>(Hout) * Wout;
  const std::size_t total = static_cast<std::size_t>(B) * D * L;
  const std::size_t step =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += step) {
    const std::size_t ell = idx % L;
    const std::size_t r = (idx / L) % D;
    const int b = idx / (D * L);
    const int c = r / K / K;
    const int p = (r / K) % K;
    const int q = r % K;
    const int h = ell / Wout;
    const int w = ell % Wout;
    U[idx] = X[off4(b, c, h+p, w+q, C, H, W)];
  }
}
