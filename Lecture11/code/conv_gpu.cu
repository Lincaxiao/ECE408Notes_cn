#include <cuda_runtime.h>
#include "tensor_index.h"

template<int T>
__global__ void conv_forward_kernel(
    int M, int C, int H, int W, int K, int tiles_w,
    const float* X, const float* filter, float* Y) {
  const int Hout = H - K + 1;
  const int Wout = W - K + 1;
  const int b = blockIdx.z;
  const int m = blockIdx.x;
  const int h = (blockIdx.y / tiles_w) * T + threadIdx.y;
  const int w = (blockIdx.y % tiles_w) * T + threadIdx.x;
  if (h >= Hout || w >= Wout) return;

  float acc = 0.0f;
  for (int c = 0; c < C; ++c)
    for (int p = 0; p < K; ++p)
      for (int q = 0; q < K; ++q) {
        const auto xi = off4(b, c, h+p, w+q, C, H, W);
        const auto fi = off4(m, c, p, q, C, K, K);
        acc += X[xi] * filter[fi];
      }
  Y[off4(b, m, h, w, M, Hout, Wout)] = acc;
}

cudaError_t launch_conv_forward(
    int B, int M, int C, int H, int W, int K,
    const float* dX, const float* dFilter, float* dY) {
  if (B <= 0 || M <= 0 || C <= 0 || K <= 0 || H < K || W < K)
    return cudaErrorInvalidValue;
  constexpr int T = 16;
  const int Hout = H - K + 1;
  const int Wout = W - K + 1;
  const int tiles_w = (Wout + T - 1) / T;
  const int tiles_h = (Hout + T - 1) / T;
  const dim3 block(T, T, 1);
  const dim3 grid(M, tiles_h * tiles_w, B);
  conv_forward_kernel<T><<<grid, block>>>(
      M, C, H, W, K, tiles_w, dX, dFilter, dY);
  return cudaGetLastError();
}
