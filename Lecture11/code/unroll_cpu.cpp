#include "tensor_index.h"

void unroll_cpu(
    int B, int C, int H, int W, int K,
    const float* X, float* U) {
  const int Hout = H - K + 1;
  const int Wout = W - K + 1;
  const std::size_t D = static_cast<std::size_t>(C) * K * K;
  const std::size_t L = static_cast<std::size_t>(Hout) * Wout;
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int p = 0; p < K; ++p)
        for (int q = 0; q < K; ++q) {
          const std::size_t r =
              (static_cast<std::size_t>(c) * K + p) * K + q;
          for (int h = 0; h < Hout; ++h)
            for (int w = 0; w < Wout; ++w) {
              const auto ell = static_cast<std::size_t>(h) * Wout + w;
              U[(static_cast<std::size_t>(b) * D + r) * L + ell] =
                  X[off4(b, c, h+p, w+q, C, H, W)];
            }
        }
}
