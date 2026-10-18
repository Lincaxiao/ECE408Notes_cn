#include "tensor_index.h"

void conv_forward_cpu(
    int B, int M, int C, int H, int W, int K,
    const float* X, const float* filter, float* Y) {
  const int Hout = H - K + 1;
  const int Wout = W - K + 1;
  for (int b = 0; b < B; ++b)
    for (int m = 0; m < M; ++m)
      for (int h = 0; h < Hout; ++h)
        for (int w = 0; w < Wout; ++w) {
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
}
