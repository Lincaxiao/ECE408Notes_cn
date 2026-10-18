#include <cmath>
#include "tensor_index.h"

static float sigmoid_cpu(float z) {
  const float e = std::exp(-std::fabs(z));
  return z >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

void pool_forward_cpu(
    int B, int M, int Hout, int Wout, int N,
    const float* Y, const float* bias, float* S) {
  const int Hs = Hout / N;
  const int Ws = Wout / N;
  for (int b = 0; b < B; ++b)
    for (int m = 0; m < M; ++m)
      for (int x = 0; x < Hs; ++x)
        for (int y = 0; y < Ws; ++y) {
          float acc = 0.0f;
          for (int p = 0; p < N; ++p)
            for (int q = 0; q < N; ++q)
              acc += Y[off4(b, m, N*x+p, N*y+q,
                            M, Hout, Wout)];
          const float mean = acc / (static_cast<float>(N) * N);
          S[off4(b, m, x, y, M, Hs, Ws)] =
              sigmoid_cpu(mean + bias[m]);
        }
}
