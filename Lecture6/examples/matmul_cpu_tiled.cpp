void matmul_cpu_tiled(
    const float* A, const float* B, float* C, int N, int T)
{
    for (int rt = 0; rt < N / T; ++rt) {
        for (int ct = 0; ct < N / T; ++ct) {
            for (int qt = 0; qt < N / T; ++qt) {
                for (int row = rt * T;
                     row < (rt + 1) * T; ++row) {
                    for (int col = ct * T;
                         col < (ct + 1) * T; ++col) {
                        float sum = 0.0f;
                        for (int j = qt * T;
                             j < (qt + 1) * T; ++j) {
                            sum += A[row * N + j]
                                 * B[j * N + col];
                        }
                        if (qt == 0) {
                            C[row * N + col] = sum;
                        } else {
                            C[row * N + col] += sum;
                        }
                    }
                }
            }
        }
    }
}
