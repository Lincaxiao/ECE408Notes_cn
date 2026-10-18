#pragma once
#include <cstddef>

#ifdef __CUDACC__
#define LEC11_HD __host__ __device__
#else
#define LEC11_HD
#endif

LEC11_HD inline std::size_t off4(
    int i0, int i1, int i2, int i3,
    int d1, int d2, int d3) {
  return ((static_cast<std::size_t>(i0) * d1 + i1)
          * d2 + i2) * d3 + i3;
}

#undef LEC11_HD
