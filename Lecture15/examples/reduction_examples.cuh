#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;

// 第15讲归约代码，要求 CUDA C++17。
// 正文中的九个完整函数保留原样；另补全 reduce_hybrid 与
// reduce_two_stage_group，分别展示混合树和正确的协作组调用范围。
// 验证范围：已核对索引、同步语义并进行 CPU 语义模拟；
// 当前环境未使用 nvcc 编译本文件，也未在 GPU 上执行。
//
// 启动条件：所有核函数使用一维网格、一维线程块。
// 以 B 为模板参数的核函数要求 blockDim.x == B，
// B 同时满足函数条件和当前设备限制。输入输出的有效区间不重叠。
// 网格块数必须处于当前设备 maxGridSize[0] 的允许范围内；
// launch_atomic_sum 的 unsigned int 转换依赖这一前提。
//
// 输出容量：未粗化的部分和输出至少 ceil(n / (2*B)) 项；
// 粗化的部分和输出至少 ceil(n / (2*B*C)) 项；原子输出为一项。
// 原子模式的输出在整个网格开始前清零一次，不能在各块内清零。
// launch_atomic_sum 在同一流中提交清零和计算，返回时结果可能尚未就绪。
// 空输入由 launch_atomic_sum 清零并跳过核函数；多轮驱动也需单独处理空输入。

__global__ void reduce_atomic(const float* input,
                              size_t n, float* sum) {
    size_t i = size_t(blockIdx.x) * blockDim.x
             + threadIdx.x;
    if (i < n) {
        atomicAdd(sum, input[i]);
    }
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu,
                                  value, offset);
    }
    return value;
}

__device__ __forceinline__
float warp_sum_group(const cg::thread_block_tile<32>& tile,
                     float value) {
    for (int offset = tile.size() / 2;
         offset > 0; offset /= 2) {
        value += tile.shfl_down(value, offset);
    }
    return value;
}

// 以下两个原地全局树要求 B 为二的幂，输入已补齐至 2*B*gridDim.x 项。
template<int B>
__global__ void reduce_interleaved(float* data,
                                   float* partial) {
    unsigned int t = threadIdx.x;
    size_t base = size_t(2) * B * blockIdx.x;
    size_t i = base + 2 * t;
    for (unsigned int s = 1; s <= B; s *= 2) {
        if (t % s == 0) {
            data[i] += data[i + s];
        }
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = data[base];
}

template<int B>
__global__ void reduce_contiguous(float* data,
                                  float* partial) {
    unsigned int t = threadIdx.x;
    size_t base = size_t(2) * B * blockIdx.x;
    size_t i = base + t;
    for (unsigned int s = B; s > 0; s /= 2) {
        if (t < s) {
            data[i] += data[i + s];
        }
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = data[base];
}

// 以下非原地核函数在读取处检查输入长度，尾部位置按加法单位元处理。
template<int B>
__global__ void reduce_shared(const float* input,
                              size_t n, float* partial) {
    static_assert(B > 0 && (B & (B - 1)) == 0,
                  "B must be a power of two");
    __shared__ float values[B];
    unsigned int t = threadIdx.x;
    size_t i = size_t(2) * B * blockIdx.x + t;

    float a = (i < n) ? input[i] : 0.0f;
    float b = (i + B < n) ? input[i + B] : 0.0f;
    values[t] = a + b;
    __syncthreads();

    for (unsigned int s = B / 2; s > 0; s /= 2) {
        if (t < s) values[t] += values[t + s];
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = values[0];
}

template<int B>
__global__ void reduce_two_stage(const float* input,
                                 size_t n, float* partial) {
    static_assert(B >= 32 && B <= 1024 && B % 32 == 0,
                  "B must contain complete warps");
    constexpr int Q = B / 32;
    __shared__ float warp_totals[Q];
    unsigned int t = threadIdx.x;
    size_t i = size_t(2) * B * blockIdx.x + t;

    float a = (i < n) ? input[i] : 0.0f;
    float b = (i + B < n) ? input[i + B] : 0.0f;
    float value = warp_sum(a + b);
    if (t % 32 == 0) warp_totals[t / 32] = value;
    __syncthreads();

    if (t < 32) {
        value = (t < Q) ? warp_totals[t] : 0.0f;
        value = warp_sum(value);
        if (t == 0) partial[blockIdx.x] = value;
    }
}

template<int B, int C, bool AtomicOutput>
__global__ void reduce_coarsened(const float* input,
                                 size_t n, float* output) {
    static_assert(B >= 32 && B <= 1024 && B % 32 == 0,
                  "B must contain complete warps");
    static_assert(C >= 1, "C must be positive");
    constexpr int Q = B / 32;
    __shared__ float warp_totals[Q];
    unsigned int t = threadIdx.x;
    size_t i = size_t(2) * C * B * blockIdx.x + t;

    float value = (i < n) ? input[i] : 0.0f;
    for (size_t tile = 1; tile < size_t(2) * C; ++tile) {
        size_t j = i + tile * B;
        if (j < n) value += input[j];
    }

    value = warp_sum(value);
    if (t % 32 == 0) warp_totals[t / 32] = value;
    __syncthreads();

    if (t < 32) {
        value = (t < Q) ? warp_totals[t] : 0.0f;
        value = warp_sum(value);
        if (t == 0) {
            if constexpr (AtomicOutput) {
                atomicAdd(output, value);
            } else {
                output[blockIdx.x] = value;
            }
        }
    }
}

// 同一流内的异步原子求和包装函数，sum 指向已分配的设备浮点数。
template<int B, int C>
cudaError_t launch_atomic_sum(const float* input,
                             size_t n, float* sum,
                             cudaStream_t stream = nullptr) {
    cudaError_t status =
        cudaMemsetAsync(sum, 0, sizeof(float), stream);
    if (status != cudaSuccess || n == 0) return status;

    constexpr size_t chunk = size_t(2) * B * C;
    size_t blocks = n / chunk + (n % chunk != 0);
    reduce_coarsened<B, C, true>
        <<<static_cast<unsigned int>(blocks), B, 0, stream>>>
        (input, n, sum);
    return cudaGetLastError();
}

// 将正文的共享内存装载和混合归约片段连接为完整核函数。
template<int B>
__global__ void reduce_hybrid(const float* input,
                              size_t n, float* partial) {
    static_assert(B >= 64 && B <= 1024 && (B & (B - 1)) == 0,
                  "B must be a power of two, from 64 to 1024");
    __shared__ float values[B];
    unsigned int t = threadIdx.x;
    size_t i = size_t(2) * B * blockIdx.x + t;

    float a = (i < n) ? input[i] : 0.0f;
    float b = (i + B < n) ? input[i + B] : 0.0f;
    values[t] = a + b;
    __syncthreads();

    for (unsigned int s = B / 2; s > 32; s /= 2) {
        if (t < s) values[t] += values[t + s];
        __syncthreads();
    }

    if (t < 32) {
        float value = values[t] + values[t + 32];
        value = warp_sum(value);
        if (t == 0) partial[blockIdx.x] = value;
    }
}

// 分组由线程块全员在分支前创建；第二阶段只使用已建立的第一个子组。
template<int B>
__global__ void reduce_two_stage_group(const float* input,
                                 size_t n, float* partial) {
    static_assert(B >= 32 && B <= 1024 && B % 32 == 0,
                  "B must contain complete warps");
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile =
        cg::tiled_partition<32>(block);
    constexpr int Q = B / 32;
    __shared__ float warp_totals[Q];
    unsigned int t = threadIdx.x;
    size_t i = size_t(2) * B * blockIdx.x + t;

    float a = (i < n) ? input[i] : 0.0f;
    float b = (i + B < n) ? input[i + B] : 0.0f;
    float value = warp_sum_group(tile, a + b);
    if (t % 32 == 0) warp_totals[t / 32] = value;
    __syncthreads();

    if (t < 32) {
        value = (t < Q) ? warp_totals[t] : 0.0f;
        value = warp_sum_group(tile, value);
        if (t == 0) partial[blockIdx.x] = value;
    }
}
