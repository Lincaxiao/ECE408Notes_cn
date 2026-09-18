// ECE408 Lecture 8 - examples assembled from the supplied lecture.
// Convolution convention: no filter reversal; zero padding; float data.
// Build: nvcc -std=c++17 -O2 convolution_examples.cu -o convolution_examples
// These kernels use the same definitions as the Chinese lecture notes.
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int MASK_WIDTH = 5;
__constant__ float Mc[MASK_WIDTH];

__global__ void convolution_1D_basic_kernel(
    const float* N, const float* M, float* P,
    int Mask_Width, int Width)
{
    int i = int(blockIdx.x) * int(blockDim.x)
          + int(threadIdx.x);
    if (i >= Width) return;

    float Pvalue = 0.0f;
    int N_start_point = i - Mask_Width / 2;
    for (int j = 0; j < Mask_Width; ++j) {
        int q = N_start_point + j;
        if (q >= 0 && q < Width) {
            Pvalue += N[q] * M[j];
        }
    }
    P[i] = Pvalue;
}

__global__ void convolution_1D_constant_kernel(
    const float* N, float* P, int Mask_Width, int Width)
{
    int i = int(blockIdx.x) * int(blockDim.x)
          + int(threadIdx.x);
    if (i >= Width) return;

    float Pvalue = 0.0f;
    int N_start_point = i - Mask_Width / 2;
    for (int j = 0; j < Mask_Width; ++j) {
        int q = N_start_point + j;
        if (q >= 0 && q < Width) {
            Pvalue += N[q] * Mc[j];
        }
    }
    P[i] = Pvalue;
}

template<int B>
__global__ void convolution_1D_tiled_three_loads(
    const float* N, float* P, int Width)
{
    constexpr int radius = MASK_WIDTH / 2;
    static_assert(MASK_WIDTH % 2 == 1, "Odd mask required");
    static_assert(B >= radius, "B must cover each halo");
    __shared__ float N_ds[B + MASK_WIDTH - 1];
    int t = int(threadIdx.x);
    int s = int(blockIdx.x) * B;
    int i = s + t;

    if (t >= B - radius) {
        int q = s - B + t;
        N_ds[t - (B - radius)] =
            (0 <= q && q < Width) ? N[q] : 0.0f;
    }
    N_ds[radius + t] = (i < Width) ? N[i] : 0.0f;
    if (t < radius) {
        int q = s + B + t;
        N_ds[radius + B + t] =
            (0 <= q && q < Width) ? N[q] : 0.0f;
    }
    __syncthreads();

    float Pvalue = 0.0f;
    for (int j = 0; j < MASK_WIDTH; ++j) {
        Pvalue += N_ds[t + j] * Mc[j];
    }
    if (i < Width) P[i] = Pvalue;
}

template<int B>
__global__ void convolution_1D_tiled_two_loads(
    const float* N, float* P, int Width)
{
    constexpr int radius = MASK_WIDTH / 2;
    static_assert(MASK_WIDTH % 2 == 1, "Odd mask required");
    static_assert(B >= MASK_WIDTH - 1, "Two loads need B >= K-1");
    __shared__ float N_ds[B + MASK_WIDTH - 1];
    int t = int(threadIdx.x);
    int i = int(blockIdx.x) * B + t;
    int start = i - radius;

    N_ds[t] = (0 <= start && start < Width)
            ? N[start] : 0.0f;
    if (t < MASK_WIDTH - 1) {
        start += B;
        N_ds[t + B] = (0 <= start && start < Width)
                    ? N[start] : 0.0f;
    }
    __syncthreads();

    float Pvalue = 0.0f;
    for (int j = 0; j < MASK_WIDTH; ++j) {
        Pvalue += N_ds[t + j] * Mc[j];
    }
    if (i < Width) P[i] = Pvalue;
}

template<int B>
__global__ void convolution_1D_core_shared(
    const float* N, float* P, int Width)
{
    constexpr int radius = MASK_WIDTH / 2;
    static_assert(MASK_WIDTH % 2 == 1, "Odd mask required");
    __shared__ float N_ds[B];
    int t = int(threadIdx.x);
    int s = int(blockIdx.x) * B;
    int i = s + t;

    N_ds[t] = (i < Width) ? N[i] : 0.0f;
    __syncthreads();

    float Pvalue = 0.0f;
    for (int j = 0; j < MASK_WIDTH; ++j) {
        int N_index = i - radius + j;
        if (0 <= N_index && N_index < Width) {
            if (s <= N_index && N_index < s + B) {
                Pvalue += N_ds[t - radius + j] * Mc[j];
            } else {
                Pvalue += N[N_index] * Mc[j];
            }
        }
    }
    if (i < Width) P[i] = Pvalue;
}

static void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(status));
    }
}

struct DeviceArray {
    float* data = nullptr;
    explicit DeviceArray(std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data),
                              count * sizeof(float)), "cudaMalloc");
    }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

static std::vector<float> cpu_reference(
    const std::vector<float>& n, const std::array<float, MASK_WIDTH>& mask)
{
    const int w = static_cast<int>(n.size());
    std::vector<float> out(n.size(), 0.0f);
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < MASK_WIDTH; ++j) {
            const int q = i - MASK_WIDTH/2 + j;
            if (0 <= q && q < w) out[i] += n[q] * mask[j];
        }
    }
    return out;
}

template<int B>
static int run_tests(const std::array<float, MASK_WIDTH>& mask)
{
    const char* names[] = {"direct", "constant", "three-load",
                           "two-load", "core-only"};
    check_cuda(cudaMemcpyToSymbol(Mc, mask.data(),
                                  MASK_WIDTH * sizeof(float)),
               "cudaMemcpyToSymbol");
    int passed = 0;
    for (const int width : {1,2,3,4,5,7,16,127,128,129,257,513}) {
        std::vector<float> input(width), output(width);
        for (int i = 0; i < width; ++i)
            input[i] = static_cast<float>((17*i) % 23 - 11);
        const auto expected = cpu_reference(input, mask);
        const std::size_t bytes = input.size() * sizeof(float);
        DeviceArray nd(input.size()), pd(input.size()), md(MASK_WIDTH);
        check_cuda(cudaMemcpy(nd.data, input.data(), bytes,
                              cudaMemcpyHostToDevice), "copy input");
        check_cuda(cudaMemcpy(md.data, mask.data(), MASK_WIDTH*sizeof(float),
                              cudaMemcpyHostToDevice), "copy global mask");
        const int grid = 1 + (width - 1) / B;
        for (int method = 0; method < 5; ++method) {
            switch (method) {
                case 0:
                    convolution_1D_basic_kernel<<<grid, B>>>(
                        nd.data, md.data, pd.data, MASK_WIDTH, width);
                    break;
                case 1:
                    convolution_1D_constant_kernel<<<grid, B>>>(
                        nd.data, pd.data, MASK_WIDTH, width);
                    break;
                case 2:
                    convolution_1D_tiled_three_loads<B><<<grid, B>>>(
                        nd.data, pd.data, width);
                    break;
                case 3:
                    convolution_1D_tiled_two_loads<B><<<grid, B>>>(
                        nd.data, pd.data, width);
                    break;
                default:
                    convolution_1D_core_shared<B><<<grid, B>>>(
                        nd.data, pd.data, width);
                    break;
            }
            check_cuda(cudaGetLastError(), "kernel launch");
            check_cuda(cudaDeviceSynchronize(), "kernel execution");
            check_cuda(cudaMemcpy(output.data(), pd.data, bytes,
                                  cudaMemcpyDeviceToHost), "copy output");
            for (int i = 0; i < width; ++i) {
                const float tolerance = 1.0e-4f + 1.0e-5f*std::abs(expected[i]);
                if (!std::isfinite(output[i]) ||
                    std::abs(output[i] - expected[i]) > tolerance) {
                    throw std::runtime_error(std::string(names[method])
                        + " mismatch at W=" + std::to_string(width)
                        + ", B=" + std::to_string(B)
                        + ", i=" + std::to_string(i));
                }
            }
            ++passed;
        }
    }
    return passed;
}

int main()
{
    try {
        int count = 0;
        const std::array<float, MASK_WIDTH> lecture_mask = {3,4,5,4,3};
        const std::array<float, MASK_WIDTH> asymmetric_mask = {1,-2,3,-4,5};
        count += run_tests<4>(lecture_mask);
        count += run_tests<128>(lecture_mask);
        count += run_tests<4>(asymmetric_mask);
        count += run_tests<128>(asymmetric_mask);
        std::cout << "PASS: " << count << " GPU/CPU comparisons\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
