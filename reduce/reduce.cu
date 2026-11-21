#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = (call);                                           \
        if (error != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

int main() {
    constexpr int n = 4 * 1024 * 1024;
    constexpr int THREADS_PER_BLOCK = 1024;
    constexpr int BLOCKS = n / THREADS_PER_BLOCK;
    std::vector<int> input(n, 1);
    std::vector<int> output(BLOCKS, 0);

    int *d_input;
    int *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, BLOCKS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    reduce0<<<BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_input, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost));
    int result = std::accumulate(output.begin(), output.end(), 0);
    int cpu_result = std::accumulate(input.begin(), input.end(), 0);
    std::printf("GPU result: %d, CPU result: %d -> %s\n", result, cpu_result,
                result == cpu_result ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
