#include "kernels.cuh"
#include <vector>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <execution>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <string>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define RAW(x) thrust::raw_pointer_cast(x)
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

void device_scan_ILP(long long* input, int numBlocks, int blockSize, int size)
{
    if (size <= blockSize) {
        // Quick CUB based scan
        void* tempStorage = nullptr;
        size_t tempStorageBytes = 0;

        CUDA_CHECK(cub::DeviceScan::InclusiveSum(tempStorage, tempStorageBytes, input, input, size));
        thrust::device_vector<unsigned char> tempStorageBuffer(tempStorageBytes);
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            RAW(tempStorageBuffer.data()), tempStorageBytes, input, input, size));

        return;
    }
    thrust::device_vector<long long> blockOutput(numBlocks);
    scan_ILP<<<numBlocks, blockSize, sizeof(long long) * blockSize>>>(input, thrust::raw_pointer_cast(blockOutput.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
    device_scan_ILP(thrust::raw_pointer_cast(blockOutput.data()), numBlocks/(blockSize * STRIPSIZE * 2), blockSize, numBlocks);
    add_offset_ILP<<<numBlocks, blockSize, 0>>>(input, RAW(blockOutput.data()), size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void device_scan(long long* input, int numBlocks, int blockSize, int size)
{
    if (size <= blockSize) {
        // Quick CUB based scan
        void* tempStorage = nullptr;
        size_t tempStorageBytes = 0;

        CUDA_CHECK(cub::DeviceScan::InclusiveSum(tempStorage, tempStorageBytes, input, input, size));
        thrust::device_vector<unsigned char> tempStorageBuffer(tempStorageBytes);
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            RAW(tempStorageBuffer.data()), tempStorageBytes, input, input, size));

        return;
    }
    thrust::device_vector<long long> blockOutput(numBlocks);
    scan<<<numBlocks, blockSize, sizeof(long long) * blockSize>>>(input, thrust::raw_pointer_cast(blockOutput.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
    device_scan(thrust::raw_pointer_cast(blockOutput.data()), numBlocks/(blockSize * STRIPSIZE), blockSize, numBlocks);
    add_offset<<<numBlocks, blockSize, 0>>>(input, RAW(blockOutput.data()), size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

constexpr int NUM_BLOCKS = ARRAY_SIZE / (BLOCK_SIZE * STRIPSIZE);
// warp 0 indexes warp_totals[0..31], so LB kernels need >= 32 slots.
constexpr size_t LB_SMEM = sizeof(long long) * 64;

struct KernelEntry {
    const char* name;
    void (*launch)(long long* input, blockVal* blockOutput);
};

static const KernelEntry kKernels[] = {
    { "scan",              [](long long* in, blockVal*)    { device_scan(in, NUM_BLOCKS, BLOCK_SIZE, ARRAY_SIZE); } },
    { "scan_ILP",          [](long long* in, blockVal*)    { device_scan_ILP(in, NUM_BLOCKS / 2, BLOCK_SIZE, ARRAY_SIZE); } },
    { "scan_LB",           [](long long* in, blockVal* bo) { scan_LB<<<NUM_BLOCKS, BLOCK_SIZE, LB_SMEM>>>(in, bo); } },
    { "scan_LB_Final",     [](long long* in, blockVal* bo) { scan_LB_Final<<<NUM_BLOCKS / 2, BLOCK_SIZE, LB_SMEM>>>(in, bo); } },
    { "scan_LB_Final2",    [](long long* in, blockVal* bo) { scan_LB_Final2<<<NUM_BLOCKS / 2, BLOCK_SIZE, LB_SMEM>>>(in, bo); } },
};

static bool run_kernel(const KernelEntry& k, const std::vector<long long>& init,
                       const std::vector<long long>& expected, thrust::device_vector<long long>& input,
                       blockVal* blockOutput, int iters)
{
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    const int zero = 0;
    float total_ms = 0.f, min_ms = 1e30f;

    for (int it = 0; it < iters; ++it) {
        thrust::copy(init.begin(), init.end(), input.begin());
        CUDA_CHECK(cudaMemset(blockOutput, 0, sizeof(blockVal) * NUM_BLOCKS));
        CUDA_CHECK(cudaMemcpyToSymbol(counter, &zero, sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(s));
        k.launch(RAW(input.data()), blockOutput);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
    }
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));

    thrust::host_vector<long long> result = input;
    auto mm = std::mismatch(std::execution::par_unseq, expected.begin(), expected.end(), result.begin());
    bool ok = (mm.first == expected.end());

    printf("%-18s iters=%-4d avg=%.4f ms  min=%.4f ms  %s", k.name, iters, total_ms / iters, min_ms,
           ok ? "PASS\n" : "FAIL");
    if (!ok) {
        size_t idx = mm.first - expected.begin();
        printf(" (first mismatch at %zu: expected %lld, got %lld)\n", idx, *mm.first, *mm.second);
    }
    return ok;
}

static void usage(const char* prog)
{
    printf("Usage: %s [kernel|all] [iters]\n  kernels:", prog);
    for (const auto& k : kKernels) printf(" %s", k.name);
    printf("\n  default: all 1\n");
}

int main(int argc, char** argv)
{
    std::string which = (argc > 1) ? argv[1] : "all";
    int iters = (argc > 2) ? std::atoi(argv[2]) : 1;
    if (which == "-h" || which == "--help" || iters <= 0) { usage(argv[0]); return iters <= 0; }

    std::vector<long long> init(ARRAY_SIZE);
    std::iota(init.begin(), init.end(), 1);
    std::vector<long long> expected(ARRAY_SIZE);
    std::inclusive_scan(init.begin(), init.end(), expected.begin());

    thrust::device_vector<long long> input(ARRAY_SIZE);
    blockVal* blockOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&blockOutput, sizeof(blockVal) * NUM_BLOCKS));

    bool all_ok = true, found = false;
    for (const auto& k : kKernels) {
        if (which != "all" && which != k.name) continue;
        found = true;
        all_ok &= run_kernel(k, init, expected, input, blockOutput, iters);
    }
    if (!found) { fprintf(stderr, "Unknown kernel '%s'\n", which.c_str()); usage(argv[0]); }

    CUDA_CHECK(cudaFree(blockOutput));
    return (found && all_ok) ? 0 : 1;
}
