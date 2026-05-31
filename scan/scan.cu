#include "kernels.cuh"
#include <vector>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <execution>
#include <cstdio>
#include <cassert>
#include <cstdlib>
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

int main()
{
    cudaEvent_t s, e;
    std::vector<long long> vec(ARRAY_SIZE);
    std::iota(vec.begin(), vec.end(), 1);

    float ms;
    int numBlocks = ARRAY_SIZE/(BLOCK_SIZE * STRIPSIZE);

    cudaEventCreate(&s);
    cudaEventCreate(&e);
    thrust::device_vector<long long> input = vec;
    blockVal* blockOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&blockOutput, sizeof(blockVal) * numBlocks));
    CUDA_CHECK(cudaMemset(blockOutput, 0, sizeof(blockVal) * numBlocks));

    std::inclusive_scan(vec.begin(), vec.end(), vec.begin());
    cudaEventRecord(s);
    // device_scan_ILP(RAW(input.data()), ARRAY_SIZE/(BLOCK_SIZE * STRIPSIZE * 2), BLOCK_SIZE, ARRAY_SIZE);
    // device_scan(RAW(input.data()), ARRAY_SIZE/(BLOCK_SIZE * STRIPSIZE), BLOCK_SIZE, ARRAY_SIZE);
    // scan_LB<<<numBlocks, BLOCK_SIZE, sizeof(long long) * (BLOCK_SIZE/32 + 1)>>>(RAW(input.data()), blockOutput);
    // scan_LB_ILP2<<<numBlocks/2, BLOCK_SIZE, sizeof(long long) * (BLOCK_SIZE/32 + 1)>>>(RAW(input.data()), blockOutput);
    scan_LB_Final2<<<numBlocks/2, BLOCK_SIZE, sizeof(long long) * (64)>>>(RAW(input.data()), blockOutput);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("Kernel complete in %f milliseconds!\n", ms);

    thrust::host_vector<long long> result = input;
    std::for_each(
        std::execution::par_unseq, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)vec.size()), [&] (int index) {
        assert(vec[index] == (long long)result[index]);
    });
    printf("All values matching!\n");

    CUDA_CHECK(cudaFree(blockOutput));
    return 0;
}
