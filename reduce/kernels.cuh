#include <cuda_runtime.h>


__global__ void reduce0(int *input, int *output) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int localId = threadIdx.x;
    unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[localId] = input[globalId];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int stride=1; stride < blockDim.x; stride *= 2) {
        if (localId % (2*stride) == 0) {
            sdata[localId] += sdata[localId + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (localId == 0) output[blockIdx.x] = sdata[0];
}
