#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/atomic>

#define STRIPSIZE 4
constexpr int BLOCK_SIZE = 256;
static constexpr int ARRAY_SIZE = 4194304;
enum STATUS { NOT_READY, AGGREGATE_READY, PREFIX_READY };

typedef struct {
    long long aggregate;        // written once at AGGREGATE_READY publish; stable thereafter
    long long inclusive_prefix; // written once at PREFIX_READY publish; stable thereafter
    cuda::atomic<STATUS, cuda::thread_scope_device> status;
} blockVal;

__global__ void add_offset(long long* input, long long* offsets, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long off = (blockIdx.x == 0) ? 0LL : offsets[blockIdx.x - 1];

    for (int i = 0; i < STRIPSIZE; i++) input[tid * STRIPSIZE + i] += off;
}

__global__ void add_offset_ILP(long long* input, long long* offsets, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long off = (blockIdx.x == 0) ? 0LL : offsets[blockIdx.x - 1];

    longlong4_32a data1 = reinterpret_cast<longlong4_32a*>(input)[tid * 2];
    longlong4_32a data2 = reinterpret_cast<longlong4_32a*>(input)[tid * 2 + 1];
    data1 = { data1.x  + off, data1.y  + off, data1.z  + off, data1.w  + off };
    data2 = { data2.x + off, data2.y + off, data2.z + off, data2.w + off };
    reinterpret_cast<longlong4_32a*>(input)[tid * 2]     = data1;
    reinterpret_cast<longlong4_32a*>(input)[tid * 2 + 1] = data2;
}

__global__ void scan(long long* input, long long* output) {
    extern __shared__ long long warp_totals[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    longlong4_32a data = reinterpret_cast<longlong4_32a*>(input)[tid];

    long long strip_prefix[STRIPSIZE] = { data.x, data.y, data.z, data.w };
    for(int i=1; i<STRIPSIZE; ++i) strip_prefix[i] += strip_prefix[i - 1];

    // Try initing thread_prefix_sum with 0 and remove deletion of strip_prefix[K-1]
    int lane = threadIdx.x & (warpSize - 1);
    long long thread_prefix_sum = strip_prefix[STRIPSIZE - 1];
    for (int offset=1; offset<warpSize; offset *= 2) {
        long long val = __shfl_up_sync(0xffffffff, thread_prefix_sum, offset);
        if (lane >= offset) thread_prefix_sum += val;
    }
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix[i] += thread_prefix_sum - strip_prefix[STRIPSIZE - 1];

    int warp_id = threadIdx.x / warpSize;
    if (lane == (warpSize - 1)) warp_totals[warp_id] = thread_prefix_sum;
    __syncthreads();

    int num_warps = blockDim.x/warpSize;
    if (warp_id == 0) {
        long long warp_prefix_sum = warp_totals[threadIdx.x];
        for (int offset = 1; offset < warpSize; offset *= 2) {
            long long val = __shfl_up_sync(0xffffffff, warp_prefix_sum, offset);
            if (lane >= offset) warp_prefix_sum += val;
        }
        warp_totals[threadIdx.x] = warp_prefix_sum;
    }
    __syncthreads();

    long long offset = (warp_id == 0) ? 0 : warp_totals[warp_id - 1];
    reinterpret_cast<longlong4_32a*>(input)[tid] =
        { strip_prefix[0] + offset, strip_prefix[1] + offset,
          strip_prefix[2] + offset, strip_prefix[3] + offset };

    if (threadIdx.x == 0) output[blockIdx.x] = warp_totals[num_warps - 1];
}

__global__ void scan_ILP(long long* input, long long* output) {
    extern __shared__ long long warp_totals[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    longlong4_32a data  = reinterpret_cast<longlong4_32a*>(input)[tid * 2];
    longlong4_32a data2 = reinterpret_cast<longlong4_32a*>(input)[tid * 2 + 1];

    long long strip_prefix[STRIPSIZE]  = { data.x,  data.y,  data.z,  data.w  };
    long long strip_prefix2[STRIPSIZE] = { data2.x, data2.y, data2.z, data2.w };
    for(int i=1; i<STRIPSIZE; ++i) strip_prefix[i]  += strip_prefix[i - 1];
    for(int i=1; i<STRIPSIZE; ++i) strip_prefix2[i] += strip_prefix2[i - 1];

    long long own_data_total = strip_prefix[STRIPSIZE - 1];
    long long thread_total   = own_data_total + strip_prefix2[STRIPSIZE - 1];

    int lane = threadIdx.x & (warpSize - 1);
    long long thread_prefix_sum = thread_total;
    for (int offset = 1; offset < warpSize; offset *= 2) {
        long long val = __shfl_up_sync(0xffffffff, thread_prefix_sum, offset);
        if (lane >= offset) thread_prefix_sum += val;
    }
    long long exclusive = thread_prefix_sum - thread_total;
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix[i]  += exclusive;
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix2[i] += exclusive + own_data_total;

    int warp_id = threadIdx.x / warpSize;
    if (lane == (warpSize - 1)) warp_totals[warp_id] = thread_prefix_sum;
    __syncthreads();

    int num_warps = blockDim.x / warpSize;
    if (warp_id == 0) {
        long long warp_prefix_sum = warp_totals[threadIdx.x];
        for (int offset = 1; offset < warpSize; offset *= 2) {
            long long val = __shfl_up_sync(0xffffffff, warp_prefix_sum, offset);
            if (lane >= offset) warp_prefix_sum += val;
        }
        warp_totals[threadIdx.x] = warp_prefix_sum;
    }
    __syncthreads();

    long long offset = (warp_id == 0) ? 0 : warp_totals[warp_id - 1];
    data  = { strip_prefix[0]  + offset, strip_prefix[1]  + offset,
              strip_prefix[2]  + offset, strip_prefix[3]  + offset };
    data2 = { strip_prefix2[0] + offset, strip_prefix2[1] + offset,
              strip_prefix2[2] + offset, strip_prefix2[3] + offset };
    reinterpret_cast<longlong4_32a*>(input)[tid * 2]     = data;
    reinterpret_cast<longlong4_32a*>(input)[tid * 2 + 1] = data2;

    if (threadIdx.x == 0) output[blockIdx.x] = warp_totals[num_warps - 1];
}

__device__ int counter = 0;

__global__ void scan_LB(long long* input, volatile blockVal* output) {
    extern __shared__ long long warp_totals[];
    int num_warps = blockDim.x / warpSize;

    // Dynamic block id: Thread 0 bumps global counter and broadcasts blockId through shared-mem
    if (threadIdx.x == 0) warp_totals[num_warps] = atomicAdd(&counter, 1);
    __syncthreads();
    int newBlockId = (int)warp_totals[num_warps];

    // Data partitioning must follow the dynamic block id, Otherwise the lookback 
    // chain gets aggregates from completely unrelated input segments.
    int tid = threadIdx.x + newBlockId * blockDim.x;

    longlong4_32a data = reinterpret_cast<longlong4_32a*>(input)[tid];
    long long strip_prefix[STRIPSIZE] = { data.x, data.y, data.z, data.w };
    for(int i=1; i<STRIPSIZE; ++i) strip_prefix[i] += strip_prefix[i - 1];

    int lane = threadIdx.x & (warpSize - 1);
    long long thread_prefix_sum = strip_prefix[STRIPSIZE - 1];
    for (int offset=1; offset<warpSize; offset *= 2) {
        long long val = __shfl_up_sync(0xffffffff, thread_prefix_sum, offset);
        if (lane >= offset) thread_prefix_sum += val;
    }
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix[i] += thread_prefix_sum - strip_prefix[STRIPSIZE - 1];

    int warp_id = threadIdx.x / warpSize;
    if (lane == (warpSize - 1)) warp_totals[warp_id] = thread_prefix_sum;
    __syncthreads();

    if (warp_id == 0) {
        long long warp_prefix_sum = warp_totals[threadIdx.x];
        for (int offset = 1; offset < warpSize; offset *= 2) {
            long long val = __shfl_up_sync(0xffffffff, warp_prefix_sum, offset);
            if (lane >= offset) warp_prefix_sum += val;
        }
        warp_totals[threadIdx.x] = warp_prefix_sum;
    }
    __syncthreads();

    long long predecessor_prefix = 0;
    if (threadIdx.x == 0) {
        long long my_aggregate = warp_totals[num_warps - 1];
        if (newBlockId == 0) {
            // First tile has no predecessors, so its inclusive prefix is just
            // its aggregate. Publish straight to PREFIX_READY.
            output[newBlockId].inclusive_prefix = my_aggregate;
            __threadfence();
            output[newBlockId].status = PREFIX_READY;
        }
        else {
            // AGGREGATE publish: release pattern with a dedicated field that
            // is written exactly once. Separating it from `inclusive_prefix`
            // avoids the AGG/PREFIX race where a single shared `value` field
            // could be observed as the newer (PREFIX) bits while status still
            // read AGGREGATE_READY.
            output[newBlockId].aggregate = my_aggregate;
            __threadfence();
            output[newBlockId].status = AGGREGATE_READY;

            int current_predecessor = newBlockId - 1;
            while (1) {                                             // Lookback
                // Volatile load forces a fresh fetch each iteration.
                STATUS state = output[current_predecessor].status;
                if (state == NOT_READY) continue;
                __threadfence();

                if (state == PREFIX_READY) {
                    // Pairs with the producer's release fence at PREFIX publish.
                    predecessor_prefix += output[current_predecessor].inclusive_prefix;
                    break;
                }
                // AGGREGATE_READY: `aggregate` is write-once, so it's safe
                // to read even if status later flips to PREFIX_READY.
                predecessor_prefix += output[current_predecessor].aggregate;
                current_predecessor -= 1;
            }
            // PREFIX publish to a separate field; status flip is the release.
            output[newBlockId].inclusive_prefix = my_aggregate + predecessor_prefix;
            __threadfence();
            output[newBlockId].status = PREFIX_READY;
        }
        // Broadcast predecessor_prefix to the rest of the block. Reusing the
        // same scratch slot that held newBlockId — safe because every thread
        // already cached newBlockId into a register at the top of the kernel.
        warp_totals[num_warps] = predecessor_prefix;
    } // threadIdx.x == 0
    __syncthreads();

    predecessor_prefix = warp_totals[num_warps];
    long long offset = (warp_id == 0) ? 0 : warp_totals[warp_id - 1];
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix[i] += offset + predecessor_prefix;

    reinterpret_cast<longlong4_32a*>(input)[tid] =
        { strip_prefix[0], strip_prefix[1], strip_prefix[2], strip_prefix[3] };
}

__global__ void scan_LB_Final(long long* input, blockVal* output) {
    extern __shared__ long long warp_totals[];
    __shared__ long long predecessor_prefix;
    __shared__ long long block_aggregate;
    int num_warps = blockDim.x / warpSize;

    // Assign input tiles in execution order to make lookback progress safe.
    if (threadIdx.x == 0) warp_totals[num_warps] = atomicAdd(&counter, 1);
    __syncthreads();
    int newBlockId = (int)warp_totals[num_warps];
    int tid = threadIdx.x + newBlockId * blockDim.x;

    // Scan each thread's two contiguous vector strips.
    longlong4_32a data = reinterpret_cast<longlong4_32a*>(input)[tid * 2];
    longlong4_32a data2 = reinterpret_cast<longlong4_32a*>(input)[tid * 2 + 1];
    long long strip_prefix[STRIPSIZE] = { data.x, data.y, data.z, data.w };
    long long strip_prefix2[STRIPSIZE] = { data2.x, data2.y, data2.z, data2.w };
    for(int i=1; i<STRIPSIZE; ++i) strip_prefix[i] += strip_prefix[i - 1];
    strip_prefix2[0] += strip_prefix[STRIPSIZE - 1];
    for(int i=1; i<STRIPSIZE; ++i) strip_prefix2[i] += strip_prefix2[i - 1];

    int lane = threadIdx.x & (warpSize - 1);
    long long thread_prefix_sum = strip_prefix2[STRIPSIZE - 1];
    // Scan thread totals within each warp.
    for (int offset=1; offset<warpSize; offset *= 2) {
        long long val = __shfl_up_sync(0xffffffff, thread_prefix_sum, offset);
        if (lane >= offset) thread_prefix_sum += val;
    }
    long long exclusive = thread_prefix_sum - strip_prefix2[STRIPSIZE - 1];
    for(int i=0; i<STRIPSIZE; ++i) { strip_prefix[i]  += exclusive; strip_prefix2[i] += exclusive; }

    int warp_id = threadIdx.x / warpSize;
    if (lane == (warpSize - 1)) warp_totals[warp_id] = thread_prefix_sum;
    __syncthreads();

    // Warp 0 scans the per-warp totals to form the block-local scan.
    if (warp_id == 0) {
        long long warp_prefix_sum = warp_totals[threadIdx.x];
        for (int offset = 1; offset < warpSize; offset *= 2) {
            long long val = __shfl_up_sync(0xffffffff, warp_prefix_sum, offset);
            if (lane >= offset) warp_prefix_sum += val;
        }
        warp_totals[threadIdx.x] = warp_prefix_sum;
    }

    if (threadIdx.x == 0) {
        predecessor_prefix = 0;
        block_aggregate = warp_totals[num_warps - 1];

        // Publish either the complete first prefix or this tile's aggregate.
        if (newBlockId == 0) {
            output[newBlockId].inclusive_prefix = block_aggregate;
            output[newBlockId].status.store(PREFIX_READY, cuda::memory_order_release);
        } else {
            output[newBlockId].aggregate = block_aggregate;
            output[newBlockId].status.store(AGGREGATE_READY, cuda::memory_order_release);
        }
    }
    __syncthreads();

    // Warp 0 inspects up to 32 predecessors per lookback step.
    if (warp_id == 0 && newBlockId != 0) {
        long long accumulated_prefix = 0;
        int current_predecessor = newBlockId - lane - 1;
        
        while (1) {
            bool valid_predecessor = current_predecessor >= 0;
            STATUS state = NOT_READY;
            while ((state == NOT_READY) && valid_predecessor) {
                state = output[current_predecessor].status.load(cuda::memory_order_acquire);
            }

            // The nearest ready prefix terminates the aggregate chain.
            unsigned mask = __ballot_sync(0xffffffff, valid_predecessor && (state == PREFIX_READY));
            unsigned leader = mask ? (__ffs(mask) - 1) : 32;
            long long prefix = 0;
            if (lane == leader) prefix = output[current_predecessor].inclusive_prefix;
            if (leader != 32) prefix = __shfl_sync(0xffffffff, prefix, leader);

            valid_predecessor = valid_predecessor && (lane < leader);
            long long agg = valid_predecessor && (state == AGGREGATE_READY) ? output[current_predecessor].aggregate : 0;
            // Reduce aggregates newer than the terminating prefix.
            for (unsigned offset = 16; offset > 0; offset /= 2) agg += __shfl_down_sync(0xffffffff, agg, offset);

            if (lane == 0) {
                accumulated_prefix += agg;
                if (leader != 32) {
                    accumulated_prefix += prefix;
                    predecessor_prefix = accumulated_prefix;
                    // Publish this tile's completed inclusive prefix.
                    output[newBlockId].inclusive_prefix = block_aggregate + predecessor_prefix;
                    output[newBlockId].status.store(PREFIX_READY, cuda::memory_order_release);
                }
            }

            if (leader != 32) break;
            current_predecessor -= 32;
        }
    }
    __syncthreads();

    // Apply block-local and predecessor offsets to every scanned element.
    long long offset = (warp_id == 0) ? 0 : warp_totals[warp_id - 1];
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix[i]  += offset + predecessor_prefix;
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix2[i] += offset + predecessor_prefix;
    
    reinterpret_cast<longlong4_32a*>(input)[tid * 2] = { strip_prefix[0], strip_prefix[1], strip_prefix[2], strip_prefix[3] };
    reinterpret_cast<longlong4_32a*>(input)[tid * 2 + 1] = { strip_prefix2[0], strip_prefix2[1], strip_prefix2[2], strip_prefix2[3] };
}

__global__ void __launch_bounds__(BLOCK_SIZE, 6) scan_LB_Final2(long long* input, blockVal* output) {
    extern __shared__ long long warp_totals[];
    __shared__ long long predecessor_prefix;

    int num_warps = blockDim.x / warpSize;
    
    // Dynamic block id: Thread 0 bumps global counter and broadcasts blockId through shared-mem
    if (threadIdx.x == 0) warp_totals[num_warps] = atomicAdd(&counter, 1);
    __syncthreads();
    int newBlockId = (int)warp_totals[num_warps];
    int tid = threadIdx.x + newBlockId * blockDim.x;
    
    longlong4_32a data = reinterpret_cast<longlong4_32a*>(input)[tid * 2];
    longlong4_32a data2 = reinterpret_cast<longlong4_32a*>(input)[tid * 2 + 1];
    long long strip_prefix[STRIPSIZE] = { data.x, data.y, data.z, data.w };
    long long strip_prefix2[STRIPSIZE] = { data2.x, data2.y, data2.z, data2.w };
    for(int i=1; i<STRIPSIZE; ++i) strip_prefix[i] += strip_prefix[i - 1];
    for(int i=1; i<STRIPSIZE; ++i) strip_prefix2[i] += strip_prefix2[i - 1];
    
    unsigned lane = threadIdx.x & (warpSize - 1);
    long long thread_prefix_sum = strip_prefix2[STRIPSIZE - 1] + strip_prefix[STRIPSIZE - 1];
    for (int offset=1; offset<warpSize; offset *= 2) {
        long long val = __shfl_up_sync(0xffffffff, thread_prefix_sum, offset);
        if (lane >= offset) thread_prefix_sum += val;
    }
    long long exclusive = thread_prefix_sum - strip_prefix2[STRIPSIZE - 1];
    for(int i=0; i<STRIPSIZE; ++i) {
        strip_prefix[i] += exclusive - strip_prefix[STRIPSIZE - 1];
        strip_prefix2[i] += exclusive;
    }
    
    int warp_id = threadIdx.x / warpSize;
    if (lane == (warpSize - 1)) warp_totals[warp_id] = thread_prefix_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        long long warp_prefix_sum = (lane < num_warps) ? warp_totals[lane] : 0;
        for (int offset = 1; offset < warpSize; offset *= 2) {
            long long val = __shfl_up_sync(0xffffffff, warp_prefix_sum, offset);
            if (lane >= offset) warp_prefix_sum += val;
        }
        long long block_aggregate = __shfl_sync(0xffffffff, warp_prefix_sum, num_warps - 1);
        if (lane < num_warps) warp_totals[lane] = warp_prefix_sum;
        __syncwarp();

        if (lane == 0) {
            predecessor_prefix = 0;

            if (newBlockId == 0) {
                output[newBlockId].inclusive_prefix = block_aggregate;
                output[newBlockId].status.store(PREFIX_READY, cuda::memory_order_release);
            } else {
                output[newBlockId].aggregate = block_aggregate;
                output[newBlockId].status.store(AGGREGATE_READY, cuda::memory_order_release);
            }
        }
        __syncwarp();

        if (newBlockId != 0) {
            long long accumulated_prefix = 0;
            int current_predecessor = newBlockId - lane - 1;

            while (current_predecessor >= 0) {
                unsigned leader = 32U;
                unsigned validLaneMask = __activemask();
                STATUS state = NOT_READY;

                while (validLaneMask != 0) {
                    state = output[current_predecessor].status.load(cuda::memory_order_acquire);

                    unsigned prefixReadyMask = __ballot_sync(validLaneMask, state == PREFIX_READY);
                    leader = prefixReadyMask ? (__ffs(prefixReadyMask) - 1) : 32U;
                    if (leader != 32U) {
                        validLaneMask = __ballot_sync(validLaneMask, (lane < leader) && (state == NOT_READY));
                    } else {
                        validLaneMask = __ballot_sync(validLaneMask, state == NOT_READY);
                    }
                }
                // __syncwarp();

                if (leader == 0U) {
                    if (lane == 0) {
                        predecessor_prefix = accumulated_prefix + output[current_predecessor].inclusive_prefix;
                        output[newBlockId].inclusive_prefix = block_aggregate + predecessor_prefix;
                        output[newBlockId].status.store(PREFIX_READY, cuda::memory_order_release);
                    }
                }
                else {
                    long long prefix = 0;
                    validLaneMask = __activemask();
                    if (lane == leader) prefix = output[current_predecessor].inclusive_prefix;
                    if (leader != 32) prefix = __shfl_sync(validLaneMask, prefix, leader);
    
                    long long agg = (lane < leader) ? output[current_predecessor].aggregate : 0;
                    for (unsigned offset = 16; offset > 0; offset /= 2) agg += __shfl_down_sync(validLaneMask, agg, offset);
                    
                    if (lane == 0) {
                        accumulated_prefix += agg;
                        if (leader != 32) {
                            accumulated_prefix += prefix;
                            predecessor_prefix = accumulated_prefix;
                            output[newBlockId].inclusive_prefix = block_aggregate + predecessor_prefix;
                            output[newBlockId].status.store(PREFIX_READY, cuda::memory_order_release);
                        }
                    }
                }

                if (leader != 32) break;
                current_predecessor -= 32;
            }
        }
    }
    __syncthreads();

    long long offset = (warp_id == 0) ? 0 : warp_totals[warp_id - 1];
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix[i]  += offset + predecessor_prefix;
    for(int i=0; i<STRIPSIZE; ++i) strip_prefix2[i] += offset + predecessor_prefix;
    
    reinterpret_cast<longlong4_32a*>(input)[tid * 2] = { strip_prefix[0], strip_prefix[1], strip_prefix[2], strip_prefix[3] };
    reinterpret_cast<longlong4_32a*>(input)[tid * 2 + 1] = { strip_prefix2[0], strip_prefix2[1], strip_prefix2[2], strip_prefix2[3] };
}
