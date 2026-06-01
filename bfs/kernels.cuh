#pragma once

#include <cuda_runtime.h>

// Rudimentary top-down BFS. Each kernel launch handles one BFS level.
// Every thread checks one node; only nodes with dist == level expand.
// atomicCAS claims unvisited neighbors once and writes their next level.
__global__ void
bfs_kernel(const int* offsets, const int* edges, int n, int level, int* dist, int* changed)
{
    int src_node = blockIdx.x * blockDim.x + threadIdx.x;
    if (src_node >= n || dist[src_node] != level) return;

    for (int edge_index = offsets[src_node]; edge_index < offsets[src_node + 1]; ++edge_index) {
        int dest_node = edges[edge_index];
        if (atomicCAS(&dist[dest_node], -1, level + 1) == -1) *changed = 1;
    }
}
