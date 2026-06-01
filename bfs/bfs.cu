#include "kernels.cuh"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int SOURCE = 1;

struct Graph {
    int n = 0;
    std::vector<int> offsets;
    std::vector<int> edges;
};

std::filesystem::path graph_file_path(const char* argv0)
{
    auto dir = std::filesystem::absolute(argv0).parent_path();
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        auto name = entry.path().filename().string();
        if (entry.path().extension() == ".txt" &&
            name.rfind("cpu_bfs_source_", 0) != 0) {
            return entry.path();
        }
    }
    return dir / "roadNet-CA.txt";
}

Graph load_graph(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open graph file");

    int header_nodes = -1;
    int max_node = -1;
    std::string line;
    std::vector<std::pair<int, int>> edge_list;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            auto pos = line.find("Nodes:");
            if (pos != std::string::npos) {
                std::stringstream ss(line.substr(pos + 6));
                ss >> header_nodes;
            }
            continue;
        }

        int u, v;
        std::stringstream ss(line);
        if (ss >> u >> v) {
            edge_list.emplace_back(u, v);
            max_node = std::max(max_node, std::max(u, v));
        }
    }

    int n = std::max(header_nodes, max_node + 1);

    Graph g;
    g.n = n;
    g.offsets.assign(n + 1, 0);
    for (auto [u, _] : edge_list) ++g.offsets[u + 1];
    std::inclusive_scan(g.offsets.begin(), g.offsets.end(), g.offsets.begin());

    g.edges.resize(edge_list.size());
    std::vector<int> cursor = g.offsets;
    for (auto [u, v] : edge_list) g.edges[cursor[u]++] = v;
    return g;
}

std::vector<int> cpu_bfs(const Graph& g, int source)
{
    std::vector<std::atomic<int>> dist(g.n);
    std::for_each(std::execution::par_unseq, dist.begin(), dist.end(),
                  [](std::atomic<int>& x) { x.store(-1, std::memory_order_relaxed); });
    dist[source].store(0, std::memory_order_relaxed);

    std::vector<int> frontier{source};
    for (int level = 0; !frontier.empty(); ++level) {
        std::vector<int> next(g.n);
        std::atomic<int> next_size{0};

        std::for_each(std::execution::par, frontier.begin(), frontier.end(), [&](int u) {
            for (int e = g.offsets[u]; e < g.offsets[u + 1]; ++e) {
                int v = g.edges[e];
                int expected = -1;
                if (dist[v].compare_exchange_strong(expected, level + 1,
                                                    std::memory_order_relaxed)) {
                    next[next_size.fetch_add(1, std::memory_order_relaxed)] = v;
                }
            }
        });

        next.resize(next_size.load(std::memory_order_relaxed));
        frontier.swap(next);
    }

    std::vector<int> out(g.n);
    std::for_each(std::execution::par_unseq, out.begin(), out.end(), [&](int& x) {
        int i = (int)(&x - out.data());
        x = dist[i].load(std::memory_order_relaxed);
    });
    return out;
}

bool read_cpu_result(const std::filesystem::path& path, int n, int m, int source,
                     std::vector<int>& dist)
{
    std::ifstream in(path);
    int file_n, file_m, file_source;
    if (!(in >> file_n >> file_m >> file_source) || file_n != n || file_m != m ||
        file_source != source) {
        return false;
    }

    dist.resize(n);
    for (int& x : dist) {
        if (!(in >> x)) return false;
    }
    return true;
}

void write_cpu_result(const std::filesystem::path& path, const std::vector<int>& dist,
                      int edge_count, int source)
{
    std::ofstream out(path);
    out << dist.size() << ' ' << edge_count << ' ' << source << '\n';
    for (int x : dist) out << x << '\n';
}

std::vector<int> gpu_bfs(const Graph& g, int source)
{
    thrust::device_vector<int> offsets = g.offsets;
    thrust::device_vector<int> edges = g.edges;
    thrust::device_vector<int> device_dist(g.n, -1);
    thrust::device_vector<int> changed(1);
    device_dist[source] = 0;

    int* d_offsets = thrust::raw_pointer_cast(offsets.data());
    int* d_edges = thrust::raw_pointer_cast(edges.data());
    int* d_dist = thrust::raw_pointer_cast(device_dist.data());
    int* d_changed = thrust::raw_pointer_cast(changed.data());

    int block = 256;
    int grid = (g.n + block - 1) / block;
    for (int level = 0, any_changed = 1; any_changed; ++level) {
        changed[0] = 0;
        bfs_kernel<<<grid, block>>>(d_offsets, d_edges, g.n, level, d_dist, d_changed);
        CUDA_CHECK(cudaGetLastError());
        any_changed = changed[0];
    }

    std::vector<int> dist(g.n);
    thrust::copy(device_dist.begin(), device_dist.end(), dist.begin());
    return dist;
}

int main(int argc, char** argv)
{
    (void)argc;
    auto graph_path = graph_file_path(argv[0]);
    auto cpu_path = graph_path.parent_path() / ("cpu_bfs_source_" + std::to_string(SOURCE) + ".txt");

    Graph g = load_graph(graph_path);
    assert(SOURCE >= 0 && SOURCE < g.n);
    std::cout << "Loaded " << g.n << " nodes and " << g.edges.size() << " edges\n";

    std::vector<int> cpu_dist;
    if (!read_cpu_result(cpu_path, g.n, (int)g.edges.size(), SOURCE, cpu_dist)) {
        cpu_dist = cpu_bfs(g, SOURCE);
        write_cpu_result(cpu_path, cpu_dist, (int)g.edges.size(), SOURCE);
    }

    std::vector<int> gpu_dist = gpu_bfs(g, SOURCE);
    bool ok = std::equal(std::execution::par_unseq, cpu_dist.begin(), cpu_dist.end(),
                         gpu_dist.begin());
    std::cout << (ok ? "BFS output matches\n" : "Incorrect values!!!\n");
    return ok ? 0 : 1;
}
