#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <chrono>

#define MAX_STATES    20000000    // more room for Schur(5)
#define MAX_Z         160         // upper bound we're trying to push
#define NUM_COLORS    5           // now solving for Schur(5)
#define Z0            20          // Monte Carlo rollout depth
#define NUM_ROLLOUTS  200000      // rollout trials
#define MAX_PREFIX    10000       // how many rollouts to keep

#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

struct State {
    uint64_t mask[NUM_COLORS];
};

__host__ __device__
bool isValid(uint64_t m, int z) {
    for (int x = 1; x <= z / 2; ++x) {
        if ((m & (1ULL << x)) && (m & (1ULL << (z - x))))
            return false;
    }
    return true;
}

State truncatedRollout(const State& seed, int zLimit, std::mt19937_64& rng) {
    State s = seed;
    int colors[NUM_COLORS] = {0, 1, 2, 3, 4};
    for (int z = 6; z <= zLimit; ++z) {
        std::shuffle(colors, colors + NUM_COLORS, rng);
        bool placed = false;
        for (int c : colors) {
            if (isValid(s.mask[c], z)) {
                s.mask[c] |= (1ULL << z);
                placed = true;
                break;
            }
        }
        if (!placed) break;
    }
    return s;
}

int computeDepth(const State& s) {
    uint64_t all = 0;
    for (int c = 0; c < NUM_COLORS; ++c) all |= s.mask[c];
    for (int z = MAX_Z; z >= 1; --z) {
        if (all & (1ULL << z)) return z;
    }
    return 0;
}

__global__
void processLevel(const State* in, int inCount, State* out, int* outCount, int z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inCount) return;

    State s = in[idx];
    for (int c = 0; c < NUM_COLORS; ++c) {
        uint64_t m = s.mask[c];
        if (!isValid(m, z)) continue;

        State t = s;
        t.mask[c] |= (1ULL << z);

        int pos = atomicAdd(outCount, 1);
        if (pos < MAX_STATES) {
            out[pos] = t;
        }
    }
}

int main() {
    auto t_total_start = std::chrono::high_resolution_clock::now();

    // Initial coloring 1–5 (based on Schur(4))
    State initial{};
    initial.mask[0] = (1ULL << 1) | (1ULL << 4); // red
    initial.mask[1] = (1ULL << 2) | (1ULL << 3); // blue
    initial.mask[2] = (1ULL << 5);               // green
    initial.mask[3] = 0;                         // cyan
    initial.mask[4] = 0;                         // magenta (new color)

    // Monte Carlo rollouts
    std::vector<State> prefixes;
    prefixes.reserve(MAX_PREFIX);
    std::mt19937_64 rng(42);

    auto t_rollout_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ROLLOUTS; ++i) {
        State s = truncatedRollout(initial, Z0, rng);
        if (computeDepth(s) == Z0) {
            prefixes.push_back(s);
            if ((int)prefixes.size() >= MAX_PREFIX) break;
        }
    }
    auto t_rollout_end = std::chrono::high_resolution_clock::now();

    if (prefixes.empty()) {
        prefixes.push_back(initial); // fallback
    }

    printf("Collected %zu prefixes at Z0=%d in %.3f seconds\n",
           prefixes.size(), Z0,
           std::chrono::duration<double>(t_rollout_end - t_rollout_start).count());

    // GPU allocation
    State *d_cur, *d_nxt;
    int *d_count;
    CHECK_CUDA(cudaMalloc(&d_cur, MAX_STATES * sizeof(State)));
    CHECK_CUDA(cudaMalloc(&d_nxt, MAX_STATES * sizeof(State)));
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_cur, prefixes.data(),
                          prefixes.size() * sizeof(State),
                          cudaMemcpyHostToDevice));
    int curCount = prefixes.size();

    // BFS from z = Z0+1 up to MAX_Z
    int maxReached = Z0;
    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    for (int z = Z0 + 1; z <= MAX_Z; ++z) {
        CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));

        int threads = 256;
        int blocks = (curCount + threads - 1) / threads;
        processLevel<<<blocks, threads>>>(d_cur, curCount, d_nxt, d_count, z);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        int nextCount;
        CHECK_CUDA(cudaMemcpy(&nextCount, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (nextCount == 0) break;

        maxReached = z;
        curCount = std::min(nextCount, MAX_STATES);
        std::swap(d_cur, d_nxt);
    }
    auto t_gpu_end = std::chrono::high_resolution_clock::now();

    printf("GPU BFS reached z = %d in %.3f seconds\n", maxReached,
           std::chrono::duration<double>(t_gpu_end - t_gpu_start).count());

    cudaFree(d_cur);
    cudaFree(d_nxt);
    cudaFree(d_count);

    auto t_total_end = std::chrono::high_resolution_clock::now();
    printf("TOTAL runtime: %.3f seconds\n",
           std::chrono::duration<double>(t_total_end - t_total_start).count());

    return 0;
}
