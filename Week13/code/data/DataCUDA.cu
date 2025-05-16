

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <unordered_set>

#define BITMASK_SIZE 3
#define SAMPLE_SIZE 1000
#define MAX_Z 100
#define CHECK_CUDA(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

struct State
{
    uint64_t colors[5][BITMASK_SIZE]; // 0=red, 1=blue, 2=green, 3=cyan, 4=magenta
};

size_t MAX_STATES; // Host-side maximum states
__device__ __host__ inline void set_bit(uint64_t *mask, int pos)
{
    int idx = pos / 64;
    int bit = pos % 64;
    mask[idx] |= (1ULL << bit);
}

__device__ __host__ inline bool check_bit(const uint64_t *mask, int pos)
{
    int idx = pos / 64;
    int bit = pos % 64;
    return (mask[idx] & (1ULL << bit)) != 0;
}

__global__ void processLevel(State *current, int count, State *next,
                             int *next_count, int z, int max_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count || *next_count >= max_states)
        return;

    State s = current[idx];
    for (int c = 0; c < 5; ++c)
    {
        bool valid = true;
        for (int x = 1; x <= z / 2; ++x)
        {
            if (check_bit(s.colors[c], x) && check_bit(s.colors[c], z - x))
            {
                valid = false;
                break;
            }
        }
        if (valid)
        {
            State new_state = s;
            set_bit(new_state.colors[c], z);
            int pos = atomicAdd(next_count, 1);
            if (pos < max_states)
                next[pos] = new_state;
        }
    }
}

int main()
{
    // Initialize GPU memory parameters
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    MAX_STATES = (free_mem * 0.8) / (2 * sizeof(State));
    const int MAX_STATES_DEVICE = static_cast<int>(MAX_STATES);
    printf("Using MAX_STATES = %zu (%.1f GB)\n", MAX_STATES,
           (2 * MAX_STATES * sizeof(State)) / 1e9);

    State initial{};
    set_bit(initial.colors[0], 1); // Red: 1
    set_bit(initial.colors[1], 2); // Blue: 2
    set_bit(initial.colors[1], 3); // Green: 3
    set_bit(initial.colors[0], 4); // Cyan: 4
    set_bit(initial.colors[2], 5); // Magenta: 5

    State *d_current, *d_next;
    int *d_count;

    CHECK_CUDA(cudaMalloc(&d_current, MAX_STATES * sizeof(State)));
    CHECK_CUDA(cudaMalloc(&d_next, MAX_STATES * sizeof(State)));
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_current, &initial, sizeof(State), cudaMemcpyHostToDevice));

    int current_count = 1;
    auto start = std::chrono::high_resolution_clock::now();

    for (int z = 6; z <= MAX_Z; ++z)
    {
        CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));

        dim3 block(256);
        dim3 grid((current_count + block.x - 1) / block.x);

        processLevel<<<grid, block>>>(d_current, current_count, d_next,
                                      d_count, z, MAX_STATES_DEVICE);
        CHECK_CUDA(cudaDeviceSynchronize());

        int next_count;
        CHECK_CUDA(cudaMemcpy(&next_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (next_count > 0)
        {
            std::vector<State> sampled_states(SAMPLE_SIZE);
            std::unordered_set<int> seen;
            std::mt19937 rng(std::random_device{}());
            std::uniform_int_distribution<int> dist(0, next_count - 1);

            int copied = 0;
            while (copied < SAMPLE_SIZE && seen.size() < next_count)
            {
                int idx = dist(rng);
                if (seen.insert(idx).second)
                {
                    int safe_idx = (idx < next_count) ? idx : (next_count - 1);
                    safe_idx = std::min(safe_idx, (int)MAX_STATES - 1);
                    // std::cout << "MAXSTATES: " << MAX_STATES << " safe_idx: " << safe_idx  << std::endl;
                    CHECK_CUDA(cudaMemcpy(&sampled_states[copied], d_next + safe_idx, sizeof(State), cudaMemcpyDeviceToHost));
                    ++copied;
                }
            }
            std::ofstream out("states_z" + std::to_string(z) + ".txt");
            for (int i = 0; i < std::min(SAMPLE_SIZE, next_count); ++i)
            {
                for (int c = 0; c < 5; ++c)
                {
                    for (int j = 0; j < BITMASK_SIZE; ++j)
                        out << sampled_states[i].colors[c][j] << ((j == BITMASK_SIZE - 1) ? "," : " ");
                }
                out << "\n";
            }
            out.close();
        }

        printf("z=%03d: States=%d\n", z, next_count);

        if (next_count == 0)
        {
            printf("\nLower bound: S(5) ≥ %d\n", z - 1);
            break;
        }

        std::swap(d_current, d_next);
        current_count = (next_count < MAX_STATES) ? next_count : MAX_STATES;
    }

    auto end = std::chrono::high_resolution_clock::now();
    printf("\nTotal time: %.2f seconds\n",
           std::chrono::duration<double>(end - start).count());

    CHECK_CUDA(cudaFree(d_current));
    CHECK_CUDA(cudaFree(d_next));
    CHECK_CUDA(cudaFree(d_count));
    return 0;
}