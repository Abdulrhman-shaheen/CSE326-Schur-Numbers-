#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>  

#define MAX_STATES 10000000  
#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

struct State {
    uint64_t red;
    uint64_t blue;
    uint64_t green;
    uint64_t cyan;
};

__global__ void processLevel(State *current_states, int current_count, State *next_states, int *next_count, int z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_count) return;

    State s = current_states[idx];

    for (int color = 0; color < 4; ++color) {
        uint64_t mask;
        switch (color) {
            case 0: mask = s.red; break;
            case 1: mask = s.blue; break;
            case 2: mask = s.green; break;
            default: mask = s.cyan; break;
        }

        bool valid = true;
        for (int x = 1; x <= z / 2; ++x) {
            if ((mask & (1ULL << x))) {
                int y = z - x;
                if ((mask & (1ULL << y))) {
                    valid = false;
                    break;
                }
            }
        }

        if (valid) {
            State new_state = s;
            switch (color) {
                case 0: new_state.red |= (1ULL << z); break;
                case 1: new_state.blue |= (1ULL << z); break;
                case 2: new_state.green |= (1ULL << z); break;
                case 3: new_state.cyan |= (1ULL << z); break;
            }

            int pos = atomicAdd(next_count, 1);
            if (pos < MAX_STATES) {  
                next_states[pos] = new_state;
            }
        }
    }
}

int main() {
    State initial{};
    initial.red = (1ULL << 1) | (1ULL << 4);
    initial.blue = (1ULL << 2) | (1ULL << 3);
    initial.green = (1ULL << 5);
    initial.cyan = 0;

    State *d_current, *d_next;
    int *d_count;

    CHECK_CUDA(cudaMalloc(&d_current, MAX_STATES * sizeof(State)));
    CHECK_CUDA(cudaMalloc(&d_next, MAX_STATES * sizeof(State)));
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_current, &initial, sizeof(State), cudaMemcpyHostToDevice));

    int current_count = 1;
    State *h_final = nullptr;

    
    auto program_start = std::chrono::high_resolution_clock::now();

    for (int z = 6; z <= 44; ++z) {
        CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));

        dim3 threads(256);
        dim3 blocks((current_count + threads.x - 1) / threads.x);

    
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));

        processLevel<<<blocks, threads>>>(d_current, current_count, d_next, d_count, z);
        CHECK_CUDA(cudaDeviceSynchronize());

        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("Kernel execution time for z=%d: %.3f ms\n", z, milliseconds);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        int next_count;
        CHECK_CUDA(cudaMemcpy(&next_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

        printf("Processing z=%d, generated %d states\n", z, next_count);

        if (next_count == 0) {
            printf("No valid continuation beyond z=%d\n", z);
            break;
        }

        std::swap(d_current, d_next);
        current_count = next_count > MAX_STATES ? MAX_STATES : next_count;  

        if (z == 44) {
            h_final = new State[current_count];
            CHECK_CUDA(cudaMemcpy(h_final, d_current, current_count * sizeof(State), cudaMemcpyDeviceToHost));
        }
    }

    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> program_duration = program_end - program_start;
    printf("Total program execution time: %.3f seconds\n", program_duration.count());

    if (h_final) {
        State solution = h_final[0];
        printf("\nFinal Coloring:\n");
        printf("Red: "); for (int i=1;i<=44;i++) if (solution.red & (1ULL<<i)) printf("%d ",i);
        printf("\nBlue: "); for (int i=1;i<=44;i++) if (solution.blue & (1ULL<<i)) printf("%d ",i);
        printf("\nGreen: "); for (int i=1;i<=44;i++) if (solution.green & (1ULL<<i)) printf("%d ",i);
        printf("\nCyan: "); for (int i=1;i<=44;i++) if (solution.cyan & (1ULL<<i)) printf("%d ",i);
        printf("\n");
        delete[] h_final;
    }

    cudaFree(d_current);
    cudaFree(d_next);
    cudaFree(d_count);
    return 0;
}
