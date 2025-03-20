#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

__global__ void check_validity(int* colors, int* sizes, bool* is_valid, int num_colors, int target) {
    int idx = threadIdx.x;
    if (idx >= num_colors) return;
    
    int L = 0;
    int R = sizes[idx] - 1;
    int* arr = colors + idx * 20; // Assuming max size 20 per color
    
    while (L < R) {
        int sum = arr[L] + arr[R];
        if (sum == target) {
            is_valid[idx] = false;
            return;
        } else if (sum < target) {
            L++;
        } else {
            R--;
        }
    }
    is_valid[idx] = true;
}

int main() {
    vector<vector<int>> host_colors = {
        {1, 4, 9, 12, 19, 26, 33, 36, 41},
        {2, 3, 10, 11, 16, 29, 30, 34, 35, 42, 43},
        {5, 6, 7, 8, 17, 18, 27, 28, 37, 38, 39, 40},
        {13, 14, 15, 20, 21, 22, 23, 24, 25, 31, 32}
    };
    vector<string> color_names = {"red", "blue", "purple", "teal"};
    int num_colors = host_colors.size();
    int target = 44;
    
    int* d_colors;
    int* d_sizes;
    bool* d_valid;
    
    int max_size = 20;
    int h_sizes[5];
    bool h_valid[5];
    int h_colors[5 * max_size] = {0};
    
    for (int i = 0; i < num_colors; i++) {
        h_sizes[i] = host_colors[i].size();
        for (int j = 0; j < host_colors[i].size(); j++) {
            h_colors[i * max_size + j] = host_colors[i][j];
        }
    }
    
    cudaMalloc(&d_colors, sizeof(int) * max_size * num_colors);
    cudaMalloc(&d_sizes, sizeof(int) * num_colors);
    cudaMalloc(&d_valid, sizeof(bool) * num_colors);
    
    cudaMemcpy(d_colors, h_colors, sizeof(int) * max_size * num_colors, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, h_sizes, sizeof(int) * num_colors, cudaMemcpyHostToDevice);
    
    auto start = chrono::high_resolution_clock::now();
    check_validity<<<1, num_colors>>>(d_colors, d_sizes, d_valid, num_colors, target);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_valid, d_valid, sizeof(bool) * num_colors, cudaMemcpyDeviceToHost);
    
    cout << "Execution Time: " << chrono::duration<double, milli>(end - start).count() << " ms\n";
    for (int i = 0; i < num_colors; i++) {
        cout << "Color " << color_names[i] << " is " << (h_valid[i] ? "valid" : "invalid") << "\n";
    }
    
    cudaFree(d_colors);
    cudaFree(d_sizes);
    cudaFree(d_valid);
    return 0;
}
