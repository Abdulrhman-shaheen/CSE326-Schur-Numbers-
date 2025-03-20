#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void tmv(float *a, float *b, float *c, int w, int h) {
    float sum = 0;
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < h; i++)
        sum += a[i * w + tx] * b[i];
    c[tx] = sum;
}