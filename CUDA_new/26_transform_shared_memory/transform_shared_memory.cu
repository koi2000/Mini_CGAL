#include "freshman.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define BDIMX 8
#define BDIMY 8
#define IPAD 2

void transformMatrix2D_CPU(float* in, float* out, int nx, int ny) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            out[i * nx + j] = in[j * nx + i];
        }
    }
}

__global__ void warmup(float* in, float* out, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * nx;
    if (ix < nx && iy < ny) {
        out[idx] = in[idx];
    }
}
__global__ void copyRow(float* in, float* out, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * nx;
    if (ix < nx && iy < ny) {
        out[idx] = in[idx];
    }
}

__global__ void transformNaiveRow(float* in, float* out, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx_row = ix + iy * nx;
    int idx_col = ix * ny + iy;
    if (ix < nx && iy < ny) {
        out[idx_col] = in[idx_row];
    }
}

__global__ void transformSmem(float* in, float* out, int nx, int ny) {
    __shared__ float tile[BDIMY][BDIMX];
    unsigned int ix, iy, transform_in_idx, transform_out_idx;
    ix = threadIdx.x + blockDim.x + blockIdx.x;
    iy = threadIdx.y + blockDim.y * blockIdx.y;
    transform_in_idx = iy * nx + ix;
    
}