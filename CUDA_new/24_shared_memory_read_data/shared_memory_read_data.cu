#include "freshman.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16

#define IPAD 1

__global__ void warmup(int* out) {
    // 位于线程块的共享存储器空间中
    // 与线程块具有相同的生命周期
    // 仅可通过块内的所有线程访问
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    // __syncthreads()将确保线程块中的每个线程都执行完
    // __syncthreads()前面的语句后，才会执行下一条语句
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
    // printf("out[%d] = %d\n", idx, out[idx]);
}

__global__ void setRowReadRow(int* out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int* out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadRow(int* out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadCol(int* out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(int* out) {
    // 动态分配内存，在编译期并不确定大小
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}

int main(int argc, char** argv) {
    // set up device
    initDevice(0);
    int kernel = 0;
    if (argc >= 2)
        kernel = atoi(argv[1]);
    int nElem = BDIMX * BDIMY;
    printf("Vector size:%d\n", nElem);
    int nByte = sizeof(int) * nElem;
    int* out;
    CHECK(cudaMalloc((int**)&out, nByte));
    cudaSharedMemConfig MemConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&MemConfig));
    printf("--------------------------------------------\n");
    switch (MemConfig) {
        case cudaSharedMemBankSizeFourByte: printf("the device is cudaSharedMemBankSizeFourByte: 4-Byte\n"); break;
        case cudaSharedMemBankSizeEightByte: printf("the device is cudaSharedMemBankSizeEightByte: 8-Byte\n"); break;
    }
    printf("--------------------------------------------\n");
    dim3 block(BDIMY, BDIMX);
    dim3 grid(1, 1);
    dim3 block_rect(BDIMX_RECT, BDIMX_RECT);
    dim3 grid_rect(1, 1);
    warmup<<<grid, block>>>(out);
    printf("wramup!\n");
    double iStart, iElaps;
    iStart = cpuSecond();
    switch (kernel) {
        case 0: {
            setRowReadRow<<<grid,block>>>(out);
        } break;
        default: break;
    }
    cudaFree(out);
    // cudaDeviceReset();
    return 0;
}