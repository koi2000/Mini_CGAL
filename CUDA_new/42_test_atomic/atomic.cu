#include "freshman.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define N 500000000
int arr[N];
__global__ void test_atomic(int* arr, int* max, int num) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        atomicMax(max, arr[tid]);
    }
}

__global__ void findMaxKernel(int* array, int* result, int size) {
    extern __shared__ int sharedArray[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // 将数据从全局内存复制到共享内存
    if (tid < size) {
        sharedArray[localIdx] = array[tid];
    } else {
        sharedArray[localIdx] = 0;  // 超出数组范围的部分用0填充
    }

    __syncthreads();

    // 使用 shared memory 在每个线程块内找到最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            sharedArray[localIdx] = max(sharedArray[localIdx], sharedArray[localIdx + stride]);
        }
        __syncthreads();
    }

    // 将每个线程块的最大值存储在全局内存中
    if (localIdx == 0) {
        atomicMax(result, sharedArray[0]);
    }
}

int main() {
    
    for (int i = 0; i < N; i++) {
        arr[i] = i;
    }
    int* darr;
    int* dmax;
    int maxx = 0;
    dim3 block(512);
    dim3 grid((N - 1) / block.x + 1);

    cudaMalloc(&darr, sizeof(int) * N);
    cudaMalloc(&dmax, sizeof(int));
    cudaMemcpy(darr, arr, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dmax, &maxx, sizeof(int), cudaMemcpyHostToDevice);

    double iStart, iElaps;
    iStart = cpuSecond();
    test_atomic<<<grid, block>>>(darr, dmax, N);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("cuda Execution configuration<<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);
    cudaMemcpy(&maxx, dmax, sizeof(int), cudaMemcpyDeviceToHost);
    maxx = 0;
    iStart = cpuSecond();
    for (int i = 0; i < N; i++) {
        maxx = max(maxx, arr[i]);
    }
    iElaps = cpuSecond() - iStart;
    printf("cpu Execution configuration<<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);
    return 0;
}