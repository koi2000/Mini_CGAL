#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <string>

__global__ void hello_world(char* dbuffer, int num) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        int offset = tid;
        char i = *(char*)(dbuffer + offset);
        printf("%c", i + '0');
    }
}

int main(int argc, char** argv) {
    dim3 grid(1, 1, 1), block(3, 4, 1);
    char* buffer = new char[100];
    memset(buffer, 0, 100);
    for (int i = 4; i < 100; i += 4) {
        buffer[i] = i / 4 % 10;
    }
    char* dbuffer;
    cudaMalloc(&dbuffer, sizeof(char) * 100);
    cudaMemcpy(dbuffer, buffer, sizeof(char) * 100, cudaMemcpyHostToDevice);
    hello_world<<<1, 256>>>(dbuffer, 100);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
    cudaDeviceReset();
    return 0;
}
