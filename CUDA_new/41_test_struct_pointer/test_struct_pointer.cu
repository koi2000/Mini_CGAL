#include <iostream>
#include <cuda_runtime.h>

struct MyStruct {
    int* data;
    int size;
};

__global__ void deviceKernel(MyStruct* deviceStruct) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 在设备上访问设备上分配的内存
    if (tid < deviceStruct->size) {
        printf("Device access: %d\n", deviceStruct->data[tid]);
    }
}

int main() {
    const int size = 5;

    // 在主机上分配内存
    int* hostData = new int[size];
    for (int i = 0; i < size; ++i) {
        hostData[i] = i;
    }

    // 创建结构体并设置指针
    MyStruct hostStruct;
    hostStruct.size = size;

    // 在设备上分配内存以存储结构体的数据
    cudaMalloc((void**)&hostStruct.data, size * sizeof(int));

    // 将数据从主机拷贝到设备
    cudaMemcpy(hostStruct.data, hostData, size * sizeof(int), cudaMemcpyHostToDevice);

    // 在设备上分配内存以存储结构体
    MyStruct* deviceStruct;
    cudaMalloc((void**)&deviceStruct, sizeof(MyStruct));

    // 将结构体从主机拷贝到设备
    cudaMemcpy(deviceStruct, &hostStruct, sizeof(MyStruct), cudaMemcpyHostToDevice);

    // 在设备上执行核函数
    deviceKernel<<<1, size>>>(deviceStruct);

    // 等待设备执行完成
    cudaDeviceSynchronize();

    // 释放内存
    delete[] hostData;
    cudaFree(hostStruct.data);
    cudaFree(deviceStruct);

    return 0;
}
