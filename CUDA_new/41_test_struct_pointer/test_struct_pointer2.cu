#include <cuda_runtime.h>
#include <iostream>
// 示例结构体，包含指针
struct MyStruct {
    int* data;
    int size;
};

// 在设备上执行的核函数
__global__ void deviceKernel(MyStruct* deviceStruct) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 在设备上进行处理，示例中将每个元素乘以2
    if (tid < deviceStruct->size) {
        deviceStruct->data[tid] *= 2;
    }
}

int main() {
    const int size = 10;

    // 在主机上创建结构体，并分配内存
    MyStruct hostStruct;
    hostStruct.size = size;
    hostStruct.data = new int[size];

    // 初始化数据
    for (int i = 0; i < size; ++i) {
        hostStruct.data[i] = i;
    }

    // 在设备上分配内存，用于存储结构体
    MyStruct* deviceStruct;
    cudaMalloc((void**)&deviceStruct, sizeof(MyStruct));

    // 将结构体从主机拷贝到设备
    cudaMemcpy(deviceStruct, &hostStruct, sizeof(MyStruct), cudaMemcpyHostToDevice);

    // 在设备上执行核函数
    deviceKernel<<<1, size>>>(deviceStruct);

    // 将结果从设备拷贝回主机
    cudaMemcpy(&hostStruct, deviceStruct, sizeof(MyStruct), cudaMemcpyDeviceToHost);

    // 在此处进行其他操作
    for(int i=0;i<size;i++){
        std::cout<<hostStruct.data[i]<<std::endl;
    }
    // 释放内存
    delete[] hostStruct.data;
    cudaFree(deviceStruct);

    return 0;
}
