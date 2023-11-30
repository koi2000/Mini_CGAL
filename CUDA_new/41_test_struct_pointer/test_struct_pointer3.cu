#include <cuda_runtime.h>
#include <stdio.h>
struct MyStruct {
    int data;
    // 其他结构体成员...
};

__global__ void copyStructOnGPU(MyStruct* input, MyStruct* output, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        output[idx].data = input[idx].data;
        // 处理其他结构体成员...
        printf("%d\n", output[idx]);
    }
}

int main() {
    // 假设有一个结构体数组，包含numElements个元素
    int numElements = 100;
    MyStruct* h_inputArray = new MyStruct[numElements];
    MyStruct* h_outputArray = new MyStruct[numElements];

    // 在主机上初始化结构体数组
    for (int i = 0; i < numElements; ++i) {
        h_inputArray[i].data = i;
        // 初始化其他结构体成员...
    }

    // 在GPU上分配空间
    MyStruct* d_inputArray;
    MyStruct* d_outputArray;
    cudaMalloc((void**)&d_inputArray, numElements * sizeof(MyStruct));
    cudaMalloc((void**)&d_outputArray, numElements * sizeof(MyStruct));

    // 将结构体数据从主机拷贝到GPU
    cudaMemcpy(d_inputArray, h_inputArray, numElements * sizeof(MyStruct), cudaMemcpyHostToDevice);

    // 定义CUDA线程块和网格大小
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // 调用CUDA核函数
    copyStructOnGPU<<<gridSize, blockSize>>>(d_inputArray, d_outputArray, numElements);

    // 将结果从GPU拷贝回主机
    cudaMemcpy(h_outputArray, d_outputArray, numElements * sizeof(MyStruct), cudaMemcpyDeviceToHost);

    // 在这里处理h_outputArray中的数据，然后释放内存

    // 释放在主机上的内存
    delete[] h_inputArray;
    delete[] h_outputArray;

    // 释放在GPU上的内存
    cudaFree(d_inputArray);
    cudaFree(d_outputArray);

    return 0;
}
