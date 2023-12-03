#include <iostream>
#include <cuda_runtime.h>
// CUDA核函数定义
__global__ void myCudaKernel(int *a, int *b, int *c, int size) {
    // 获取当前线程的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保索引不超出数组大小
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

class MyClass {
public:
    void myMemberFunction(int *a, int *b, int *c, int size) {
        // 在成员函数中分配设备内存
        int *d_a, *d_b, *d_c;
        cudaMalloc((void **)&d_a, size * sizeof(int));
        cudaMalloc((void **)&d_b, size * sizeof(int));
        cudaMalloc((void **)&d_c, size * sizeof(int));

        // 将数据从主机复制到设备
        cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

        // 定义线程块和线程数
        dim3 blockSize(256, 1, 1);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1, 1);

        // 调用全局CUDA核函数
        myCudaKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);

        // 将结果从设备复制回主机
        cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

        // 释放在设备上分配的内存
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
};

int main() {
    const int size = 10;
    int a[size], b[size], c[size];

    // 初始化数组 a 和 b
    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // 创建类的实例
    MyClass myObject;

    // 在类的成员函数中调用CUDA核函数
    myObject.myMemberFunction(a, b, c, size);

    // 打印结果
    for (int i = 0; i < size; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}