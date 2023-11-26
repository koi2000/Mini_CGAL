#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void hello_world(void) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    printf("GPU: Hello world at %d %d %d\n", x, y, z);
}

int main(int argc, char** argv) {
    dim3 grid(1, 1, 1), block(3, 4, 1);
    hello_world<<<grid, block>>>();
    cudaDeviceReset();
    return 0;
}
