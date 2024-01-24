#ifndef CUDA_FUNCTION_H
#    define CUDA_FUNCTION _H
#include <stdint.h>
__device__ float readFloatOnCuda(char* buffer, int offset);

__device__ int16_t readInt16OnCuda(char* buffer, int offset);

__device__ uint16_t readuInt16OnCuda(char* buffer, int offset);

__device__ int readIntOnCuda(char* buffer, int offset);

__device__ unsigned char readCharOnCuda(char* buffer, int offset);

__device__ float* readPointOnCuda(char* buffer, int offset);
#endif