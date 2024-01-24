#include "cuda_functions.cuh"

__device__ float readFloatOnCuda(char* buffer, int offset) {
    float f = *(float*)(buffer + offset);
    return f;
}

__device__ int16_t readInt16OnCuda(char* buffer, int offset) {
    int16_t i = *(int16_t*)(buffer + offset);
    return i;
}

__device__ uint16_t readuInt16OnCuda(char* buffer, int offset) {
    uint16_t i = *(uint16_t*)(buffer + offset);
    return i;
}

__device__ int readIntOnCuda(char* buffer, int offset) {
    int i = *(int*)(buffer + offset);
    return i;
}

__device__ unsigned char readCharOnCuda(char* buffer, int offset) {
    unsigned char i = *(unsigned char*)(buffer + offset);
    return i;
}

__device__ float* readPointOnCuda(char* buffer, int offset) {
    float coord[3];
    for (unsigned i = 0; i < 3; ++i) {
        coord[i] = readFloatOnCuda(buffer, offset);
        offset += sizeof(float);
    }
    // MCGAL::Point pt(coord[0], coord[1], coord[2]);
    return coord;
}
