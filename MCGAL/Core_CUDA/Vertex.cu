#include "include/Vertex.cuh"
#include "include/global.cuh"
#include <stdlib.h>
namespace MCGAL {

void Vertex::addHalfedge(Halfedge* halfedge) {
    halfedges[halfedges_size++] = halfedge->poolId;
}

void Vertex::addHalfedge(int halfedge) {
    halfedges[halfedges_size++] = halfedge;
}

__device__ void Vertex::addHalfedgeOnCuda(int halfedge) {
    int tpIndex = halfedges_size;
    atomicAdd(&halfedges_size, 1);
    halfedges[tpIndex] = halfedge;
}

Halfedge* Vertex::getHalfedgeByIndex(int index) {
    assert(index < halfedges_size);
    return contextPool.getHalfedgeByIndex(halfedges[index]);
}
// #pragma optimize( "", off )
void Vertex::eraseHalfedgeByIndex(int index) {
    assert(index < halfedges_size);
    for (int i = index; i < halfedges_size - 1; ++i) {
        halfedges[i] = halfedges[i + 1];
    }
    --halfedges_size;
}

void Vertex::eraseHalfedgeByPointer(Halfedge* halfedge) {
    int index = 0;
    for (int i = 0; i < halfedges_size; i++) {
        if (getHalfedgeByIndex(i)->poolId == halfedge->poolId) {
            index = i;
            break;
        }
    }
    eraseHalfedgeByIndex(index);
}
// #pragma optimize( "", on )

// __device__ void Vertex::addHalfedgeOnCuda(int halfedge) {
//     // halfedges[halfedges_size++] = halfedge;
//     auto ti = blockDim.x * blockIdx.x + threadIdx.x;
//     // For each thread in a wrap
//     for (int i = 0; i < 32; i++) {
//         // Check if it is this thread's turn
//         if (ti % 32 != i)
//             continue;

//         // Lock
//         while (atomicExch(&lock, 0) == 0)
//             ;
//         // Work
//         halfedges[halfedges_size] = halfedge;
//         atomicAdd(&halfedges_size, 1);
//         // Unlock
//         lock = 1;
//     }
// }

void Vertex::eraseHalfedgeByIndexOnCuda(int index) {
    assert(index < halfedges_size);
    for (int i = index; i < halfedges_size - 1; ++i) {
        halfedges[i] = halfedges[i + 1];
    }
    --halfedges_size;
}

}  // namespace MCGAL