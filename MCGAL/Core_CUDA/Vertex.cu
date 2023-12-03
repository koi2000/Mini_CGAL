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

Halfedge* Vertex::getHalfedgeByIndex(int index) {
    assert(index < halfedges_size);
    return contextPool.getHalfedgeByIndex(halfedges[index]);
}

void Vertex::eraseHalfedgeByIndex(int index) {
    assert(index < halfedges_size);
    for (int i = index; i < halfedges_size - 1; ++i) {
        halfedges[i] = halfedges[i + 1];
    }
    --halfedges_size;
}

void Vertex::eraseHalfedgeByPointer(Halfedge* halfedge) {
    int index = 0;
    int flag = 0;
    for (int i = 0; i < halfedges_size; i++) {
        if (getHalfedgeByIndex(i) == halfedge) {
            index = i;
            flag = 1;
            break;
        }
    }
    eraseHalfedgeByIndex(index);
}

// cuda
__device__ void Vertex::addHalfedgeOnCuda(Halfedge* halfedge) {
    halfedges[halfedges_size++] = halfedge->poolId;
}

__device__ void Vertex::addHalfedgeOnCuda(int halfedge) {
    halfedges[halfedges_size++] = halfedge;
}

}  // namespace MCGAL