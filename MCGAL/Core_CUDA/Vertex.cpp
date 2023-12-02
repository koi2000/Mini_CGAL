#include "include/Vertex.h"
#include "include/global.h"
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
    if(flag==0){
        int a = 9;
    }
    eraseHalfedgeByIndex(index);
}

}  // namespace MCGAL