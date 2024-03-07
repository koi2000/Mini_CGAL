#include "include/ContextPool.h"

namespace MCGAL {

ContextPool::ContextPool() {
    vindex = new int(0);
    hindex = new int(0);
    findex = new int(0);
    vpool = new MCGAL::Vertex[VERTEX_POOL_SIZE];
    for (int i = 0; i < VERTEX_POOL_SIZE; i++) {
        vpool[i].poolId = i;
    }
    hpool = new MCGAL::Halfedge[HALFEDGE_POOL_SIZE];
    for (int i = 0; i < HALFEDGE_POOL_SIZE; i++) {
        hpool[i].poolId = i;
    }
    fpool = new MCGAL::Facet[FACET_POOL_SIZE];
    for (int i = 0; i < FACET_POOL_SIZE; i++) {
        fpool[i].poolId = i;
    }
}

ContextPool::~ContextPool() {
    if (vpool != nullptr) {
        delete[] vpool;
        vpool = nullptr;
    }
    if (hpool != nullptr) {
        delete[] hpool;
        hpool = nullptr;
    }
    if (fpool != nullptr) {
        delete[] fpool;
        fpool = nullptr;
    }
}

}  // namespace MCGAL