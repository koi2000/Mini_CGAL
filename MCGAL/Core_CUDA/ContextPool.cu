#include "core.h"

namespace MCGAL {

ContextPool::ContextPool() {
    mallocOnUnifiedMemory();
    // mallocOnCpu();
    // copyToCuda();
}

ContextPool::~ContextPool() {
    freeOnUnifiedMemory();
    // freeOnCpu();
    // freeCuda();
}

void ContextPool::mallocOnUnifiedMemory() {
    dvpool = new MCGAL::Vertex[VERTEX_POOL_SIZE];
    CHECK(cudaMallocManaged(&vpool, VERTEX_POOL_SIZE * sizeof(Vertex)));
    for (int i = 0; i < VERTEX_POOL_SIZE; i++) {
        dvpool[i].poolId = i;
    }
    CHECK(cudaMemcpy(vpool, dvpool, VERTEX_POOL_SIZE * sizeof(Vertex), cudaMemcpyHostToDevice));
    
    dhpool = new MCGAL::Halfedge[HALFEDGE_POOL_SIZE];
    CHECK(cudaMallocManaged(&hpool, HALFEDGE_POOL_SIZE * sizeof(Halfedge)));
    for (int i = 0; i < HALFEDGE_POOL_SIZE; i++) {
        dhpool[i].poolId = i;
    }
    CHECK(cudaMemcpy(hpool, dhpool, HALFEDGE_POOL_SIZE * sizeof(Halfedge), cudaMemcpyHostToDevice));
    
    CHECK(cudaMallocManaged(&fpool, FACET_POOL_SIZE * sizeof(Facet)));
    dfpool = new MCGAL::Facet[FACET_POOL_SIZE];
    for (int i = 0; i < FACET_POOL_SIZE; i++) {
        dfpool[i].poolId = i;
    }
    CHECK(cudaMemcpy(fpool, dfpool, FACET_POOL_SIZE * sizeof(Facet), cudaMemcpyHostToDevice));
}

void ContextPool::freeOnUnifiedMemory() {
    if (vpool != nullptr) {
        cudaFree(vpool);
        delete[] dvpool;
        dvpool = nullptr;
    }
    if (hpool != nullptr) {
        cudaFree(hpool);
        delete[] dhpool;
        dhpool = nullptr;
    }
    if (fpool != nullptr) {
        cudaFree(fpool);
        delete[] dfpool;
        dfpool = nullptr;
    }
}

void ContextPool::mallocOnCpu() {
    vpool = new MCGAL::Vertex[VERTEX_POOL_SIZE];
    hpool = new MCGAL::Halfedge[HALFEDGE_POOL_SIZE];
    fpool = new MCGAL::Facet[FACET_POOL_SIZE];
    for (int i = 0; i < VERTEX_POOL_SIZE; i++) {
        vpool[i] = MCGAL::Vertex();
    }
    for (int i = 0; i < HALFEDGE_POOL_SIZE; i++) {
        hpool[i] = MCGAL::Halfedge();
    }
    for (int i = 0; i < FACET_POOL_SIZE; i++) {
        fpool[i] = MCGAL::Facet();
    }
}

void ContextPool::freeOnCpu() {
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

void ContextPool::copyToCuda() {
    CHECK(cudaMalloc(&dvpool, VERTEX_POOL_SIZE * sizeof(Vertex)));
    CHECK(cudaMalloc(&dhpool, HALFEDGE_POOL_SIZE * sizeof(Halfedge)));
    CHECK(cudaMalloc(&dfpool, FACET_POOL_SIZE * sizeof(Facet)));
    cudaMemcpy(dvpool, vpool, VERTEX_POOL_SIZE * sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(dhpool, hpool, HALFEDGE_POOL_SIZE * sizeof(Halfedge), cudaMemcpyHostToDevice);
    cudaMemcpy(dfpool, fpool, FACET_POOL_SIZE * sizeof(Facet), cudaMemcpyHostToDevice);
}

void ContextPool::freeCuda() {
    cudaFree(dvpool);
    cudaFree(dhpool);
    cudaFree(dfpool);
}

}  // namespace MCGAL