#include "core.h"

namespace MCGAL {

ContextPool::ContextPool() {
    // mallocOnUnifiedMemory();
    mallocOnCpu();
}

ContextPool::~ContextPool() {
    // freeOnUnifiedMemory();
    freeOnCpu();
}

void ContextPool::mallocOnUnifiedMemory() {
    vpool = new MCGAL::Vertex*[VERTEX_POOL_SIZE];
    CHECK(cudaMallocManaged(vpool, VERTEX_POOL_SIZE * sizeof(Vertex*)));
    hpool = new MCGAL::Halfedge*[HALFEDGE_POOL_SIZE];
    CHECK(cudaMallocManaged(hpool, HALFEDGE_POOL_SIZE * sizeof(Halfedge*)));
    fpool = new MCGAL::Facet*[FACET_POOL_SIZE];
    CHECK(cudaMallocManaged(fpool, FACET_POOL_SIZE * sizeof(Facet*)));
    for (int i = 0; i < VERTEX_POOL_SIZE; i++) {
        vpool[i] = new MCGAL::Vertex();
        CHECK(cudaMallocManaged(&vpool[i], sizeof(Vertex)));
    }
    for (int i = 0; i < HALFEDGE_POOL_SIZE; i++) {
        hpool[i] = new MCGAL::Halfedge();
        CHECK(cudaMallocManaged(&hpool[i], sizeof(Halfedge)));
    }
    for (int i = 0; i < FACET_POOL_SIZE; i++) {
        fpool[i] = new MCGAL::Facet();
        CHECK(cudaMallocManaged(&fpool[i], sizeof(Facet)));
    }
}

void ContextPool::freeOnUnifiedMemory() {
    if (vpool != nullptr) {
        for (int i = 0; i < VERTEX_POOL_SIZE; i++) {
            cudaFree(vpool[i]);
            delete vpool[i];
            vpool[i] = nullptr;
        }
        delete[] vpool;
        cudaFree(vpool);
        vpool = nullptr;
    }
    if (hpool != nullptr) {
        for (int i = 0; i < HALFEDGE_POOL_SIZE; i++) {
            cudaFree(hpool[i]);
            delete hpool[i];
            hpool[i] = nullptr;
        }
        cudaFree(hpool);
        delete[] hpool;
        hpool = nullptr;
    }
    if (fpool != nullptr) {
        for (int i = 0; i < FACET_POOL_SIZE; i++) {
            cudaFree(fpool[i]);
            delete fpool[i];
            fpool[i] = nullptr;
        }
        cudaFree(fpool);
        delete[] fpool;
        fpool = nullptr;
    }
}

void ContextPool::mallocOnCpu() {
    vpool = new MCGAL::Vertex*[VERTEX_POOL_SIZE];
    hpool = new MCGAL::Halfedge*[HALFEDGE_POOL_SIZE];
    fpool = new MCGAL::Facet*[FACET_POOL_SIZE];
    for (int i = 0; i < VERTEX_POOL_SIZE; i++) {
        vpool[i] = new MCGAL::Vertex();
    }
    for (int i = 0; i < HALFEDGE_POOL_SIZE; i++) {
        hpool[i] = new MCGAL::Halfedge();
    }
    for (int i = 0; i < FACET_POOL_SIZE; i++) {
        fpool[i] = new MCGAL::Facet();
    }
}

void ContextPool::freeOnCpu() {
    if (vpool != nullptr) {
        for (int i = 0; i < VERTEX_POOL_SIZE; i++) {
            delete vpool[i];
            vpool[i] = nullptr;
        }
        delete[] vpool;
        vpool = nullptr;
    }
    if (hpool != nullptr) {
        for (int i = 0; i < HALFEDGE_POOL_SIZE; i++) {
            delete hpool[i];
            hpool[i] = nullptr;
        }
        delete[] hpool;
        hpool = nullptr;
    }
    if (fpool != nullptr) {
        for (int i = 0; i < FACET_POOL_SIZE; i++) {
            delete fpool[i];
            fpool[i] = nullptr;
        }
        delete[] fpool;
        fpool = nullptr;
    }
}

void ContextPool::copyToCuda() {
    cudaMalloc((void**)&dvpool, VERTEX_POOL_SIZE * sizeof(Vertex*));
    for (int i = 0; i < VERTEX_POOL_SIZE; ++i) {
        cudaMalloc((void**)&(dvpool[i]), sizeof(Vertex));
        cudaMemcpy(dvpool[i], vpool[i], sizeof(Vertex), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**)&dhpool, HALFEDGE_POOL_SIZE * sizeof(Halfedge*));
    for (int i = 0; i < HALFEDGE_POOL_SIZE; ++i) {
        cudaMalloc((void**)&(dhpool[i]), sizeof(Halfedge));
        cudaMemcpy(dhpool[i], hpool[i], sizeof(Halfedge), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**)&dfpool, FACET_POOL_SIZE * sizeof(Facet*));
    for (int i = 0; i < FACET_POOL_SIZE; ++i) {
        cudaMalloc((void**)&(dfpool[i]), sizeof(Facet));
        cudaMemcpy(dfpool[i], fpool[i], sizeof(Facet), cudaMemcpyHostToDevice);
    }
}

void ContextPool::freeCuda() {
    for (int i = 0; i < VERTEX_POOL_SIZE; ++i) {
        cudaFree(dvpool[i]);
    }
    cudaFree(dvpool);

    for (int i = 0; i < HALFEDGE_POOL_SIZE; ++i) {
        cudaFree(dhpool[i]);
    }
    cudaFree(dhpool);

    for (int i = 0; i < FACET_POOL_SIZE; ++i) {
        cudaFree(dfpool[i]);
    }
    cudaFree(dfpool);
}

}  // namespace MCGAL