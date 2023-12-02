#ifndef CONTEXTPOOL_H
#define CONTEXTPOOL_H
#include <vector>
#include "Configuration.h"
#include "Vertex.h"
#include "Halfedge.h"
#include "Facet.h"
#include "cuda_util.h"
namespace MCGAL {

class ContextPool {
  private:
    // try to use cuda zero copy
    MCGAL::Vertex* vpool = nullptr;
    MCGAL::Halfedge* hpool = nullptr;
    MCGAL::Facet* fpool = nullptr;

    MCGAL::Vertex* dvpool;
    MCGAL::Halfedge* dhpool;
    MCGAL::Facet* dfpool;

    int vindex = 0;
    int hindex = 0;
    int findex = 0;
    ContextPool();

  public:
    ~ContextPool();

    static ContextPool& getInstance() {
        static ContextPool contextPool;
        return contextPool;
    }

    void copyToCuda();
    void freeCuda();
    void mallocOnUnifiedMemory();
    void freeOnUnifiedMemory();

    void mallocOnCpu();
    void freeOnCpu();

    ContextPool(const ContextPool&) = delete;
    ContextPool& operator=(const ContextPool&) = delete;

    MCGAL::Vertex* getVertexByIndex(int index) {
        return &vpool[index];
    }

    MCGAL::Halfedge* getHalfedgeByIndex(int index) {
        return &hpool[index];
    }

    MCGAL::Facet* getFacetByIndex(int index) {
        return &fpool[index];
    }

    inline MCGAL::Vertex* allocateVertexFromPool() {
        return &vpool[vindex++];
    }

    inline MCGAL::Vertex* allocateVertexFromPool(MCGAL::Point& p) {
        vpool[vindex].setPoint(p);
        return &vpool[vindex++];
    }

    inline MCGAL::Halfedge* allocateHalfedgeFromPool() {
        return &hpool[hindex++];
    }

    inline MCGAL::Halfedge* allocateHalfedgeFromPool(MCGAL::Vertex* v1, MCGAL::Vertex* v2) {
        hpool[hindex].reset(v1, v2);
        return &hpool[hindex++];
    }

    inline MCGAL::Facet* allocateFaceFromPool() {
        return &fpool[findex++];
    }

    inline MCGAL::Facet* allocateFaceFromPool(MCGAL::Halfedge* h) {
        fpool[findex].reset(h);
        return &fpool[findex++];
    }

    inline MCGAL::Facet* allocateFaceFromPool(std::vector<MCGAL::Vertex*> vts) {
        fpool[findex].reset(vts);
        return &fpool[findex++];
    }

    inline int preAllocVertex(int size) {
        int ret = vindex;
        vindex += size;
        return ret;
    }

    inline int preAllocHalfedge(int size) {
        int ret = hindex;
        hindex += size;
        return ret;
    }

    inline int preAllocFace(int size) {
        int ret = findex;
        findex += size;
        return ret;
    }
};

}  // namespace MCGAL
#endif