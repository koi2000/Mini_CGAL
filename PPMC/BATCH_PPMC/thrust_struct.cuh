#ifndef THRUST_STRUCT_H
#define THRUST_STRUCT_H

#include "../MCGAL/Core_CUDA/include/core.cuh"
/**
 * @brief 
*/
struct ExtractFacetPoolId {
    __host__ __device__ int operator()(const MCGAL::Facet& f) const {
        return f.poolId;
    }
};

struct FilterFacetByMeshId {
    MCGAL::Facet* fpool;  

    FilterFacetByMeshId(MCGAL::Facet* _fpool) : fpool(_fpool) {}
    __host__ __device__ bool operator()(const int& a) const {
        return fpool[a].meshId != -1;
    }
};

struct FilterFacetBySplitable {
    MCGAL::Facet* fpool;

    FilterFacetBySplitable(MCGAL::Facet* _fpool) : fpool(_fpool) {}
    __device__ bool operator()(const int& a) const {
        return fpool[a].meshId != -1 && fpool[a].isSplittableOnCuda();
    }
};

struct ExtractHalfedgePoolId {
    __host__ __device__ int operator()(const MCGAL::Halfedge& h) const {
        return h.poolId;
    }
};

struct FilterHalfedgeByMeshId {
    MCGAL::Halfedge* hpool;  

    FilterHalfedgeByMeshId(MCGAL::Halfedge* _hpool) : hpool(_hpool) {}
    __host__ __device__ bool operator()(const int& a) const {
        return hpool[a].meshId != -1;
    }
};

struct FilterHalfedgeByAdded {
    MCGAL::Halfedge* hpool;

    FilterHalfedgeByAdded(MCGAL::Halfedge* _hpool) : hpool(_hpool) {}
    __device__ bool operator()(const int& a) const {
        return hpool[a].meshId != -1 && hpool[a].isAddedOnCuda();
    }
};

struct SortFacetByForder {
    MCGAL::Facet* fpool;

    SortFacetByForder(MCGAL::Facet* _fpool) : fpool(_fpool) {}
    __host__ __device__ bool operator()(const int& fid1, const int& fid2) const {
        MCGAL::Facet* f1 = &fpool[fid1];
        MCGAL::Facet* f2 = &fpool[fid2];
        if (f1->forder == ~(unsigned long long)0) {
            return false;
        } else if (f2->forder == ~(unsigned long long)0) {
            return true;
        }
        if (f1->meshId == f2->meshId) {
            return f1->forder < f2->forder;
        }
        return f1->meshId < f2->meshId;
        // return f1->forder < f2->forder;
    }
};

struct SortHalfedgeByHorder {
    MCGAL::Halfedge* hpool;

    SortHalfedgeByHorder(MCGAL::Halfedge* _hpool) : hpool(_hpool) {}
    __host__ __device__ bool operator()(const int& hid1, const int& hid2) const {
        MCGAL::Halfedge* h1 = &hpool[hid1];
        MCGAL::Halfedge* h2 = &hpool[hid2];
        if (h1->horder == ~(unsigned long long)0) {
            return false;
        } else if (h2->horder == ~(unsigned long long)0) {
            return true;
        }
        if (h1->meshId == h2->meshId) {
            return h1->horder < h2->horder;
        }
        return h1->meshId < h2->meshId;
    }
};


/**
 * @brief 用于在cuda上compact facet order
*/
struct UpdateFacetOrderFunctor {
    int* fids;
    MCGAL::Facet* fpool;

    UpdateFacetOrderFunctor(int* _fids, MCGAL::Facet* _fpool) : fids(_fids), fpool(_fpool) {}

    __host__ __device__ void operator()(int index) const {
        int fidIndex = fids[index];

        if (fpool[fidIndex].forder != ~(unsigned long long)0) {
            MCGAL::Facet& facet = fpool[fidIndex];
            facet.forder = static_cast<unsigned long long>(index);
        }
    }
};

/**
 * @brief 用于在cuda上compact halfedge order
*/
struct UpdateHalfedgeOrderFunctor {
    int* hids;
    MCGAL::Halfedge* hpool;

    UpdateHalfedgeOrderFunctor(int* _hids, MCGAL::Halfedge* _hpool) : hids(_hids), hpool(_hpool) {}

    __host__ __device__ void operator()(int index) const {
        int hidIndex = hids[index];

        if (hpool[hidIndex].horder != ~(unsigned long long)0) {
            MCGAL::Halfedge& halfedge = hpool[hidIndex];
            halfedge.horder = static_cast<unsigned long long>(index);
        }
    }
};

/**
 * @brief 筛选掉所有未被更新order的facet
*/
struct FilterFacetByForder {
    MCGAL::Facet* fpool;  

    FilterFacetByForder(MCGAL::Facet* _fpool) : fpool(_fpool) {}
    __host__ __device__ bool operator()(const int& a) const {
        return fpool[a].forder != ~(unsigned long long)0;
    }
};


#endif