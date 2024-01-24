#include "DecompressTool.cuh"

__global__ void
readBaseMeshOnCuda(char* buffer, int* stOffsets, int num, int* vh_departureConquest, int* nbDecimations) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {}
}

DeCompressTool::~DeCompressTool() {
    stOffsets.clear();
    stOffsets.shrink_to_fit();
    lods.clear();
    lods.shrink_to_fit();
    nbDecimations.clear();
    nbDecimations.shrink_to_fit();
    vh_departureConquest.clear();
    vh_departureConquest.shrink_to_fit();
    splitableCounts.clear();
    splitableCounts.shrink_to_fit();
    insertedCounts.clear();
    insertedCounts.shrink_to_fit();

    delete[] buffer;
    cudaFree(dbuffer);
    cudaFree(dvh_departureConquest);
    cudaFree(dstOffsets);
    cudaFree(dfaceIndexes);
    cudaFree(dvertexIndexes);
    cudaFree(dstHalfedgeIndexes);
    cudaFree(dstFacetIndexes);
}

/**
 * 思考一种比较好的处理方式，是从多个路径读取多个文件还是从一个读出来然后解析
 * 这里确定一下，从多个文件里读取
 */
DeCompressTool::DeCompressTool(char** path, int number, bool is_base) {
    int dataOffset = 0;
    buffer = new char[BUFFER_SIZE];
    for (int i = 0; i < number; i++) {
        std::ifstream fin(path[i], std::ios::binary);
        int len2;
        fin.read((char*)&len2, sizeof(int));
        char* p_data = new char[len2];
        memset(p_data, 0, len2);
        stOffsets.push_back(dataOffset);
        fin.read(p_data, len2);
        memcpy(buffer + dataOffset, p_data, len2);
        dataOffset += len2;
        free(p_data);
    }
    CHECK(cudaMalloc((int**)&dfaceIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc((int**)&dvertexIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc((int**)&dstHalfedgeIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc((int**)&dstFacetIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dbuffer, dataOffset));
    CHECK(cudaMalloc(&dstOffsets, stOffsets.size() * sizeof(int)));
    CHECK(cudaMalloc(&dSplittabelCount, stOffsets.size() * sizeof(int)));
    CHECK(cudaMemcpy(dbuffer, buffer, dataOffset, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstOffsets, stOffsets.data(), stOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
    batch_size = number;
    if (is_base) {
        vh_departureConquest.resize(2 * number);
        nbDecimations.resize(number);
        splitableCounts.resize(number);
        insertedCounts.resize(number);
        dim3 block(256, 1, 1);
        dim3 grid((number + block.x - 1) / block.x, 1, 1);
#pragma omp parallel for
        for (int i = 0; i < number; i++) {
            readBaseMesh(i, &stOffsets[i]);
        }
    }
}

void DeCompressTool::decode(int lod) {
    if (lod < i_decompPercentage) {
        return;
    }
    i_decompPercentage = lod;
    b_jobCompleted = false;
    while (!b_jobCompleted) {
        startNextDecompresssionOp();
    }
}

__global__ void resetStateOnCuda(MCGAL::Halfedge* hpool, MCGAL::Facet* fpool, int num) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        MCGAL::Facet* fit = &fpool[tid];
        if (fit->isRemovedOnCuda()) {
            fit->setMeshIdOnCuda(-1);
            for (int i = 0; i < fit->halfedge_size; i++) {
                MCGAL::Halfedge* hit = fit->getHalfedgeByIndexOnCuda(hpool, i);
                if (hit->isRemovedOnCuda()) {
                    hit->setMeshIdOnCuda(-1);
                }
                hit->resetStateOnCuda();
            }
        } else {
            fit->resetStateOnCuda();
            for (int i = 0; i < fit->halfedge_size; i++) {
                MCGAL::Halfedge* hit = fit->getHalfedgeByIndexOnCuda(hpool, i);
                if (hit->isRemovedOnCuda()) {
                    hit->setMeshIdOnCuda(-1);
                }
                hit->resetStateOnCuda();
            }
        }
    }
}

void DeCompressTool::startNextDecompresssionOp() {
    // check if the target LOD is reached
    if (i_curDecimationId * 100.0 / nbDecimations[0] >= i_decompPercentage) {
        if (i_curDecimationId == nbDecimations[0]) {}
        b_jobCompleted = true;
        return;
    }
    std::vector<int> twos;
    // 1. reset the states. note that the states of the vertices need not to be reset
    //
    int number = MCGAL::contextPool.findex;
    dim3 block(256, 1, 1);
    dim3 grid((number + block.x - 1) / block.x, 1, 1);
    int vsize = MCGAL::contextPool.vindex;
    int hsize = MCGAL::contextPool.hindex;
    int fsize = MCGAL::contextPool.findex;
    CHECK(cudaMemcpy(MCGAL::contextPool.dvpool, MCGAL::contextPool.vpool, vsize * sizeof(MCGAL::Vertex),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MCGAL::contextPool.dhpool, MCGAL::contextPool.hpool, hsize * sizeof(MCGAL::Halfedge),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MCGAL::contextPool.dfpool, MCGAL::contextPool.fpool, fsize * sizeof(MCGAL::Facet),
                     cudaMemcpyHostToDevice));
    resetStateOnCuda<<<grid, block>>>(MCGAL::contextPool.dhpool, MCGAL::contextPool.dfpool, number);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(MCGAL::contextPool.vpool, MCGAL::contextPool.dvpool, vsize * sizeof(MCGAL::Vertex),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(MCGAL::contextPool.hpool, MCGAL::contextPool.dhpool, hsize * sizeof(MCGAL::Halfedge),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(MCGAL::contextPool.fpool, MCGAL::contextPool.dfpool, fsize * sizeof(MCGAL::Facet),
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < splitableCounts.size(); i++) {
        splitableCounts[i] = 0;
        insertedCounts[i] = 0;
    }
    i_curDecimationId++;  // increment the current decimation operation id.
    // 2. decoding the removed vertices and add to target facets
    struct timeval start = get_cur_time();
    // #pragma omp parallel for num_threads(batch_size)
    //     for (int i = 0; i < batch_size; i++) {
    //         RemovedVerticesDecodingStep(i);
    //     }
    BatchRemovedVerticesDecodingStep();
    logt("%d RemovedVerticesDecodingStep", start, i_curDecimationId);
    // 3. decoding the inserted edge and marking the ones added
    // #pragma omp parallel for num_threads(batch_size)
    //     for (int i = 0; i < batch_size; i++) {
    //         InsertedEdgeDecodingStep(i);
    //     }
    BatchInsertedEdgeDecodingStep();
    logt("%d InsertedEdgeDecodingStep", start, i_curDecimationId);
    // 4. truly insert the removed vertices
    insertRemovedVertices();
    logt("%d insertRemovedVertices", start, i_curDecimationId);
// 5. truly remove the added edges
#pragma omp parallel for num_threads(batch_size)
    for (int i = 0; i < batch_size; i++) {
        removeInsertedEdges(i);
    }
    logt("%d removeInsertedEdges", start, i_curDecimationId);
}

MCGAL::Halfedge* DeCompressTool::pushHehInit(int meshId) {
    MCGAL::Halfedge* hehBegin;
    MCGAL::Vertex* v1 = MCGAL::contextPool.getVertexByIndex(vh_departureConquest[meshId * 2 + 1]);
    MCGAL::Vertex* v0 = MCGAL::contextPool.getVertexByIndex(vh_departureConquest[meshId * 2]);
    for (int i = 0; i < v1->halfedges_size; i++) {
        MCGAL::Halfedge* hit = v1->getHalfedgeByIndex(i);
        if (hit->opposite()->vertex_ == vh_departureConquest[meshId * 2]) {
            hehBegin = hit->opposite();
            break;
        }
    }
    // assert(hehBegin->vertex() == vh_departureConquest[0]);
    // Push it to the queue.
    return hehBegin;
}

// bool cmpForder(MCGAL::Facet* f1, MCGAL::Facet* f2) {
//     return f1->forder < f2->forder;
// }

// bool cmpHorder(MCGAL::Halfedge* h1, MCGAL::Halfedge* h2) {
//     return h1->horder < h2->horder;
// }

bool cmpForder(int fid1, int fid2) {
    MCGAL::Facet* f1 = MCGAL::contextPool.getFacetByIndex(fid1);
    MCGAL::Facet* f2 = MCGAL::contextPool.getFacetByIndex(fid2);
    if (f1->forder == ~(unsigned long long)0) {
        return false;
    } else if (f2->forder == ~(unsigned long long)0) {
        return true;
    }
    if (f1->meshId == f2->meshId) {
        return f1->forder < f2->forder;
    }
    return f1->meshId < f2->meshId;
}

bool cmpHorder(int hid1, int hid2) {
    MCGAL::Halfedge* h1 = MCGAL::contextPool.getHalfedgeByIndex(hid1);
    MCGAL::Halfedge* h2 = MCGAL::contextPool.getHalfedgeByIndex(hid2);
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

void DeCompressTool::RemovedVerticesDecodingOnCuda() {
    int size = MCGAL::contextPool.findex;
    int* fids = new int[size];
    int index = 0;
    int* fsizes = new int[batch_size];
    int* fsizesSum = new int[batch_size + 1];
    memset(fsizes, 0, batch_size * sizeof(int));
    memset(fsizesSum, 0, (batch_size + 1) * sizeof(int));
    for (int i = 0; i < size; i++) {
        if (MCGAL::contextPool.fpool[i].meshId != -1) {
            fids[index++] = i;
            fsizes[MCGAL::contextPool.fpool[i].meshId]++;
        }
    }
    for (int i = 1; i < batch_size; i++) {
        fsizesSum[i] = fsizesSum[i - 1] + fsizes[i];
    }
    fsizesSum[batch_size] = index;
    int* firstQueue = new int[size];
    int* secondQueue = new int[size];
    int currentQueueSize = batch_size;
    int nextQueueSize = 0;
    int level = 0;
    for (int i = 0; i < batch_size; i++) {
        MCGAL::Halfedge* hit = pushHehInit(i);
        hit->facet()->forder = 0;
        firstQueue[i] = hit->poolId;
    }
}

__global__ void initForder(MCGAL::Facet* fpool, int* ids, int num) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        int fid = ids[tid];
        fpool[fid].forder = 0;
    }
}

void DeCompressTool::InsertedEdgeDecodingOnCuda() {}

__global__ void computeNextQueue(MCGAL::Vertex* vpool,
                                 MCGAL::Halfedge* hpool,
                                 MCGAL::Facet* fpool,
                                 int* currentQueue,
                                 int* nextQueue,
                                 int* nextQueueSize,
                                 int currentQueueSize) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < currentQueueSize) {
        int current = currentQueue[tid];
        MCGAL::Halfedge* h = &hpool[current];
        MCGAL::Facet* f = h->dfacet(fpool);
        if (f->isProcessedOnCuda()) {
            return;
        }
        MCGAL::Halfedge* hIt = h;
        unsigned long long idx = 1;
        do {
            MCGAL::Halfedge* hOpp = hIt->dopposite(hpool);
            unsigned long long order = f->forder << 4 | idx;
            atomicMin(&hOpp->dfacet(fpool)->forder, order);
            if (hOpp->dfacet(fpool)->forder == order && !hOpp->dfacet(fpool)->isProcessedOnCuda()) {
                idx++;
                if (hOpp->dfacet(fpool)->indexInQueue != -1) {
                    nextQueue[hOpp->dfacet(fpool)->indexInQueue] = hOpp->poolId;
                } else {
                    int position = atomicAdd(nextQueueSize, 1);
                    hOpp->dfacet(fpool)->indexInQueue = position;
                    nextQueue[position] = hOpp->poolId;
                }
            }
            hIt = hIt->dnext(hpool);
        } while (hIt != h);
    }
}

__global__ void
setProcessedProcessedFlagOnCuda(MCGAL::Halfedge* hpool, MCGAL::Facet* fpool, int* currentQueue, int currentQueueSize) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < currentQueueSize) {
        MCGAL::Halfedge* h = &hpool[currentQueue[tid]];
        h->dfacet(fpool)->indexInQueue = -1;
        h->dfacet(fpool)->setProcessedFlagOnCuda();
    }
}

// 函数对象用于提取结构体中的 member2
struct ExtractFacetPoolId {
    __host__ __device__ int operator()(const MCGAL::Facet& f) const {
        return f.poolId;
    }
};

struct FilterByMeshId {
    MCGAL::Facet* fpool;  // 外部变量

    // 构造函数
    FilterByMeshId(MCGAL::Facet* _fpool) : fpool(_fpool) {}
    __host__ __device__ bool operator()(const int& a) const {
        // 根据第三个成员是否为-1作为条件
        return fpool[a].meshId != -1;
    }
};

struct SortByForder {
    MCGAL::Facet* fpool;  // 外部变量

    // 构造函数
    SortByForder(MCGAL::Facet* _fpool) : fpool(_fpool) {}
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
    }
};

__global__ void meshIdCount(MCGAL::Facet* fpool, int* fsizes, int num) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        atomicAdd(&(fpool[tid].meshId), 1);
    }
}

__global__ void readFacetSymbolOnCuda(MCGAL::Facet* fpool,
                                      int* fids,
                                      int* fsizesSum,
                                      int* stOffsets,
                                      int* offsets,
                                      char* buffer,
                                      int num) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        MCGAL::Facet* facet = &fpool[fids[tid]];
        // 需要知道自己在自己这个mesh中是第几个
        int offset = stOffsets[facet->meshId] + tid - fsizesSum[facet->meshId];
        char symbol = readCharOnCuda(buffer, offset);
        if (symbol) {
            facet->setSplittableOnCuda();
            offsets[tid] = 1;
        } else {
            facet->setUnsplittableOnCuda();
        }
    }
}

__global__ void
readPointOnCuda(MCGAL::Facet* fpool, int* fids, int* fsizesSum, int* stOffsets, int* offsets, char* buffer, int num) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        MCGAL::Facet* facet = &fpool[fids[tid]];
        if (facet->isSplittableOnCuda()) {
            int st = facet->meshId == 0 ? 0 : offsets[fsizesSum[facet->meshId] - 1];
            int offset = offsets[tid] - 1 - st;
            float* p = readPointOnCuda(buffer, stOffsets[facet->meshId] + offset * sizeof(float) * 3);
            facet->setRemovedVertexPosOnCuda(p);
        }
    }
}

__global__ void arrayAdd(int* arr1, int* arr2, int num) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        arr1[tid] += arr2[tid];
    }
}

__global__ void calSplitableCounts(int* stOffsets, int* splitableCounts, int* offsets, int* fsizesSum, int num) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        int ed = offsets[fsizesSum[tid + 1] - 1];
        int st = fsizesSum[tid] == 0 ? 0 : offsets[fsizesSum[tid] - 1];
        splitableCounts[tid] = ed - st;
        stOffsets[tid] += (splitableCounts[tid]) * sizeof(float) * 3;
    }
}

/**
 * 不考虑meshId为-1的情况，会找个时候把meshId为-1的清理掉。这里需要注意很多东西，因为三个数据结构都在互相持有
 * 如果直接compact的话不是很好，基本就是reorganize了
 * 直接就需要创建一个拷贝
 * 是否可以使用 统一内存分配？ 需要考虑缺页中断的数量
 * 先尝试使用，有问题再说
 */
void DeCompressTool::BatchRemovedVerticesDecodingStep() {
    int size = MCGAL::contextPool.findex;

    thrust::device_vector<int> origin_fids(size);
    thrust::device_vector<int> fids(size);
    // 使用 thrust::transform 提取facet中的 poolId
    thrust::transform(MCGAL::contextPool.fpool, MCGAL::contextPool.fpool + size, origin_fids.begin(),
                      ExtractFacetPoolId());
    thrust::copy_if(origin_fids.begin(), origin_fids.end(), fids.begin(), FilterByMeshId(MCGAL::contextPool.dfpool));
    // 获取紧凑后的数组大小
    int index = count_if(origin_fids.begin(), origin_fids.end(), FilterByMeshId(MCGAL::contextPool.dfpool));
    // 将信息拷贝过来
    int* fsizes;
    int* fsizesSum;
    CHECK(cudaMalloc(&fsizes, batch_size * sizeof(int)));
    CHECK(cudaMalloc(&fsizesSum, (batch_size + 1) * sizeof(int)));
    dim3 block(256, 1, 1);
    dim3 grid((index + block.x - 1) / block.x, 1, 1);
    meshIdCount<<<grid, block>>>(MCGAL::contextPool.dfpool, fsizes, index);

    for (int i = 1; i < batch_size; i++) {
        fsizesSum[i] = fsizesSum[i - 1] + fsizes[i];
    }
    fsizesSum[batch_size] = index;

    int* d_firstQueue;
    int* d_secondQueue;
    int* d_nextQueueSize;
    int nextQueueSize = 0;
    cudaMalloc((void**)&d_firstQueue, size);
    cudaMalloc((void**)&d_secondQueue, size);
    cudaMalloc((void**)d_nextQueueSize, sizeof(int));
    cudaMemcpy(d_nextQueueSize, &nextQueueSize, sizeof(int), cudaMemcpyHostToDevice);
    int* h_firstQueue = new int[batch_size];
    int* stIds = new int[batch_size];
    int currentQueueSize = batch_size;

    int level = 0;
    for (int i = 0; i < batch_size; i++) {
        MCGAL::Halfedge* hit = pushHehInit(i);
        // hit->facet()->forder = 0;
        stIds[i] = hit->facet_;
        h_firstQueue[i] = hit->poolId;
    }
    // set forder by cuda
    initForder<<<1, batch_size>>>(MCGAL::contextPool.dfpool, stIds, batch_size);
    // copy first to queue
    CHECK(cudaMemcpy(d_firstQueue, &h_firstQueue, batch_size * sizeof(int), cudaMemcpyHostToDevice));

    int threshold = 64 / 4 - 1;
    int firstCount = 0;
    int secondCount = 0;
    while (currentQueueSize > 0) {
        int* d_currentQueue;
        int* d_nextQueue;
        if (level % 2 == 0) {
            d_currentQueue = d_firstQueue;
            d_nextQueue = d_secondQueue;
        } else {
            d_currentQueue = d_secondQueue;
            d_nextQueue = d_firstQueue;
        }
        dim3 block(256, 1, 1);
        dim3 grid((currentQueueSize + block.x - 1) / block.x, 1, 1);
        computeNextQueue<<<grid, block>>>(MCGAL::contextPool.dvpool, MCGAL::contextPool.dhpool,
                                          MCGAL::contextPool.dfpool, d_currentQueue, d_nextQueue, d_nextQueue,
                                          currentQueueSize);

        ++level;
        // 到达阈值后开始compact
        if (level == threshold) {
            // sort(faces.begin() + firstCount, faces.begin() + secondCount, cmpForder);
            // 需要一个新的临时的array，以order进行排序
            // sort(fids + firstCount, fids + index, cmpForder);
            thrust::sort(fids.begin() + firstCount, fids.begin() + index, SortByForder(MCGAL::contextPool.dfpool));
            for (int i = firstCount; i < index; i++) {
                if (MCGAL::contextPool.fpool[fids[i]].forder != (~(unsigned long long)0)) {
                    MCGAL::Facet* facet = &MCGAL::contextPool.fpool[fids[i]];
                    facet->forder = i;
                    secondCount = i;
                }
            }
            firstCount = secondCount;
            int power = 1;
            int x = secondCount + 1;
            while (x > 1) {
                x /= 2;
                power++;
            }
            threshold += (64 - power) / 4 - 1;
        }

        setProcessedProcessedFlagOnCuda<<<grid, block>>>(MCGAL::contextPool.dhpool, MCGAL::contextPool.dfpool,
                                                         d_currentQueue, currentQueueSize);
        CHECK(cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost));
        currentQueueSize = nextQueueSize;
        CHECK(cudaMemcpy(&d_nextQueueSize, &nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost));
    }
    // sort
    // sort(fids, fids + index, cmpForder);
    thrust::sort(fids.begin() + firstCount, fids.begin() + index, SortByForder(MCGAL::contextPool.dfpool));
    int* d_offset;

    // 需要计算前缀和
    CHECK(cudaMalloc(&d_offset, index * sizeof(int)));
    // std::vector<int> offsets(index);

    readFacetSymbolOnCuda<<<grid, block>>>(MCGAL::contextPool.fpool, thrust::raw_pointer_cast(fids.data()), fsizesSum,
                                           dstOffsets, d_offset, dbuffer, index);

    thrust::inclusive_scan(thrust::device, d_offset, d_offset + index, d_offset);
    arrayAdd<<<1, batch_size>>>(dstOffsets, fsizes, batch_size);
    readPointOnCuda<<<grid, block>>>(MCGAL::contextPool.fpool, thrust::raw_pointer_cast(fids.data()), fsizesSum,
                                     dstOffsets, d_offset, dbuffer, index);

    calSplitableCounts<<<1, batch_size>>>(dstOffsets, dSplittabelCount, d_offset, fsizesSum, batch_size);
}

void DeCompressTool::BatchInsertedEdgeDecodingStep() {
    int size = MCGAL::contextPool.hindex;
    int* firstQueue = new int[size];
    int* secondQueue = new int[size];
    int currentQueueSize = batch_size;
    int nextQueueSize = 0;
    int level = 0;
    int threshold = 64 / 4 - 1;
    int firstCount = 0;
    int secondCount = 0;
    int* hids = new int[size];
    // 将信息拷贝过来
    int index = 0;
    int* hsizes = new int[batch_size];
    int* hsizesSum = new int[batch_size + 1];
    memset(hsizes, 0, batch_size * sizeof(int));
    memset(hsizesSum, 0, (batch_size + 1) * sizeof(int));
    for (int i = 0; i < size; i++) {
        MCGAL::Halfedge* halfedge = &MCGAL::contextPool.hpool[i];
        if (halfedge->meshId != -1) {
            hids[index++] = i;
            hsizes[halfedge->facet()->meshId]++;
        }
    }
    for (int i = 1; i < batch_size; i++) {
        hsizesSum[i] = hsizesSum[i - 1] + hsizes[i];
    }
    hsizesSum[batch_size] = index;

    for (int i = 0; i < batch_size; i++) {
        MCGAL::Halfedge* hit = pushHehInit(i);
        hit->horder = 0;
        firstQueue[i] = hit->poolId;
    }

    while (currentQueueSize > 0) {
        int* currentQueue;
        int* nextQueue;
        if (level % 2 == 0) {
            currentQueue = firstQueue;
            nextQueue = secondQueue;
        } else {
            currentQueue = secondQueue;
            nextQueue = firstQueue;
        }
#pragma omp parallel for num_threads(128)
        for (int i = 0; i < currentQueueSize; i++) {
            int current = currentQueue[i];
            MCGAL::Halfedge* h = MCGAL::contextPool.getHalfedgeByIndex(current);
            if (h->isProcessed()) {
                continue;
            }
            MCGAL::Halfedge* hIt = h->next();
            unsigned long long idx = 1;
            while (hIt->opposite() != h) {
                unsigned long long order = h->horder << 4 | idx;

#pragma omp atomic compare
                hIt->horder = order < hIt->horder ? order : hIt->horder;

                if (hIt->horder == order) {
                    idx++;
                    int position;
#pragma omp critical
                    { position = nextQueueSize++; }
                    nextQueue[position] = hIt->poolId;
                }
                hIt = hIt->opposite()->next();
            };
        }
        ++level;
        // 到达阈值后开始compact
        if (level == threshold) {
            // sort(halfedges.begin() + firstCount, halfedges.begin() + secondCount, cmpHorder);
            sort(hids + firstCount, hids + index, cmpHorder);

            for (int i = firstCount; i < index; i++) {
                if (MCGAL::contextPool.hpool[hids[i]].horder != (~(unsigned long long)0)) {
                    MCGAL::contextPool.hpool[hids[i]].horder = i;
                    secondCount = i;
                }
            }
            firstCount = secondCount;

            int power = 1;
            int x = secondCount + 1;
            while (x > 1) {
                x /= 2;
                power++;
            }
            threshold += (64 - power) / 4 - 1;
        }
        // offFile << "\n";
        currentQueueSize = nextQueueSize;
        nextQueueSize = 0;
    }
    sort(hids, hids + index, cmpHorder);
    // 并行读取
#pragma omp parallel for num_threads(128)
    for (int i = 0; i < index; i++) {
        MCGAL::Halfedge* halfedge = MCGAL::contextPool.getHalfedgeByIndex(hids[i]);
        int offset = stOffsets[halfedge->meshId] + i - hsizesSum[halfedge->meshId];
        char symbol = readCharByOffset(offset);
        if (symbol) {
            halfedge->setAdded();
        }
    }
    for (int i = 0; i < batch_size; i++) {
        stOffsets[i] += hsizes[i];
    }
    delete firstQueue;
    delete secondQueue;
}

void DeCompressTool::RemovedVerticesDecodingStep(int meshId) {
    std::queue<MCGAL::Halfedge*> gateQueue;
    int splitable_count = 0;
    gateQueue.push(pushHehInit(meshId));
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        MCGAL::Facet* f = h->facet();

        // If the face is already processed, pick the next halfedge:
        if (f->isConquered())
            continue;

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h;
        do {
            MCGAL::Halfedge* hOpp = hIt->opposite();
            // TODO: wait
            // assert(!hOpp->is_border());
            if (!hOpp->facet()->isConquered())
                gateQueue.push(hOpp);
            hIt = hIt->next();
        } while (hIt != h);

        // Decode the face symbol.
        unsigned sym = readChar(&stOffsets[meshId]);
        if (sym == 1) {
            MCGAL::Point rmved = readPoint(&stOffsets[meshId]);
            f->setSplittable();
            splitable_count++;
            f->setRemovedVertexPos(rmved);
        } else {
            f->setUnsplittable();
        }
    }
    splitableCounts[meshId] = splitable_count;
}

void DeCompressTool::InsertedEdgeDecodingStep(int meshId) {
    std::queue<MCGAL::Halfedge*> gateQueue;
    int inserted_edgecount = 0;
    gateQueue.push(pushHehInit(meshId));
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        // Test if the edge has already been conquered.
        if (h->isProcessed())
            continue;

        // Mark the halfedge as processed.
        h->setProcessed();
        // h->opposite()->setProcessed();

        unsigned sym = readChar(&stOffsets[meshId]);
        // Determine if the edge is original or not.
        // Mark the edge to be removed.
        if (sym != 0) {
            h->setAdded();
            inserted_edgecount++;
        }

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h->next();
        while (hIt->opposite() != h) {
            if (!hIt->isProcessed() && !hIt->isNew())
                gateQueue.push(hIt);
            hIt = hIt->opposite()->next();
        }
        assert(!hIt->isNew());
    }
}

inline __device__ void insert_tip_cuda(MCGAL::Halfedge* hs, MCGAL::Halfedge* h, MCGAL::Halfedge* v) {
    h->setNextOnCuda(v->dnext(hs));
    v->setNextOnCuda(h->dopposite(hs));
}

// kernel function
__global__ void createCenterVertexOnCuda(MCGAL::Vertex* vpool,
                                         MCGAL::Halfedge* hpool,
                                         MCGAL::Facet* fpool,
                                         int* vertexIndexes,
                                         int* faceIndexes,
                                         int* stHalfedgeIndexes,
                                         int* stFacetIndexes,
                                         int num,
                                         double clockRate,
                                         int id) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        unsigned int startTime = clock64();

        int faceId = faceIndexes[tid];
        MCGAL::Facet* facet = &fpool[faceId];
        int vertexId = vertexIndexes[tid];
        MCGAL::Vertex* vnew = &vpool[vertexId];
        int stHalfedgeIndex = stHalfedgeIndexes[tid];
        int stFacetIndex = stFacetIndexes[tid];

        MCGAL::Halfedge* h = facet->getHalfedgeByIndexOnCuda(hpool, 0);
        MCGAL::Halfedge* hnew = &hpool[stHalfedgeIndex++];
        hnew->resetOnCuda(h->dend_vertex(vpool), vnew);

        MCGAL::Halfedge* oppo_new = &hpool[stHalfedgeIndex++];
        oppo_new->resetOnCuda(vnew, h->dend_vertex(vpool));
        hnew->setOppositeOnCuda(oppo_new);
        oppo_new->setOppositeOnCuda(hnew);
        insert_tip_cuda(hpool, hnew->dopposite(hpool), h);
        MCGAL::Halfedge* g = hnew->dopposite(hpool)->dnext(hpool);
        MCGAL::Halfedge* hed = hnew;
        while (g->dnext(hpool)->poolId != hed->poolId) {
            MCGAL::Halfedge* gnew = &hpool[stHalfedgeIndex++];
            gnew->resetOnCuda(g->dend_vertex(vpool), vnew);

            MCGAL::Halfedge* oppo_gnew = &hpool[stHalfedgeIndex++];
            oppo_gnew->resetOnCuda(vnew, g->dend_vertex(vpool));

            gnew->setOppositeOnCuda(oppo_gnew);
            oppo_gnew->setOppositeOnCuda(gnew);
            gnew->setNextOnCuda(hnew->dopposite(hpool));
            insert_tip_cuda(hpool, gnew->dopposite(hpool), g);
            g = gnew->dopposite(hpool)->dnext(hpool);
            hnew = gnew;
        }

        hed->setNextOnCuda(hnew->dopposite(hpool));
        for (int i = 1; i < h->dfacet(fpool)->halfedge_size; i += 1) {
            MCGAL::Halfedge* hit = &hpool[h->dfacet(fpool)->halfedges[i]];
            fpool[stFacetIndex++].resetOnCuda(vpool, hpool, hit);
        }
        h->dfacet(fpool)->resetOnCuda(vpool, hpool, h);
    }
}

// 多线程预处理
// 将最后一步放到cuda上
void DeCompressTool::insertRemovedVertices() {
    struct timeval start = get_cur_time();
    int splitable_count = 0;
    for (int i = 0; i < splitableCounts.size(); i++) {
        splitable_count += splitableCounts[i];
    }

    std::vector<int> faceIndexes(splitable_count);
    std::vector<int> vertexIndexes(splitable_count);
    std::vector<int> stHalfedgeIndexes(splitable_count);
    std::vector<int> stFacetIndexes(splitable_count);
    int index = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double clockRate = prop.clockRate;
    int findex = MCGAL::contextPool.findex;
//
#pragma omp parallel for num_threads(50) schedule(dynamic)
    for (int i = 0; i < findex; i++) {
        MCGAL::Facet* fit = MCGAL::contextPool.getFacetByIndex(i);
        if (fit->meshId != -1 && fit->isSplittable()) {
            faceIndexes[index] = fit->poolId;
            int hcount = fit->halfedge_size * 2;
            int fcount = fit->halfedge_size - 1;
            // atomic
            int findex;
#pragma omp critical
            { findex = MCGAL::contextPool.preAllocFace(fcount); }
            for (int i = 0; i < fcount; i++) {
                MCGAL::contextPool.getFacetByIndex(findex + i)->setMeshId(fit->meshId);
            }
            stFacetIndexes[index] = findex;
            int hindex;
#pragma omp critical
            {
                hindex = MCGAL::contextPool.preAllocHalfedge(hcount);
                stHalfedgeIndexes[index] = hindex;
            }
            for (int j = 0; j < hcount; j++) {
                MCGAL::Halfedge* h = MCGAL::contextPool.getHalfedgeByIndex(hindex + j);
                h->setMeshId(fit->meshId);
            }
            MCGAL::Vertex* vnew;
#pragma omp critical
            {
                vertexIndexes[index] = MCGAL::contextPool.getVindex();
                // atomic
                vnew = MCGAL::contextPool.allocateVertexFromPool(fit->getRemovedVertexPos());
            }
            vnew->setMeshId(fit->meshId);
#pragma omp atomic
            index++;
            for (int k = 0; k < fit->halfedge_size; k++) {
                MCGAL::Halfedge* h = fit->getHalfedgeByIndex(k);
                h->end_vertex()->addHalfedge(hindex + k * 2);
                vnew->addHalfedge(hindex + k * 2 + 1);
            }
        }
    }
    // add it to mesh
    int num = splitable_count;
    dim3 block(256, 1, 1);
    dim3 grid((num + block.x - 1) / block.x, 1, 1);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
    CHECK(cudaMemcpy(dfaceIndexes, faceIndexes.data(), faceIndexes.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dvertexIndexes, vertexIndexes.data(), vertexIndexes.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstHalfedgeIndexes, stHalfedgeIndexes.data(), stHalfedgeIndexes.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstFacetIndexes, stFacetIndexes.data(), stFacetIndexes.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    int vsize = MCGAL::contextPool.vindex;
    int hsize = MCGAL::contextPool.hindex;
    int fsize = MCGAL::contextPool.findex;
    // log("size is %d %d %d", vsize, hsize, fsize);
    CHECK(cudaMemcpy(MCGAL::contextPool.dvpool, MCGAL::contextPool.vpool, vsize * sizeof(MCGAL::Vertex),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MCGAL::contextPool.dhpool, MCGAL::contextPool.hpool, hsize * sizeof(MCGAL::Halfedge),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MCGAL::contextPool.dfpool, MCGAL::contextPool.fpool, fsize * sizeof(MCGAL::Facet),
                     cudaMemcpyHostToDevice));
    createCenterVertexOnCuda<<<grid, block>>>(MCGAL::contextPool.dvpool, MCGAL::contextPool.dhpool,
                                              MCGAL::contextPool.dfpool, dvertexIndexes, dfaceIndexes,
                                              dstHalfedgeIndexes, dstFacetIndexes, num, clockRate, i_curDecimationId);
    cudaDeviceSynchronize();
    double t = logt("%d kernel function", start, i_curDecimationId);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
    CHECK(cudaMemcpy(MCGAL::contextPool.vpool, MCGAL::contextPool.dvpool, vsize * sizeof(MCGAL::Vertex),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(MCGAL::contextPool.hpool, MCGAL::contextPool.dhpool, hsize * sizeof(MCGAL::Halfedge),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(MCGAL::contextPool.fpool, MCGAL::contextPool.dfpool, fsize * sizeof(MCGAL::Facet),
                     cudaMemcpyDeviceToHost));
    logt("%d cuda memory copy back", start, i_curDecimationId);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

__device__ MCGAL::Halfedge* find_prevOncuda(MCGAL::Halfedge* hpool, MCGAL::Halfedge* h) {
    MCGAL::Halfedge* g = h;
    int idx = 0;
    while (g->dnext(hpool) != h) {
        if (idx >= 120) {
            printf("error\n");
            break;
        }
        idx++;
        g = g->dnext(hpool);
    }

    return g;
}

inline __device__ void remove_tipOnCuda(MCGAL::Halfedge* hpool, MCGAL::Halfedge* h) {
    // h->next = h->next->opposite->next;
    h->setNextOnCuda(h->dnext(hpool)->dopposite(hpool)->dnext(hpool));
}

__device__ void joinFacetDevice(MCGAL::Vertex* vpool, MCGAL::Halfedge* hpool, MCGAL::Facet* fpool, MCGAL::Halfedge* h) {
    MCGAL::Halfedge* hprev = find_prevOncuda(hpool, h);
    MCGAL::Halfedge* gprev = find_prevOncuda(hpool, h->dopposite(hpool));
    atomicAdd(&h->count, 1);
    // atomicAdd(&hprev->count, 1);
    remove_tipOnCuda(hpool, hprev);
    remove_tipOnCuda(hpool, gprev);
    gprev->dfacet(fpool)->setRemovedOnCuda();
    hprev->dfacet(fpool)->resetOnCuda(vpool, hpool, hprev);
}

__global__ void joinFacetOnCuda(MCGAL::Vertex* vpool,
                                MCGAL::Halfedge* hpool,
                                MCGAL::Facet* fpool,
                                int* edgeIndexes,
                                int* stIndexes,
                                int* thNumberes,
                                int num,
                                double clockRate) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        int stIndex = stIndexes[tid];
        int thNumber = thNumberes[tid];
        for (int i = 0; i < thNumber; i++) {
            MCGAL::Halfedge* h = &hpool[edgeIndexes[stIndex + i]];
            joinFacetDevice(vpool, hpool, fpool, h);
        }
    }
}

/**
 * Remove all the marked edges on cuda
 */
// void DeCompressTool::removeInsertedEdgesOnCuda() {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     double clockRate = prop.clockRate;
//     struct timeval start = get_cur_time();
//     // std::vector<int> edgeIndex(inserted_edgecount);
//     std::vector<int> edgeIndexes;
//     std::vector<int> stIndex;
//     std::vector<int> thNumber;
//     for (int i = 0; i < faces.size(); i++) {
//         MCGAL::Facet* node = faces[i];
//         if (node->isVisited()) {
//             continue;
//         }
//         // 记录这一轮bfs所有可用的面
//         std::vector<int> ids;
//         std::queue<MCGAL::Facet*> fqueue;
//         fqueue.push(node);
//         while (!fqueue.empty()) {
//             MCGAL::Facet* fit = fqueue.front();
//             fqueue.pop();
//             if (fit->isVisited()) {
//                 continue;
//             }
//             fit->setVisitedFlag();
//             int flag = 0;
//             for (int j = 0; j < fit->halfedge_size; j++) {
//                 MCGAL::Halfedge* hit = fit->getHalfedgeByIndex(j);
//                 MCGAL::Facet* fit2 = hit->opposite()->facet();
//                 if (hit->isAdded() && !hit->isVisited()) {
//                     ids.push_back(hit->poolId);
//                     hit->setVisited();
//                     hit->opposite()->setRemoved();
//                     // fit2->setRemoved();
//                     hit->vertex()->eraseHalfedgeByPointer(hit);
//                     hit->opposite()->vertex()->eraseHalfedgeByPointer(hit->opposite());
//                     fqueue.push(fit2);
//                 } else if (hit->opposite()->isAdded() && !hit->opposite()->isVisited()) {
//                     ids.push_back(hit->poolId);
//                     hit->opposite()->setVisited();
//                     hit->setRemoved();
//                     // fit2->setRemoved();
//                     hit->vertex()->eraseHalfedgeByPointer(hit);
//                     hit->opposite()->vertex()->eraseHalfedgeByPointer(hit->opposite());
//                     fqueue.push(fit2);
//                 }
//             }
//         }
//         if (!ids.empty()) {
//             stIndex.push_back(edgeIndexes.size());
//             for (int j = 0; j < ids.size(); j++) {
//                 edgeIndexes.push_back(ids[j]);
//             }
//             thNumber.push_back(ids.size());
//         }
//     }

//     logt("%d collect halfedge information", start, i_curDecimationId);
//     int* dedgeIndexes;
//     int* dstIndex;
//     int* dthNumber;
//     std::vector<int> edgeIndexesCnt(inserted_edgecount, 0);
//     CHECK(cudaMalloc(&dedgeIndexes, edgeIndexes.size() * sizeof(int)));
//     CHECK(cudaMalloc(&dstIndex, stIndex.size() * sizeof(int)));
//     CHECK(cudaMalloc(&dthNumber, thNumber.size() * sizeof(int)));
//     CHECK(cudaMemcpy(dedgeIndexes, edgeIndexes.data(), edgeIndexes.size() * sizeof(int), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(dstIndex, stIndex.data(), stIndex.size() * sizeof(int), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(dthNumber, thNumber.data(), thNumber.size() * sizeof(int), cudaMemcpyHostToDevice));
//     int vsize = MCGAL::contextPool.vindex;
//     int hsize = MCGAL::contextPool.hindex;
//     int fsize = MCGAL::contextPool.findex;
//     int num = stIndex.size();
//     CHECK(cudaMemcpy(MCGAL::contextPool.dvpool, MCGAL::contextPool.vpool, vsize * sizeof(MCGAL::Vertex),
//                      cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(MCGAL::contextPool.dhpool, MCGAL::contextPool.hpool, hsize * sizeof(MCGAL::Halfedge),
//                      cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(MCGAL::contextPool.dfpool, MCGAL::contextPool.fpool, fsize * sizeof(MCGAL::Facet),
//                      cudaMemcpyHostToDevice));
//     dim3 block(256, 1, 1);
//     dim3 grid((num + block.x - 1) / block.x, 1, 1);
//     logt("%d cuda memcpy copy", start, i_curDecimationId);
//     joinFacetOnCuda<<<grid, block>>>(MCGAL::contextPool.dvpool, MCGAL::contextPool.dhpool, MCGAL::contextPool.dfpool,
//                                      dedgeIndexes, dstIndex, dthNumber, num, clockRate);
//     cudaDeviceSynchronize();
//     logt("%d join facet kernel", start, i_curDecimationId);
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         printf("ERROR: %s:%d,", __FILE__, __LINE__);
//         printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
//         exit(1);
//     }
//     CHECK(cudaMemcpy(MCGAL::contextPool.vpool, MCGAL::contextPool.dvpool, vsize * sizeof(MCGAL::Vertex),
//                      cudaMemcpyDeviceToHost));
//     CHECK(cudaMemcpy(MCGAL::contextPool.hpool, MCGAL::contextPool.dhpool, hsize * sizeof(MCGAL::Halfedge),
//                      cudaMemcpyDeviceToHost));
//     CHECK(cudaMemcpy(MCGAL::contextPool.fpool, MCGAL::contextPool.dfpool, fsize * sizeof(MCGAL::Facet),
//                      cudaMemcpyDeviceToHost));
//     cudaFree(dedgeIndexes);
//     cudaFree(dstIndex);
//     cudaFree(dthNumber);
//     // exit(0);
//     return;
// }

void DeCompressTool::removeInsertedEdges(int meshId) {
    std::queue<MCGAL::Halfedge*> gateQueue;
    gateQueue.push(pushHehInit(meshId));
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        if (h->isVisited())
            continue;

        if (h->isRemoved()) {
            continue;
        }
        // Mark the face as processed.
        h->setVisited();

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h;
        do {
            MCGAL::Halfedge* hOpp = hIt->opposite();
            // TODO: wait
            // assert(!hOpp->is_border());
            if (!hOpp->isVisited())
                gateQueue.push(hOpp);
            hIt = hIt->next();
        } while (hIt != h);

        if (hIt->isRemoved()) {
            hIt->setVisited();
            continue;
        }
        if (hIt->isAdded()) {
            join_facet(hIt);
            hIt->setVisited();
        }
    }
    return;
}

MCGAL::Halfedge* DeCompressTool::find_prev(MCGAL::Halfedge* h) const {
    MCGAL::Halfedge* g = h;
    while (g->next() != h)
        g = g->next();
    return g;
}

inline void DeCompressTool::remove_tip(MCGAL::Halfedge* h) const {
    h->next_ = h->next()->opposite()->next_;
}

MCGAL::Halfedge* DeCompressTool::join_facet(MCGAL::Halfedge* h) {
    MCGAL::Halfedge* hprev = find_prev(h);
    MCGAL::Halfedge* gprev = find_prev(h->opposite());
    remove_tip(hprev);
    remove_tip(gprev);
    h->opposite()->setRemoved();
    h->setRemoved();
    h->setMeshId(-1);
    h->opposite()->setMeshId(-1);
    h->vertex()->eraseHalfedgeByPointer(h);
    h->opposite()->vertex()->eraseHalfedgeByPointer(h->opposite());
    gprev->facet()->setRemoved();
    hprev->facet()->reset(hprev);
    return hprev;
}

void DeCompressTool::readBaseMesh(int meshId, int* offset) {
    // read the number of level of detail
    int i_nbDecimations = readuInt16(offset);
    nbDecimations[meshId] = i_nbDecimations;
    // set the mesh bounding box
    unsigned i_nbVerticesBaseMesh = readInt(offset);
    unsigned i_nbFacesBaseMesh = readInt(offset);

    std::deque<MCGAL::Point>* p_pointDeque = new std::deque<MCGAL::Point>();
    std::deque<uint32_t*>* p_faceDeque = new std::deque<uint32_t*>();
    // Read the vertex positions.
    for (unsigned i = 0; i < i_nbVerticesBaseMesh; ++i) {
        MCGAL::Point pos = readPoint(offset);
        p_pointDeque->push_back(pos);
    }
    // read the face vertex indices
    // Read the face vertex indices.
    for (unsigned i = 0; i < i_nbFacesBaseMesh; ++i) {
        int nv = readInt(offset);
        uint32_t* f = new uint32_t[nv + 1];
        // Write in the first cell of the array the face degree.
        f[0] = nv;
        for (unsigned j = 1; j < nv + 1; ++j) {
            f[j] = readInt(offset);
        }
        p_faceDeque->push_back(f);
    }
    // Let the builder do its job.
    buildFromBuffer(meshId, p_pointDeque, p_faceDeque);

    // Free the memory.
    for (unsigned i = 0; i < p_faceDeque->size(); ++i) {
        delete[] p_faceDeque->at(i);
    }
    delete p_faceDeque;
    delete p_pointDeque;
}

void DeCompressTool::buildFromBuffer(int meshId,
                                     std::deque<MCGAL::Point>* p_pointDeque,
                                     std::deque<uint32_t*>* p_faceDeque) {
    std::vector<MCGAL::Vertex*> vertices;
    // add vertex to Mesh
    for (std::size_t i = 0; i < p_pointDeque->size(); ++i) {
        MCGAL::Point p = p_pointDeque->at(i);
        MCGAL::Vertex* vt = MCGAL::contextPool.allocateVertexFromPool(p);
        vt->setMeshId(meshId);
        vertices.push_back(vt);
    }
    vh_departureConquest[meshId * 2] = vertices[0]->poolId;
    vh_departureConquest[meshId * 2 + 1] = vertices[1]->poolId;
    // read face and add to Mesh
    for (int i = 0; i < p_faceDeque->size(); ++i) {
        uint32_t* ptr = p_faceDeque->at(i);
        int num_face_vertices = ptr[0];
        std::vector<MCGAL::Vertex*> vts;
        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index = ptr[j + 1];
            vts.push_back(vertices[vertex_index]);
        }
        MCGAL::Facet* face = MCGAL::contextPool.allocateFaceFromPool(vts);
        face->setMeshId(meshId);
        for (int k = 0; k < face->halfedge_size; k++) {
            MCGAL::Halfedge* hit = face->getHalfedgeByIndex(k);
            hit->setMeshId(face->meshId);
        }
    }
    // clear vector
    vertices.clear();
}

void DeCompressTool::dumpto(std::string prefix) {
    std::vector<std::vector<MCGAL::Vertex*>> vertices(batch_size);
    std::vector<std::vector<MCGAL::Facet*>> facets(batch_size);
    int vindex = MCGAL::contextPool.vindex;
    int findex = MCGAL::contextPool.findex;
#pragma omp parallel for num_threads(60)
    for (int i = 0; i < vindex; i++) {
        MCGAL::Vertex* v = MCGAL::contextPool.getVertexByIndex(i);
        if (v->meshId != -1) {
            vertices[v->meshId].push_back(v);
        }
    }
#pragma omp parallel for num_threads(60)
    for (int i = 0; i < findex; i++) {
        MCGAL::Facet* f = MCGAL::contextPool.getFacetByIndex(i);
        if (f->meshId != -1) {
            if (f->isRemoved())
                continue;
            facets[f->meshId].push_back(f);
        }
    }
    char path[256];
#pragma omp parallel for num_threads(batch_size)
    for (int i = 0; i < batch_size; i++) {
        std::sprintf(path, prefix.c_str(), i);
        dumpto(vertices[i], facets[i], path);
    }
}

void DeCompressTool::dumpto(std::vector<MCGAL::Vertex*> vertices, std::vector<MCGAL::Facet*> facets, char* path) {
    std::ofstream offFile(path);
    if (!offFile.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }
    // write header
    offFile << "OFF\n";
    offFile << vertices.size() << " " << facets.size() << " 0\n";
    offFile << "\n";
    // write vertex
    int id = 0;
    for (MCGAL::Vertex* vertex : vertices) {
        offFile << vertex->x() << " " << vertex->y() << " " << vertex->z() << "\n";
        vertex->setId(id++);
    }

    for (MCGAL::Facet* face : facets) {
        if (face->isRemoved())
            continue;
        offFile << face->vertex_size << " ";
        MCGAL::Halfedge* hst = MCGAL::contextPool.getHalfedgeByIndex(face->halfedges[0]);
        MCGAL::Halfedge* hed = hst;
        // for (int i = 0; i < face->halfedge_size; i++) {
        //     hst = face->getHalfedgeByIndex(i);
        //     offFile << hst->vertex()->getId() << " ";
        // }

        do {
            offFile << hst->vertex()->getId() << " ";
            hst = hst->next();
        } while (hst != hed);
        offFile << "\n";
    }

    offFile.close();
}
