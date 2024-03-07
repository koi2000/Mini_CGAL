#include "DecompressTool.h"

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
}

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
    batch_size = number;
    if (is_base) {
        vh_departureConquest.resize(2 * batch_size);
        nbDecimations.resize(batch_size);
        splitableCounts.resize(batch_size);
        insertedCounts.resize(batch_size);
        // #pragma omp parallel for
        for (int i = 0; i < number; i++) {
            readBaseMesh(i, &stOffsets[i]);
        }
    }
}

void DeCompressTool::decode(int lod) {
    startNextDecompresssionOp();
}

void DeCompressTool::resetState() {
    int size = *MCGAL::contextPool.findex;
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < size; i++) {
        MCGAL::Facet* fit = MCGAL::contextPool.getFacetByIndex(i);
        if (fit->isRemoved()) {
            fit->setMeshId(-1);
            for (int i = 0; i < fit->halfedges.size(); i++) {
                MCGAL::Halfedge* hit = fit->halfedges[i];
                if (hit->isRemoved()) {
                    hit->setMeshId(-1);
                }
                hit->resetState();
            }
        } else {
            fit->resetState();
            for (int i = 0; i < fit->halfedges.size(); i++) {
                MCGAL::Halfedge* hit = fit->halfedges[i];
                if (hit->isRemoved()) {
                    hit->setMeshId(-1);
                }
                hit->resetState();
            }
        }
    }
}

void DeCompressTool::startNextDecompresssionOp() {
    struct timeval start = get_cur_time();
    resetState();
    logt("%d resetState", start, i_curDecimationId);
    for (int i = 0; i < splitableCounts.size(); i++) {
        splitableCounts[i] = 0;
        insertedCounts[i] = 0;
    }
    i_curDecimationId++;
#pragma omp parallel for num_threads(batch_size) schedule(dynamic)
    for (int i = 0; i < batch_size; i++) {
        RemovedVerticesDecodingStep(i);
    }
    logt("%d RemovedVerticesDecodingStep", start, i_curDecimationId);
#pragma omp parallel for num_threads(batch_size) schedule(dynamic)
    for (int i = 0; i < batch_size; i++) {
        InsertedEdgeDecodingStep(i);
    }
    logt("%d InsertedEdgeDecodingStep", start, i_curDecimationId);
    insertRemovedVertices();
    logt("%d insertRemovedVertices", start, i_curDecimationId);
    removedInsertedEdges();
    logt("%d removedInsertedEdges", start, i_curDecimationId);
}

MCGAL::Halfedge* DeCompressTool::pushHehInit(int meshId) {
    MCGAL::Halfedge* hehBegin;
    MCGAL::Vertex* v1 = MCGAL::contextPool.getVertexByIndex(vh_departureConquest[meshId * 2 + 1]);
    MCGAL::Vertex* v0 = MCGAL::contextPool.getVertexByIndex(vh_departureConquest[meshId * 2]);
    for (int i = 0; i < v1->halfedges.size(); i++) {
        MCGAL::Halfedge* hit = v1->halfedges[i];
        if (hit->opposite->vertex->poolId == vh_departureConquest[meshId * 2]) {
            hehBegin = hit->opposite;
            break;
        }
    }
    return hehBegin;
}

void DeCompressTool::RemovedVerticesDecodingStep(int meshId) {
    std::queue<MCGAL::Halfedge*> gateQueue;
    int splitable_count = 0;
    gateQueue.push(pushHehInit(meshId));
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        MCGAL::Facet* f = h->face;

        // If the face is already processed, pick the next halfedge:
        if (f->isConquered())
            continue;

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h;
        do {
            MCGAL::Halfedge* hOpp = hIt->opposite;
            // TODO: wait
            // assert(!hOpp->is_border());
            if (!hOpp->face->isConquered())
                gateQueue.push(hOpp);
            hIt = hIt->next;
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
        MCGAL::Halfedge* hIt = h->next;
        while (hIt->opposite != h) {
            if (!hIt->isProcessed() && !hIt->isNew())
                gateQueue.push(hIt);
            hIt = hIt->opposite->next;
        }
        assert(!hIt->isNew());
    }
    insertedCounts[meshId] = inserted_edgecount;
}

template <typename T, typename Op> void DeCompressTool::omp_scan(int n, const T* in, T* out, Op op) {
    omp_set_dynamic(true);
    int number_of_processors = 64;
    if (n < number_of_processors) {
        number_of_processors = n;
    }
    // omp_set_num_threads(number_of_processors);
    int parallel_chunk = n / number_of_processors;

#pragma omp parallel for
    for (int i = 0; i < n; i += parallel_chunk) {
        out[i] = in[i];
        for (int j = i + 1; j < n && j < i + parallel_chunk; j++) {
            out[j] = op(out[j - 1], in[j]);
        }
    }

    for (int i = 2 * parallel_chunk - 1; i < n; i += parallel_chunk) {
        out[i] = op(out[i], out[i - parallel_chunk]);
    }

#pragma omp parallel for
    for (int i = parallel_chunk; i < n; i += parallel_chunk) {
        int temp = i + parallel_chunk;
        if (temp > n) {
            temp = n + 1;
        }
#pragma omp parallel for
        for (int j = i; j < temp - 1; j++) {
            out[j] = op(out[j], out[i - 1]);
        }
    }
}

void DeCompressTool::initStIndexes(int* vertexIndexes,
                                   int* faceIndexes,
                                   int* stFacetIndexes,
                                   int* stHalfedgeIndexes,
                                   int num) {
    // #pragma omp parallel for
    for (int i = 0; i < num; i++) {
        MCGAL::Facet* fit = MCGAL::contextPool.getFacetByIndex(faceIndexes[i]);
        int hcount = fit->halfedges.size() * 2;
        int fcount = fit->halfedges.size() - 1;
        vertexIndexes[i] = 1;
        stFacetIndexes[i] = fcount;
        stHalfedgeIndexes[i] = hcount;
    }
}

void DeCompressTool::arrayAddConstant(int* array, int constant, int num) {
#pragma omp parallel for
    for (int i = 0; i < num; i++) {
        array[i] = array[i] + constant;
    }
}

void DeCompressTool::insert_tip(MCGAL::Halfedge* h, MCGAL::Halfedge* v) {
    h->next = v->next;
    v->next = h->opposite;
}

void DeCompressTool::createCenterVertex(int* vertexIndexes,
                                        int* faceIndexes,
                                        int* stHalfedgeIndexes,
                                        int* stFacetIndexes,
                                        int num) {
#pragma omp parallel for schedule(dynamic)
    for (int tid = 0; tid < num; tid++) {
        int faceId = faceIndexes[tid];
        MCGAL::Facet* facet = &MCGAL::contextPool.fpool[faceId];
        int vertexId = vertexIndexes[tid];
        MCGAL::Vertex* vnew = &MCGAL::contextPool.vpool[vertexId];
        int stHalfedgeIndex = stHalfedgeIndexes[tid];
        int stFacetIndex = stFacetIndexes[tid];

        MCGAL::Halfedge* h = facet->halfedges[0];
        MCGAL::Halfedge* hnew = &MCGAL::contextPool.hpool[stHalfedgeIndex++];
        hnew->setVertex(h->end_vertex, vnew);

        MCGAL::Halfedge* oppo_new = &MCGAL::contextPool.hpool[stHalfedgeIndex++];
        oppo_new->setVertex(vnew, h->end_vertex);
        hnew->opposite = oppo_new;
        oppo_new->opposite = hnew;
        insert_tip(hnew->opposite, h);
        MCGAL::Halfedge* g = hnew->opposite->next;
        MCGAL::Halfedge* hed = hnew;
        while (g->next->poolId != hed->poolId) {
            MCGAL::Halfedge* gnew = &MCGAL::contextPool.hpool[stHalfedgeIndex++];
            gnew->setVertex(g->end_vertex, vnew);

            MCGAL::Halfedge* oppo_gnew = &MCGAL::contextPool.hpool[stHalfedgeIndex++];
            oppo_gnew->setVertex(vnew, g->end_vertex);

            gnew->opposite = oppo_gnew;
            oppo_gnew->opposite = gnew;
            gnew->next = hnew->opposite;
            insert_tip(gnew->opposite, g);
            g = gnew->opposite->next;
            hnew = gnew;
        }

        hed->next = hnew->opposite;
        for (int i = 1; i < h->face->halfedges.size(); i += 1) {
            MCGAL::Halfedge* hit = h->face->halfedges[i];
            MCGAL::contextPool.fpool[stFacetIndex++].reset(hit);
        }
        h->face->reset(h);
    }
}

void DeCompressTool::preAllocInit(int* vertexIndexes,
                                  int* faceIndexes,
                                  int* stFacetIndexes,
                                  int* stHalfedgeIndexes,
                                  int num) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num; i++) {
        MCGAL::Facet* fit = MCGAL::contextPool.getFacetByIndex(faceIndexes[i]);

        int hcount = fit->halfedges.size() * 2;
        int fcount = fit->halfedges.size() - 1;
        int stfindex = stFacetIndexes[i];
        for (int i = 0; i < fcount; i++) {
            MCGAL::contextPool.fpool[stfindex + i].setMeshId(fit->meshId);
        }
        int stHindex = stHalfedgeIndexes[i];
        for (int j = 0; j < hcount; j++) {
            MCGAL::contextPool.hpool[stHindex + j].setMeshId(fit->meshId);
            MCGAL::contextPool.hpool[stHindex + j].resetState();
        }
        MCGAL::Vertex* vnew = &MCGAL::contextPool.vpool[vertexIndexes[i]];
        vnew->setMeshId(fit->meshId);
        vnew->setPoint(fit->getRemovedVertexPos());
    }
}

MCGAL::Halfedge* DeCompressTool::find_prev(MCGAL::Halfedge* h) {
    MCGAL::Halfedge* g = h;
    int idx = 0;
    while (g->next != h) {
        if (idx >= 120) {
            // printf("error\n");
            break;
        }
        idx++;
        g = g->next;
    }

    return g;
}

inline void DeCompressTool::remove_tip(MCGAL::Halfedge* h) {
    // h->next = h->next->opposite->next;
    h->next = h->next->opposite->next;
}

void DeCompressTool::joinFacet(int* fids, int num) {
#pragma omp parallel for schedule(dynamic)
    for (int tid = 0; tid < num; tid++) {
        MCGAL::Facet* facet = &MCGAL::contextPool.fpool[fids[tid]];

        for (int i = 0; i < facet->halfedges.size(); i++) {
            if (facet->halfedges[i]->isAdded()) {
                // printf("%d %d\n", i_curId, facet->halfedges[i]);
                MCGAL::Halfedge* h = facet->halfedges[i];
                MCGAL::Halfedge* hprev = find_prev(h);
                MCGAL::Halfedge* gprev = find_prev(h->opposite);
                // atomicAdd(&hprev->count, 1);
                remove_tip(hprev);
                remove_tip(gprev);
                h->setRemoved();
                h->opposite->setRemoved();

                h->opposite->setMeshId(-1);
                h->setMeshId(-1);
                gprev->face->setRemoved();
                gprev->face->setMeshId(-1);
                hprev->face->reset(hprev);
            }
        }
    }
}

void DeCompressTool::insertRemovedVertices() {
    int size = *MCGAL::contextPool.findex;
    int splitable_count = 0;
    for (int i = 0; i < splitableCounts.size(); i++) {
        splitable_count += splitableCounts[i];
    }
    int* faceIndexes = new int[splitable_count];
    int* vertexIndexes = new int[splitable_count + 1];
    vertexIndexes[0] = 0;
    int* stHalfedgeIndexes = new int[splitable_count + 1];
    stHalfedgeIndexes[0] = 0;
    int* stFacetIndexes = new int[splitable_count + 1];
    stFacetIndexes[0] = 0;
    //
    int* sourceArray = new int[size];
    int* conditionArray = new int[size];
    int* prefixSum = new int[size];
    memset(prefixSum, 0, sizeof(int) * size);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        sourceArray[i] = i;
        if (MCGAL::contextPool.getFacetByIndex(i)->isSplittable() &&
            MCGAL::contextPool.getFacetByIndex(i)->meshId != -1) {
            conditionArray[i] = 1;
        } else {
            conditionArray[i] = 0;
        }
    }
    omp_scan(size, conditionArray, prefixSum, std::plus<int>());
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        if (conditionArray[i]) {
            faceIndexes[prefixSum[i] - 1] = sourceArray[i];
        }
    }
    initStIndexes(vertexIndexes + 1, faceIndexes, stFacetIndexes + 1, stHalfedgeIndexes + 1, splitable_count);

    omp_scan(splitable_count + 1, vertexIndexes, vertexIndexes, std::plus<int>());
    omp_scan(splitable_count + 1, stFacetIndexes, stFacetIndexes, std::plus<int>());
    omp_scan(splitable_count + 1, stHalfedgeIndexes, stHalfedgeIndexes, std::plus<int>());

    arrayAddConstant(vertexIndexes, *MCGAL::contextPool.vindex, splitable_count + 1);
    arrayAddConstant(stHalfedgeIndexes, *MCGAL::contextPool.hindex, splitable_count + 1);
    arrayAddConstant(stFacetIndexes, *MCGAL::contextPool.findex, splitable_count + 1);
    *MCGAL::contextPool.vindex = vertexIndexes[splitable_count];
    *MCGAL::contextPool.hindex = stHalfedgeIndexes[splitable_count];
    *MCGAL::contextPool.findex = stFacetIndexes[splitable_count];
    preAllocInit(vertexIndexes, faceIndexes, stFacetIndexes, stHalfedgeIndexes, splitable_count);

    createCenterVertex(vertexIndexes, faceIndexes, stHalfedgeIndexes, stFacetIndexes, splitable_count);
    delete faceIndexes;
    delete vertexIndexes;
    delete stHalfedgeIndexes;
    delete stFacetIndexes;
    delete sourceArray;
    delete conditionArray;
    delete prefixSum;
}

void DeCompressTool::removedInsertedEdges() {
    int size = *MCGAL::contextPool.hindex;
    int inserted_edgecount = 0;
    for (int i = 0; i < insertedCounts.size(); i++) {
        inserted_edgecount += insertedCounts[i];
    }
    int* edgeIndexes = new int[inserted_edgecount];
    int* sourceArray = new int[size];
    int* conditionArray = new int[size];
    int* prefixSum = new int[size];
    memset(prefixSum, 0, sizeof(int) * size);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        sourceArray[i] = i;
        if (MCGAL::contextPool.getHalfedgeByIndex(i)->isAdded() &&
            MCGAL::contextPool.getHalfedgeByIndex(i)->meshId != -1) {
            conditionArray[i] = 1;
        } else {
            conditionArray[i] = 0;
        }
    }
    omp_scan(size, conditionArray, prefixSum, std::plus<int>());
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        if (conditionArray[i]) {
            edgeIndexes[prefixSum[i] - 1] = sourceArray[i];
        }
    }
// 转为facet id
#pragma omp parallel for
    for (int i = 0; i < inserted_edgecount; ++i) {
        edgeIndexes[i] = MCGAL::contextPool.hpool[edgeIndexes[i]].face->poolId;
    }
    quickSort_parallel(edgeIndexes, inserted_edgecount, 64);

    prefixSum[0] = 1;
// 计算前缀和
#pragma omp parallel for
    for (int i = 1; i < inserted_edgecount; ++i) {
        if (edgeIndexes[i] != edgeIndexes[i - 1]) {
            prefixSum[i] = 1;
        } else {
            prefixSum[i] = 0;
        }
    }

    omp_scan(inserted_edgecount, prefixSum, prefixSum, std::plus<int>());
// 根据前缀和构建去重后的数组
#pragma omp parallel for
    for (int i = 0; i < inserted_edgecount; ++i) {
        if (prefixSum[i] == 1) {
            edgeIndexes[prefixSum[i] - 1] = edgeIndexes[i];
        }
    }

    joinFacet(edgeIndexes, inserted_edgecount);

    delete edgeIndexes;
    delete sourceArray;
    delete conditionArray;
    delete prefixSum;
}