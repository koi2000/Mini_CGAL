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
        } else {
            fit->resetStateOnCuda();
            for (int i = 0; i < fit->halfedge_size; i++) {
                fit->getHalfedgeByIndexOnCuda(hpool, i)->resetStateOnCuda();
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
#pragma omp parallel for num_threads(batch_size)
    for (int i = 0; i < batch_size; i++) {
        RemovedVerticesDecodingStep(i);
    }

    logt("%d RemovedVerticesDecodingStep", start, i_curDecimationId);
    // 3. decoding the inserted edge and marking the ones added
#pragma omp parallel for num_threads(batch_size)
    for (int i = 0; i < batch_size; i++) {
        InsertedEdgeDecodingStep(i);
    }
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
        h->opposite()->setProcessed();

        // Test if there is a symbol for this edge.
        // There is no symbol if the two faces of an edge are unsplitable.
        if (h->facet()->isSplittable() || h->opposite()->facet()->isSplittable()) {
            // Decode the edge symbol.
            unsigned sym = readChar(&stOffsets[meshId]);
            // Determine if the edge is original or not.
            // Mark the edge to be removed.
            if (sym != 0) {
                h->setAdded();
                inserted_edgecount++;
            }
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
    // int* faceIndexes = new int[splitable_count];
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
            for (int i = 0; i < fit->halfedge_size; i++) {
                MCGAL::Halfedge* h = fit->getHalfedgeByIndex(i);
                h->end_vertex()->addHalfedge(hindex + i * 2);
                vnew->addHalfedge(hindex + i * 2 + 1);
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
    // cudaFree(dfaceIndexes);
    // cudaFree(dvertexIndexes);
    // cudaFree(dstHalfedgeIndexes);
    // cudaFree(dstFacetIndexes);
    // delete faceIndexes;
}

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
