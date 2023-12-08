// #include "../MCGAL/Core_CUDA/global.cuh"
#include "mymesh.cuh"
#include "util.h"
#include <nvToolsExt.h>

void MyMesh::decode(int lod) {
    assert(lod >= 0 && lod <= 100);
    // assert(!this->is_compression_mode());
    if (lod < i_decompPercentage) {
        return;
    }
    i_decompPercentage = lod;
    b_jobCompleted = false;
    while (!b_jobCompleted) {
        startNextDecompresssionOp();
    }
}

void MyMesh::startNextDecompresssionOp() {
    // check if the target LOD is reached
    if (i_curDecimationId * 100.0 / i_nbDecimations >= i_decompPercentage) {
        if (i_curDecimationId == i_nbDecimations) {}
        b_jobCompleted = true;
        return;
    }
    // 1. reset the states. note that the states of the vertices need not to be reset
    for (auto fit = faces.begin(); fit != faces.end();) {
        if ((*fit)->isRemoved()) {
            fit = faces.erase(fit);
        } else {
            (*fit)->resetState();
            // for (MCGAL::Halfedge* hit : (*fit)->halfedges) {
            //     hit->resetState();
            // }
            for (int i = 0; i < (*fit)->halfedge_size; i++) {
                (*fit)->getHalfedgeByIndex(i)->resetState();
            }
            fit++;
        }
    }
    splitable_count = 0;
    i_curDecimationId++;  // increment the current decimation operation id.
    // 2. decoding the removed vertices and add to target facets
    struct timeval start = get_cur_time();
    RemovedVerticesDecodingStep();
    logt("%d RemovedVerticesDecodingStep", start, i_curDecimationId);
    // 3. decoding the inserted edge and marking the ones added
    InsertedEdgeDecodingStep();
    logt("%d InsertedEdgeDecodingStep", start, i_curDecimationId);
    // 4. truly insert the removed vertices
    // insertRemovedVertices();
    insertRemovedVerticesOnCuda();
    logt("%d insertRemovedVertices", start, i_curDecimationId);
    // 5. truly remove the added edges
    removeInsertedEdges();
    logt("%d removeInsertedEdges", start, i_curDecimationId);
}

void MyMesh::readBaseMesh() {
    // read the number of level of detail
    i_nbDecimations = readuInt16();
    // set the mesh bounding box
    unsigned i_nbVerticesBaseMesh = readInt();
    unsigned i_nbFacesBaseMesh = readInt();

    std::deque<MCGAL::Point>* p_pointDeque = new std::deque<MCGAL::Point>();
    std::deque<uint32_t*>* p_faceDeque = new std::deque<uint32_t*>();
    // Read the vertex positions.
    for (unsigned i = 0; i < i_nbVerticesBaseMesh; ++i) {
        MCGAL::Point pos = readPoint();
        p_pointDeque->push_back(pos);
    }
    // read the face vertex indices
    // Read the face vertex indices.
    for (unsigned i = 0; i < i_nbFacesBaseMesh; ++i) {
        int nv = readInt();
        uint32_t* f = new uint32_t[nv + 1];
        // Write in the first cell of the array the face degree.
        f[0] = nv;
        for (unsigned j = 1; j < nv + 1; ++j) {
            f[j] = readInt();
        }
        p_faceDeque->push_back(f);
    }
    // Let the builder do its job.
    buildFromBuffer(p_pointDeque, p_faceDeque);

    // Free the memory.
    for (unsigned i = 0; i < p_faceDeque->size(); ++i) {
        delete[] p_faceDeque->at(i);
    }
    delete p_faceDeque;
    delete p_pointDeque;
}

void MyMesh::buildFromBuffer(std::deque<MCGAL::Point>* p_pointDeque, std::deque<uint32_t*>* p_faceDeque) {
    this->vertices.clear();
    // this->halfedges.clear();
    // used to create faces
    std::vector<MCGAL::Vertex*> vertices;
    // add vertex to Mesh
    for (std::size_t i = 0; i < p_pointDeque->size(); ++i) {
        MCGAL::Point p = p_pointDeque->at(i);
        MCGAL::Vertex* vt = MCGAL::contextPool.allocateVertexFromPool(p);
        vt->setId(i);
        this->vertices.push_back(vt);
        vertices.push_back(vt);
    }
    this->vh_departureConquest[0] = vertices[0];
    this->vh_departureConquest[1] = vertices[1];
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
        this->add_face(face);
        // this->faces
    }
    // clear vector
    vertices.clear();
}

void MyMesh::RemovedVerticesDecodingStep() {
    pushHehInit();
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
        unsigned sym = readChar();
        if (sym == 1) {
            MCGAL::Point rmved = readPoint();
            f->setSplittable();
            splitable_count++;
            f->setRemovedVertexPos(rmved);
        } else {
            f->setUnsplittable();
        }
    }
}

/**
 * One step of the inserted edge coding conquest.
 */
void MyMesh::InsertedEdgeDecodingStep() {
    pushHehInit();
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
            unsigned sym = readChar();
            // Determine if the edge is original or not.
            // Mark the edge to be removed.
            if (sym != 0)
                h->setAdded();
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
        // nvtxRangePush("Your Kernel");
        unsigned int startTime = clock64();

        int faceId = faceIndexes[tid];
        MCGAL::Facet* facet = &fpool[faceId];
        int vertexId = vertexIndexes[tid];
        MCGAL::Vertex* vnew = &vpool[vertexId];
        int stHalfedgeIndex = stHalfedgeIndexes[tid];
        int stFacetIndex = stFacetIndexes[tid];
        // nvtxRangePop();
        // if (tid == 300) {
        //     unsigned int endTime = clock64();
        //     float elapsedTime = (endTime - startTime) / clockRate;
        //     printf("get from pool execution time: %.6f ms\n", elapsedTime);
        // }

        MCGAL::Halfedge* h = facet->getHalfedgeByIndexOnCuda(hpool, 0);
        MCGAL::Halfedge* hnew = &hpool[stHalfedgeIndex++];
        hnew->resetOnCuda(h->dend_vertex(vpool), vnew);

        MCGAL::Halfedge* oppo_new = &hpool[stHalfedgeIndex++];
        oppo_new->resetOnCuda(vnew, h->dend_vertex(vpool));
        hnew->setOppositeOnCuda(oppo_new);
        oppo_new->setOppositeOnCuda(hnew);
        insert_tip_cuda(hpool, hnew->dopposite(hpool), h);
        // if (tid == 300) {
        //     unsigned int endTime = clock64();
        //     float elapsedTime = (endTime - startTime) / clockRate;
        //     printf("first execution time: %.6f ms\n", elapsedTime);
        // }

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
        // if (tid == 300) {
        //     unsigned int endTime = clock64();
        //     float elapsedTime = (endTime - startTime) / clockRate;
        //     printf("main loop execution time: %.6f ms\n", elapsedTime);
        // }

        hed->setNextOnCuda(hnew->dopposite(hpool));
        // #pragma unroll 2
        for (int i = 1; i < h->dfacet(fpool)->halfedge_size; i += 1) {
            MCGAL::Halfedge* hit = &hpool[h->dfacet(fpool)->halfedges[i]];
            fpool[stFacetIndex++].resetOnCuda(vpool, hpool, hit);
        }
        // if (tid == 300) {
        //     unsigned int endTime = clock64();
        //     float elapsedTime = (endTime - startTime) / clockRate;
        //     printf("facet reset execution time: %.6f ms\n", elapsedTime);
        // }

        h->dfacet(fpool)->resetOnCuda(vpool, hpool, h);
    }
}

__global__ void createCenterVertexOnCuda10(MCGAL::Vertex* vpool,
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
        int hnewId = stHalfedgeIndex++;
        int oppoId = stHalfedgeIndex++;
        MCGAL::Halfedge* hnew = &hpool[hnewId];
        // hnew->resetOnCuda(h->dend_vertex(vpool), vnew);
        hnew->vertex_ = h->end_vertex_;
        hnew->end_vertex_ = vertexId;

        MCGAL::Halfedge* oppo_new = &hpool[oppoId];
        // oppo_new->resetOnCuda(vnew, h->dend_vertex(vpool));
        oppo_new->end_vertex_ = h->end_vertex_;
        oppo_new->vertex_ = vertexId;

        hnew->setOppositeOnCuda(oppo_new);
        hnew->opposite_ = oppoId;
        oppo_new->opposite_ = hnewId;
        // oppo_new->setOppositeOnCuda(hnew);
        // insert_tip_cuda(hpool, hnew->dopposite(hpool), h);
        oppo_new->next_ = h->next_;
        h->next_ = oppoId;

        MCGAL::Halfedge* g = hnew->dopposite(hpool)->dnext(hpool);
        MCGAL::Halfedge* hed = hnew;
        while (g->dnext(hpool)->poolId != hed->poolId) {
            int gId = stHalfedgeIndex++;
            int oppogId = stHalfedgeIndex++;
            MCGAL::Halfedge* gnew = &hpool[gId];
            gnew->vertex_ = g->end_vertex_;
            gnew->end_vertex_ = vertexId;

            MCGAL::Halfedge* oppo_gnew = &hpool[oppogId];
            gnew->end_vertex_ = g->end_vertex_;
            gnew->vertex_ = vertexId;

            gnew->opposite_ = oppogId;
            oppo_gnew->opposite_ = gId;
            gnew->next_ = hnew->opposite_;
            // gnew->setNextOnCuda(hnew->dopposite(hpool));
            // insert_tip_cuda(hpool, gnew->dopposite(hpool), g);
            oppo_gnew->next_ = g->next_;
            g->next_ = gId;
            g = oppo_gnew->dnext(hpool);
            hnew = gnew;
        }

        hed->next_ = hnew->opposite_;
        if (tid == 300) {
            unsigned int endTime = clock64();
            float elapsedTime = (endTime - startTime) / clockRate;
            printf("main loop execution time: %.6f ms\n", elapsedTime);
        }
        // hed->setNextOnCuda(hnew->dopposite(hpool));
        // // #pragma unroll 2
        for (int i = 1; i < h->dfacet(fpool)->halfedge_size; i += 1) {
            MCGAL::Halfedge* hit = &hpool[h->dfacet(fpool)->halfedges[i]];
            fpool[stFacetIndex++].resetOnCuda(vpool, hpool, hit);
        }
        if (tid == 300) {
            unsigned int endTime = clock64();
            float elapsedTime = (endTime - startTime) / clockRate;
            printf("facet reset execution time: %.6f ms\n", elapsedTime);
        }

        h->dfacet(fpool)->resetOnCuda(vpool, hpool, h);
    }
}

void MyMesh::insertRemovedVerticesOnCuda() {
    struct timeval start = get_cur_time();
    std::vector<int> faceIndexes(splitable_count);
    std::vector<int> vertexIndexes(splitable_count);
    std::vector<int> stHalfedgeIndexes(splitable_count);
    std::vector<int> stFacetIndexes(splitable_count);
    int index = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double clockRate = prop.clockRate;
    for (int i = 0; i < faces.size(); i++) {
        MCGAL::Facet* fit = faces[i];
        if (fit->isSplittable()) {
            faceIndexes[index] = fit->poolId;
            int hcount = fit->halfedge_size * 2;
            int fcount = fit->halfedge_size - 1;
            int findex = MCGAL::contextPool.preAllocFace(fcount);
            for (int i = 0; i < fcount; i++) {
                this->faces.push_back(MCGAL::contextPool.getFacetByIndex(findex + i));
            }
            stFacetIndexes[index] = findex;

            int hindex = MCGAL::contextPool.preAllocHalfedge(hcount);
            stHalfedgeIndexes[index] = hindex;
            vertexIndexes[index] = (MCGAL::contextPool.getVindex());
            MCGAL::Vertex* vnew = MCGAL::contextPool.allocateVertexFromPool(fit->getRemovedVertexPos());
            this->vertices.push_back(vnew);
            index++;
            for (int i = 0; i < fit->halfedge_size; i++) {
                MCGAL::Halfedge* h = fit->getHalfedgeByIndex(i);
                h->end_vertex()->addHalfedge(hindex + i * 2);
                vnew->addHalfedge(hindex + i * 2 + 1);
            }
        }
    }
    logt("%d collect face information", start, i_curDecimationId);
    // add it to mesh
    int num = faceIndexes.size();
    dim3 block(256, 1, 1);
    dim3 grid((num + block.x - 1) / block.x, 1, 1);

    CHECK(cudaMemcpy(dfaceIndexes, faceIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dvertexIndexes, vertexIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstHalfedgeIndexes, stHalfedgeIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstFacetIndexes, stFacetIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
    int vsize = MCGAL::contextPool.vindex;
    int hsize = MCGAL::contextPool.hindex;
    int fsize = MCGAL::contextPool.findex;
    log("size is %d %d %d", vsize, hsize, fsize);
    CHECK(cudaMemcpyAsync(MCGAL::contextPool.dvpool, MCGAL::contextPool.vpool, vsize * sizeof(MCGAL::Vertex),
                          cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(MCGAL::contextPool.dhpool, MCGAL::contextPool.hpool, hsize * sizeof(MCGAL::Halfedge),
                          cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(MCGAL::contextPool.dfpool, MCGAL::contextPool.fpool, fsize * sizeof(MCGAL::Facet),
                          cudaMemcpyHostToDevice));

    logt("%d cuda memory copy", start, i_curDecimationId);

    if (i_curDecimationId == 2) {
        grid.x = 19;
#ifdef GRID_SIZE
        grid.x = GRID_SIZE;
#endif
                     block.x = 512;
#ifdef BLOCK_SIZE
        block.x = BLOCK_SIZE;
#endif
                      block.y = 1;
    }
    // if (i_curDecimationId == 2) {
    //     log("%d %d", grid.x, block.x);
    //     createCenterVertexOnCuda<<<grid, block>>>(
    //         MCGAL::contextPool.vpool, MCGAL::contextPool.hpool, MCGAL::contextPool.fpool, dvertexIndexes, dfaceIndexes,
    //         dstHalfedgeIndexes, dstFacetIndexes, num, clockRate, i_curDecimationId);
    //     cudaDeviceSynchronize();
    //     logt("%d kernel function", start, i_curDecimationId);
    //     exit(0);
    // }
    createCenterVertexOnCuda<<<grid, block>>>(MCGAL::contextPool.dvpool, MCGAL::contextPool.dhpool,
                                              MCGAL::contextPool.dfpool, dvertexIndexes, dfaceIndexes,
                                              dstHalfedgeIndexes, dstFacetIndexes, num, clockRate, i_curDecimationId);
    cudaDeviceSynchronize();
    double t = logt("%d kernel function", start, i_curDecimationId);
    if (i_curDecimationId == 2) {
#if defined(GRID_SIZE) && defined(BLOCK_SIZE)
    printf("%d %d %lf \n", GRID_SIZE, BLOCK_SIZE, t);
#endif
    exit(0);
    }
    CHECK(cudaMemcpyAsync(MCGAL::contextPool.vpool, MCGAL::contextPool.dvpool, vsize * sizeof(MCGAL::Vertex),
                          cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpyAsync(MCGAL::contextPool.hpool, MCGAL::contextPool.dhpool, hsize * sizeof(MCGAL::Halfedge),
                          cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpyAsync(MCGAL::contextPool.fpool, MCGAL::contextPool.dfpool, fsize * sizeof(MCGAL::Facet),
                          cudaMemcpyDeviceToHost));
    logt("%d cuda memory copy back", start, i_curDecimationId);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

// void MyMesh::insertRemovedVerticesOnCuda() {
//     cudaSetDevice(0);  // 选择设备号为0的GPU
//     struct timeval start = get_cur_time();
//     std::vector<int> faceIndexes(splitable_count);
//     std::vector<int> vertexIndexes(splitable_count);
//     std::vector<int> stHalfedgeIndexes(splitable_count);
//     std::vector<int> stFacetIndexes(splitable_count);
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     double clockRate = prop.clockRate;
//     int index = 0;
//     for (int i = 0; i < faces.size(); i++) {
//         MCGAL::Facet* fit = faces[i];
//         if (fit->isSplittable()) {
//             faceIndexes[index] = fit->poolId;
//             int hcount = fit->halfedge_size * 2;
//             int fcount = fit->halfedge_size - 1;
//             int findex = MCGAL::contextPool.preAllocFace(fcount);
//             for (int i = 0; i < fcount; i++) {
//                 this->faces.push_back(MCGAL::contextPool.getFacetByIndex(findex + i));
//             }
//             stFacetIndexes[index] = findex;

//             int hindex = MCGAL::contextPool.preAllocHalfedge(hcount);
//             stHalfedgeIndexes[index] = hindex;
//             vertexIndexes[index] = (MCGAL::contextPool.getVindex());
//             MCGAL::Vertex* vnew = MCGAL::contextPool.allocateVertexFromPool(fit->getRemovedVertexPos());
//             this->vertices.push_back(vnew);
//             index++;
//             for (int i = 0; i < fit->halfedge_size; i++) {
//                 MCGAL::Halfedge* h = fit->getHalfedgeByIndex(i);
//                 h->end_vertex()->addHalfedge(hindex + i * 2);
//                 vnew->addHalfedge(hindex + i * 2 + 1);
//             }
//             // pre add all the halfedge and vertex
//             // hindex++;
//             // for (int i = 1; i < fit->halfedge_size; i++) {
//             //     MCGAL::Halfedge* hit = fit->getHalfedgeByIndex(i);
//             //     int fc = (findex + i - 1);
//             //     MCGAL::Facet* f = MCGAL::contextPool.getFacetByIndex(fc);
//             //     f->halfedge_size = 0;
//             //     f->vertex_size = 0;
//             //     f->addHalfedge(hit);
//             //     hit->setFacet(f);
//             //     MCGAL::contextPool.getHalfedgeByIndex(hindex)->setFacet(f);
//             //     f->addHalfedge(hindex++);
//             //     MCGAL::contextPool.getHalfedgeByIndex(hindex)->setFacet(f);
//             //     f->addHalfedge(hindex++);

//             //     f->addVertex(hit->end_vertex());
//             //     f->addVertex(hit->vertex());
//             //     f->addVertex(vnew);
//             // }
//         }
//     }
//     logt("%d collect face information", start, i_curDecimationId);
//     // add it to mesh
//     int num = faceIndexes.size();
//     dim3 block(512, 1, 1);
//     dim3 grid((num + block.x - 1) / block.x, 1, 1);

//     CHECK(cudaMemcpy(dfaceIndexes, faceIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(dvertexIndexes, vertexIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(dstHalfedgeIndexes, stHalfedgeIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(dstFacetIndexes, stFacetIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));

//     logt("%d cuda memory copy", start, i_curDecimationId);
//     if (i_curDecimationId == 2) {
//         grid.x = 15;
//         block.x = 512;
//         block.y = 1;
//     }
//     if (i_curDecimationId == 2) {
//         log("%d %d", grid.x, block.x);
//         createCenterVertexOnCuda10<<<grid, block>>>(
//             MCGAL::contextPool.vpool, MCGAL::contextPool.hpool, MCGAL::contextPool.fpool, dvertexIndexes,
//             dfaceIndexes, dstHalfedgeIndexes, dstFacetIndexes, num, clockRate, i_curDecimationId);
//     } else {
//         createCenterVertexOnCuda<<<grid, block>>>(
//             MCGAL::contextPool.vpool, MCGAL::contextPool.hpool, MCGAL::contextPool.fpool, dvertexIndexes,
//             dfaceIndexes, dstHalfedgeIndexes, dstFacetIndexes, num, clockRate, i_curDecimationId);
//     }

//     cudaDeviceSynchronize();
//     logt("%d kernel function", start, i_curDecimationId);
//     if (i_curDecimationId == 10) {
//         exit(0);
//     }
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         printf("ERROR: %s:%d,", __FILE__, __LINE__);
//         printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
//         exit(1);
//     }
// }

/**
 * Insert center vertices.
 */
void MyMesh::insertRemovedVertices() {
    // Add the first halfedge to the queue.
    pushHehInit();
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        MCGAL::Facet* f = h->facet();

        // If the face is already processed, pick the next halfedge:
        if (f->isProcessed())
            continue;

        // Mark the face as processed.
        f->setProcessedFlag();

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h;
        do {
            MCGAL::Halfedge* hOpp = hIt->opposite();
            // TODO: wait
            // assert(!hOpp->is_border());
            if (!hOpp->facet()->isProcessed())
                gateQueue.push(hOpp);
            hIt = hIt->next();
        } while (hIt != h);
        assert(!h->isNew());

        if (f->isSplittable()) {
            // Insert the vertex.
            MCGAL::Halfedge* hehNewVertex = create_center_vertex(h);
            hehNewVertex->vertex()->setPoint(f->getRemovedVertexPos());

            // Mark all the created edges as new.
            MCGAL::Vertex* Hvc = hehNewVertex->vertex();
            for (int i = 0; i < Hvc->halfedges_size; i++) {
                MCGAL::Halfedge* hit = Hvc->getHalfedgeByIndex(i);
                hit->setNew();
                hit->opposite()->setNew();
                hit->facet()->setProcessedFlag();
            }
        }
    }
}

/**
 * Remove all the marked edges
 */
void MyMesh::removeInsertedEdges() {
    pushHehInit();
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
            join_face(hIt);
            hIt->setVisited();
        }
    }
    return;
}