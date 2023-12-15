// #include "../MCGAL/Core_CUDA/global.cuh"
#include "himesh.cuh"
#include "util.h"
#include <map>
#include <nvToolsExt.h>
#include <omp.h>

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
    std::vector<int> twos;
    // 1. reset the states. note that the states of the vertices need not to be reset
    for (auto fit = faces.begin(); fit != faces.end();) {
        // if ((*fit)->count >= 2) {
        //     twos.push_back((*fit)->poolId);
        //     printf("%d", (*fit)->poolId);
        // }
        // fit++;
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
    inserted_edgecount = 0;
    i_curDecimationId++;  // increment the current decimation operation id.
    // 2. decoding the removed vertices and add to target facets
    struct timeval start = get_cur_time();
    RemovedVerticesDecodingStep();
    logt("%d RemovedVerticesDecodingStep", start, i_curDecimationId);
    // 3. decoding the inserted edge and marking the ones added
    InsertedEdgeDecodingStep();
    logt("%d InsertedEdgeDecodingStep", start, i_curDecimationId);
    // 4. truly insert the removed vertices
    insertRemovedVertices();
    // insertRemovedVerticesOnCuda();
    logt("%d insertRemovedVertices", start, i_curDecimationId);
    // 5. truly remove the added edges
    removeInsertedEdgesOnCuda();
    // removeInsertedEdges();
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

#ifndef UNIFIED
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
    // log("size is %d %d %d", vsize, hsize, fsize);
    CHECK(cudaMemcpy(MCGAL::contextPool.dvpool, MCGAL::contextPool.vpool, vsize * sizeof(MCGAL::Vertex),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MCGAL::contextPool.dhpool, MCGAL::contextPool.hpool, hsize * sizeof(MCGAL::Halfedge),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MCGAL::contextPool.dfpool, MCGAL::contextPool.fpool, fsize * sizeof(MCGAL::Facet),
                     cudaMemcpyHostToDevice));

    logt("%d cuda memory copy", start, i_curDecimationId);

#    ifdef TEST
    if (i_curDecimationId == 2) {
        grid.x = 19;
#        ifdef GRID_SIZE
        grid.x = GRID_SIZE;
#        endif
        block.x = 512;
#        ifdef BLOCK_SIZE
        block.x = BLOCK_SIZE;
#        endif
        block.y = 1;
    }
#    endif
    createCenterVertexOnCuda<<<grid, block>>>(MCGAL::contextPool.dvpool, MCGAL::contextPool.dhpool,
                                              MCGAL::contextPool.dfpool, dvertexIndexes, dfaceIndexes,
                                              dstHalfedgeIndexes, dstFacetIndexes, num, clockRate, i_curDecimationId);
    cudaDeviceSynchronize();
    double t = logt("%d kernel function", start, i_curDecimationId);
#    ifdef TEST
    if (i_curDecimationId == 2) {
#        if defined(GRID_SIZE) && defined(BLOCK_SIZE)
        printf("%d %d %lf \n", GRID_SIZE, BLOCK_SIZE, t);
#        endif
        exit(0);
    }
#    endif
    cudaError_t error = cudaGetLastError();
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
#else

void MyMesh::insertRemovedVerticesOnCuda() {
    cudaSetDevice(0);
    struct timeval start = get_cur_time();
    std::vector<int> faceIndexes(splitable_count);
    std::vector<int> vertexIndexes(splitable_count);
    std::vector<int> stHalfedgeIndexes(splitable_count);
    std::vector<int> stFacetIndexes(splitable_count);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double clockRate = prop.clockRate;
    int index = 0;
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
    dim3 block(512, 1, 1);
    dim3 grid((num + block.x - 1) / block.x, 1, 1);

    CHECK(cudaMemcpy(dfaceIndexes, faceIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dvertexIndexes, vertexIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstHalfedgeIndexes, stHalfedgeIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstFacetIndexes, stFacetIndexes.data(), num * sizeof(int), cudaMemcpyHostToDevice));

    logt("%d cuda memory copy", start, i_curDecimationId);
    if (i_curDecimationId == 2) {
        grid.x = 19;
#    ifdef GRID_SIZE
        grid.x = GRID_SIZE;
#    endif
        block.x = 512;
#    ifdef BLOCK_SIZE
        block.x = BLOCK_SIZE;
#    endif
        block.y = 1;
    }
    createCenterVertexOnCuda<<<grid, block>>>(MCGAL::contextPool.vpool, MCGAL::contextPool.hpool,
                                              MCGAL::contextPool.fpool, dvertexIndexes, dfaceIndexes,
                                              dstHalfedgeIndexes, dstFacetIndexes, num, clockRate, i_curDecimationId);
    cudaDeviceSynchronize();
    double t = logt("%d kernel function", start, i_curDecimationId);
    if (i_curDecimationId == 2) {
#    if defined(GRID_SIZE) && defined(BLOCK_SIZE)
        printf("%d %d %lf \n", GRID_SIZE, BLOCK_SIZE, t);
#    endif
        exit(0);
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
#endif

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

__global__ void resetHalfedgeOnCuda(MCGAL::Vertex* vpool,
                                    MCGAL::Halfedge* hpool,
                                    MCGAL::Facet* fpool,
                                    int* edgeIndexes,
                                    int* edgeIndexesCnt,
                                    int num,
                                    double clockRate) {
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < num) {
        MCGAL::Halfedge* hprev = &hpool[edgeIndexes[tid]];
        hprev->dfacet(fpool)->resetOnCuda(vpool, hpool, hprev);
    }
}

// __global__ void joinFacetOnCuda(MCGAL::Vertex* vpool,
//                                 MCGAL::Halfedge* hpool,
//                                 MCGAL::Facet* fpool,
//                                 int* facetIndexes,
//                                 int* stIndexes,
//                                 int* thNumberes,
//                                 int num,
//                                 double clockRate) {
//     int tid = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
//     if (tid < num) {
//         int stIndex = stIndexes[tid];
//         int thNumber = thNumberes[tid];
//         int idx = 0;
//         int edge[100];
//         for (int i = 0; i < thNumber; i++) {
//             MCGAL::Facet* fit = &fpool[facetIndexes[stIndex + i]];
//             for (int j = 0; j < fit->halfedge_size; j++) {
//                 if (hpool[fit->halfedges[j]].isAddedOnCuda()) {
//                     edge[idx++] = fit->halfedges[j];
//                 }
//             }
//         }
//         // printf("%d\t", idx);
//         for (int i = 0; i < idx; i++) {
//             MCGAL::Halfedge* h = &hpool[edge[i]];
//             MCGAL::Facet* lockFacet = h->dfacet(fpool);
//             // MCGAL::Facet* lockOppoFacet = h->dopposite(hpool)->dfacet(fpool);
//             MCGAL::Halfedge* hprev = find_prevOncuda(hpool, h);
//             MCGAL::Halfedge* gprev = find_prevOncuda(hpool, h->dopposite(hpool));
//             remove_tipOnCuda(hpool, hprev);
//             remove_tipOnCuda(hpool, gprev);
//             // int hnext = h->dopposite(hpool)->next_;
//             // int gnext = h->next_;
//             // hprev->next_ = hnext;
//             // gprev->next_ = gnext;
//             // gprev->dfacet(fpool)->setRemovedOnCuda();
//             lockFacet->resetOnCuda(vpool, hpool, hprev);
//             if (lockFacet->halfedge_size >= 50) {
//                 printf("%d\t", thNumber);
//             }
//         }
//     }
// }

/**
 * Remove all the marked edges on cuda
 */
/**
 * 以面为单位进行bfs，将所有联通的面作为一个block进行处理
 */
// void MyMesh::removeInsertedEdgesOnCuda() {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     double clockRate = prop.clockRate;
//     struct timeval start = get_cur_time();
//     std::vector<int> facetIndexes;
//     std::vector<int> stIndexes;
//     std::vector<int> thNumbers;
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
//             ids.push_back(fit->poolId);
//             for (int j = 0; j < fit->halfedge_size; j++) {
//                 MCGAL::Halfedge* hit = fit->getHalfedgeByIndex(j);
//                 if (hit->isAdded() && !hit->isVisited()) {
//                     // ids.push_back(hit->poolId);
//                     hit->setVisited();
//                     hit->opposite()->setRemoved();
//                     MCGAL::Facet* fit2 = hit->opposite()->facet();
//                     fit2->setRemoved();
//                     hit->vertex()->eraseHalfedgeByPointer(hit);
//                     hit->opposite()->vertex()->eraseHalfedgeByPointer(hit->opposite());
//                     // 入队
//                     fqueue.push(fit2);
//                 }
//             }
//         }
//         if (!ids.empty()) {
//             stIndexes.push_back(facetIndexes.size());
//             for (int j = 0; j < ids.size(); j++) {
//                 facetIndexes.push_back(ids[j]);
//             }
//             thNumbers.push_back(ids.size());
//         }
//     }
//     int* dfacetIndexes;
//     int* dstIndexes;
//     int* dthNumbers;
//     CHECK(cudaMalloc(&dfacetIndexes, facetIndexes.size() * sizeof(int)));
//     CHECK(cudaMalloc(&dstIndexes, stIndexes.size() * sizeof(int)));
//     CHECK(cudaMalloc(&dthNumbers, thNumbers.size() * sizeof(int)));
//     CHECK(cudaMemcpy(dfacetIndexes, facetIndexes.data(), facetIndexes.size() * sizeof(int), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(dstIndexes, stIndexes.data(), stIndexes.size() * sizeof(int), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(dthNumbers, thNumbers.data(), thNumbers.size() * sizeof(int), cudaMemcpyHostToDevice));
//     int vsize = MCGAL::contextPool.vindex;
//     int hsize = MCGAL::contextPool.hindex;
//     int fsize = MCGAL::contextPool.findex;
//     int num = stIndexes.size();
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
//                                      dfacetIndexes, dstIndexes, dthNumbers, num, clockRate);
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
//     cudaFree(dfacetIndexes);
//     cudaFree(dstIndexes);
//     cudaFree(dthNumbers);
// }

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
            // join_face(h);
            MCGAL::Halfedge* hprev = find_prevOncuda(hpool, h);
            MCGAL::Halfedge* gprev = find_prevOncuda(hpool, h->dopposite(hpool));
            remove_tipOnCuda(hpool, hprev);
            remove_tipOnCuda(hpool, gprev);
            // h->dopposite(hpool)->setRemoved();

            gprev->dfacet(fpool)->setRemovedOnCuda();
            hprev->dfacet(fpool)->resetOnCuda(vpool, hpool, hprev);
        }
    }
}

void MyMesh::removeInsertedEdgesOnCuda() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double clockRate = prop.clockRate;
    struct timeval start = get_cur_time();
    // inserted_edgecount数量应该会比facet大一点
    // 记录三个数组，第一个是所有待处理的边id，类似一个pool
    // 第二个每个thread的起始index
    // 第三个每个thread需要处理的数量
    // std::vector<int> edgeIndex(inserted_edgecount);
    std::vector<int> edgeIndexes;
    std::vector<int> stIndexes;
    std::vector<int> thNumberes;
    for (int i = 0; i < faces.size(); i++) {
        MCGAL::Facet* node = faces[i];
        if (node->isVisited()) {
            continue;
        }
        // 记录这一轮bfs所有可用的面
        std::vector<int> ids;
        std::queue<MCGAL::Facet*> fqueue;
        fqueue.push(node);
        while (!fqueue.empty()) {
            MCGAL::Facet* fit = fqueue.front();
            fqueue.pop();
            if (fit->isVisited()) {
                continue;
            }
            fit->setVisitedFlag();
            for (int j = 0; j < fit->halfedge_size; j++) {
                MCGAL::Halfedge* hit = fit->getHalfedgeByIndex(j);
                MCGAL::Facet* fit2 = hit->opposite()->facet();
                if (hit->isAdded() && !hit->isVisited()) {
                    // MCGAL::Facet* fit2 = hit->opposite()->facet();
                    // edgeIndex[idx++] = hit->poolId;
                    ids.push_back(hit->poolId);
                    hit->setVisited();
                    hit->opposite()->setRemoved();
                    // fit2->setRemoved();
                    hit->vertex()->eraseHalfedgeByPointer(hit);
                    hit->opposite()->vertex()->eraseHalfedgeByPointer(hit->opposite());
                    // fqueue.push(fit2);
                    // 入队
                }
                fqueue.push(fit2);
            }
        }
        if (!ids.empty()) {
            stIndexes.push_back(edgeIndexes.size());
            for (int j = 0; j < ids.size(); j++) {
                edgeIndexes.push_back(ids[j]);
            }
            thNumberes.push_back(ids.size());
        }
    }
    int stIndex = stIndexes[0];
    int thNumber = thNumberes[0];
    auto s = 0LL;
    // #pragma omp parallel for reduction(+ : s)
    //     for (auto i = 1; i <= 2; i++) {
    //         s += i;
    //     }
    logt("%d join facet kernel", start, i_curDecimationId);
#pragma omp parallel for num_threads(60)
    for (int i = 0; i < thNumber; i++) {
        MCGAL::Halfedge* h = &MCGAL::contextPool.hpool[edgeIndexes[stIndex + i]];
        // join_face(h);
        MCGAL::Halfedge* hprev = find_prev(h);
        MCGAL::Halfedge* gprev = find_prev(h->opposite());
        int hnext = h->opposite()->next_;
        int gnext = h->next_;
        hprev->next_ = hnext;
        gprev->next_ = gnext;
        // remove_tip(hprev);
        // remove_tip(gprev);
        h->opposite()->setRemoved();
        gprev->facet()->setRemoved();
        hprev->facet()->reset(hprev);
    }
    logt("%d parallel", start, i_curDecimationId);
    return;
}

// Halfedge* Mesh::join_face(Halfedge* h) {
//     Halfedge* hprev = find_prev(h);
//     Halfedge* gprev = find_prev(h->opposite());
//     remove_tip(hprev);
//     remove_tip(gprev);
//     h->opposite()->setRemoved();
//     h->vertex()->eraseHalfedgeByPointer(h);
//     h->opposite()->vertex()->eraseHalfedgeByPointer(h->opposite());
//     gprev->facet()->setRemoved();
//     hprev->facet()->reset(hprev);
//     return hprev;
// }

/**
 * Remove all the marked edges on cuda
 */
// void MyMesh::removeInsertedEdgesOnCuda() {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     double clockRate = prop.clockRate;
//     struct timeval start = get_cur_time();
//     // inserted_edgecount数量应该会比facet大一点
//     // 记录三个数组，第一个是所有待处理的边id，类似一个pool
//     // 第二个每个thread的起始index
//     // 第三个每个thread需要处理的数量
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
//                     // MCGAL::Facet* fit2 = hit->opposite()->facet();
//                     // edgeIndex[idx++] = hit->poolId;
//                     ids.push_back(hit->poolId);
//                     hit->setVisited();
//                     hit->opposite()->setRemoved();
//                     // fit2->setRemoved();
//                     hit->vertex()->eraseHalfedgeByPointer(hit);
//                     hit->opposite()->vertex()->eraseHalfedgeByPointer(hit->opposite());
//                     // fqueue.push(fit2);
//                 }
//                 for (int i = 0; i < fit2->halfedge_size; i++) {
//                     MCGAL::Halfedge* hit = fit2->getHalfedgeByIndex(i);
//                     if (hit->isAdded()) {
//                         fqueue.push(fit2);
//                         break;
//                     }
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
//     dim3 block(32, 1, 1);
//     dim3 grid((num + block.x - 1) / block.x, 1, 1);
//     logt("%d cuda memcpy copy", start, i_curDecimationId);
//     printf("grid %d block %d\n", grid.x, block.x);
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