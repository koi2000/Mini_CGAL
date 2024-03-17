#include "himesh.h"
#include "util.h"
void HiMesh::decode(int lod) {
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

void HiMesh::startNextDecompresssionOp() {
    // check if the target LOD is reached
    if (i_curDecimationId * 100.0 / i_nbDecimations >= i_decompPercentage) {
        if (i_curDecimationId == i_nbDecimations) {}
        b_jobCompleted = true;
        return;
    }
    // 1. reset the states. note that the states of the vertices need not to be reset
    // for (MCGAL::Facet* fit : faces) {
    //     fit->resetState();
    //     for (MCGAL::Halfedge* hit : fit->halfedges) {
    //         hit->resetState();
    //     }
    // }
    for (auto fit = faces.begin(); fit != faces.end();) {
        if ((*fit)->isRemoved()) {
            fit = faces.erase(fit);
        } else {
            (*fit)->resetState();
            for (MCGAL::Halfedge* hit : (*fit)->halfedges) {
                hit->resetState();
            }
            fit++;
        }
    }

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
    logt("%d insertRemovedVertices", start, i_curDecimationId);
    // 5. truly remove the added edges
    removeInsertedEdges();
    logt("%d removeInsertedEdges", start, i_curDecimationId);
}

// 每个点都多了一个id，在读一个id
void HiMesh::readBaseMesh() {
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

void HiMesh::buildFromBuffer(std::deque<MCGAL::Point>* p_pointDeque, std::deque<uint32_t*>* p_faceDeque) {
    this->vertices.clear();
    // this->halfedges.clear();
    // used to create faces
    std::vector<MCGAL::Vertex*> vertices;
    // add vertex to Mesh
    for (std::size_t i = 0; i < p_pointDeque->size(); ++i) {
        float x, y, z;
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

bool cmpForder(MCGAL::Facet* f1, MCGAL::Facet* f2) {
    if (f1->groupId == f2->groupId) {
        return f1->forder < f2->forder;
    }
    return f1->groupId < f2->groupId;
}

bool cmpHorder(MCGAL::Halfedge* h1, MCGAL::Halfedge* h2) {
    if (h1->groupId == h2->groupId) {
        return h1->horder < h2->horder;
    }
    return h1->groupId < h2->groupId;
}

void HiMesh::RemovedVerticesDecodingStep() {
    // 从buffer中读出meta信息
    int sampleNumber = readInt();
    sampleNumbers.push_back(sampleNumber);
    std::vector<int> stVertex(sampleNumber);
    for (int i = 0; i < 2 * sampleNumber; i++) {
        stVertex.push_back(readInt());
    }
    stVerteices.push_back(stVertex);
    std::vector<int> facetNumberInGroup;
    for (int i = 0; i < sampleNumber; i++) {
        facetNumberInGroup.push_back(readInt());
    }
    int size = *MCGAL::contextPool.findex;
    // 将起始点放入到队列里，充分利用conquered属性
    int* firstQueue = new int[size];
    int* secondQueue = new int[size];
    int currentQueueSize = sampleNumber;
    int nextQueueSize = 0;
    for (int i = 0; i < sampleNumber; i++) {
        int poolId1 = MCGAL::contextPool.vid2PoolId[stVertex[i * 2]];
        int poolId2 = MCGAL::contextPool.vid2PoolId[stVertex[i * 2 + 1]];
        MCGAL::Vertex* v1 = MCGAL::contextPool.getVertexByIndex(poolId1);
        MCGAL::Vertex* v2 = MCGAL::contextPool.getVertexByIndex(poolId2);
        for (int j = 0; j < v1->halfedges.size(); j++) {
            if (v1->halfedges[j]->end_vertex == v2) {
                v1->halfedges[j]->face->forder = i;
                firstQueue[i] = v1->halfedges[j]->poolId;
                break;
            }
        }
    }
    int level = 0;
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
        for (int i = 0; i < currentQueueSize; i++) {
            int current = currentQueue[i];
            MCGAL::Halfedge* h = MCGAL::contextPool.getHalfedgeByIndex(current);
            MCGAL::Facet* f = h->face;
            if (f->isProcessed()) {
                continue;
            }
            MCGAL::Halfedge* hIt = h;
            unsigned long long idx = 1;
            do {
                MCGAL::Halfedge* hOpp = hIt->opposite;
                unsigned long long order = f->forder << 4 | idx;

                if (!hOpp->face->isConquered()) {
                    hOpp->face->forder = order < hOpp->face->forder ? order : hOpp->face->forder;
                    if (hOpp->face->groupId != -1) {
                        hOpp->face->groupId = min(h->face->groupId, hOpp->face->groupId);
                    }
                    int position = nextQueueSize++;
                    nextQueue[position] = hOpp->poolId;
                }
                hIt = hIt->next;
            } while (hIt != h);
        }
        currentQueueSize = nextQueueSize;
        nextQueueSize = 0;
    }
    // 排好序之后直接开始按顺序读取
    sort(faces.begin(), faces.end(), cmpForder);
    std::vector<int> splittableIndex;
    for (int i = 0; i < faces.size(); i++) {
        char symbol = readCharByOffset(dataOffset + i);
        if (symbol) {
            faces[i]->setSplittable();
            splittableIndex.push_back(i);

        } else {
            faces[i]->setUnsplittable();
        }
    }
    dataOffset += faces.size();
    for (int i = 0; i < splittableIndex.size(); i++) {
        faces[splittableIndex[i]]->setRemovedVertexPos(readPointByOffset(dataOffset + i * 4 * sizeof(int)));
    }
    dataOffset += splittableIndex.size() * 4 * sizeof(int);
}

/**
 * One step of the inserted edge coding conquest.
 */
void HiMesh::InsertedEdgeDecodingStep() {
    int sampleNumber = sampleNumbers[i_curDecimationId];
    std::vector<int> stVertex = stVerteices[i_curDecimationId];
    std::vector<int> halfedgeNumberInGroup;
    for (int i = 0; i < sampleNumber; i++) {
        halfedgeNumberInGroup.push_back(readInt());
    }
    int size = *MCGAL::contextPool.hindex;
    // 将起始点放入到队列里，充分利用conquered属性
    int* firstQueue = new int[size];
    int* secondQueue = new int[size];
    int currentQueueSize = sampleNumber;
    int nextQueueSize = 0;
    for (int i = 0; i < sampleNumber; i++) {
        int poolId1 = MCGAL::contextPool.vid2PoolId[stVertex[i * 2]];
        int poolId2 = MCGAL::contextPool.vid2PoolId[stVertex[i * 2 + 1]];
        MCGAL::Vertex* v1 = MCGAL::contextPool.getVertexByIndex(poolId1);
        MCGAL::Vertex* v2 = MCGAL::contextPool.getVertexByIndex(poolId2);
        for (int j = 0; j < v1->halfedges.size(); j++) {
            if (v1->halfedges[j]->end_vertex == v2) {
                v1->halfedges[j]->face->forder = i;
                firstQueue[i] = v1->halfedges[j]->poolId;
                break;
            }
        }
    }
    int level = 0;
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
        for (int i = 0; i < currentQueueSize; i++) {
            int current = currentQueue[i];
            MCGAL::Halfedge* h = MCGAL::contextPool.getHalfedgeByIndex(current);
            if (h->isProcessed()) {
                continue;
            }
            MCGAL::Halfedge* hIt = h->next;
            unsigned long long idx = 1;
            while (hIt->opposite != h) {
                unsigned long long order = h->horder << 4 | idx;

                if (!hIt->isProcessed()) {
                    hIt->horder = order < hIt->horder ? order : hIt->horder;
                    if (hIt->groupId != -1) {
                        hIt->groupId = min(h->face->groupId, hIt->groupId);
                    }
                    int position = nextQueueSize++;
                    nextQueue[position] = hIt->poolId;
                }
                hIt = hIt->opposite->next;
            };
        }
        currentQueueSize = nextQueueSize;
        nextQueueSize = 0;
    }
    // 排好序之后直接开始按顺序读取
    sort(halfedges.begin(), halfedges.end(), cmpHorder);

    for (int i = 0; i < halfedges.size(); i++) {
        char symbol = readCharByOffset(dataOffset + i);
        if (symbol) {
            halfedges[i]->setAdded();
        }
    }
    dataOffset += halfedges.size();
}

/**
 * Insert center vertices.
 */
void HiMesh::insertRemovedVertices() {
    // Add the first halfedge to the queue.
    pushHehInit();
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        MCGAL::Facet* f = h->face;

        // If the face is already processed, pick the next halfedge:
        if (f->isProcessed())
            continue;

        // Mark the face as processed.
        f->setProcessedFlag();

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h;
        do {
            MCGAL::Halfedge* hOpp = hIt->opposite;
            // TODO: wait
            // assert(!hOpp->is_border());
            if (!hOpp->face->isProcessed())
                gateQueue.push(hOpp);
            hIt = hIt->next;
        } while (hIt != h);
        assert(!h->isNew());

        if (f->isSplittable()) {
            // Insert the vertex.
            MCGAL::Halfedge* hehNewVertex = create_center_vertex(h);
            hehNewVertex->vertex->setPoint(f->getRemovedVertexPos());

            // Mark all the created edges as new.
            MCGAL::Vertex* Hvc = hehNewVertex->vertex;
            for (MCGAL::Halfedge* hit : Hvc->halfedges) {
                hit->setNew();
                hit->opposite->setNew();
                hit->face->setProcessedFlag();
            }
        }
    }
}

/**
 * Remove all the marked edges
 */
void HiMesh::removeInsertedEdges() {
    for (auto fit = faces.begin(); fit != faces.end();) {
        if ((*fit)->isRemoved()) {
            fit = faces.erase(fit);
        } else {
            fit++;
        }
    }
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
            MCGAL::Halfedge* hOpp = hIt->opposite;
            // TODO: wait
            // assert(!hOpp->is_border());
            if (!hOpp->isVisited())
                gateQueue.push(hOpp);
            hIt = hIt->next;
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