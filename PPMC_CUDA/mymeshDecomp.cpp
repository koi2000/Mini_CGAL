#include "../MCGAL/Core_CUDA/global.h"
#include "mymesh.h"
#include "util.h"

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
            for (MCGAL::Halfedge* hit : (*fit)->halfedges) {
                hit->resetState();
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
    insertRemovedVertices();
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

void MyMesh::RemovedVerticesDecodingStep() {
    //
    pushHehInit();
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
        h->opposite->setProcessed();

        // Test if there is a symbol for this edge.
        // There is no symbol if the two faces of an edge are unsplitable.
        if (h->face->isSplittable() || h->opposite->face->isSplittable()) {
            // Decode the edge symbol.
            unsigned sym = readChar();
            // Determine if the edge is original or not.
            // Mark the edge to be removed.
            if (sym != 0)
                h->setAdded();
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
}

/**
 * Insert center vertices.
 */
void MyMesh::insertRemovedVertices() {
    // Add the first halfedge to the queue.
    pushHehInit();
    // 需要交到cuda上的信息有，原来面的数据
    // pool全部要交上去
    // prealloc的信息

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