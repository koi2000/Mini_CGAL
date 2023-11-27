#include "mymesh.h"
#include "util.h"
void MyMesh::decode(int lod) {
    assert(lod >= 0 && lod <= 100);
    assert(!this->is_compression_mode());
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
    for (MyMesh::Halfedge_iterator hit = halfedges_begin(); hit != halfedges_end(); ++hit)
        hit->resetState();
    for (MyMesh::Face_iterator fit = facets_begin(); fit != facets_end(); ++fit)
        fit->resetState();

    i_curDecimationId++;  // increment the current decimation operation id.
    // 2. decoding the removed vertices and add to target facets
    struct timeval start = hispeed::get_cur_time();
    RemovedVerticesDecodingStep();
    // hispeed::logt("%d RemovedVerticesDecodingStep", start, i_curDecimationId);
    // 3. decoding the inserted edge and marking the ones added
    InsertedEdgeDecodingStep();
    // hispeed::logt("%d InsertedEdgeDecodingStep", start, i_curDecimationId);
    // 4. truly insert the removed vertices
    insertRemovedVertices();
    // hispeed::logt("%d insertRemovedVertices", start, i_curDecimationId);
    // 5. truly remove the added edges
    removeInsertedEdges();
    // hispeed::logt("%d removeInsertedEdges", start, i_curDecimationId);
}

void MyMesh::readBaseMesh() {
    // for (unsigned i = 0; i < 3; i++) {
    //     mbb.low[i] = readFloat();
    // }
    // for (unsigned i = 0; i < 3; i++) {
    //     mbb.high[i] = readFloat();
    // }
    // read the number of level of detail
    i_nbDecimations = readuInt16();
    // set the mesh bounding box
    unsigned i_nbVerticesBaseMesh = readInt();
    unsigned i_nbFacesBaseMesh = readInt();

    std::deque<Point>* p_pointDeque = new std::deque<Point>();
    std::deque<uint32_t*>* p_faceDeque = new std::deque<uint32_t*>();
    // Read the vertex positions.
    for (unsigned i = 0; i < i_nbVerticesBaseMesh; ++i) {
        Point pos = readPoint();
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
    MyMeshBaseBuilder<HalfedgeDS> builder(p_pointDeque, p_faceDeque);
    delegate(builder);

    // Free the memory.
    for (unsigned i = 0; i < p_faceDeque->size(); ++i) {
        delete[] p_faceDeque->at(i);
    }
    delete p_faceDeque;
    delete p_pointDeque;
}

void MyMesh::RemovedVerticesDecodingStep() {
    //
    pushHehInit();
    int count = 0;
    while (!gateQueue.empty()) {
        Halfedge_handle h = gateQueue.front();
        gateQueue.pop();

        Face_handle f = h->facet();

        // If the face is already processed, pick the next halfedge:
        if (f->isConquered())
            continue;
        count++;
        // Add the other halfedges to the queue
        Halfedge_handle hIt = h;
        do {
            Halfedge_handle hOpp = hIt->opposite();
            assert(!hOpp->is_border());
            if (!hOpp->facet()->isConquered())
                gateQueue.push(hOpp);
            hIt = hIt->next();
        } while (hIt != h);

        // Decode the face symbol.
        unsigned sym = readChar();
        if (sym == 1) {
            Point rmved = readPoint();
            f->setSplittable();
            f->setRemovedVertexPos(rmved);
        } else {
            f->setUnsplittable();
        }
    }
    // hispeed::log("%d RemovedVerticesDecodingStep count %d", this->i_curDecimationId, count);
}

/**
 * One step of the inserted edge coding conquest.
 */
void MyMesh::InsertedEdgeDecodingStep() {
    pushHehInit();
    int count = 0;
    while (!gateQueue.empty()) {
        Halfedge_handle h = gateQueue.front();
        gateQueue.pop();

        // Test if the edge has already been conquered.
        if (h->isProcessed())
            continue;
        count++;
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
        Halfedge_handle hIt = h->next();
        while (hIt->opposite() != h) {
            if (!hIt->isProcessed() && !hIt->isNew())
                gateQueue.push(hIt);
            hIt = hIt->opposite()->next();
        }
        assert(!hIt->isNew());
    }
    // hispeed::log("%d InsertedEdgeDecodingStep count %d", this->i_curDecimationId, count);
}

/**
 * Insert center vertices.
 */
void MyMesh::insertRemovedVertices() {
    // Add the first halfedge to the queue.
    pushHehInit();
    int count = 0;
    while (!gateQueue.empty()) {
        Halfedge_handle h = gateQueue.front();
        gateQueue.pop();

        Face_handle f = h->facet();

        // If the face is already processed, pick the next halfedge:
        if (f->isProcessed())
            continue;
        count++;
        // Mark the face as processed.
        f->setProcessedFlag();

        // Add the other halfedges to the queue
        Halfedge_handle hIt = h;
        do {
            Halfedge_handle hOpp = hIt->opposite();
            assert(!hOpp->is_border());
            if (!hOpp->facet()->isProcessed())
                gateQueue.push(hOpp);
            hIt = hIt->next();
        } while (hIt != h);
        assert(!h->isNew());

        if (f->isSplittable()) {
            // Insert the vertex.
            Halfedge_handle hehNewVertex = create_center_vertex(h);
            hehNewVertex->vertex()->point() = f->getRemovedVertexPos();

            // Mark all the created edges as new.
            Halfedge_around_vertex_circulator Hvc = hehNewVertex->vertex_begin();
            Halfedge_around_vertex_circulator Hvc_end = Hvc;
            CGAL_For_all(Hvc, Hvc_end) {
                Hvc->setNew();
                Hvc->opposite()->setNew();
            }
        }
    }
    // hispeed::log("%d insertRemovedVertices count %d", this->i_curDecimationId, count);
}

/**
 * Remove all the marked edges
*/
void MyMesh::removeInsertedEdges(){
    for (MyMesh::Halfedge_iterator hit = halfedges_begin(); hit != halfedges_end(); hit++) {
        if (hit->isAdded()) {
            join_facet(hit);
        }
    }
}