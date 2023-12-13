//
// Created by DELL on 2023/11/9.
//
#include "math.h"
#include "himesh.cuh"
#include "util.h"

void MyMesh::encode(int lod) {
    b_jobCompleted = false;
    while (!b_jobCompleted) {
        startNextCompresssionOp();
    }
}

void MyMesh::startNextCompresssionOp() {
    for (auto it = halfedges.begin(); it != halfedges.end();) {
        if ((*it)->isRemoved()) {
            it = halfedges.erase(it);
        } else {
            it++;
        }
    }
    // 1. reset the stats
    for (MCGAL::Vertex* vit : vertices) {
        vit->resetState();
    }

    // for (MCGAL::Facet* fit : faces) {
    //     fit->resetState();
    //     for (int i = 0; i < fit->halfedge_size; i++) {
    //         fit->getHalfedgeByIndex(i)->resetState();
    //     }
    // }
    int cnt = 0;
    for (auto fit = faces.begin(); fit != faces.end();) {
        if ((*fit)->isRemoved()) {
            fit = faces.erase(fit);
        } else {
            (*fit)->resetState();
            for (int i = 0; i < (*fit)->halfedge_size; i++) {
                (*fit)->getHalfedgeByIndex(i)->resetState();
                cnt++;
            }
            fit++;
        }
    }
    i_nbRemovedVertices = 0;  // Reset the number of removed vertices.
    while (!gateQueue.empty()) {
        gateQueue.pop();
    }

    // 2. do one round of decimation
    // choose a halfedge that can be processed:
    if (i_curDecimationId < 10) {
        size_t i_heInitId = size_of_halfedges() / 2;
        MCGAL::Halfedge* hitInit = halfedges[i_heInitId];
        hitInit->setInQueue();
        // MCGAL::Halfedge* hitInit = *vh_departureConquest[0]->halfedges.begin();
        gateQueue.push(hitInit);
    }
    // bfs all the facet
    /**
     * 本质是对所有的facet进行bfs，对于一个面，如果被处理过就不能再被处理了，如果没有被处理
     * 就根据情况来，如果点可以被删除，就删除，并标记为splittable，如果不能被删，就不删，标记为unsplittable
     * 这里是通过bfs halfedge来达到bfs facet的目的的
     * 按道理来说，一个面被标记为splittable，说明这个面中有多个面被删除，多个边被删除，这些被删的不能再进入bfs
     * 被标记为conqured的点也不能被删除，
     * 大概率问题出在判断是否为 流型结构 上
     */
    int count = 0;
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();
        // TODO: wait
        // assert(!h->is_border());
        MCGAL::Facet* f = h->facet();
        // if (h->isRemoved()) {
        //     h->removeFromQueue();
        //     continue;
        // }
        // if the face is already processed, pick the next halfedge:
        if (f->isConquered()) {
            h->removeFromQueue();
            continue;
        }
        count++;
        // the face is not processed. Count the number of non conquered vertices that can be split
        bool hasRemovable = false;
        MCGAL::Halfedge* unconqueredVertexHE;

        for (MCGAL::Halfedge* hh = h->next(); hh != h; hh = hh->next()) {
            if (isRemovable(hh->end_vertex())) {
                hasRemovable = true;
                unconqueredVertexHE = hh;
                break;
            }
        }

        // if all face vertices are conquered, then the current face is a null patch:
        if (!hasRemovable) {
            f->setUnsplittable();
            // and add the outer halfedges to the queue. Also mark the vertices of the face conquered
            MCGAL::Halfedge* hh = h;
            do {
                hh->vertex()->setConquered();
                MCGAL::Halfedge* hOpp = hh->opposite();
                // TODO: wait
                // assert(!hOpp->is_border());
                if (!hOpp->facet()->isConquered()) {
                    gateQueue.push(hOpp);
                    hOpp->setInQueue();
                }
            } while ((hh = hh->next()) != h);
            h->removeFromQueue();
        } else {
            // in that case, cornerCut that vertex.
            h->removeFromQueue();
            vertexCut(unconqueredVertexHE);
        }
    }
    log("%d number is %d", i_curDecimationId, count);
    log("%d removed number is %d", i_curDecimationId, i_nbRemovedVertices);
    // 3. do the encoding job
    if (i_nbRemovedVertices == 0) {
        b_jobCompleted = true;
        i_nbDecimations = i_curDecimationId--;
        // Write the compressed data to the buffer.
        writeBaseMesh();
        int i_deci = i_curDecimationId;
        assert(i_deci >= 0);
        while (i_deci >= 0) {
            // encodeHausdorff(i_deci);
            encodeRemovedVertices(i_deci);
            encodeInsertedEdges(i_deci);
            i_deci--;
        }
    } else {
        // 3dpro: compute and encode the Hausdorff distance for all the facets in this LOD
        // computeHausdorfDistance();
        // HausdorffCodingStep();
        RemovedVertexCodingStep();
        InsertedEdgeCodingStep();
        // finish this round of decimation and start the next
        i_curDecimationId++;  // Increment the current decimation operation id.
    }
}

MCGAL::Halfedge* MyMesh::vertexCut(MCGAL::Halfedge* startH) {
    MCGAL::Vertex* v = startH->end_vertex();

    // make sure that the center vertex can be removed
    assert(!v->isConquered());
    assert(v->vertex_degree() > 2);

    MCGAL::Halfedge* h = startH->opposite();
    MCGAL::Halfedge* end(h);
    int removed = 0;
    do {
        // TODO: wait
        // assert(!h->is_border());
        MCGAL::Facet* f = h->facet();
        assert(!f->isConquered() && !f->isRemoved());  // we cannot cut again an already cut face, or a NULL patch
        /*
         * the old facets around the vertex will be removed in the vertex cut operation
         * and being replaced with a merged one. but the replacing group information
         * will be inherited by the new facet.
         *
         */

        // if the face is not a triangle, cut the corner to make it a triangle
        if (f->facet_degree() > 3) {
            // loop around the face to find the appropriate other halfedge
            MCGAL::Halfedge* hSplit(h->next());
            for (; hSplit->next()->next() != h; hSplit = hSplit->next())
                ;
            MCGAL::Halfedge* hCorner = split_facet(h, hSplit);
            // mark the new halfedges as added
            hCorner->setAdded();
            hCorner->opposite()->setAdded();
            // the corner one inherit the original facet
            // while the fRest is a newly generated facet
        }
        // mark the vertex as conquered
        h->end_vertex()->setConquered();
        // h->end_vertex->setConquered();
        removed++;
    } while ((h = h->opposite()->next()) != end);

    // copy the position of the center vertex:
    MCGAL::Point vPos = startH->end_vertex()->point();
    // remove the center vertex
    MCGAL::Halfedge* hNewFace = erase_center_vertex(startH);
    MCGAL::Facet* added_face = hNewFace->facet();

    // now mark the new face as having a removed vertex
    added_face->setSplittable();
    // keep the removed vertex position.
    added_face->setRemovedVertexPos(vPos);

    // scan the outside halfedges of the new face and add them to
    // the queue if the state of its face is unknown. Also mark it as in_queue
    h = hNewFace;
    do {
        MCGAL::Halfedge* hOpp = h->opposite();
        // TODO: wait
        // assert(!hOpp->is_border());
        if (!hOpp->facet()->isConquered()) {
            gateQueue.push(hOpp);
            hOpp->setInQueue();
        }
    } while ((h = h->next()) != hNewFace);
    // Increment the number of removed vertices.
    i_nbRemovedVertices++;
    removedPoints.push_back(vPos);
    return hNewFace;
}

void MyMesh::RemovedVertexCodingStep() {
    // resize the vectors to add the current conquest symbols
    geometrySym.push_back(std::deque<MCGAL::Point>());
    connectFaceSym.push_back(std::deque<unsigned>());

    // Add the first halfedge to the queue.
    pushHehInit();
    // bfs to add all the point
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        MCGAL::Facet* f = h->facet();

        // If the face is already processed, pick the next halfedge:
        if (f->isProcessed())
            continue;

        // Determine face symbol.
        unsigned sym = f->isSplittable();

        // Push the symbols.
        connectFaceSym[i_curDecimationId].push_back(sym);

        // Determine the geometry symbol.
        if (sym) {
            MCGAL::Point rmved = f->getRemovedVertexPos();
            geometrySym[i_curDecimationId].push_back(rmved);
            // record the removed points during compressing.
        }

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
    }
}

void MyMesh::InsertedEdgeCodingStep() {
    connectEdgeSym.push_back(std::deque<unsigned>());
    pushHehInit();
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();
        if (h->isProcessed()) {
            continue;
        }
        // Mark the halfedge as processed.
        h->setProcessed();
        h->opposite()->setProcessed();

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h->next();
        while (hIt->opposite() != h) {
            if (!hIt->isProcessed())
                gateQueue.push(hIt);
            hIt = hIt->opposite()->next();
        }

        // Don't write a symbol if the two faces of an edgde are unsplitable.
        // this can help to save some space, since it is guaranteed that the edge is not inserted
        bool b_toCode = h->facet()->isUnsplittable() && h->opposite()->facet()->isUnsplittable() ? false : true;

        // Determine the edge symbol.
        unsigned sym;
        if (h->isOriginal())
            sym = 0;
        else
            sym = 1;

        // Store the symbol if needed.
        if (b_toCode)
            connectEdgeSym[i_curDecimationId].push_back(sym);
    }
}

void MyMesh::writeBaseMesh() {
    unsigned i_nbVerticesBaseMesh = size_of_vertices();
    unsigned i_nbFacesBaseMesh = size_of_facets();
    // Write the number of level of decimations.
    writeInt16(i_nbDecimations);

    // Write the number of vertices and faces on 16 bits.
    writeInt(i_nbVerticesBaseMesh);
    writeInt(i_nbFacesBaseMesh);
    size_t id = 0;
    for (unsigned j = 0; j < 2; ++j) {
        MCGAL::Point p = vh_departureConquest[j]->point();
        writePoint(p);
        vh_departureConquest[j]->setId(id++);
    }
    // Write the other vertices.
    for (MCGAL::Vertex* vit : vertices) {
        if (vit == vh_departureConquest[0] || vit == vh_departureConquest[1])
            continue;
        MCGAL::Point point = vit->point();
        writePoint(point);
        // Set an id to the vertex.
        vit->setId(id++);
    }
    // Write the base mesh face vertex indices.
    for (MCGAL::Facet* fit : faces) {
        unsigned i_faceDegree = fit->facet_degree();
        writeInt(i_faceDegree);
        MCGAL::Halfedge* st = fit->getHalfedgeByIndex(0);
        MCGAL::Halfedge* ed = st;
        do {
            writeInt(st->vertex()->getId());
            st = st->next();
        } while (st != ed);
        // for (int i = 0; i < fit->halfedge_size; i++) {
        //     writeInt(fit->getHalfedgeByIndex(i)->vertex()->getId());
        // }
    }
}

/**
 * Encode an inserted edge list.
 */
void MyMesh::encodeInsertedEdges(unsigned i_operationId) {
    std::deque<unsigned>& symbols = connectEdgeSym[i_operationId];
    assert(symbols.size() > 0);

    unsigned i_len = symbols.size();
    for (unsigned i = 0; i < i_len; ++i) {
        writeChar(symbols[i]);
    }
}

/**
 * Encode the geometry and the connectivity of a removed vertex list.
 */
void MyMesh::encodeRemovedVertices(unsigned i_operationId) {
    std::deque<unsigned>& connSym = connectFaceSym[i_operationId];
    std::deque<MCGAL::Point>& geomSym = geometrySym[i_operationId];

    unsigned i_lenGeom = geomSym.size();
    unsigned i_lenConn = connSym.size();
    assert(i_lenGeom > 0);
    assert(i_lenConn > 0);

    unsigned k = 0;
    for (unsigned i = 0; i < i_lenConn; ++i) {
        // Encode the connectivity.
        unsigned sym = connSym[i];
        writeChar(sym);
        // Encode the geometry if necessary.
        if (sym) {
            writePoint(geomSym[k]);
            k++;
        }
    }
}