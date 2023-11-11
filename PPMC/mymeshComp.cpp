//
// Created by DELL on 2023/11/9.
//
#include "configuration.h"
#include "frenetRotation.h"
#include "mymesh.h"

#include "math.h"

void MyMesh::encode(int lod) {
    b_jobCompleted = false;
    while (!b_jobCompleted) {
        startNextCompresssionOp();
    }
}

void MyMesh::startNextCompresssionOp() {
    // 1. reset the stats
    for (MyMesh::Vertex_iterator vit = vertices_begin(); vit != vertices_end(); ++vit)
        vit->resetState();
    for (MyMesh::Halfedge_iterator hit = halfedges_begin(); hit != halfedges_end(); ++hit)
        hit->resetState();
    for (MyMesh::Face_iterator fit = facets_begin(); fit != facets_end(); ++fit)
        fit->resetState();
    i_nbRemovedVertices = 0;  // Reset the number of removed vertices.

    // 2. do one round of decimation
    // choose a halfedge that can be processed:
    if (i_curDecimationId < 10) {
        // teng: we always start from the middle, DO NOT use the rand function
        // size_t i_heInitId = (float)rand() / RAND_MAX * size_of_halfedges();
        size_t i_heInitId = size_of_halfedges() / 2;
        Halfedge_iterator hitInit = halfedges_begin();
        for (unsigned i = 0; i < i_heInitId; ++i)
            ++hitInit;
        hitInit->setInQueue();
        gateQueue.push((Halfedge_handle)hitInit);
    }

    while (!gateQueue.empty()) {
        Halfedge_handle h = gateQueue.front();
        gateQueue.pop();

        // pick a the first halfedge from the queue. f is the adjacent face.
        assert(!h->is_border());
        Face_handle f = h->facet();

        // if the face is already processed, pick the next halfedge:
        if (f->isConquered()) {
            h->removeFromQueue();
            continue;
        }
        // the face is not processed. Count the number of non conquered vertices that can be split
        bool hasRemovable = false;
        Halfedge_handle unconqueredVertexHE;

        for (Halfedge_handle hh = h->next(); hh != h; hh = hh->next()) {
            if (isRemovable(hh->vertex())) {
                hasRemovable = true;
                unconqueredVertexHE = hh;
                break;
            }
        }

        // if all face vertices are conquered, then the current face is a null patch:
        if (!hasRemovable) {
            f->setUnsplittable();
            // and add the outer halfedges to the queue. Also mark the vertices of the face conquered
            Halfedge_handle hh = h;
            do {
                hh->vertex()->setConquered();
                Halfedge_handle hOpp = hh->opposite();
                assert(!hOpp->is_border());
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
}

void MyMesh::beginDecimationConquest() {
    for (MyMesh::Vertex_iterator vit = vertices_begin(); vit != vertices_end(); vit++) {
        vit->resetState();
    }
    for (MyMesh::Halfedge_iterator hit = halfedges_begin(); hit != halfedges_end(); hit++) {
        hit->resetState();
    }
    for (MyMesh::Face_iterator fit = facets_begin(); fit != facets_end(); ++fit) {
        fit->resetState();
    }
    // Select the first gate to begin the decimation.
    size_t i_heInitId = (float)rand() / RAND_MAX * size_of_halfedges();
    Halfedge_iterator hitInit = halfedges_begin();
    for (unsigned i = 0; i < i_heInitId; ++i)
        ++hitInit;

    hitInit->setInQueue();
    gateQueue.push((Halfedge_handle)hitInit);

    // Reset the number of removed vertices.
    i_nbRemovedVertices = 0;

    // Set the current operation.
    operation = DecimationConquest;
}

void MyMesh::decimationStep() {
    while (!gateQueue.empty()) {
        Halfedge_handle h = gateQueue.front();
        gateQueue.pop();
        // pick a the first halfedge from the queue. f is the adjacent face.
        assert(!h->is_border());
        Face_handle f = h->facet();
        // if the face is already processed, pick the next halfedge:
        if (f->isConquered()) {
            h->removeFromQueue();
            continue;
        }
        // the face is not processed. Count the number of non conquered vertices that can be split
        int numSplittableVerts = 0;
        // 拿到第一个可以删除的点
        Halfedge_handle unconqueredVertexHE;
        // assert(h->vertex()->isConquered());
        // 统计有多少可以拆除的边
        for (Halfedge_handle hh = h->next(); hh != h; hh = hh->next()) {
            // 检查度，以及是否会破环流型结构
            if (isRemovable(hh->vertex())) {
                if (numSplittableVerts == 0)
                    unconqueredVertexHE = hh;
                ++numSplittableVerts;
            }
        }
        // 如果所有的vertex都已经被处理，那么当前面就是一个空补丁
        if (numSplittableVerts == 0) {
            f->setUnsplittable();
            Halfedge_handle hh = h;
            // 遍历当前面所有halfedge，将反向边插入并设置当前点状态
            do {
                hh->vertex()->setConquered();
                Halfedge_handle hOpp = hh->opposite();
                assert(!hOpp->is_border());
                if (!hOpp->facet()->isConquered()) {
                    gateQueue.push(hOpp);
                    hOpp->setInQueue();
                }
            } while ((hh = hh->next()) != h);
            h->removeFromQueue();
            return;
        } else {
            h->removeFromQueue();
            vertexCut(unconqueredVertexHE);
            return;
        }
    }

    if (i_nbRemovedVertices == 0) {
        // 如果一个点都没有删除，则开始尝试违反凸模式
        if (!b_testConvexity) {
            for (MyMesh::Vertex_iterator vit = vertices_begin(); vit != vertices_end(); vit++) {
                if (isRemovable(vit)) {}
            }
            operation = Idle;
            b_jobCompleted = true;
            i_curDecimationId--;
            // TODO: 待完成
            writeCompressedData();
            // TODO: 待完成
            writeCompressedFile();
        } else {
            // 如果没有需要删除
            b_testConvexity = false;
            i_levelNotConvexId = i_curDecimationId;
            beginDecimationConquest();
        }
    } else {
        // TODO: 待完成
        determineResiduals();
        if (b_useLiftingScheme) {
            lift(false);
        }
        operation = RemovedVertexCoding;
        // TODO: 待完成
        beginRemovedVertexCodingConquest();
    }
}

// 顶点删除以及新建边
// 存被删除的点的位置信息
MyMesh::Halfedge_handle MyMesh::vertexCut(Halfedge_handle startH) {
    Vertex_handle v = startH->vertex();

    // make sure that the center vertex can be removed
    assert(!v->isConquered());
    assert(v->vertex_degree() > 2);

    unordered_set<replacing_group*> rep_groups;
    replacing_group* new_rg = new replacing_group();

    Halfedge_handle h = startH->opposite(), end(h);
    int removed = 0;
    do {
        assert(!h->is_border());
        Face_handle f = h->facet();
        assert(!f->isConquered());  // we cannot cut again an already cut face, or a NULL patch

        /*
         * the old facets around the vertex will be removed in the vertex cut operation
         * and being replaced with a merged one. but the replacing group information
         * will be inherited by the new facet.
         *
         */
        if (f->rg != NULL) {
            rep_groups.emplace(f->rg);
            assert(f->rg->ref-- > 0);
        }

        // if the face is not a triangle, cut the corner to make it a triangle
        if (f->facet_degree() > 3) {
            // loop around the face to find the appropriate other halfedge
            Halfedge_handle hSplit(h->next());
            for (; hSplit->next()->next() != h; hSplit = hSplit->next())
                ;
            Halfedge_handle hCorner = split_facet(h, hSplit);
            // mark the new halfedges as added
            hCorner->setAdded();
            hCorner->opposite()->setAdded();
            // the corner one inherit the original facet
            // while the fRest is a newly generated facet
            Face_handle fCorner = hCorner->face();
            Face_handle fRest = hCorner->opposite()->face();
            assert(fCorner->rg == f->rg);
            if (f->rg) {
                fRest->rg = f->rg;
                // assert(fCorner->rg != NULL && fRest->rg == NULL);
                fRest->rg->ref++;
            }
            // log("split %ld + %ld %ld", fCorner->facet_degree(), fRest->facet_degree(), f->facet_degree());
        }
        f->rg = NULL;

        // mark the vertex as conquered
        h->vertex()->setConquered();
        removed++;
    } while ((h = h->opposite()->next()) != end);

    // copy the position of the center vertex:
    Point vPos = startH->vertex()->point();
    new_rg->removed_vertices.emplace(vPos);

    int bf = size_of_facets();
    // remove the center vertex
    Halfedge_handle hNewFace = erase_center_vertex(startH);
    Face_handle added_face = hNewFace->facet();
    assert(added_face->rg == NULL);
    added_face->rg = new_rg;
    new_rg->ref++;

    // log("test: %d = %d - %ld merged %ld replacing groups", removed, bf, size_of_facets(), rep_groups.size());
    merge(rep_groups, new_rg);

    // now mark the new face as having a removed vertex
    added_face->setSplittable();
    // keep the removed vertex position.
    added_face->setRemovedVertexPos(vPos);

    // scan the outside halfedges of the new face and add them to
    // the queue if the state of its face is unknown. Also mark it as in_queue
    h = hNewFace;
    do {
        Halfedge_handle hOpp = h->opposite();
        assert(!hOpp->is_border());
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

// 记录一下residuals 编码
void MyMesh::determineResiduals() {
    // TODO:
    pushHehInit();
    while (!gateQueue.empty()) {
        Halfedge_handle h = gateQueue.front();
        gateQueue.pop();

        Face_handle f = h->facet();

        // If the face is already processed, pick the next halfedge:
        if (f->isProcessed())
            continue;

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

        if (f->isSplittable())
            f->setResidual(getQuantizedPos(f->getRemovedVertexPos()) - getQuantizedPos(barycenter(h)));
    }
}

void MyMesh::beginRemovedVertexCodingConquest() {
    typeOfOperation.push_back(DECIMATION_OPERATION_ID);
    pushHehInit();

    geometrySym.push_back(std::deque<VectorInt>());
    connectFaceSym.push_back(std::deque<std::pair<unsigned, unsigned>>());
    f_avgSurfaceFaceWithCenterRemoved = 0;
    f_avgSurfaceFaceWithoutCenterRemoved = 0;
    i_nbFacesWithCenterRemoved = 0;
    i_nbFacesWithoutCenterRemoved = 0;
    i_nbGoodPredictions = 0;

    operation = RemovedVertexCoding;
    printf("Removed vertex coding begining.\n");
}

void MyMesh::RemovedVertexCodingStep() {
    while (!gateQueue.empty()) {
        Halfedge_handle h = gateQueue.front();
        gateQueue.pop();
        Face_handle f = h->face();
        // 如果当前面已经被处理，选择下一个
        if (f->isProcessed()) {
            continue;
        }
        unsigned sym, symPred;
        bool b_split = f->isSplittable();
        // TODO:
        float f_faceSurface = faceSurface(h);
        float
    }
}