//
// Created by DELL on 2023/11/9.
//
#include "configuration.h"
#include "frenetRotation.h"
#include "mymesh.h"

#include "math.h"

void MyMesh::startNextCompresssionOp() {
    beginDecimationConquest();
    while (operation == DecimationConquest)
        decimationStep();
    beginRemovedVertexCodingConquest();
    while (operation == RemovedVertexCoding)
        RemovedVertexCodingStep();

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
    // 确保中心点可被删除
    assert(!v->isConquered());
    assert(v->vertex_degree() > 2);
    Halfedge_handle h = startH->opposite(), end(h);
    do {
        assert(!h->is_border());
        Face_handle f = h->facet();
        // 一个面不能删两次，null patch也不能删
        assert(!f->isConquered());
        // 如果面不是三角形，删掉corner
        if (f->facet_degree() > 3) {
            //
            Halfedge_handle hSplit(h->next());
            // 找到与当前节点相隔两个的点
            for (; hSplit->next()->next() != h; hSplit = hSplit->next())
                ;
            Halfedge_handle hCorner = split_facet(h, hSplit);
            hCorner->setAdded();
            hCorner->opposite()->setAdded();
        }
        // mark the vertex as conquered
        h->vertex()->setConquered();
    } while ((h = h->opposite()->next()) != end);
    // copy the position of the center vertex:
    Point vPos = startH->vertex()->point();

    // remove the center vertex
    Halfedge_handle hNewFace = erase_center_vertex(startH);

    // now mark the new face as having a removed vertex
    hNewFace->facet()->setSplittable();
    // keep the removed vertex position.
    hNewFace->facet()->setRemovedVertexPos(vPos);
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
    while (!gateQueue.empty()){
        Halfedge_handle h = gateQueue.front();
        gateQueue.pop();
        Face_handle f = h->face();
        // 如果当前面已经被处理，选择下一个
        if (f->isProcessed()){
            continue;
        }
        unsigned sym,symPred;
        bool b_split = f->isSplittable();
        // TODO:
        float f_faceSurface = faceSurface(h);
        float

    }
}