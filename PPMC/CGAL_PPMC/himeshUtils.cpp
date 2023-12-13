//
// Created by DELL on 2023/11/10.
//
#include "himesh.h"

void MyMesh::pushHehInit() {
    Halfedge_handle hehBegin;
    Halfedge_around_vertex_circulator hit(vh_departureConquest[0]->vertex_begin());
    while (1) {
        hehBegin = hit->opposite();
        if (hehBegin->vertex() == vh_departureConquest[1])
            break;
        ++hit;
    }
    // Push it to the queue.
    gateQueue.push(hehBegin);
}
