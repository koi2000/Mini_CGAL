//
// Created by DELL on 2023/11/9.
//
#include "mymesh.h"
//

bool MyMesh::willViolateManifold(const std::vector<Halfedge_const_handle>& polygon) const {
    unsigned i_degree = polygon.size();

    // Test if a patch vertex is not connected to one vertex
    // that is not one of its direct neighbor.
    // Test also that two vertices of the patch will not be doubly connected
    // after the vertex cut opeation.
    for (unsigned i = 0; i < i_degree; ++i) {
        Halfedge_around_vertex_const_circulator Hvc = polygon[i]->vertex()->vertex_begin();
        Halfedge_around_vertex_const_circulator Hvc_end = Hvc;
        CGAL_For_all(Hvc, Hvc_end) {
            // Look if the current vertex belongs to the patch.
            Vertex_const_handle vh = Hvc->opposite()->vertex();
            for (unsigned j = 0; j < i_degree; ++j) {
                if (vh == polygon[j]->vertex()) {
                    unsigned i_prev = i == 0 ? i_degree - 1 : i - 1;
                    unsigned i_next = i == i_degree - 1 ? 0 : i + 1;

                    if ((j == i_prev && polygon[i]->facet_degree() != 3)  // The vertex cut operation is forbidden.
                        || (j == i_next &&
                            polygon[i]->opposite()->facet_degree() != 3))  // The vertex cut operation is forbidden.
                        return true;
                }
            }
        }
    }

    return false;
}

bool MyMesh::isRemovable(Vertex_handle v) const {
    //	if(size_of_vertices()<10){
    //		return false;
    //	}
    if (v != vh_departureConquest[0] && v != vh_departureConquest[1] && !v->isConquered() && v->vertex_degree() > 2 &&
        v->vertex_degree() <= 8) {
        // test convexity
        std::vector<Vertex_const_handle> vh_oneRing;
        std::vector<Halfedge_const_handle> heh_oneRing;

        vh_oneRing.reserve(v->vertex_degree());
        heh_oneRing.reserve(v->vertex_degree());
        // vh_oneRing.push_back(v);
        Halfedge_around_vertex_const_circulator hit(v->vertex_begin()), end(hit);
        do {
            vh_oneRing.push_back(hit->opposite()->vertex());
            heh_oneRing.push_back(hit->opposite());
        } while (++hit != end);
        //
        bool removable = !willViolateManifold(heh_oneRing);
        // && isProtruding(heh_oneRing);
        //&& isConvex(vh_oneRing)
        return removable;
    }
    return false;
}