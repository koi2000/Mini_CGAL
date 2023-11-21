#include "core.h"

namespace MCGAL {
// create a new half edge, setup the opposite of this half edge if needed
Halfedge::Halfedge(Vertex* v1, Vertex* v2) {
    vertex = v1;
    end_vertex = v2;
    vertex->halfedges.insert(this);
    end_vertex->opposite_half_edges.insert(this);
    Vertex* h1 = new Vertex(0.031457, 0.037887, -0.000065);
    Vertex* h2 = new Vertex(0.033419, 0.039939, -0.002264);
    if (compareFloat(v1->x(), h1->x()) && compareFloat(v1->y(), h1->y()) && compareFloat(v1->z(), h1->z())) {
        if (compareFloat(v2->x(), h2->x()) && compareFloat(v2->y(), h2->y()) && compareFloat(v2->z(), h2->z())) {
            printf("create the vertex\n");
        }
    }
    // in case this is the second half edge
    for (Halfedge* h : v2->halfedges) {
        if (h->end_vertex == v1) {
            if (h->opposite) {
                printf("create half edge:\n");
                v1->print();
                v2->print();
                h->opposite->face->print_off();
            }
            assert(h->opposite == NULL);
            h->opposite = this;
            this->opposite = h;
        }
    }
}

Halfedge::~Halfedge() {
    // reset the opposite;
    if (opposite != NULL) {
        assert(opposite->opposite = this);
        opposite->opposite = NULL;
    }

    // detach from the vertices
    assert(vertex && end_vertex);
    // assert(vertex->halfedges.find(this) != vertex->halfedges.end());
    // assert(end_vertex->opposite_half_edges.find(this) != end_vertex->opposite_half_edges.end());
    // if (vertex->halfedges.find(this) != vertex->halfedges.end()) {
    //     vertex->halfedges.erase(this);
    // }
    // if (end_vertex->opposite_half_edges.find(this) != end_vertex->opposite_half_edges.end()) {
    //     end_vertex->opposite_half_edges.erase(this);
    // }
}
}  // namespace MCGAL
