#include "core.h"

namespace MCGAL {
// create a new half edge, setup the opposite of this half edge if needed
Halfedge::Halfedge(Vertex* v1, Vertex* v2) {
    vertex = v1;
    end_vertex = v2;
    vertex->halfedges.push_back(this);
    // end_vertex->opposite_half_edges.insert(this);
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
            break;
        }
    }
}

void Halfedge::reset(Vertex* v1, Vertex* v2) {
    vertex = v1;
    end_vertex = v2;
    vertex->halfedges.push_back(this);
    // end_vertex->opposite_half_edges.insert(this);
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
            break;
        }
    }
}

Halfedge::~Halfedge() {
    // reset the opposite;
    if (opposite != NULL) {
        assert(opposite->opposite = this);
        opposite->opposite = NULL;
    }
}
}  // namespace MCGAL
