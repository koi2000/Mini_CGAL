#include "include/Halfedge.h"
#include "include/global.h"
namespace MCGAL {
// create a new half edge, setup the opposite of this half edge if needed
Halfedge::Halfedge(Vertex* v1, Vertex* v2) {
    vertex_ = v1->poolId;
    end_vertex_ = v2->poolId;
    
    vertex()->addHalfedge(this);
    // in case this is the second half edge
    for (int i = 0; i < v2->halfedges_size; i++) {
        Halfedge* h = v2->getHalfedgeByIndex(i);
        if (h->end_vertex() == v1) {
            if (h->opposite()) {
                printf("create half edge:\n");
                v1->print();
                v2->print();
                h->opposite()->facet()->print_off();
            }
            assert(h->opposite() == NULL);
            // h->opposite = this;
            // this->opposite = h;
            h->setOpposite(this);
            this->setOpposite(h);
            break;
        }
    }
}

void Halfedge::reset(Vertex* v1, Vertex* v2) {
    vertex_ = v1->poolId;
    end_vertex_ = v2->poolId;
    // vertex()->halfedges.push_back(this);
    vertex()->addHalfedge(this);
    // end_vertex->opposite_half_edges.insert(this);
    // in case this is the second half edge
    // for (Halfedge* h : v2->halfedges) {
    for (int i = 0; i < v2->halfedges_size; i++) {
        Halfedge* h = contextPool.getHalfedgeByIndex(v2->halfedges[i]);
        if (h->end_vertex() == v1) {
            if (h->opposite()) {
                printf("create half edge:\n");
                v1->print();
                v2->print();
                h->opposite()->facet()->print_off();
            }
            assert(h->opposite() == NULL);
            // h->opposite = this;
            // this->opposite = h;
            h->setOpposite(this);
            this->setOpposite(h);
            break;
        }
    }
}

Halfedge::~Halfedge() {}

Vertex* Halfedge::vertex() {
    if (vertex_ == -1) {
        return nullptr;
    }
    return contextPool.getVertexByIndex(vertex_);
};
Vertex* Halfedge::end_vertex() {
    if (end_vertex_ == -1) {
        return nullptr;
    }
    return contextPool.getVertexByIndex(end_vertex_);
};
Facet* Halfedge::facet() {
    if (facet_ == -1) {
        return nullptr;
    }
    return contextPool.getFacetByIndex(facet_);
};

Halfedge* Halfedge::opposite() {
    if (opposite_ == -1) {
        return nullptr;
    }
    return contextPool.getHalfedgeByIndex(opposite_);
};

Halfedge* Halfedge::next() {
    if (next_ == -1) {
        return nullptr;
    }
    return contextPool.getHalfedgeByIndex(next_);
};

void Halfedge::setOpposite(Halfedge* opposite) {
    opposite_ = opposite->poolId;
};

void Halfedge::setOpposite(int opposite) {
    opposite_ = opposite;
};

void Halfedge::setNext(Halfedge* next) {
    next_ = next->poolId;
};

void Halfedge::setNext(int next) {
    next_ = next;
};

void Halfedge::setFacet(Facet* facet) {
    facet_ = facet->poolId;
};

void Halfedge::setFacet(int facet) {
    facet_ = facet;
};

}  // namespace MCGAL
