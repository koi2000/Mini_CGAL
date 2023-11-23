#include "core.h"

namespace MCGAL {
void Face::remove(Halfedge* rh) {
    halfedges.erase(rh);
    for (Halfedge* h : halfedges) {
        if (h->next == rh) {
            h->next = NULL;
        }
    }
    delete rh;
}

Face::Face(const Face& face) {
    // this->vertices = face.vertices;
    // this->halfedges = face.halfedges;
    this->flag = face.flag;
    this->processedFlag = face.processedFlag;
}

Face::~Face() {
    halfedges.clear();
    vertices.clear();
}

Face::Face(Halfedge* hit) {
    Halfedge* st(hit);
    Halfedge* ed(hit);
    std::vector<Halfedge*> edges;
    do {
        edges.push_back(st);
        st = st->next;
    } while (st != ed);
    this->reset(edges);
}

Face* Face::clone() {
    return new Face(*this);
}

Face::Face(std::vector<Vertex*>& vs) {
    Halfedge* prev = nullptr;
    Halfedge* head = nullptr;
    for (int i = 0; i < vs.size(); i++) {
        vertices.insert(vs[i]);
        Vertex* nextv = vs[(i + 1) % vs.size()];
        Halfedge* hf = new Halfedge(vs[i], nextv);
        halfedges.insert(hf);
        // vs[i]->halfedges.insert(hf);
        hf->face = this;
        if (prev != NULL) {
            prev->next = hf;
        } else {
            head = hf;
        }
        if (i == vs.size() - 1) {
            hf->next = head;
        }
        prev = hf;
    }
}

Face::Face(std::vector<Vertex*>& vs, Mesh* mesh) {
    Halfedge* prev = nullptr;
    Halfedge* head = nullptr;
    for (int i = 0; i < vs.size(); i++) {
        vertices.insert(vs[i]);
        Vertex* nextv = vs[(i + 1) % vs.size()];
        Halfedge* hf = std::move(mesh->allocateHalfedgeFromPool(vs[i], nextv));
        halfedges.insert(hf);
        // vs[i]->halfedges.insert(hf);
        hf->face = this;
        if (prev != NULL) {
            prev->next = hf;
        } else {
            head = hf;
        }
        if (i == vs.size() - 1) {
            hf->next = head;
        }
        prev = hf;
    }
}

void Face::reset(std::vector<Vertex*>& vs, Mesh* mesh) {
    Halfedge* prev = nullptr;
    Halfedge* head = nullptr;
    for (int i = 0; i < vs.size(); i++) {
        vertices.insert(vs[i]);
        Vertex* nextv = vs[(i + 1) % vs.size()];
        Halfedge* hf = mesh->allocateHalfedgeFromPool(vs[i], nextv);
        halfedges.insert(hf);
        // vs[i]->halfedges.insert(hf);
        hf->face = this;
        if (prev != NULL) {
            prev->next = hf;
        } else {
            head = hf;
        }
        if (i == vs.size() - 1) {
            hf->next = head;
        }
        prev = hf;
    }
}

void Face::reset(Halfedge* h) {
    Halfedge* st = h;
    Halfedge* ed = h;
    std::vector<Halfedge*> edges;
    do {
        edges.push_back(st);
        st = st->next;
    } while (st != ed);
    reset(edges);
}

void Face::reset(std::vector<Halfedge*>& hs) {
    this->halfedges.clear();
    this->vertices.clear();
    for (int i = 0; i < hs.size(); i++) {
        // hs[i]->next = hs[(i + 1) % hs.size()];
        // hs[i]->face = this;
        this->halfedges.insert(hs[i]);
        this->vertices.insert(hs[i]->vertex);
        hs[i]->face = this;
    }
}

void Face::print() {
    printf("totally %ld vertices:\n", vertices.size());
    int idx = 0;
    for (Vertex* v : vertices) {
        printf("%d:\t", idx++);
        v->print();
    }
}

void Face::print_off() {
    printf("OFF\n%ld 1 0\n", vertices.size());
    for (Vertex* v : vertices) {
        v->print();
    }
    printf("%ld\t", vertices.size());
    for (int i = 0; i < vertices.size(); i++) {
        printf("%d ", i);
    }
    printf("\n");
}

bool Face::equal(const Face& rhs) const {
    if (vertices.size() != rhs.vertices.size()) {
        return false;
    }
    for (Vertex* vertix : vertices) {
        if (!rhs.vertices.count(vertix)) {
            return false;
        }
    }

    return true;
}
bool Face::operator==(const Face& rhs) const {
    return this->equal(rhs);
}

int Face::facet_degree() {
    return vertices.size();
}

}  // namespace MCGAL
