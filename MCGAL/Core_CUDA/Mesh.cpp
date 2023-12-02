#include "core.h"
#include "global.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
namespace MCGAL {

Mesh::~Mesh() {
    vertices.clear();
    faces.clear();
}

Facet* Mesh::add_face(std::vector<Vertex*>& vs) {
    Facet* f = contextPool.allocateFaceFromPool(vs);
    faces.push_back(f);
    return f;
}

Facet* Mesh::add_face(Facet* face) {
    faces.push_back(face);
    return face;
}

// Halfedge* Mesh::split_facet(Halfedge* h, Halfedge* g) {
//     Facet* origin = h->face;
//     // early expose
//     Facet* fnew = new Facet();
//     // create new halfedge
//     Halfedge* hnew = new Halfedge(h->end_vertex, g->end_vertex);
//     Halfedge* oppo_hnew = new Halfedge(g->end_vertex, h->end_vertex);
//     // set the opposite
//     // set the connect information
//     hnew->next = g->next;
//     oppo_hnew->next = h->next;
//     h->next = hnew;
//     g->next = oppo_hnew;
//     // add the new halfedge to origin face

//     // delete old halfedge from origin face

//     // create new face depend on vertexs
//     Halfedge* gst = g;
//     Halfedge* ged = gst;
//     Halfedge* hst = h;
//     Halfedge* hed = hst;
//     std::vector<Halfedge*> origin_face;
//     std::vector<Halfedge*> new_face;
//     do {
//         new_face.push_back(hst);
//         hst = hst->next;
//     } while (hst != hed);
//     do {
//         origin_face.push_back(gst);
//         gst = gst->next;
//     } while (gst != ged);
//     // origin_face.push_back(oppo_hnew);
//     origin->reset(origin_face);
//     fnew->reset(new_face);
//     // add halfedge and face to mesh
//     // this->halfedges.insert(hnew);
//     // this->halfedges.insert(oppo_hnew);
//     this->faces.push_back(fnew);
//     return hnew;
// }

Halfedge* Mesh::create_center_vertex(Halfedge* h) {
    // Vertex* vnew = new Vertex();
    Vertex* vnew = contextPool.allocateVertexFromPool();
    this->vertices.push_back(vnew);
    Halfedge* hnew = contextPool.allocateHalfedgeFromPool(h->end_vertex(), vnew);
    Halfedge* oppo_new = contextPool.allocateHalfedgeFromPool(vnew, h->end_vertex());
    // add new halfedge to current mesh and set opposite
    // set the next element
    // now the next of hnew and prev of oppo_new is unknowen
    insert_tip(hnew->opposite(), h);
    Halfedge* g = hnew->opposite()->next();
    std::vector<Halfedge*> origin_around_halfedge;

    Halfedge* hed = hnew;
    while (g->next() != hed) {
        Halfedge* gnew = contextPool.allocateHalfedgeFromPool(g->end_vertex(), vnew);
        Halfedge* oppo_gnew = contextPool.allocateHalfedgeFromPool(vnew, g->end_vertex());
        origin_around_halfedge.push_back(g);
        // gnew->next = hnew->opposite;
        gnew->setNext(hnew->opposite());
        insert_tip(gnew->opposite(), g);

        g = gnew->opposite()->next();
        hnew = gnew;
    }
    // hed->next = hnew->opposite;
    hed->setNext(hnew->opposite());
    h->facet()->reset(h);
    // collect all the halfedge
    for (Halfedge* hit : origin_around_halfedge) {
        Facet* face = contextPool.allocateFaceFromPool(hit);
        this->faces.push_back(face);
    }
    return oppo_new;
}

// inline void Mesh::close_tip(Halfedge* h, Vertex* v) const {
//     h->next = h->opposite;
//     h->vertex = v;
// }

inline void Mesh::insert_tip(Halfedge* h, Halfedge* v) const {
    // h->next = v->next;
    h->setNext(v->next());
    v->setNext(h->opposite());
    // v->next = h->opposite;
}

Halfedge* Mesh::find_prev(Halfedge* h) const {
    Halfedge* g = h;
    while (g->next() != h)
        g = g->next();
    return g;
}

// Halfedge* Mesh::erase_center_vertex(Halfedge* h) {
//     Halfedge* g = h->next->opposite;
//     Halfedge* hret = find_prev(h);
//     Facet* face = new Facet();
//     faces.push_back(face);
//     while (g != h) {
//         Halfedge* gprev = find_prev(g);
//         remove_tip(gprev);
//         // if (g->face != face) {
//         // faces.erase(g->face);
//         // }
//         Halfedge* gnext = g->next->opposite;
//         // this->halfedges.erase(g);
//         // this->halfedges.erase(g->opposite);
//         // g->vertex->halfedges.erase(g);
//         // g->opposite->vertex->halfedges.erase(g->opposite);

//         g = gnext;
//     }
//     // faces.erase(h->face);
//     remove_tip(hret);
//     // vertices.erase(h->end_vertex);
//     for (Halfedge* hit : h->end_vertex->halfedges) {
//         // hit->end_vertex->halfedges.erase(hit->opposite);
//     }
//     h->end_vertex->halfedges.clear();

//     // this->halfedges.erase(h);
//     // this->halfedges.erase(h->opposite);
//     // h->vertex->halfedges.erase(h);
//     // h->opposite->vertex->halfedges.erase(h->opposite);
//     set_face_in_face_loop(hret, face);
//     return hret;
// }

// void Mesh::set_face_in_face_loop(Halfedge* h, Facet* f) const {
//     f->halfedges.clear();
//     f->vertices.clear();
//     Halfedge* end = h;
//     do {
//         h->face = f;
//         f->halfedges.push_back(h);
//         f->vertices.push_back(h->vertex);
//         h = h->next;
//     } while (h != end);
// }

inline void Mesh::remove_tip(Halfedge* h) const {
    // h->next = h->next->opposite->next;
    h->setNext(h->next()->opposite()->next());
}

Halfedge* Mesh::join_face(Halfedge* h) {
    Halfedge* hprev = find_prev(h);
    Halfedge* gprev = find_prev(h->opposite());
    remove_tip(hprev);
    remove_tip(gprev);
    h->opposite()->setRemoved();

    // h->vertex->halfedges.erase(h);
    // for (auto it = h->vertex->halfedges.begin(); it != h->vertex->halfedges.end(); it++) {
    //     if ((*it) == h) {
    //         h->vertex->halfedges.erase(it);
    //         break;
    //     }
    // }
    h->vertex()->eraseHalfedgeByPointer(h);
    h->opposite()->vertex()->eraseHalfedgeByPointer(h->opposite());
    // for (auto it = h->opposite->vertex->halfedges.begin(); it != h->opposite->vertex->halfedges.end(); it++) {
    //     if ((*it) == h) {
    //         h->opposite->vertex->halfedges.erase(it);
    //         break;
    //     }
    // }

    // h->opposite->vertex->halfedges.erase(h->opposite);
    // this->faces.erase(gprev->face);
    gprev->facet()->setRemoved();
    // delete gprev->face;
    hprev->facet()->reset(hprev);
    return hprev;
}

bool Mesh::loadOFF(std::string path) {
    std::ifstream fp(path);
    if (!fp.is_open()) {
        std::cerr << "Error: Unable to open file " << path << std::endl;
        return false;
    }

    std::stringstream file;
    file << fp.rdbuf();  // Read the entire file content into a stringstream

    std::string format;
    file >> format >> nb_vertices >> nb_faces >> nb_edges;

    if (format != "OFF") {
        std::cerr << "Error: Invalid OFF file format" << std::endl;
        return false;
    }

    std::vector<Vertex*> vertices;
    for (std::size_t i = 0; i < nb_vertices; ++i) {
        float x, y, z;
        file >> x >> y >> z;
        Vertex* vt = new Vertex(x, y, z);
        vt->setVid(i);
        this->vertices.push_back(vt);
        vertices.push_back(vt);
    }

    for (int i = 0; i < nb_faces; ++i) {
        int num_face_vertices;
        file >> num_face_vertices;
        std::vector<Vertex*> vts;
        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index;
            file >> vertex_index;
            vts.push_back(vertices[vertex_index]);
        }
        this->add_face(vts);
    }
    vertices.clear();
    fp.close();
    return true;
}

std::istream& operator>>(std::istream& input, Mesh& mesh) {
    std::string format;
    // read off header
    input >> format >> mesh.nb_vertices >> mesh.nb_faces >> mesh.nb_edges;
    if (format != "OFF") {
        std::cerr << "Error: Invalid OFF file format" << std::endl;
    }

    // vector used to create face
    std::vector<Vertex*> vertices;
    // add vertex into Mesh
    for (std::size_t i = 0; i < mesh.nb_vertices; ++i) {
        float x, y, z;
        input >> x >> y >> z;
        Vertex* vt = new Vertex(x, y, z);
        vt->setVid(i);
        mesh.vertices.push_back(vt);
        vertices.push_back(vt);
    }

    // write face into mesh
    for (int i = 0; i < mesh.nb_faces; ++i) {
        int num_face_vertices;
        input >> num_face_vertices;
        // std::vector<Facet*> faces;
        std::vector<Vertex*> vts;

        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index;
            input >> vertex_index;
            vts.push_back(vertices[vertex_index]);
        }
        Facet* face = mesh.add_face(vts);
        // for (Halfedge* halfedge : face->halfedges) {
        //     mesh.halfedges.insert(halfedge);
        // }
    }
    // clear vector
    vertices.clear();
    return input;
}

void Mesh::dumpto(std::string path) {
    for (auto fit = faces.begin(); fit != faces.end();) {
        if ((*fit)->isRemoved()) {
            // delete *fit;
            fit = faces.erase(fit);
        } else {
            fit++;
        }
    }

    std::ofstream offFile(path);
    if (!offFile.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }
    // write header
    offFile << "OFF\n";
    offFile << this->vertices.size() << " " << this->faces.size() << " 0\n";
    offFile << "\n";
    // write vertex
    int id = 0;
    for (Vertex* vertex : this->vertices) {
        offFile << vertex->x() << " " << vertex->y() << " " << vertex->z() << "\n";
        vertex->setId(id++);
    }

    for (Facet* face : this->faces) {
        if (face->isRemoved())
            continue;
        offFile << face->vertex_size << " ";
        Halfedge* hst = contextPool.getHalfedgeByIndex(face->halfedges[0]);
        Halfedge* hed = hst;
        do {
            offFile << hst->vertex()->getId() << " ";
            hst = hst->next();
        } while (hst != hed);
        offFile << "\n";
    }

    offFile.close();
}

}  // namespace MCGAL