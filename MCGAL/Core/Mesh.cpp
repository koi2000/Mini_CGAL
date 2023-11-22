#include "core.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
namespace MCGAL {

Mesh::~Mesh() {
    for (Face* f : faces) {
        delete f;
    }
    for (Vertex* p : vertices) {
        // assert(p->halfedges.size() == (int)0 && p->opposite_half_edges.size() == 0);
        delete p;
    }
    for (Halfedge* e : halfedges) {
        delete e;
    }
    vertices.clear();
    faces.clear();
    halfedges.clear();
}

Face* Mesh::add_face(std::vector<Vertex*>& vs) {
    Face* f = new Face(vs);
    for (Halfedge* hit : f->halfedges) {
        this->halfedges.insert(hit);
    }
    faces.insert(f);
    return f;
}

Halfedge* Mesh::split_facet(Halfedge* h, Halfedge* g) {
    Face* origin = h->face;
    // early expose
    Face* fnew = new Face();
    // create new halfedge
    Halfedge* hnew = new Halfedge(h->end_vertex, g->end_vertex);
    Halfedge* oppo_hnew = new Halfedge(g->end_vertex, h->end_vertex);
    // set the opposite
    // hnew->opposite = oppo_hnew;
    // oppo_hnew->opposite = hnew;
    // set the connect information
    hnew->next = g->next;
    oppo_hnew->next = h->next;
    h->next = hnew;
    g->next = oppo_hnew;
    // add the new halfedge to origin face

    // delete old halfedge from origin face

    // create new face depend on vertexs
    Halfedge* gst = g;
    Halfedge* ged = gst;
    Halfedge* hst = h;
    Halfedge* hed = hst;
    std::vector<Halfedge*> origin_face;
    std::vector<Halfedge*> new_face;
    do {
        new_face.push_back(hst);
        hst = hst->next;
    } while (hst != hed);
    // new_face.push_back(hnew);
    // hst = h->next;
    // hed = g->next;
    do {
        origin_face.push_back(gst);
        gst = gst->next;
    } while (gst != ged);
    // origin_face.push_back(oppo_hnew);
    origin->reset(origin_face);
    fnew->reset(new_face);
    // add halfedge and face to mesh
    this->halfedges.insert(hnew);
    this->halfedges.insert(oppo_hnew);
    this->faces.insert(fnew);
    return hnew;
}

Halfedge* Mesh::create_center_vertex(Halfedge* h) {
    // this->faces.erase(h->face);
    // Face* origin = h->face;
    Vertex* vnew = new Vertex();
    this->vertices.insert(vnew);
    Halfedge* hnew = new Halfedge(h->end_vertex, vnew);
    Halfedge* oppo_new = new Halfedge(vnew, h->end_vertex);
    // add new halfedge to current mesh and set opposite
    this->halfedges.insert(hnew);
    this->halfedges.insert(oppo_new);
    // hnew->opposite = oppo_new;
    // oppo_new->opposite = hnew;
    // set the next element
    // now the next of hnew and prev of oppo_new is unknowen
    insert_tip(hnew->opposite, h);
    Halfedge* g = hnew->opposite->next;
    std::vector<Halfedge*> origin_around_halfedge;
    // origin_around_halfedge.push_back(h);

    Halfedge* hed = hnew;
    while (g->next != hed) {
        Halfedge* gnew = new Halfedge(g->end_vertex, vnew);
        Halfedge* oppo_gnew = new Halfedge(vnew, g->end_vertex);
        this->halfedges.insert(gnew);
        this->halfedges.insert(oppo_gnew);
        // gnew->opposite = oppo_gnew;
        // oppo_gnew->opposite = gnew;
        origin_around_halfedge.push_back(g);
        gnew->next = hnew->opposite;
        insert_tip(gnew->opposite, g);

        g = gnew->opposite->next;
        hnew = gnew;
    }
    hed->next = hnew->opposite;
    h->face->reset(h);
    // collect all the halfedge
    for (Halfedge* hit : origin_around_halfedge) {
        Face* face = new Face(hit);
        this->faces.insert(face);
    }
    return oppo_new;
}

void Mesh::close_tip(Halfedge* h, Vertex* v) const {
    h->next = h->opposite;
    h->vertex = v;
}

void Mesh::insert_tip(Halfedge* h, Halfedge* v) const {
    h->next = v->next;
    v->next = h->opposite;
}

Halfedge* Mesh::find_prev(Halfedge* h) const {
    Halfedge* g = h;
    while (g->next != h)
        g = g->next;
    return g;
}

Halfedge* Mesh::erase_center_vertex(Halfedge* h) {
    Halfedge* g = h->next->opposite;
    Halfedge* hret = find_prev(h);
    Face* face = new Face();
    faces.insert(face);
    while (g != h) {
        Halfedge* gprev = find_prev(g);
        remove_tip(gprev);
        // if (g->face != face) {
        faces.erase(g->face);
        // }
        Halfedge* gnext = g->next->opposite;
        this->halfedges.erase(g);
        this->halfedges.erase(g->opposite);
        g->vertex->halfedges.erase(g);
        g->opposite->vertex->halfedges.erase(g->opposite);

        g = gnext;
    }
    faces.erase(h->face);
    remove_tip(hret);
    vertices.erase(h->end_vertex);
    for (Halfedge* hit : h->end_vertex->halfedges) {
        hit->end_vertex->halfedges.erase(hit->opposite);
    }
    h->end_vertex->halfedges.clear();

    this->halfedges.erase(h);
    this->halfedges.erase(h->opposite);
    h->vertex->halfedges.erase(h);
    h->opposite->vertex->halfedges.erase(h->opposite);
    set_face_in_face_loop(hret, face);
    return hret;
}

void Mesh::set_face_in_face_loop(Halfedge* h, Face* f) const {
    f->halfedges.clear();
    f->vertices.clear();
    Halfedge* end = h;
    do {
        h->face = f;
        f->halfedges.insert(h);
        f->vertices.insert(h->vertex);
        h = h->next;
    } while (h != end);
}

void Mesh::remove_tip(Halfedge* h) const {
    h->next = h->next->opposite->next;
}

Halfedge* Mesh::join_face(Halfedge* h) {
    Halfedge* hprev = find_prev(h);
    Halfedge* gprev = find_prev(h->opposite);
    remove_tip(hprev);
    remove_tip(gprev);
    // hds->edges_erase(h);
    // halfedges.erase(h);
    // this->halfedges.erase(h);
    // this->halfedges.erase(h->opposite);
    h->opposite->setRemoved();

    h->vertex->halfedges.erase(h);
    h->opposite->vertex->halfedges.erase(h->opposite);
    // if (gprev->face != h->face)
    this->faces.erase(gprev->face);
    delete gprev->face;
    hprev->face->reset(hprev);
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
        this->vertices.insert(vt);
        vertices.push_back(vt);
    }

    for (int i = 0; i < nb_faces; ++i) {
        int num_face_vertices;
        file >> num_face_vertices;
        std::vector<Vertex*> vts;
        std::vector<int> idxs;
        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index;
            file >> vertex_index;
            vts.push_back(vertices[vertex_index]);
            idxs.push_back(vertex_index);
        }
        this->add_face(vts);
        this->face_index.push_back(idxs);
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
        mesh.vertices.insert(vt);
        vertices.push_back(vt);
    }

    // write face into mesh
    for (int i = 0; i < mesh.nb_faces; ++i) {
        int num_face_vertices;
        input >> num_face_vertices;
        // std::vector<Face*> faces;
        std::vector<Vertex*> vts;
        std::vector<int> idxs;
        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index;
            input >> vertex_index;
            vts.push_back(vertices[vertex_index]);
            idxs.push_back(vertex_index);
        }
        Face* face = mesh.add_face(vts);
        for (Halfedge* halfedge : face->halfedges) {
            mesh.halfedges.insert(halfedge);
        }
        mesh.face_index.push_back(idxs);
    }
    // clear vector
    vertices.clear();
    return input;
}

void Mesh::dumpto(std::string path) {
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

    for (Face* face : this->faces) {
        offFile << face->vertices.size() << " ";
        Halfedge* hst = *face->halfedges.begin();
        Halfedge* hed = *face->halfedges.begin();
        do {
            offFile << hst->vertex->getId() << " ";
            hst = hst->next;
        } while (hst != hed);
        offFile << "\n";
    }

    offFile.close();
}

}  // namespace MCGAL