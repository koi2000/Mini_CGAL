//
// Created by DELL on 2023/11/9.
//
#include "himesh.h"
#include "util.h"
// #include "configuration.h"
#include <algorithm>

HiMesh::HiMesh(string& str, bool completeop) : MCGAL::Mesh() {
    boost::replace_all(str, "|", "\n");
    assert(str.size() != 0 && "input string should not be empty!");
    struct timeval start = get_cur_time();
    auto very_start = start;
    srand(PPMC_RANDOM_CONSTANT);
    i_mode = COMPRESSION_MODE_ID;
    const size_t d_capacity = 3 * str.size();
    p_data = new char[d_capacity];
    // Fill the buffer with 0
    std::istringstream is;
    is.str(str.c_str());
    is >> *this;
    if (size_of_facets() == 0) {
        std::cerr << "failed to parse the OFF file into Polyhedron" << endl;
        exit(EXIT_FAILURE);
    }

    // TODO: wait
    // if (keep_largest_connected_components(1) != 0) {
    //     std::cerr << "Can't compress the mesh." << std::endl;
    //     std::cerr << "The codec doesn't handle meshes with several connected components." << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // wait
    // if (!is_closed()) {
    //     std::cerr << "Can't compress the mesh." << std::endl;
    //     std::cerr << "The codec doesn't handle meshes with borders." << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // Set the vertices of the edge that is the departure of the coding and decoding conquests.
    vh_departureConquest[0] = (*halfedges.begin())->vertex;
    vh_departureConquest[1] = (*halfedges.begin())->opposite->vertex;

    if (completeop) {
        encode(0);
    }
}

// in decompression mode
HiMesh::HiMesh(char* data, size_t dsize, bool owndata) : MCGAL::Mesh() {
    assert(dsize > 0);
    srand(PPMC_RANDOM_CONSTANT);
    own_data = owndata;
    i_mode = DECOMPRESSION_MODE_ID;
    if (owndata) {
        p_data = new char[dsize];
        memcpy(p_data, data, dsize);
    } else {
        p_data = data;
    }
    readBaseMesh();
    if (dataOffset % 4 != 0) {
        dataOffset = (dataOffset / 4 + 1) * 4;
    }
    // Set the vertices of the edge that is the departure of the coding and decoding conquests.
    // vh_departureConquest[0] = (*halfedges.begin())->vertex;
    // vh_departureConquest[1] = (*halfedges.begin())->end_vertex;
}

HiMesh::~HiMesh() {
    if (own_data && p_data != NULL) {
        delete[] p_data;
    }
    // TODO:
    // clear_aabb_tree();
}

HiMesh::HiMesh(char* path) : MCGAL::Mesh() {
    this->loadBuffer(path);
    // assert(dsize > 0);
    srand(PPMC_RANDOM_CONSTANT);
    readBaseMesh();
    // Set the vertices of the edge that is the departure of the coding and decoding conquests.
    // vh_departureConquest[0] = vertices.;
    // vh_departureConquest[1] = (*halfedges.begin())->end_vertex;
}

void HiMesh::pushHehInit() {
    MCGAL::Halfedge* hehBegin;
    // std::unordered_set<MCGAL::Halfedge*> hset = vh_departureConquest[0]->halfedges;
    auto hit = vh_departureConquest[1]->halfedges.begin();
    auto hed = vh_departureConquest[1]->halfedges.end();
    for (; hit != hed; hit++) {
        hehBegin = (*hit)->opposite;
        if (hehBegin->vertex == vh_departureConquest[0])
            break;
    }
    assert(hehBegin->vertex == vh_departureConquest[0]);
    // Push it to the queue.
    gateQueue.push(hehBegin);
}

MCGAL::Facet* HiMesh::add_face_by_pool(std::vector<MCGAL::Vertex*>& vs) {
    MCGAL::Facet* f = allocateFaceFromPool(vs, this);
    // for (MCGAL::Halfedge* hit : f->halfedges) {
    //     this->halfedges.insert(hit);
    // }
    faces.push_back(f);
    return f;
}

// bool HiMesh::willViolateManifold(const std::vector<MCGAL::Halfedge*>& polygon) const {
//     unsigned i_degree = polygon.size();
//     // Test if a patch vertex is not connected to one vertex
//     // that is not one of its direct neighbor.
//     // Test also that two vertices of the patch will not be doubly connected
//     // after the vertex cut opeation.
//     for (unsigned i = 0; i < i_degree; ++i) {
//         MCGAL::Halfedge* Hvc = *polygon[i]->vertex->halfedges.begin();
//         MCGAL::Halfedge* Hvc_end = Hvc;
//         for (MCGAL::Halfedge* hvc : polygon[i]->vertex->halfedges) {
//             MCGAL::Vertex* vh = Hvc->opposite->vertex;
//             for (unsigned j = 0; j < i_degree; ++j) {
//                 if (vh == polygon[j]->vertex) {
//                     unsigned i_prev = i == 0 ? i_degree - 1 : i - 1;
//                     unsigned i_next = i == i_degree - 1 ? 0 : i + 1;
//                     if ((j == i_prev &&
//                          polygon[i]->face->facet_degree() != 3)  // The vertex cut operation is forbidden.
//                         || (j == i_next &&
//                             polygon[i]->opposite->face->facet_degree() != 3))  // The vertex cut operation is
//                             forbidden.
//                         return true;
//                 }
//             }
//         }
//     }
//     return false;
// }

// bool HiMesh::isConvex(const std::vector<MCGAL::Vertex*>& polygon) const {
//     // 遍历所有点
//     for (unsigned i = 0; i < polygon.size(); i++) {
//         MCGAL::Vertex* vt = polygon[i];
//         // 遍历这个点的所有半边
//         for (unsigned j = 0; j < vt->halfedges.size(); j++) {
//             MCGAL::Halfedge* hit = vt->halfedges[j];
//             // 查看是否有半边指向了点集里的其他点
//             for (unsigned k = 0; k < polygon.size(); k++) {
//                 if (hit->end_vertex == polygon[k] && i != k && hit->face->facet_degree() != 3) {
//                     return false;
//                 }
//             }
//         }
//     }
//     return true;
// }

bool HiMesh::isRemovable(MCGAL::Vertex* v) const {
    //	if(size_of_vertices()<10){
    //		return false;
    //	}
    if (v != vh_departureConquest[0] && v != vh_departureConquest[1] && !v->isConquered() && v->vertex_degree() > 2 &&
        v->vertex_degree() <= 8) {
        // test convexity
        std::vector<MCGAL::Vertex*> vh_oneRing;
        std::vector<MCGAL::Halfedge*> heh_oneRing;
        // vh_oneRing.reserve(v->vertex_degree());
        heh_oneRing.reserve(v->vertex_degree());
        for (MCGAL::Halfedge* hit : v->halfedges) {
            // vh_oneRing.push_back(hit->opposite->vertex);
            heh_oneRing.push_back(hit);
        }  // while (hit != end);
        //
        // bool removable = !willViolateManifold(heh_oneRing);
        bool removable = !willViolateManifold(heh_oneRing);
        return removable;
    }
    return false;
}

bool HiMesh::willViolateManifold(const std::vector<MCGAL::Halfedge*>& polygon) const {
    unsigned i_degree = polygon.size();
    for (unsigned i = 0; i < i_degree; ++i) {
        MCGAL::Halfedge* it = polygon[i];
        if (it->face->facet_degree() == 3) {
            continue;
        }
        for (int j = 0; j < i_degree; j++) {
            MCGAL::Halfedge* jt = polygon[j];
            if (i == j)
                continue;
            if (it->face == jt->opposite->face) {
                for (int k = 0; k < it->end_vertex->halfedges.size(); k++) {
                    if (it->end_vertex->halfedges[k]->end_vertex == jt->end_vertex) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void HiMesh::compute_mbb() {
    // mbb.reset();
    // for (Vertex_const_iterator vit = vertices_begin(); vit != vertices_end(); ++vit) {
    //     Point p = vit->point();
    //     mbb.update(p.x(), p.y(), p.z());
    // }
}

std::istream& operator>>(std::istream& input, HiMesh& mesh) {
    std::string format;
    input >> format >> mesh.nb_vertices >> mesh.nb_faces >> mesh.nb_edges;

    if (format != "OFF") {
        std::cerr << "Error: Invalid OFF file format" << std::endl;
    }
    std::vector<MCGAL::Vertex*> vertices;
    for (std::size_t i = 0; i < mesh.nb_vertices; ++i) {
        float x, y, z;
        input >> x >> y >> z;
        MCGAL::Vertex* vt = new MCGAL::Vertex(x, y, z);
        mesh.vertices.push_back(vt);
        vertices.push_back(vt);
    }
    for (int i = 0; i < mesh.nb_faces; ++i) {
        int num_face_vertices;
        input >> num_face_vertices;
        // std::vector<Facet*> faces;
        std::vector<MCGAL::Vertex*> vts;
        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index;
            input >> vertex_index;
            vts.push_back(vertices[vertex_index]);
        }
        MCGAL::Facet* face = mesh.add_face(vts);
        for (int k = 0; k < face->halfedges.size(); k++) {
            mesh.halfedges.push_back(face->halfedges[k]);
        }
    }
    vertices.clear();
    return input;
}

void HiMesh::write_to_off(const char* path) {
    this->dumpto(path);
}

void HiMesh::dumpBuffer(char* path) {
    ofstream fout(path, ios::binary);
    int len = dataOffset;
    fout.write((char*)&len, sizeof(int));
    fout.write(p_data, len);
    fout.close();
}

void HiMesh::loadBuffer(char* path) {
    ifstream fin(path, ios::binary);
    int len2;
    fin.read((char*)&len2, sizeof(int));
    dataOffset = 0;
    p_data = new char[len2];
    memset(p_data, 0, len2);
    // p_data = new char[BUFFER_SIZE];
    // memset(p_data, 0, BUFFER_SIZE);
    fin.read(p_data, len2);
}

// TODO: 待完善
// size_t HiMesh::size_of_edges() {
//     return size_of_halfedges() / 2;
// }

// size_t HiMesh::size_of_triangles() {
//     size_t tri_num = 0;
//     for (Facet_const_iterator f = facets_begin(); f != facets_end(); ++f) {
//         tri_num += f->facet_degree() - 2;
//     }
//     return tri_num;
// }

// Polyhedron* HiMesh::to_triangulated_polyhedron() {
//     Polyhedron* poly = to_polyhedron();
//     CGAL::Polygon_mesh_processing::triangulate_faces(*poly);
//     return poly;
// }

// Polyhedron* HiMesh::to_polyhedron() {
//     stringstream ss;
//     ss << *this;
//     Polyhedron* poly = new Polyhedron();
//     ss >> *poly;
//     return poly;
// }

// TODO: 待完善
// HiMesh* HiMesh::clone_mesh() {
//     stringstream ss;
//     ss << *this;
//     string str = ss.str();
//     HiMesh* nmesh = new HiMesh(str);
//     return nmesh;
// }

// TODO: 待完善
// string HiMesh::to_off() {
//     std::stringstream os;
//     os << *this;
//     return os.str();
// }

// TODO: 待完善
// void HiMesh::write_to_off(const char* path) {
//     string ct = to_off();
//     hispeed::write_file(ct, path);
// }

// TODO: 待完善
// string HiMesh::to_wkt() {
//     std::stringstream ss;
//     ss << "POLYHEDRALSURFACE Z (";
//     bool lfirst = true;
//     for (Facet_const_iterator fit = facets_begin(); fit != facets_end(); ++fit) {
//         if (lfirst) {
//             lfirst = false;
//         } else {
//             ss << ",";
//         }
//         ss << "((";
//         bool first = true;
//         Halfedge_around_facet_const_circulator hit(fit->facet_begin()), end(hit);
//         Point firstpoint;
//         do {
//             Point p = hit->vertex()->point();
//             if (!first) {
//                 ss << ",";
//             } else {
//                 firstpoint = p;
//             }
//             first = false;
//             ss << p[0] << " " << p[1] << " " << p[2];
//             // Write the current vertex id.
//         } while (++hit != end);
//         ss << "," << firstpoint[0] << " " << firstpoint[1] << " " << firstpoint[2];
//         ss << "))";
//     }
//     ss << ")";
//     return ss.str();
// }

// TODO: 待完善
// void HiMesh::write_to_wkt(const char* path) {
//     string ct = to_wkt();
//     hispeed::write_file(ct, path);
// }
