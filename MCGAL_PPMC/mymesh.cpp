//
// Created by DELL on 2023/11/9.
//
#include "mymesh.h"
#include "util.h"
// #include "configuration.h"
#include <algorithm>

int replacing_group::counter = 0;
int replacing_group::alive = 0;

MyMesh::MyMesh(string& str, bool completeop) : MCGAL::Mesh() {
    boost::replace_all(str, "|", "\n");
    assert(str.size() != 0 && "input string should not be empty!");
    struct timeval start = hispeed::get_cur_time();
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

    // TODO: 等待完成
    // if (keep_largest_connected_components(1) != 0) {
    //     std::cerr << "Can't compress the mesh." << std::endl;
    //     std::cerr << "The codec doesn't handle meshes with several connected components." << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // 等待完成
    // if (!is_closed()) {
    //     std::cerr << "Can't compress the mesh." << std::endl;
    //     std::cerr << "The codec doesn't handle meshes with borders." << std::endl;
    //     exit(EXIT_FAILURE);
    // }
    
    // Set the vertices of the edge that is the departure of the coding and decoding conquests.
    vh_departureConquest[0] = static_cast<MyVertex*>((*halfedges.begin())->vertex);
    vh_departureConquest[1] = static_cast<MyVertex*>((*halfedges.begin())->opposite->vertex);

    if (completeop) {
        encode(0);
    }
}

// in decompression mode
MyMesh::MyMesh(char* data, size_t dsize, bool owndata) : MCGAL::Mesh() {
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
    // Set the vertices of the edge that is the departure of the coding and decoding conquests.
    vh_departureConquest[0] = static_cast<MyVertex*>((*halfedges.begin())->vertex);
    vh_departureConquest[1] = static_cast<MyVertex*>((*halfedges.begin())->opposite->vertex);
}

MyMesh::~MyMesh() {
    if (own_data && p_data != NULL) {
        delete[] p_data;
    }
    // TODO:
    // clear_aabb_tree();
    for (replacing_group* rg : map_group) {
        delete rg;
    }
    map_group.clear();
}

void MyMesh::compute_mbb() {
    // mbb.reset();
    // for (Vertex_const_iterator vit = vertices_begin(); vit != vertices_end(); ++vit) {
    //     Point p = vit->point();
    //     mbb.update(p.x(), p.y(), p.z());
    // }
}

// TODO: 待完善
// size_t MyMesh::size_of_edges() {
//     return size_of_halfedges() / 2;
// }

// size_t MyMesh::size_of_triangles() {
//     size_t tri_num = 0;
//     for (Facet_const_iterator f = facets_begin(); f != facets_end(); ++f) {
//         tri_num += f->facet_degree() - 2;
//     }
//     return tri_num;
// }

// Polyhedron* MyMesh::to_triangulated_polyhedron() {
//     Polyhedron* poly = to_polyhedron();
//     CGAL::Polygon_mesh_processing::triangulate_faces(*poly);
//     return poly;
// }

// Polyhedron* MyMesh::to_polyhedron() {
//     stringstream ss;
//     ss << *this;
//     Polyhedron* poly = new Polyhedron();
//     ss >> *poly;
//     return poly;
// }

// TODO: 待完善
// MyMesh* MyMesh::clone_mesh() {
//     stringstream ss;
//     ss << *this;
//     string str = ss.str();
//     MyMesh* nmesh = new MyMesh(str);
//     return nmesh;
// }

// TODO: 待完善
// string MyMesh::to_off() {
//     std::stringstream os;
//     os << *this;
//     return os.str();
// }

// TODO: 待完善
// void MyMesh::write_to_off(const char* path) {
//     string ct = to_off();
//     hispeed::write_file(ct, path);
// }

// TODO: 待完善
// string MyMesh::to_wkt() {
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
// void MyMesh::write_to_wkt(const char* path) {
//     string ct = to_wkt();
//     hispeed::write_file(ct, path);
// }

