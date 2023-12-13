//
// Created by DELL on 2023/11/9.
//
#include "himesh.cuh"
// #include "../MCGAL/Core_CUDA/global.cuh"
#include "util.h"
#include <algorithm>

MyMesh::MyMesh(string& str, bool completeop) : MCGAL::Mesh() {
    boost::replace_all(str, "|", "\n");
    assert(str.size() != 0 && "input string should not be empty!");
    struct timeval start = get_cur_time();
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
    vh_departureConquest[0] = (*halfedges.begin())->vertex();
    vh_departureConquest[1] = (*halfedges.begin())->opposite()->vertex();
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
    MCGAL::contextPool.reset();
    readBaseMesh();
    CHECK(cudaMalloc(&dfaceIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dvertexIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dstFacetIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dstHalfedgeIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dedgeIndexes, REMOVEDEDGE_SIZE * sizeof(int)));
    // Set the vertices of the edge that is the departure of the coding and decoding conquests.
    // vh_departureConquest[0] = (*halfedges.begin())->vertex;
    // vh_departureConquest[1] = (*halfedges.begin())->end_vertex;
}

MyMesh::~MyMesh() {
    if (own_data && p_data != NULL) {
        delete[] p_data;
    }
    cudaFree(dfaceIndexes);
    cudaFree(dvertexIndexes);
    cudaFree(dstHalfedgeIndexes);
    cudaFree(dstFacetIndexes);
    cudaFree(dedgeIndexes);
    // TODO:
    // clear_aabb_tree();
}

MyMesh::MyMesh(char* path) : MCGAL::Mesh() {
    this->loadBuffer(path);
    // assert(dsize > 0);
    srand(PPMC_RANDOM_CONSTANT);
    readBaseMesh();
    CHECK(cudaMalloc(&dfaceIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dvertexIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dstFacetIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dstHalfedgeIndexes, SPLITABLE_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&dedgeIndexes, REMOVEDEDGE_SIZE * sizeof(int)));
    // Set the vertices of the edge that is the departure of the coding and decoding conquests.
}

void MyMesh::pushHehInit() {
    MCGAL::Halfedge* hehBegin;
    // MCGAL::Vertex* st = MCGAL::contextPool.getVertexByIndex(vh_departureConquest[1]->poolId);
    for (int i = 0; i < vh_departureConquest[1]->halfedges_size; i++) {
        MCGAL::Halfedge* hit = vh_departureConquest[1]->getHalfedgeByIndex(i);
        if (hit->opposite()->vertex() == vh_departureConquest[0]) {
            hehBegin = hit->opposite();
            break;
        }
    }
    assert(hehBegin->vertex() == vh_departureConquest[0]);
    // Push it to the queue.
    gateQueue.push(hehBegin);
}

MCGAL::Facet* MyMesh::add_face_by_pool(std::vector<MCGAL::Vertex*>& vs) {
    MCGAL::Facet* f = MCGAL::contextPool.allocateFaceFromPool(vs);
    // for (MCGAL::Halfedge* hit : f->halfedges) {
    //     this->halfedges.insert(hit);
    // }
    faces.push_back(f);
    return f;
}

bool MyMesh::willViolateManifold(const std::vector<MCGAL::Halfedge*>& polygon) const {
    unsigned i_degree = polygon.size();

    // Test if a patch vertex is not connected to one vertex
    // that is not one of its direct neighbor.
    // Test also that two vertices of the patch will not be doubly connected
    // after the vertex cut opeation.
    for (unsigned i = 0; i < i_degree; ++i) {
        for (int k = 0; k < polygon[i]->vertex()->halfedges_size; k++) {
            MCGAL::Halfedge* hvc = polygon[i]->vertex()->getHalfedgeByIndex(k);
            for (unsigned j = 0; j < i_degree; ++j) {
                MCGAL::Vertex* vh = hvc->opposite()->vertex();
                if (vh == polygon[j]->vertex()) {
                    unsigned i_prev = i == 0 ? i_degree - 1 : i - 1;
                    unsigned i_next = i == i_degree - 1 ? 0 : i + 1;

                    if ((j == i_prev &&
                         polygon[i]->facet()->facet_degree() != 3)  // The vertex cut operation is forbidden.
                        || (j == i_next && polygon[i]->opposite()->facet()->facet_degree() !=
                                               3))  // The vertex cut operation is forbidden.
                        return true;
                }
            }
        }
    }

    return false;
}

bool MyMesh::isRemovable(MCGAL::Vertex* v) const {
    //	if(size_of_vertices()<10){
    //		return false;
    //	}
    if (v != vh_departureConquest[0] && v != vh_departureConquest[1] && !v->isConquered() && v->vertex_degree() > 2 &&
        v->vertex_degree() <= 8) {
        // test convexity
        std::vector<MCGAL::Vertex*> vh_oneRing;
        std::vector<MCGAL::Halfedge*> heh_oneRing;

        vh_oneRing.reserve(v->vertex_degree());
        heh_oneRing.reserve(v->vertex_degree());
        for (int i = 0; i < v->halfedges_size; i++) {
            MCGAL::Halfedge* hit = v->getHalfedgeByIndex(i);
            vh_oneRing.push_back(hit->opposite()->vertex());
            heh_oneRing.push_back(hit->opposite());
        }
        bool removable = //!willViolateManifold(heh_oneRing)
                         // && isProtruding(heh_oneRing);
                         isConvex(vh_oneRing);
        return removable;
    }
    return false;
}

bool MyMesh::isConvex(const std::vector<MCGAL::Vertex*>& polygon) const {
    // 遍历所有点
    for (unsigned i = 0; i < polygon.size(); i++) {
        MCGAL::Vertex* vt = polygon[i];
        // 遍历这个点的所有半边
        for (unsigned j = 0; j < vt->halfedges_size; j++) {
            MCGAL::Halfedge* hit = vt->getHalfedgeByIndex(j);
            // 查看是否有半边指向了点集里的其他点
            for (unsigned k = 0; k < polygon.size(); k++) {
                if (hit->end_vertex() == polygon[k] && i != k && hit->facet()->facet_degree() != 3) {
                    return false;
                }
            }
        }
    }
    return true;
}

std::istream& operator>>(std::istream& input, MyMesh& mesh) {
    std::string format;
    input >> format >> mesh.nb_vertices >> mesh.nb_faces >> mesh.nb_edges;

    if (format != "OFF") {
        std::cerr << "Error: Invalid OFF file format" << std::endl;
    }
    std::vector<MCGAL::Vertex*> vertices;
    for (std::size_t i = 0; i < mesh.nb_vertices; ++i) {
        float x, y, z;
        input >> x >> y >> z;
        MCGAL::Vertex* vt = MCGAL::contextPool.allocateVertexFromPool({x, y, z});
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
        for (int k = 0; k < face->halfedge_size; k++) {
            mesh.halfedges.push_back(face->getHalfedgeByIndex(k));
        }
    }
    vertices.clear();
    return input;
}

void MyMesh::write_to_off(const char* path) {
    this->dumpto(path);
}

void MyMesh::dumpBuffer(char* path) {
    ofstream fout(path, ios::binary);
    int len = dataOffset;
    fout.write((char*)&len, sizeof(int));
    fout.write(p_data, len);
    fout.close();
}

void MyMesh::loadBuffer(char* path) {
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
