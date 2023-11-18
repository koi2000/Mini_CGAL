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
        assert(p->halfedges.size() == (int)0 && p->opposite_half_edges.size() == 0);
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

// bool Mesh::loadOFF(char* data, bool owned = false) {}

bool Mesh::loadOFF(std::string path) {
    std::ifstream fp(path);
    if (!fp.is_open()) {
        std::cerr << "Error: Unable to open file " << path << std::endl;
        return false;
    }

    std::stringstream file;
    file << fp.rdbuf();  // Read the entire file content into a stringstream

    // std:: file(path);
    // if (!file.is_open()) {
    //     std::cerr << "Error: Unable to open file " << path << std::endl;
    //     return false;
    // }
    std::string format;
    // 读取 OFF 文件格式信息
    file >> format >> nb_vertices >> nb_faces >> nb_edges;

    if (format != "OFF") {
        std::cerr << "Error: Invalid OFF file format" << std::endl;
        return false;
    }

    // 辅助数组，用于创建faces
    std::vector<Vertex*> vertices;
    // 读取顶点并添加到 Mesh
    for (std::size_t i = 0; i < nb_vertices; ++i) {
        float x, y, z;
        file >> x >> y >> z;
        Vertex* vt = new Vertex(x, y, z);
        vt->setVid(i);
        this->vertices.insert(vt);
        vertices.push_back(vt);
    }

    // 读取面信息并添加到 Mesh
    for (int i = 0; i < nb_faces; ++i) {
        int num_face_vertices;
        file >> num_face_vertices;
        // std::vector<Face*> faces;
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
    // 清空 vector
    vertices.clear();
    fp.close();
    return true;
}

std::istream& operator>>(std::istream& input, Mesh& mesh) {
    std::string format;
    // 读取 OFF 文件格式信息
    input >> format >> mesh.nb_vertices >> mesh.nb_faces >> mesh.nb_edges;

    if (format != "OFF") {
        std::cerr << "Error: Invalid OFF file format" << std::endl;
    }

    // 辅助数组，用于创建faces
    std::vector<Vertex*> vertices;
    // 读取顶点并添加到 Mesh
    for (std::size_t i = 0; i < mesh.nb_vertices; ++i) {
        float x, y, z;
        input >> x >> y >> z;
        Vertex* vt = new Vertex(x, y, z);
        vt->setVid(i);
        mesh.vertices.insert(vt);
        vertices.push_back(vt);
    }

    // 读取面信息并添加到 Mesh
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
    // 清空 vector
    vertices.clear();
    return input;
}

void Mesh::dumpto(std::string path) {
    std::ofstream offFile(path);
    if (!offFile.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }
    // 写入 OFF 文件头部信息
    offFile << "OFF\n";
    offFile << this->vertices.size() << " " << this->faces.size() << " 0\n";
    // 写入顶点坐标
    for (Vertex* vertex : this->vertices) {
        offFile << vertex->x() << " " << vertex->y() << " " << vertex->z() << "\n";
    }

    // 写入面的顶点索引
    // for (int i = 0; i < this->face_index.size(); i++) {
    //     offFile << this->face_index[i].size() << " ";
    //     for (int idx : this->face_index[i]) {
    //         offFile << idx << "";
    //     }
    //     offFile << "\n";
    // }
    for (Face* face : this->faces) {
        offFile << face->vertices.size() << " ";
        for (Vertex* vertex : face->vertices) {
            offFile << vertex->getId() << " ";
        }
        offFile << "\n";
    }

    offFile.close();
}

}  // namespace MCGAL