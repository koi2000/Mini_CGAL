#include "core.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
namespace MCGAL {

Mesh::~Mesh() {
    for (Face* f : faces) {
        delete f;
    }
    for (Vertex* p : vertices) {
        assert(p->halfedges.size() == (int)0 && p->opposite_half_edges.size() == 0);
        delete p;
    }
    vertices.clear();
    faces.clear();
}

Face* Mesh::add_face(std::vector<Vertex*>& vs) {
    Face* f = new Face(vs);
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
        vt->setId(i);
        this->vertices.insert(vt);
        vertices.push_back(vt);
    }

    // 读取面信息并添加到 Mesh
    for (int i = 0; i < nb_faces; ++i) {
        int num_face_vertices;
        file >> num_face_vertices;
        // std::vector<Face*> faces;
        std::vector<Vertex*> vts;
        // std::vector<int> idxs;
        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index;
            file >> vertex_index;
            vts.push_back(vertices[vertex_index]);
            // idxs.push_back(vertex_index);
        }
        this->add_face(vts);
        // this->face_index.push_back(idxs);
    }
    // 清空 vector
    vertices.clear();
    fp.close();
    return true;
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
    for (int i = 0; i < this->face_index.size(); i++) {
        offFile << this->face_index[i].size() << " ";
        for (int idx : this->face_index[i]) {
            offFile << idx << "";
        }
        offFile << "\n";
    }
    offFile.close();
}

}  // namespace MCGAL