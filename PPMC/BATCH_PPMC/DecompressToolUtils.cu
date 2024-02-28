#include "DecompressTool.cuh"

void DeCompressTool::readBaseMesh(int meshId, int* offset) {
    // read the number of level of detail
    int i_nbDecimations = readuInt16(offset);
    nbDecimations[meshId] = i_nbDecimations;
    // set the mesh bounding box
    unsigned i_nbVerticesBaseMesh = readInt(offset);
    unsigned i_nbFacesBaseMesh = readInt(offset);

    std::deque<MCGAL::Point>* p_pointDeque = new std::deque<MCGAL::Point>();
    std::deque<uint32_t*>* p_faceDeque = new std::deque<uint32_t*>();
    // Read the vertex positions.
    for (unsigned i = 0; i < i_nbVerticesBaseMesh; ++i) {
        MCGAL::Point pos = readPoint(offset);
        p_pointDeque->push_back(pos);
    }
    // read the face vertex indices
    // Read the face vertex indices.
    for (unsigned i = 0; i < i_nbFacesBaseMesh; ++i) {
        int nv = readInt(offset);
        uint32_t* f = new uint32_t[nv + 1];
        // Write in the first cell of the array the face degree.
        f[0] = nv;
        for (unsigned j = 1; j < nv + 1; ++j) {
            f[j] = readInt(offset);
        }
        p_faceDeque->push_back(f);
    }
    // Let the builder do its job.
    buildFromBuffer(meshId, p_pointDeque, p_faceDeque);

    // Free the memory.
    for (unsigned i = 0; i < p_faceDeque->size(); ++i) {
        delete[] p_faceDeque->at(i);
    }
    delete p_faceDeque;
    delete p_pointDeque;
}

void DeCompressTool::buildFromBuffer(int meshId,
                                     std::deque<MCGAL::Point>* p_pointDeque,
                                     std::deque<uint32_t*>* p_faceDeque) {
    std::vector<MCGAL::Vertex*> vertices;
    // add vertex to Mesh
    for (std::size_t i = 0; i < p_pointDeque->size(); ++i) {
        MCGAL::Point p = p_pointDeque->at(i);
        MCGAL::Vertex* vt = MCGAL::contextPool.allocateVertexFromPool(p);
        vt->setMeshId(meshId);
        vertices.push_back(vt);
    }
    vh_departureConquest[meshId * 2] = vertices[0]->poolId;
    vh_departureConquest[meshId * 2 + 1] = vertices[1]->poolId;
    // read face and add to Mesh
    for (int i = 0; i < p_faceDeque->size(); ++i) {
        uint32_t* ptr = p_faceDeque->at(i);
        int num_face_vertices = ptr[0];
        std::vector<MCGAL::Vertex*> vts;
        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index = ptr[j + 1];
            vts.push_back(vertices[vertex_index]);
        }
        MCGAL::Facet* face = MCGAL::contextPool.allocateFaceFromPool(vts);
        face->setMeshId(meshId);
        for (int k = 0; k < face->halfedge_size; k++) {
            MCGAL::Halfedge* hit = face->getHalfedgeByIndex(k);
            hit->setMeshId(face->meshId);
        }
    }
    // clear vector
    vertices.clear();
}

void DeCompressTool::dumpto(std::string prefix) {
    // int vsize = *MCGAL::contextPool.vindex;
    // int hsize = *MCGAL::contextPool.hindex;
    // int fsize = *MCGAL::contextPool.findex;
    // CHECK(cudaMemcpy(MCGAL::contextPool.vpool, MCGAL::contextPool.dvpool, vsize * sizeof(MCGAL::Vertex),
    //                  cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(MCGAL::contextPool.hpool, MCGAL::contextPool.dhpool, hsize * sizeof(MCGAL::Halfedge),
    //                  cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(MCGAL::contextPool.fpool, MCGAL::contextPool.dfpool, fsize * sizeof(MCGAL::Facet),
    //                  cudaMemcpyDeviceToHost));

    std::vector<std::vector<MCGAL::Vertex*>> vertices(batch_size);
    std::vector<std::vector<MCGAL::Facet*>> facets(batch_size);
    int vindex = *MCGAL::contextPool.vindex;
    int findex = *MCGAL::contextPool.findex;
#pragma omp parallel for num_threads(60)
    for (int i = 0; i < vindex; i++) {
        MCGAL::Vertex* v = MCGAL::contextPool.getVertexByIndex(i);
        if (v->meshId != -1) {
            vertices[v->meshId].push_back(v);
        }
    }
#pragma omp parallel for num_threads(60)
    for (int i = 0; i < findex; i++) {
        MCGAL::Facet* f = MCGAL::contextPool.getFacetByIndex(i);
        if (f->meshId != -1) {
            if (f->isRemoved())
                continue;
            facets[f->meshId].push_back(f);
        }
    }
    char path[256];
#pragma omp parallel for num_threads(batch_size)
    for (int i = 0; i < batch_size; i++) {
        std::sprintf(path, prefix.c_str(), i);
        dumpto(vertices[i], facets[i], path);
    }
}

void DeCompressTool::dumpto(std::vector<MCGAL::Vertex*> vertices, std::vector<MCGAL::Facet*> facets, char* path) {
    std::ofstream offFile(path);
    if (!offFile.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }
    // write header
    offFile << "OFF\n";
    offFile << vertices.size() << " " << facets.size() << " 0\n";
    offFile << "\n";
    // write vertex
    int id = 0;
    for (MCGAL::Vertex* vertex : vertices) {
        offFile << vertex->x() << " " << vertex->y() << " " << vertex->z() << "\n";
        vertex->setId(id++);
    }

    for (MCGAL::Facet* face : facets) {
        if (face->isRemoved())
            continue;
        offFile << face->vertex_size << " ";
        MCGAL::Halfedge* hst = MCGAL::contextPool.getHalfedgeByIndex(face->halfedges[0]);
        MCGAL::Halfedge* hed = hst;

        do {
            offFile << hst->vertex()->getId() << " ";
            hst = hst->next();
        } while (hst != hed);
        offFile << "\n";
    }

    offFile.close();
}
