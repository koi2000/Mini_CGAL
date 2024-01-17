#include "himesh.h"
#include "util.h"
void HiMesh::decode(int lod) {
    assert(lod >= 0 && lod <= 100);
    // assert(!this->is_compression_mode());
    if (lod < i_decompPercentage) {
        return;
    }
    i_decompPercentage = lod;
    b_jobCompleted = false;
    while (!b_jobCompleted) {
        startNextDecompresssionOp();
    }
}

void HiMesh::startNextDecompresssionOp() {
    // check if the target LOD is reached
    if (i_curDecimationId * 100.0 / i_nbDecimations >= i_decompPercentage) {
        if (i_curDecimationId == i_nbDecimations) {}
        b_jobCompleted = true;
        return;
    }
    halfedges.clear();
    halfedges.shrink_to_fit();
    halfedges.reserve(size_of_facets() * 4);
    // 1. reset the states. note that the states of the vertices need not to be reset
    for (auto fit = faces.begin(); fit != faces.end();) {
        if ((*fit)->isRemoved()) {
            fit = faces.erase(fit);
        } else {
            (*fit)->resetState();
            for (MCGAL::Halfedge* hit : (*fit)->halfedges) {
                hit->resetState();
                halfedges.push_back(hit);
            }
            fit++;
        }
    }

    i_curDecimationId++;  // increment the current decimation operation id.
    // 2. decoding the removed vertices and add to target facets
    struct timeval start = get_cur_time();
    RemovedVerticesDecodingStep();
    logt("%d RemovedVerticesDecodingStep", start, i_curDecimationId);
    // 3. decoding the inserted edge and marking the ones added
    InsertedEdgeDecodingStep();
    logt("%d InsertedEdgeDecodingStep", start, i_curDecimationId);
    for (auto fit = faces.begin(); fit != faces.end();) {
        if ((*fit)->isRemoved()) {
            fit = faces.erase(fit);
        } else {
            (*fit)->setUnProcessed();
            for (MCGAL::Halfedge* hit : (*fit)->halfedges) {
                hit->setUnProcessed();
                // halfedges.push_back(hit);
            }
            fit++;
        }
    }
    // 4. truly insert the removed vertices
    insertRemovedVertices();
    logt("%d insertRemovedVertices", start, i_curDecimationId);
    // 5. truly remove the added edges
    removeInsertedEdges();
    logt("%d removeInsertedEdges", start, i_curDecimationId);
}

void HiMesh::readBaseMesh() {
    // read the number of level of detail
    i_nbDecimations = readuInt16();
    // set the mesh bounding box
    unsigned i_nbVerticesBaseMesh = readInt();
    unsigned i_nbFacesBaseMesh = readInt();

    std::deque<MCGAL::Point>* p_pointDeque = new std::deque<MCGAL::Point>();
    std::deque<uint32_t*>* p_faceDeque = new std::deque<uint32_t*>();
    // Read the vertex positions.
    for (unsigned i = 0; i < i_nbVerticesBaseMesh; ++i) {
        MCGAL::Point pos = readPoint();
        p_pointDeque->push_back(pos);
    }
    // Read the face vertex indices.
    for (unsigned i = 0; i < i_nbFacesBaseMesh; ++i) {
        int nv = readInt();
        uint32_t* f = new uint32_t[nv + 1];
        // Write in the first cell of the array the face degree.
        f[0] = nv;
        for (unsigned j = 1; j < nv + 1; ++j) {
            f[j] = readInt();
        }
        p_faceDeque->push_back(f);
    }
    // Let the builder do its job.
    buildFromBuffer(p_pointDeque, p_faceDeque);

    // Free the memory.
    for (unsigned i = 0; i < p_faceDeque->size(); ++i) {
        delete[] p_faceDeque->at(i);
    }
    delete p_faceDeque;
    delete p_pointDeque;
}

void HiMesh::buildFromBuffer(std::deque<MCGAL::Point>* p_pointDeque, std::deque<uint32_t*>* p_faceDeque) {
    this->vertices.clear();
    // this->halfedges.clear();
    // used to create faces
    std::vector<MCGAL::Vertex*> vertices;
    // add vertex to Mesh
    for (std::size_t i = 0; i < p_pointDeque->size(); ++i) {
        float x, y, z;
        MCGAL::Point p = p_pointDeque->at(i);
        MCGAL::Vertex* vt = allocateVertexFromPool(p);
        vt->setId(i);
        this->vertices.push_back(vt);
        vertices.push_back(vt);
    }
    this->vh_departureConquest[0] = vertices[0];
    this->vh_departureConquest[1] = vertices[1];
    // read face and add to Mesh
    for (int i = 0; i < p_faceDeque->size(); ++i) {
        uint32_t* ptr = p_faceDeque->at(i);
        int num_face_vertices = ptr[0];
        std::vector<MCGAL::Vertex*> vts;
        for (int j = 0; j < num_face_vertices; ++j) {
            int vertex_index = ptr[j + 1];
            vts.push_back(vertices[vertex_index]);
        }
        MCGAL::Facet* face = allocateFaceFromPool(vts, this);
        this->add_face(face);
        // this->faces
    }
    // clear vector
    vertices.clear();
}

bool cmpForder(MCGAL::Facet* f1, MCGAL::Facet* f2) {
    // for (int i = 31; i >= 0; --i) {
    //     uint8_t a_part = (uint8_t)(f1->forder >> (i * 4)) & 0xF;
    //     uint8_t b_part = (uint8_t)(f2->forder >> (i * 4)) & 0xF;
    //     if (a_part == 0 && b_part != 0) {
    //         return true;
    //     } else if (a_part != 0 && b_part == 0) {
    //         return false;
    //     } else if (a_part == 0 && b_part == 0) {
    //         return f1->forder < f2->forder;
    //     }
    // }
    return f1->forder < f2->forder;
}

bool cmpforder(uint64_t f1, uint64_t f2) {
    for (int i = 31; i >= 0; --i) {
        uint8_t a_part = (uint8_t)(f1 >> (i * 4)) & 0xF;
        uint8_t b_part = (uint8_t)(f2 >> (i * 4)) & 0xF;
        if (a_part == 0 && b_part != 0) {
            return true;
        } else if (a_part != 0 && b_part == 0) {
            return false;
        } else if (a_part == 0 && b_part == 0) {
            return f1 < f2;
        }
    }
    return 0;
}

bool cmpHorder(MCGAL::Halfedge* f1, MCGAL::Halfedge* f2) {
    // for (int i = 31; i >= 0; --i) {
    //     uint8_t a_part = (uint8_t)(f1->horder >> (i * 4)) & 0xF;
    //     uint8_t b_part = (uint8_t)(f2->horder >> (i * 4)) & 0xF;
    //     if (a_part == 0 && b_part != 0) {
    //         return true;
    //     } else if (a_part != 0 && b_part == 0) {
    //         return false;
    //     } else if (a_part == 0 && b_part == 0) {
    //         return f1->horder < f2->horder;
    //     }
    // }
    return f1->horder < f2->horder;
}

bool cmphorder(uint64_t f1, uint64_t f2) {
    for (int i = 31; i >= 0; --i) {
        uint8_t a_part = (uint8_t)(f1 >> (i * 4)) & 0xF;
        uint8_t b_part = (uint8_t)(f2 >> (i * 4)) & 0xF;
        if (a_part == 0 && b_part != 0) {
            return true;
        } else if (a_part != 0 && b_part == 0) {
            return false;
        } else if (a_part == 0 && b_part == 0) {
            return f1 < f2;
        }
    }
    return 0;
}

// bool cmpForder(MCGAL::Facet* f1, MCGAL::Facet* f2) {
//     for (int i = 41; i >= 0; --i) {
//         uint8_t a_part = (uint8_t)(f1->forder >> (i * 3)) & 0x7;
//         uint8_t b_part = (uint8_t)(f2->forder >> (i * 3)) & 0x7;
//         if (a_part == 0 && b_part != 0) {
//             return true;
//         } else if (a_part != 0 && b_part == 0) {
//             return false;
//         } else if (a_part == 0 && b_part == 0) {
//             return f1->forder < f2->forder;
//         }
//     }
//     return 0;
// }

// bool cmpforder(uint64_t f1, uint64_t f2) {
//     for (int i = 41; i >= 0; --i) {
//         uint8_t a_part = (uint8_t)(f1 >> (i * 3)) & 0x7;
//         uint8_t b_part = (uint8_t)(f2 >> (i * 3)) & 0x7;
//         if (a_part == 0 && b_part != 0) {
//             return true;
//         } else if (a_part != 0 && b_part == 0) {
//             return false;
//         } else if (a_part == 0 && b_part == 0) {
//             return f1 < f2;
//         }
//     }
//     return 0;
// }

// bool cmpHorder(MCGAL::Halfedge* f1, MCGAL::Halfedge* f2) {
//     for (int i = 41; i >= 0; --i) {
//         uint8_t a_part = (uint8_t)(f1->horder >> (i * 3)) & 0x7;
//         uint8_t b_part = (uint8_t)(f2->horder >> (i * 3)) & 0x7;
//         if (a_part == 0 && b_part != 0) {
//             return true;
//         } else if (a_part != 0 && b_part == 0) {
//             return false;
//         } else if (a_part == 0 && b_part == 0) {
//             return f1->horder < f2->horder;
//         }
//     }
//     return 0;
// }

// bool cmphorder(uint64_t f1, uint64_t f2) {
//     for (int i = 41; i >= 0; --i) {
//         uint8_t a_part = (uint8_t)(f1 >> (i * 3)) & 0x7;
//         uint8_t b_part = (uint8_t)(f2 >> (i * 3)) & 0x7;
//         if (a_part == 0 && b_part != 0) {
//             return true;
//         } else if (a_part != 0 && b_part == 0) {
//             return false;
//         } else if (a_part == 0 && b_part == 0) {
//             return f1 < f2;
//         }
//     }
//     return 0;
// }

// void HiMesh::RemovedVerticesDecodingStep() {
//     //
//     pushHehInit();
//     std::ofstream offFile("./decode.txt");
//     while (!gateQueue.empty()) {
//         MCGAL::Halfedge* h = gateQueue.front();
//         gateQueue.pop();
//         MCGAL::Facet* f = h->face;
//         // If the face is already processed, pick the next halfedge:
//         if (f->isConquered())
//             continue;
//         std::vector<float> fts;
//         for (int j = 0; j < f->vertices.size(); j++) {
//             fts.push_back(f->vertices[j]->x());
//             fts.push_back(f->vertices[j]->y());
//             fts.push_back(f->vertices[j]->z());
//         }
//         sort(fts.begin(), fts.end());
//         for (int i = 0; i < fts.size(); i++) {
//             offFile << fts[i] << " ";
//         }
//         offFile << "\n";
//         // Add the other halfedges to the queue
//         MCGAL::Halfedge* hIt = h;
//         do {
//             MCGAL::Halfedge* hOpp = hIt->opposite;
//             // TODO: wait
//             // assert(!hOpp->is_border());
//             if (!hOpp->face->isConquered())
//                 gateQueue.push(hOpp);
//             hIt = hIt->next;
//         } while (hIt != h);
//         // Decode the face symbol.
//         unsigned sym = readChar();
//         if (sym == 1) {
//             MCGAL::Point rmved = readPoint();
//             f->setSplittable();
//             f->setRemovedVertexPos(rmved);
//         } else {
//             f->setUnsplittable();
//         }
//     }
// }

void HiMesh::RemovedVerticesDecodingStep() {
    int size = size_of_facets();
    int* firstQueue = new int[size];
    int* secondQueue = new int[size];
    int currentQueueSize = 1;
    int nextQueueSize = 0;
    int level = 0;
    char path[256];
    sprintf(path, "./RemovedVerticesDecodingStep2%d.txt", i_curDecimationId);
    std::ofstream offFile(path);

    MCGAL::Halfedge* hehBegin;
    for (int i = 0; i < vh_departureConquest[1]->halfedges.size(); i++) {
        MCGAL::Halfedge* hit = vh_departureConquest[1]->halfedges[i];
        if (hit->opposite->vertex == vh_departureConquest[0]) {
            hehBegin = hit->opposite;
            break;
        }
    }
    hehBegin->face->forder = 0;
    int threshold = 64 / 4 - 1;
    int firstCount = 0;
    int secondCount = 0;
    firstQueue[0] = hehBegin->poolId;
    while (currentQueueSize > 0) {
        int* currentQueue;
        int* nextQueue;
        if (level % 2 == 0) {
            currentQueue = firstQueue;
            nextQueue = secondQueue;
        } else {
            currentQueue = secondQueue;
            nextQueue = firstQueue;
        }
        // #pragma omp parallel for num_threads(128) schedule(dynamic)
        for (int i = 0; i < currentQueueSize; i++) {
            int current = currentQueue[i];
            MCGAL::Halfedge* h = getHalfedgeFromPool(current);
            MCGAL::Facet* f = h->face;
            if (f->isProcessed()) {
                continue;
            }
            MCGAL::Halfedge* hIt = h;
            uint64_t idx = 1;
            do {
                MCGAL::Halfedge* hOpp = hIt->opposite;
                uint64_t order = f->forder << 4 | idx;
                if (hOpp->face->forder == (~(uint64_t)0)) {
#pragma omp atomic write
                    secondCount = secondCount + 1;
                }

                if (order < hOpp->face->forder) {
                    // 仅能在队列里保留一个

                    // if (hOpp->face->forder != (~(uint64_t)0) && !hOpp->face->isProcessed()) {
                    //     nextQueue[hOpp->face->indexInQueue] = hOpp->poolId;
                    //     idx++;
                    //     hOpp->face->forder = order;
                    //     hIt = hIt->next;
                    //     continue;
                    // }

                    hOpp->face->forder = order;
                }
                if (hOpp->face->forder == order && !hOpp->face->isProcessed()) {
                    idx++;
                    int position;
                    // 只应该保留order低的那条边，不应该两个都进队列
                    // #pragma omp atomic read
                    position = nextQueueSize;
                    // #pragma omp atomic write
                    if (hOpp->face->indexInQueue != -1) {
                        nextQueue[hOpp->face->indexInQueue] = hOpp->poolId;
                    } else {
                        nextQueueSize = nextQueueSize + 1;
                        hOpp->face->indexInQueue = position;
                        // #pragma omp atomic write
                        nextQueue[position] = hOpp->poolId;
                    }
                    // nextQueueSize = nextQueueSize + 1;
                    // hOpp->face->indexInQueue = position;
                    // // #pragma omp atomic write
                    // nextQueue[position] = hOpp->poolId;
                }
                hIt = hIt->next;
            } while (hIt != h);
        }
        ++level;
        // 到达阈值后开始compact
        if (level == threshold) {
            // sort(faces.begin() + firstCount, faces.begin() + secondCount, cmpForder);
            sort(faces.begin(), faces.end(), cmpForder);
            firstCount = secondCount;
            for (int i = 0; i < secondCount + 1; i++) {
                faces[i]->forder = i;
            }
            int power = 1;
            int x = secondCount + 1;
            while (x > 1) {
                x /= 2;
                power++;
            }
            threshold += (64 - power) / 4 - 1;
        }

        // #pragma omp parallel num_threads(128)
        for (int i = 0; i < currentQueueSize; i++) {
            MCGAL::Halfedge* h = getHalfedgeFromPool(currentQueue[i]);
            h->face->indexInQueue = -1;
            h->face->setProcessedFlag();
        }

        currentQueueSize = nextQueueSize;
        nextQueueSize = 0;
    }
    // sort
    sort(faces.begin(), faces.end(), cmpForder);

    for (int i = 0; i < faces.size(); i++) {
        std::vector<float> fts;
        for (int j = 0; j < faces[i]->vertices.size(); j++) {
            fts.push_back(faces[i]->vertices[j]->x());
            fts.push_back(faces[i]->vertices[j]->y());
            fts.push_back(faces[i]->vertices[j]->z());
        }
        sort(fts.begin(), fts.end());
        for (int j = 0; j < fts.size(); j++) {
            offFile << fts[j] << " ";
        }
        offFile << "\n";
    }

    std::vector<int> offsets(faces.size());
// 并行读取
#pragma omp parallel for num_threads(128) schedule(dynamic)
    for (int i = 0; i < faces.size(); i++) {
        char symbol = readCharByOffset(dataOffset + i);
        if (symbol) {
            faces[i]->setSplittable();
            offsets[i] = 1;
        } else {
            faces[i]->setUnsplittable();
        }
    }

    // scan
    for (int i = 1; i < size; i++) {
        offsets[i] += offsets[i - 1];
    }

    dataOffset += faces.size();
#pragma omp parallel for num_threads(128) schedule(dynamic)
    for (int i = 0; i < size; i++) {
        if (faces[i]->isSplittable()) {
            MCGAL::Point p = readPointByOffset(dataOffset + (offsets[i] - 1) * sizeof(float) * 3);
            faces[i]->setRemovedVertexPos(p);
        }
    }
    dataOffset += *offsets.rbegin() * 3 * sizeof(float);
    delete firstQueue;
    delete secondQueue;
}

// void HiMesh::RemovedVerticesDecodingStep() {
//     int size = size_of_facets();
//     int* firstQueue = new int[size];
//     int* secondQueue = new int[size];
//     int currentQueueSize = 1;
//     int nextQueueSize = 0;
//     int level = 0;
//     char path[256];
//     sprintf(path, "./RemovedVerticesDecodingStep%d.txt", i_curDecimationId);
//     std::ofstream offFile(path);
//     MCGAL::Halfedge* hehBegin;
//     for (int i = 0; i < vh_departureConquest[1]->halfedges.size(); i++) {
//         MCGAL::Halfedge* hit = vh_departureConquest[1]->halfedges[i];
//         if (hit->opposite->vertex == vh_departureConquest[0]) {
//             hehBegin = hit->opposite;
//             break;
//         }
//     }
//     hehBegin->face->forder = 0;
//     int threshold = 64 / 3 - 1;
//     int firstCount = 0;
//     int secondCount = 0;
//     firstQueue[0] = hehBegin->poolId;
//     while (currentQueueSize > 0) {
//         int* currentQueue;
//         int* nextQueue;
//         if (level % 2 == 0) {
//             currentQueue = firstQueue;
//             nextQueue = secondQueue;
//         } else {
//             currentQueue = secondQueue;
//             nextQueue = firstQueue;
//         }
// #pragma omp parallel num_threads(128) schedule(dynamic)
//         for (int i = 0; i < currentQueueSize; i++) {
//             int current = currentQueue[i];
//             MCGAL::Halfedge* h = getHalfedgeFromPool(current);
//             MCGAL::Facet* f = h->face;
//             if (f->isProcessed()) {
//                 continue;
//             }
//             MCGAL::Halfedge* hIt = h;
//             uint64_t idx = 1;
//             do {
//                 MCGAL::Halfedge* hOpp = hIt->opposite;
//                 // uint64_t order = f->forder | (idx << ((41 - level) * 3));
//                 uint64_t order = f->forder << 3 | idx;
// #pragma omp atomic compare
//                 if (order < hOpp->face->forder) {
//                     if (hOpp->face->forder == (~(uint64_t)0)) {
// #pragma omp atomic
//                         secondCount++;
//                     }
//                     hOpp->face->forder = order;
//                 }
//                 if (hOpp->face->forder == order && !hOpp->face->isProcessed()) {
//                     idx++;
//                     int position = nextQueueSize;
// #pragma omp atomic
//                     nextQueueSize++;
//                     nextQueue[position] = hOpp->poolId;
//                 }
//                 hIt = hIt->next;
//             } while (hIt != h);
//         }
//         ++level;
//         // 到达阈值后开始compact
//         if (level == threshold) {
//             sort(faces.begin() + firstCount, faces.begin() + secondCount, cmpForder);
//             firstCount = secondCount;
//             for (int i = 0; i < secondCount; i++) {
//                 faces[i]->forder = i;
//             }
//             int power = 0;
//             int x = secondCount;
//             while (x > 1) {
//                 x /= 2;
//                 power++;
//             }
//             threshold += (64 - power) / 3;
//         }
//         // #pragma omp parallel num_threads(128)
//         for (int i = 0; i < currentQueueSize; i++) {
//             MCGAL::Halfedge* h = getHalfedgeFromPool(currentQueue[i]);
//             h->face->setProcessedFlag();
//         }
//         currentQueueSize = nextQueueSize;
//         nextQueueSize = 0;
//     }
//     // sort
//     sort(faces.begin(), faces.end(), cmpForder);
//     for (int i = 0; i < faces.size(); i++) {
//         offFile << faces[i]->poolId << " ";
//     }
//     std::vector<int> offsets(faces.size());
//     // 并行读取
// #pragma omp parallel num_threads(128) schedule(dynamic)
//     for (int i = 0; i < faces.size(); i++) {
//         char symbol = readCharByOffset(dataOffset + i);
//         if (symbol) {
//             faces[i]->setSplittable();
//             offsets[i] = 1;
//         } else {
//             faces[i]->setUnsplittable();
//         }
//     }
//     // scan
//     for (int i = 1; i < size; i++) {
//         offsets[i] += offsets[i - 1];
//     }
//     dataOffset += faces.size();
// #pragma omp parallel num_threads(128) schedule(dynamic)
//     for (int i = 0; i < size; i++) {
//         if (faces[i]->isSplittable()) {
//             MCGAL::Point p = readPointByOffset(dataOffset + (offsets[i] - 1) * sizeof(float) * 3);
//             faces[i]->setRemovedVertexPos(p);
//         }
//     }
//     dataOffset += *offsets.rbegin() * 3 * sizeof(float);
//     delete firstQueue;
//     delete secondQueue;
// }

/**
 * One step of the inserted edge coding conquest.
 */
void HiMesh::InsertedEdgeDecodingStep() {
    int size = halfedges.size();
    int* firstQueue = new int[size];
    int* secondQueue = new int[size];
    int currentQueueSize = 1;
    int nextQueueSize = 0;
    int level = 0;
    int threshold = 64 / 4 - 1;
    int firstCount = 0;
    int secondCount = 0;
    MCGAL::Halfedge* hehBegin;
    for (int i = 0; i < vh_departureConquest[1]->halfedges.size(); i++) {
        MCGAL::Halfedge* hit = vh_departureConquest[1]->halfedges[i];
        if (hit->opposite->vertex == vh_departureConquest[0]) {
            hehBegin = hit->opposite;
            break;
        }
    }
    hehBegin->horder = 0;
    firstQueue[0] = hehBegin->poolId;
    while (currentQueueSize > 0) {
        int* currentQueue;
        int* nextQueue;
        if (level % 2 == 0) {
            currentQueue = firstQueue;
            nextQueue = secondQueue;
        } else {
            currentQueue = secondQueue;
            nextQueue = firstQueue;
        }
        // #pragma omp parallel for num_threads(128)
        for (int i = 0; i < currentQueueSize; i++) {
            int current = currentQueue[i];
            MCGAL::Halfedge* h = getHalfedgeFromPool(current);
            if (h->isProcessed()) {
                continue;
            }
            MCGAL::Halfedge* hIt = h->next;
            uint64_t idx = 1;
            while (hIt->opposite != h) {
                uint64_t order = h->horder << 4 | idx;
                if (hIt->horder == (~(uint64_t)0)) {
#pragma omp atomic write
                    secondCount = secondCount + 1;
                }
#pragma omp critical
                {
                    if (order < hIt->horder) {
                        hIt->horder = order;
                    }
                }
#pragma omp critical
                {
                    if (hIt->horder == order) {
                        idx++;
                        int position;
                        // #pragma omp atomic read
                        position = nextQueueSize;
                        // #pragma omp atomic write
                        nextQueueSize = nextQueueSize + 1;
                        // #pragma omp atomic write
                        nextQueue[position] = hIt->poolId;
                    }
                    hIt = hIt->opposite->next;
                };
            }
        }
        ++level;
        // 到达阈值后开始compact
        if (level == threshold) {
            // sort(halfedges.begin() + firstCount, halfedges.begin() + secondCount, cmpHorder);
            sort(halfedges.begin(), halfedges.end(), cmpHorder);
            firstCount = secondCount;
            for (int i = 0; i < secondCount + 1; i++) {
                halfedges[i]->horder = i;
            }
            int power = 1;
            int x = secondCount + 1;
            while (x > 1) {
                x /= 2;
                power++;
            }
            threshold += (64 - power) / 4 - 1;
        }
        // offFile << "\n";
        currentQueueSize = nextQueueSize;
        nextQueueSize = 0;
    }
    sort(halfedges.begin(), halfedges.end(), cmpHorder);
// 并行读取
#pragma omp parallel for num_threads(128)
    for (int i = 0; i < halfedges.size(); i++) {
        char symbol = readCharByOffset(dataOffset + i);
        if (symbol) {
            halfedges[i]->setAdded();
        }
    }
    dataOffset += halfedges.size();
    delete firstQueue;
    delete secondQueue;
}

/**
 * Insert center vertices.
 */
void HiMesh::insertRemovedVertices() {
    // Add the first halfedge to the queue.
    pushHehInit();
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        MCGAL::Facet* f = h->face;

        // If the face is already processed, pick the next halfedge:
        if (f->isProcessed())
            continue;

        // Mark the face as processed.
        f->setProcessedFlag();

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h;
        do {
            MCGAL::Halfedge* hOpp = hIt->opposite;
            // TODO: wait
            // assert(!hOpp->is_border());
            if (!hOpp->face->isProcessed())
                gateQueue.push(hOpp);
            hIt = hIt->next;
        } while (hIt != h);
        assert(!h->isNew());

        if (f->isSplittable()) {
            // Insert the vertex.
            MCGAL::Halfedge* hehNewVertex = create_center_vertex(h);
            hehNewVertex->vertex->setPoint(f->getRemovedVertexPos());

            // Mark all the created edges as new.
            MCGAL::Vertex* Hvc = hehNewVertex->vertex;
            for (MCGAL::Halfedge* hit : Hvc->halfedges) {
                hit->setNew();
                hit->opposite->setNew();
                hit->face->setProcessedFlag();
            }
        }
    }
}

/**
 * Remove all the marked edges
 */
void HiMesh::removeInsertedEdges() {
    for (auto fit = faces.begin(); fit != faces.end();) {
        if ((*fit)->isRemoved()) {
            fit = faces.erase(fit);
        } else {
            fit++;
        }
    }
    pushHehInit();
    while (!gateQueue.empty()) {
        MCGAL::Halfedge* h = gateQueue.front();
        gateQueue.pop();

        if (h->isVisited())
            continue;

        if (h->isRemoved()) {
            continue;
        }
        // Mark the face as processed.
        h->setVisited();

        // Add the other halfedges to the queue
        MCGAL::Halfedge* hIt = h;
        do {
            MCGAL::Halfedge* hOpp = hIt->opposite;
            // TODO: wait
            // assert(!hOpp->is_border());
            if (!hOpp->isVisited())
                gateQueue.push(hOpp);
            hIt = hIt->next;
        } while (hIt != h);

        if (hIt->isRemoved()) {
            hIt->setVisited();
            continue;
        }
        if (hIt->isAdded()) {
            join_face(hIt);
            hIt->setVisited();
        }
    }
    return;
}