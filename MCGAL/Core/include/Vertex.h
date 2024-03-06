#ifndef VERTEX_H
#define VERTEX_H
#include <assert.h>
#include <stdexcept>
#include <stdint.h>
#include <stdlib.h>
#include "Point.h"
#include <vector>
namespace MCGAL {

class Halfedge;
class Facet;
class Vertex;

class Vertex : public Point {
    enum Flag { Unconquered = 0, Conquered = 1 };
    Flag flag = Unconquered;
    unsigned int id = 0;

  public:
    Vertex() : Point() {
        // halfedges.reserve(BUCKET_SIZE);
    }
    Vertex(const Point& p) : Point(p) {
        // halfedges.reserve(BUCKET_SIZE);
    }
    Vertex(float v1, float v2, float v3) : Point(v1, v2, v3) {
        // halfedges.reserve(BUCKET_SIZE);
    }

    int vid_ = 0;
    int poolId = -1;
    int meshId = -1;
    std::vector<Halfedge*> halfedges;
    // std::set<Halfedge*> opposite_half_edges;
    // std::unordered_set<Halfedge*> halfedges;
    // std::unordered_set<Halfedge*> opposite_half_edges;

    void setMeshId(int meshId) {
        this->meshId = meshId;
    }

    int vertex_degree() {
        return halfedges.size();
    }

    void print() {
        printf("%f %f %f\n", v[0], v[1], v[2]);
    }

    void setVid(int id) {
        this->vid_ = id;
    }

    float x() const {
        return v[0];
    }

    float y() const {
        return v[1];
    }

    float z() const {
        return v[2];
    }

    int vid() const {
        return vid_;
    }

    Point point() {
        return Point(this);
    }

    void setPoint(const Point& p) {
        this->v[0] = p.x();
        this->v[1] = p.y();
        this->v[2] = p.z();
    }

    inline void resetState() {
        flag = Unconquered;
    }

    inline bool isConquered() const {
        return flag == Conquered;
    }

    inline void setConquered() {
        flag = Conquered;
    }

    inline size_t getId() const {
        return id;
    }

    inline void setId(size_t nId) {
        id = nId;
    }

    void eraseHalfedgeByPointer(Halfedge* halfedge) {
        for (auto it = halfedges.begin(); it != halfedges.end();) {
            if ((*it) == halfedge) {
                halfedges.erase(it);
                break;
            } else {
                it++;
            }
        }
    }
};
}  // namespace MCGAL

#endif