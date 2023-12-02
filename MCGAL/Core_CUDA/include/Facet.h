#ifndef FACET_H
#define FACET_H
#include "Configuration.h"
#include "Halfedge.h"
#include "Vertex.h"
#include <vector>

namespace MCGAL {
class replacing_group;
class Mesh;
class Facet {
    typedef MCGAL::Point Point;
    enum Flag { Unknown = 0, Splittable = 1, Unsplittable = 2 };
    enum ProcessedFlag { NotProcessed, Processed };
    enum RemovedFlag { NotRemoved, Removed };

    Flag flag = Unknown;
    ProcessedFlag processedFlag = NotProcessed;
    RemovedFlag removedFlag = NotRemoved;
    Point removedVertexPos;

  public:
    // std::vector<Vertex*> vertices;
    int vertices[VERTEX_IN_FACE];
    int vertex_size = 0;
    // std::vector<Halfedge*> halfedges;
    int halfedges[HALFEDGE_IN_FACE];
    int halfedge_size = 0;
    int poolId;

  public:
    ~Facet();

    // constructor
    Facet(){};
    Facet(const Facet& face);
    Facet(Halfedge* hit);
    Facet(std::vector<Vertex*>& vs);

    // pools
    void addHalfedge(Halfedge* halfedge);
    void addHalfedge(int halfedge);
    void addVertex(Vertex* vertex);
    void addVertex(int vertex);

    Vertex* getVertexByIndex(int index);
    Halfedge* getHalfedgeByIndex(int index);

    // utils
    Facet* clone();
    void reset(Halfedge* h);
    void reset(std::vector<Halfedge*>& hs);
    void reset(std::vector<Vertex*>& vs);
    void remove(Halfedge* h);
    int facet_degree();

    // override
    bool equal(const Facet& rhs) const;
    bool operator==(const Facet& rhs) const;

    // to_string method
    void print();
    void print_off();

    // set flags
    inline void resetState() {
        flag = Unknown;
        processedFlag = NotProcessed;
        removedFlag = NotRemoved;
    }

    inline void resetProcessedFlag() {
        processedFlag = NotProcessed;
    }

    inline bool isConquered() const {
        return (flag == Splittable || flag == Unsplittable);
    }

    inline bool isSplittable() const {
        return (flag == Splittable);
    }

    inline bool isUnsplittable() const {
        return (flag == Unsplittable);
    }

    inline void setSplittable() {
        assert(flag == Unknown);
        flag = Splittable;
    }

    inline void setUnsplittable() {
        assert(flag == Unknown);
        flag = Unsplittable;
    }

    inline void setProcessedFlag() {
        processedFlag = Processed;
    }

    inline bool isProcessed() const {
        return (processedFlag == Processed);
    }

    inline Point getRemovedVertexPos() const {
        return removedVertexPos;
    }

    inline void setRemovedVertexPos(Point p) {
        removedVertexPos = p;
    }

    /* Flag 3*/
    inline void setRemoved() {
        removedFlag = Removed;
    }

    inline bool isRemoved() {
        return removedFlag == Removed;
    }

  public:
    replacing_group* rg = NULL;
};
}  // namespace MCGAL
#endif