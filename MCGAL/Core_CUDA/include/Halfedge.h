#ifndef HALFEDGE_H
#define HALFEDGE_H
#include <assert.h>
namespace MCGAL {
class Vertex;
class Facet;
class Halfedge {
    enum Flag { NotYetInQueue = 0, InQueue = 1, NoLongerInQueue = 2 };
    enum Flag2 { Original, Added, New };
    enum ProcessedFlag { NotProcessed, Processed };
    enum RemovedFlag { NotRemoved, Removed };
    enum BFSFlag { NotVisited, Visited };

    Flag flag = NotYetInQueue;
    Flag2 flag2 = Original;
    ProcessedFlag processedFlag = NotProcessed;
    RemovedFlag removedFlag = NotRemoved;
    BFSFlag bfsFlag = NotVisited;

  public:
    int poolId;
    int vertex_ = -1;
    int end_vertex_ = -1;
    int facet_ = -1;
    int next_ = -1;
    int opposite_ = -1;
    Halfedge() {
        vertex_ = -1;
        end_vertex_ = -1;
        facet_ = -1;
        next_ = -1;
        opposite_ = -1;
    };
    // Vertex* vertex = nullptr;
    // Vertex* end_vertex = nullptr;
    // Facet* face = nullptr;
    // Halfedge* next = nullptr;
    // Halfedge* opposite = nullptr;
    Halfedge(Vertex* v1, Vertex* v2);
    Halfedge(int v1, int v2);
    ~Halfedge();

    Vertex* vertex();
    Vertex* end_vertex();
    Facet* facet();
    Halfedge* opposite();
    Halfedge* next();

    void setOpposite(Halfedge* opposite);
    void setOpposite(int opposite);

    void setNext(Halfedge* next);
    void setNext(int next);

    void setFacet(Facet* facet);
    void setFacet(int facet);

    void reset(Vertex* v1, Vertex* v2);

    inline void resetState() {
        flag = NotYetInQueue;
        flag2 = Original;
        processedFlag = NotProcessed;
        removedFlag = NotRemoved;
        bfsFlag = NotVisited;
    }

    /* Flag 1 */

    inline void setInQueue() {
        flag = InQueue;
    }

    inline void removeFromQueue() {
        assert(flag == InQueue);
        flag = NoLongerInQueue;
    }

    inline bool canAddInQueue() {
        return flag == NotYetInQueue;
    }

    /* Processed flag */

    inline void resetProcessedFlag() {
        processedFlag = NotProcessed;
    }

    inline void setProcessed() {
        processedFlag = Processed;
    }

    inline bool isProcessed() const {
        return (processedFlag == Processed);
    }

    /* Flag 2 */

    inline void setAdded() {
        assert(flag2 == Original);
        flag2 = Added;
    }

    inline void setNew() {
        assert(flag2 == Original);
        flag2 = New;
    }

    inline bool isAdded() const {
        return flag2 == Added;
    }

    inline bool isOriginal() const {
        return flag2 == Original;
    }

    inline bool isNew() const {
        return flag2 == New;
    }

    /* Flag 3*/
    inline void setRemoved() {
        removedFlag = Removed;
    }

    inline bool isRemoved() {
        return removedFlag == Removed;
    }

    /* bfs Flag */
    inline void setVisited() {
        bfsFlag = Visited;
    }

    inline bool isVisited() {
        return bfsFlag == Visited;
    }
};
}  // namespace MCGAL
#endif