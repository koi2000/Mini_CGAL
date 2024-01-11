#include <assert.h>
#include <math.h>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
namespace MCGAL {

#define VERTEX_POOL_SIZE 100 * 1024
#define HALFEDGE_POOL_SIZE 500 * 1024
#define FACET_POOL_SIZE 100 * 1024

#define BUCKET_SIZE 4096
#define SMALL_BUCKET_SIZE 32

inline bool compareFloat(float f1, float f2) {
    if (fabs(f1 - f2) < 1e-6) {
        return true;
    }
    return false;
}

class Point {
  public:
    Point() {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
    }

    Point(float x, float y, float z) {
        v[0] = x;
        v[1] = y;
        v[2] = z;
    }

    Point(Point* pt) {
        assert(pt);
        for (int i = 0; i < 3; i++) {
            v[i] = pt->v[i];
        }
    };

    float x() const {
        return v[0];
    }

    float y() const {
        return v[1];
    }

    float z() const {
        return v[2];
    }

    // Hash function for Point
    struct Hash {
        size_t operator()(const Point point) const {
            // Use a combination of hash functions for each member
            size_t hash = std::hash<float>{}(point.x());
            hash_combine(hash, std::hash<float>{}(point.y()));
            hash_combine(hash, std::hash<float>{}(point.z()));
            return hash;
        }
    };

    // Equality comparison for Vertex
    struct Equal {
        bool operator()(const Point p1, const Point p2) const {
            // Compare each member for equality
            return p1.x() == p2.x() && p1.y() == p2.y() && p1.z() == p2.z();
        }
    };

    static void hash_combine(size_t& seed, size_t hash) {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    float& operator[](int index) {
        if (index >= 0 && index < 3) {
            return v[index];
        } else {
            throw std::out_of_range("Index out of range");
        }
    }

  protected:
    float v[3];
};

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
    std::vector<Halfedge*> halfedges;
    // std::set<Halfedge*> opposite_half_edges;
    // std::unordered_set<Halfedge*> halfedges;
    // std::unordered_set<Halfedge*> opposite_half_edges;

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

    // Hash function for Vertex
    struct Hash {
        size_t operator()(const Vertex* vertex) const {
            // Use a combination of hash functions for each member
            size_t hash = std::hash<float>{}(vertex->x());
            hash_combine(hash, std::hash<float>{}(vertex->y()));
            hash_combine(hash, std::hash<float>{}(vertex->z()));
            hash_combine(hash, std::hash<int>{}(vertex->vid()));
            return hash;
        }
    };

    // Equality comparison for Vertex
    struct Equal {
        bool operator()(const Vertex* v1, const Vertex* v2) const {
            // Compare each member for equality
            return v1->x() == v2->x() && v1->y() == v2->y() && v1->z() == v2->z();
            // Add other members if needed
        }
    };

    static void hash_combine(size_t& seed, size_t hash) {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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
    Halfedge(){};
    Vertex* vertex = nullptr;
    Vertex* end_vertex = nullptr;
    Facet* face = nullptr;
    Halfedge* next = nullptr;
    Halfedge* opposite = nullptr;
    Halfedge(Vertex* v1, Vertex* v2);
    ~Halfedge();
    __uint128_t horder = 0;
    int poolId = -1;

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

    inline void setUnProcessed() {
        processedFlag = NotProcessed;
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

    // Hash function for Point
    struct Hash {
        size_t operator()(Halfedge* halfedge) const {
            // Use a combination of hash functions for each member
            size_t hash = std::hash<float>{}(halfedge->vertex->x());
            hash_combine(hash, std::hash<float>{}(halfedge->vertex->y()));
            hash_combine(hash, std::hash<float>{}(halfedge->vertex->z()));
            hash_combine(hash, std::hash<float>{}(halfedge->end_vertex->x()));
            hash_combine(hash, std::hash<float>{}(halfedge->end_vertex->y()));
            hash_combine(hash, std::hash<float>{}(halfedge->end_vertex->z()));
            return hash;
        }
    };

    // Equality comparison for Vertex
    struct Equal {
        bool operator()(Halfedge* h1, Halfedge* h2) const {
            // Compare each member for equality
            return h1->vertex->x() == h2->vertex->x() && h1->vertex->y() == h2->vertex->y() &&
                   h1->vertex->z() == h2->vertex->z() && h1->end_vertex->x() == h2->end_vertex->x() &&
                   h1->end_vertex->y() == h2->end_vertex->y() && h1->end_vertex->z() == h2->end_vertex->z();
        }
    };

    static void hash_combine(size_t& seed, size_t hash) {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

class Mesh;
class Facet {
  public:
    typedef MCGAL::Point Point;
    enum Flag { Unknown = 0, Splittable = 1, Unsplittable = 2 };
    enum ProcessedFlag { NotProcessed, Processed };
    enum RemovedFlag { NotRemoved, Removed };

    Flag flag = Unknown;
    ProcessedFlag processedFlag = NotProcessed;
    RemovedFlag removedFlag = NotRemoved;
    Point removedVertexPos;
    __uint128_t forder = 0;
    int poolId = -1;

  public:
    std::vector<Vertex*> vertices;
    std::vector<Halfedge*> halfedges;

  public:
    ~Facet();

    // constructor
    Facet(){};
    Facet(const Facet& face);
    Facet(Halfedge* hit);
    Facet(std::vector<Vertex*>& vs);
    Facet(std::vector<Vertex*>& vs, Mesh* mesh);

    // utils
    Facet* clone();
    void reset(Halfedge* h);
    void reset(std::vector<Halfedge*>& hs);
    void reset(std::vector<Vertex*>& vs, Mesh* mesh);
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
        return (flag == Splittable || flag == Unsplittable || removedFlag == Removed);
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

    inline void setUnProcessed() {
        processedFlag = NotProcessed;
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
};

class Mesh {
  public:
    // std::unordered_set<Vertex*, Vertex::Hash, Vertex::Equal> vertices;
    std::vector<Vertex*> vertices;
    // std::unordered_set<Vertex*> vertices;
    // std::unordered_set<Halfedge*, Halfedge::Hash, Halfedge::Equal> halfedges;
    // std::unordered_set<Halfedge*> halfedges;
    // std::unordered_set<Facet*> faces;
    std::vector<Facet*> faces;
    std::vector<Halfedge*> halfedges;
    int nb_vertices = 0;
    int nb_faces = 0;
    int nb_edges = 0;

    MCGAL::Vertex** vpool = nullptr;
    MCGAL::Halfedge** hpool = nullptr;
    MCGAL::Facet** fpool = nullptr;
    int vindex = 0;
    int hindex = 0;
    int findex = 0;

  public:
    Mesh() {
        vpool = new MCGAL::Vertex*[VERTEX_POOL_SIZE];
        hpool = new MCGAL::Halfedge*[HALFEDGE_POOL_SIZE];
        fpool = new MCGAL::Facet*[FACET_POOL_SIZE];
        for (int i = 0; i < VERTEX_POOL_SIZE; i++) {
            vpool[i] = new MCGAL::Vertex();
            vpool[i]->poolId = i;
        }
        for (int i = 0; i < HALFEDGE_POOL_SIZE; i++) {
            hpool[i] = new MCGAL::Halfedge();
            hpool[i]->poolId = i;
        }
        for (int i = 0; i < FACET_POOL_SIZE; i++) {
            fpool[i] = new MCGAL::Facet();
            fpool[i]->poolId = i;
        }
        // vertices.reserve(FACET_POOL_SIZE);
        // faces.reserve(FACET_POOL_SIZE);
        faces.reserve(BUCKET_SIZE);
        vertices.reserve(BUCKET_SIZE);
    }
    ~Mesh();

    inline MCGAL::Vertex* getVertexFromPool(int index) {
        return vpool[index];
    }

    inline MCGAL::Halfedge* getHalfedgeFromPool(int index) {
        return hpool[index];
    }

    inline MCGAL::Facet* getFacetFromPool(int index) {
        return fpool[index];
    }

    // allocate from pool
    inline MCGAL::Vertex* allocateVertexFromPool() {
        return vpool[vindex++];
    }

    inline MCGAL::Vertex* allocateVertexFromPool(MCGAL::Point& p) {
        vpool[vindex]->setPoint(p);
        return vpool[vindex++];
    }

    inline MCGAL::Halfedge* allocateHalfedgeFromPool() {
        return hpool[hindex++];
    }

    inline MCGAL::Halfedge* allocateHalfedgeFromPool(MCGAL::Vertex* v1, MCGAL::Vertex* v2) {
        hpool[hindex]->reset(v1, v2);
        return hpool[hindex++];
    }

    inline MCGAL::Facet* allocateFaceFromPool() {
        return fpool[findex++];
    }

    inline MCGAL::Facet* allocateFaceFromPool(MCGAL::Halfedge* h) {
        fpool[findex]->reset(h);
        return fpool[findex++];
    }

    inline MCGAL::Facet* allocateFaceFromPool(std::vector<MCGAL::Vertex*> vts, Mesh* mesh) {
        fpool[findex]->reset(vts, mesh);
        return fpool[findex++];
    }

    // IOS
    bool loadOFF(std::string path);
    void dumpto(std::string path);
    void print();
    std::string to_string();
    Vertex* get_vertex(int vseq = 0);

    // element operating
    Facet* add_face(std::vector<Vertex*>& vs);
    Facet* add_face(Facet* face);
    void eraseFacetByPointer(Facet* facet);
    void eraseVertexByPointer(Vertex* vertex);

    Facet* remove_vertex(Vertex* v);
    Halfedge* merge_edges(Vertex* v);

    /*
     * statistics
     *
     * */
    size_t size_of_vertices() {
        return vertices.size();
    }

    size_t size_of_facets() {
        return faces.size();
    }

    size_t size_of_halfedges() {
        // int count = 0;
        // for (Facet* fit : faces) {
        //     for (Halfedge* hit : fit->halfedges) {
        //         count++;
        //     }
        // }
        // return count;
        return halfedges.size();
    }

    Halfedge* split_facet(Halfedge* h, Halfedge* g);

    Halfedge* create_center_vertex(Halfedge* h);

    inline void close_tip(Halfedge* h, Vertex* v) const;

    inline void insert_tip(Halfedge* h, Halfedge* v) const;

    Halfedge* find_prev(Halfedge* h) const;

    Halfedge* erase_center_vertex(Halfedge* h);

    void set_face_in_face_loop(Halfedge* h, Facet* f) const;

    inline void remove_tip(Halfedge* h) const;

    Halfedge* join_face(Halfedge* h);
};

}  // namespace MCGAL