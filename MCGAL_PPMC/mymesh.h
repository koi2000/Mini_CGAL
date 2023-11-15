#include "../MCGAL/Core/core.h"
#include "aab.h"
#include <algorithm>
#include <deque>
#include <queue>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/replace.hpp>
#define BUFFER_SIZE 10 * 1024 * 1024

const int COMPRESSION_MODE_ID = 0;
const int DECOMPRESSION_MODE_ID = 1;

#define INV_ALPHA 2
#define INV_GAMMA 2

#define PPMC_RANDOM_CONSTANT 0315


class MyVertex : public MCGAL::Vertex {
    enum Flag { Unconquered = 0, Conquered = 1 };

    Flag flag = Unconquered;
    unsigned int id = 0;

  public:
    MyVertex() : MCGAL::Vertex() {}
    MyVertex(const Point& p) : MCGAL::Vertex(p) {}

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
};

class MyHalfedge : public MCGAL::Halfedge {
    enum Flag { NotYetInQueue = 0, InQueue = 1, NoLongerInQueue = 2 };
    enum Flag2 { Original, Added, New };
    enum ProcessedFlag { NotProcessed, Processed };

    Flag flag = NotYetInQueue;
    Flag2 flag2 = Original;
    ProcessedFlag processedFlag = NotProcessed;

  public:
    MyHalfedge() {}

    inline void resetState() {
        flag = NotYetInQueue;
        flag2 = Original;
        processedFlag = NotProcessed;
    }

    /* Flag 1 */

    inline void setInQueue() {
        flag = InQueue;
    }

    inline void removeFromQueue() {
        assert(flag == InQueue);
        flag = NoLongerInQueue;
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
};

class replacing_group;
class MyTriangle;

class MyFace : public MCGAL::Face {
    typedef MCGAL::Point Point;
    enum Flag { Unknown = 0, Splittable = 1, Unsplittable = 2 };
    enum ProcessedFlag { NotProcessed, Processed };

    Flag flag = Unknown;
    ProcessedFlag processedFlag = NotProcessed;

    Point removedVertexPos;
    float proxy_hausdorff_distance = 0.0;
    float hausdorff_distance = 0.0;

  public:
    MyFace() {}

    inline void resetState() {
        flag = Unknown;
        processedFlag = NotProcessed;
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

  public:
    replacing_group* rg = NULL;
};


class MyMesh : public MCGAL::Mesh {
    // Gate queues
    std::queue<MyHalfedge*> gateQueue;

    // Processing mode: 0 for compression and 1 for decompression.
    int i_mode;
    bool b_jobCompleted = false;  // True if the job has been completed.

    unsigned i_curDecimationId = 0;
    unsigned i_nbDecimations;
    unsigned i_decompPercentage = 0;
    // Number of vertices removed during current conquest.
    unsigned i_nbRemovedVertices;

    // The vertices of the edge that is the departure of the coding and decoding conquests.
    MyVertex* vh_departureConquest[2];
    // Geometry symbol list.
    std::deque<std::deque<MCGAL::Point>> geometrySym;

    std::deque<std::deque<unsigned char>> hausdorfSym;
    std::deque<std::deque<unsigned char>> proxyhausdorfSym;

    std::deque<std::deque<float>> hausdorfSym_float;
    std::deque<std::deque<float>> proxyhausdorfSym_float;

    // Connectivity symbol list.
    std::deque<std::deque<unsigned>> connectFaceSym;
    std::deque<std::deque<unsigned>> connectEdgeSym;

    // The compressed data;
    char* p_data;
    size_t dataOffset = 0;  // the offset to read and write.

    aab mbb;  // the bounding box
    std::vector<MyTriangle*> original_facets;

    // Store the maximum Hausdorf Distance
    std::vector<MCGAL::Point> removedPoints;

    //
    std::unordered_set<replacing_group*> map_group;

    bool own_data = true;

  public:
    int id = 0;
    // constructor for encoding
    MyMesh(std::string& str, bool completeop = false);

    // constructors for decoding
    MyMesh(char* data, size_t dsize, bool own_data = true);
    MyMesh(MyMesh* mesh) : MyMesh(mesh->p_data, mesh->dataOffset, true) {}
    ~MyMesh();

    void encode(int lod = 0);
    void decode(int lod = 100);

    // Compression
    void startNextCompresssionOp();
    void RemovedVertexCodingStep();
    void InsertedEdgeCodingStep();
    void HausdorffCodingStep();

    void merge(std::unordered_set<replacing_group*>& reps, replacing_group*);
    MyHalfedge* vertexCut(MyHalfedge* startH);
    void encodeInsertedEdges(unsigned i_operationId);
    void encodeRemovedVertices(unsigned i_operationId);
    void encodeHausdorff(unsigned i_operationId);

    // Compression geometry and connectivity tests.
    bool isRemovable(MyVertex* v) const;
    bool isConvex(const std::vector<MyVertex*>& polygon) const;
    bool isPlanar(const std::vector<MyVertex*>& polygon, float epsilon) const;
    bool willViolateManifold(const std::vector<MyHalfedge*>& polygon) const;
    float removalError(MyVertex* v, const std::vector<MyVertex*>& polygon) const;

    // Decompression
    void startNextDecompresssionOp();
    void RemovedVerticesDecodingStep();
    void InsertedEdgeDecodingStep();
    void HausdorffDecodingStep();
    void insertRemovedVertices();
    void removeInsertedEdges();

    void pushHehInit();

    void alphaFolding();

    // IOs
    void writeFloat(float f);
    float readFloat();
    void writeInt16(int16_t i);
    int16_t readInt16();
    void writeuInt16(uint16_t i);
    uint16_t readuInt16();
    void writeInt(int i);
    int readInt();
    unsigned char readChar();
    void writeChar(unsigned char ch);
    void writePoint(MCGAL::Point& p);
    MCGAL::Point readPoint();

    void writeBaseMesh();
    void readBaseMesh();
    
    void compute_mbb();

    // Polyhedron* to_polyhedron();
    // Polyhedron* to_triangulated_polyhedron();
    std::string to_wkt();
    std::string to_off();
    void write_to_off(const char* path);
    void write_to_wkt(const char* path);

    size_t size_of_triangles();

    size_t size_of_edges();

    bool is_compression_mode() {
        return i_mode == COMPRESSION_MODE_ID;
    }
    size_t get_data_size() {
        return dataOffset;
    }
    const char* get_data() {
        return p_data;
    }

    MyMesh* clone_mesh();
};

class replacing_group {
  public:
    replacing_group() {
        // cout<<this<<" is constructed"<<endl;
        id = counter++;
        alive++;
    }
    ~replacing_group() {
        removed_vertices.clear();
        alive--;
    }

    void print() {
        // log("%5d (%2d refs %4d alive) - removed_vertices: %ld", id, ref, alive, removed_vertices.size());
    }

    std::unordered_set<MCGAL::Point> removed_vertices;
    // unordered_set<Triangle> removed_triangles;
    int id;
    int ref = 0;

    static int counter;
    static int alive;
};
