#include "../MCGAL/Core/include/core.h"
#include "aab.h"
#include <algorithm>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#include <deque>
#include <queue>
#define BUFFER_SIZE 10 * 1024 * 1024

const int COMPRESSION_MODE_ID = 0;
const int DECOMPRESSION_MODE_ID = 1;

#define INV_ALPHA 2
#define INV_GAMMA 2

#define PPMC_RANDOM_CONSTANT 0315

class HiMesh : public MCGAL::Mesh {
    // Gate queues
    std::queue<MCGAL::Halfedge*> gateQueue;
    std::queue<MCGAL::Halfedge*> halfedgeQueue;
    std::queue<MCGAL::Facet*> facetQueue;
    // Processing mode: 0 for compression and 1 for decompression.
    int i_mode;
    bool b_jobCompleted = false;  // True if the job has been completed.

    unsigned i_curDecimationId = 0;
    unsigned i_nbDecimations;
    unsigned i_decompPercentage = 0;
    // Number of vertices removed during current conquest.
    unsigned i_nbRemovedVertices;

    // The vertices of the edge that is the departure of the coding and decoding conquests.
    MCGAL::Vertex* vh_departureConquest[2];
    // 
    
    // Geometry symbol list.
    std::deque<std::deque<MCGAL::Point>> geometrySym;

    std::deque<std::deque<unsigned char>> hausdorfSym;
    std::deque<std::deque<unsigned char>> proxyhausdorfSym;

    std::deque<std::deque<float>> hausdorfSym_float;
    std::deque<std::deque<float>> proxyhausdorfSym_float;

    // Connectivity symbol list.
    std::deque<std::deque<unsigned>> connectFaceSym;
    std::deque<std::deque<unsigned>> connectEdgeSym;
    std::vector<std::vector<int>> stVerteices;
    
    std::vector<std::vector<int>> facetNumberInGroups;
    std::vector<std::vector<int>> halfedgeNumberInGroups;
    std::vector<std::vector<int>> stBfsIds;
    std::vector<int> sampleNumbers;

    // The compressed data;
    char* p_data;
    size_t dataOffset = 0;  // the offset to read and write.

    aab mbb;  // the bounding box

    // Store the maximum Hausdorf Distance
    std::vector<MCGAL::Point> removedPoints;

    bool own_data = true;

  public:
    int id = 0;
    HiMesh(char* bufferPath);
    // constructor for encoding
    HiMesh(std::string& str, bool completeop = false);

    // constructors for decoding
    HiMesh(char* data, size_t dsize, bool own_data = true);
    HiMesh(HiMesh* mesh) : HiMesh(mesh->p_data, mesh->dataOffset, true) {}
    ~HiMesh();

    void encode(int lod = 0);
    void decode(int lod = 100);

    // get from pool
    MCGAL::Facet* add_face_by_pool(std::vector<MCGAL::Vertex*>& vts);

    // Compression
    void startNextCompresssionOp();
    void RemovedVertexCodingStep();
    void InsertedEdgeCodingStep();
    void HausdorffCodingStep();

    MCGAL::Halfedge* vertexCut(MCGAL::Halfedge* startH);
    void encodeInsertedEdges(unsigned i_operationId);
    void encodeRemovedVertices(unsigned i_operationId);
    void encodeHausdorff(unsigned i_operationId);

    // Compression geometry and connectivity tests.
    bool isRemovable(MCGAL::Vertex* v);
    bool checkCompetition(MCGAL::Vertex* v) const;
    bool isConvex(const std::vector<MCGAL::Vertex*>& polygon) const;
    bool isPlanar(const std::vector<MCGAL::Vertex*>& polygon, float epsilon) const;
    bool willViolateManifold(const std::vector<MCGAL::Halfedge*>& polygon) const;
    float removalError(MCGAL::Vertex* v, const std::vector<MCGAL::Vertex*>& polygon) const;

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
    float readFloatByOffset(int offset);
    int16_t readInt16ByOffset(int offset);
    uint16_t readuInt16ByOffset(int offset);
    int readIntByOffset(int offset);
    unsigned char readCharByOffset(int offset);
    MCGAL::Point readPointByOffset(int offset);

    void writeBaseMesh();
    void readBaseMesh();

    void dumpBuffer(char* path);
    void loadBuffer(char* path);

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
    friend std::istream& operator>>(std::istream& in, HiMesh& A);

    void buildFromBuffer(std::deque<MCGAL::Point>* p_pointDeque, std::deque<uint32_t*>* p_faceDeque);

    HiMesh* clone_mesh();
};
