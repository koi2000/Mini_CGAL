#ifndef DECOMPRESS_TOOL
#define DECOMPRESS_TOOL
#include "../MCGAL/Core/include/core.h"
#include "quickSort.h"
#include "util.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <omp.h>
#include <queue>
#define BUFFER_SIZE 10 * 10 * 1024 * 1024
#define SPLITABLE_SIZE 10 * 10 * 1024
class DeCompressTool {
  private:
    char* buffer;
    std::vector<int> stOffsets;
    std::vector<int> lods;
    std::vector<int> nbDecimations;
    int i_nbDecimations;
    std::vector<int> vh_departureConquest;
    std::vector<int> splitableCounts;
    std::vector<int> insertedCounts;
    int i_curDecimationId = 0;
    bool isBase;
    int batch_size = 0;
    // operator
    template <typename T, typename Op> void omp_scan(int n, const T* in, T* out, Op op);
    void startNextDecompresssionOp();
    void resetState();
    MCGAL::Halfedge* pushHehInit(int meshId);
    void readBaseMesh(int meshId, int* offset);
    void buildFromBuffer(int meshId, std::deque<MCGAL::Point>* p_pointDeque, std::deque<uint32_t*>* p_faceDeque);
    void RemovedVerticesDecodingStep(int meshId);
    void InsertedEdgeDecodingStep(int meshId);

    inline void remove_tip(MCGAL::Halfedge* h);
    MCGAL::Halfedge* find_prev(MCGAL::Halfedge* h);
    inline void insert_tip(MCGAL::Halfedge* h, MCGAL::Halfedge* v);
    void initStIndexes(int* vertexIndexes, int* faceIndexes, int* stFacetIndexes, int* stHalfedgeIndexes, int num);
    void arrayAddConstant(int* array, int constant, int num);
    void preAllocInit(int* vertexIndexes, int* faceIndexes, int* stFacetIndexes, int* stHalfedgeIndexes, int num);
    void createCenterVertex(int* vertexIndexes, int* faceIndexes, int* stHalfedgeIndexes, int* stFacetIndexes, int num);
    void joinFacet(int* fids, int num);
    void insertRemovedVertices();
    void removedInsertedEdges();

    // IOs
    
    void dumpto(std::vector<MCGAL::Vertex*>& vertices, std::vector<MCGAL::Facet*>& facets, char* path);
    void writeFloat(float f, int* offset);
    float readFloat(int* offset);
    void writeInt16(int16_t i, int* offset);
    int16_t readInt16(int* offset);
    void writeuInt16(uint16_t i, int* offset);
    uint16_t readuInt16(int* offset);
    void writeInt(int i, int* offset);
    int readInt(int* offset);
    unsigned char readChar(int* offset);
    void writeChar(unsigned char ch, int* offset);
    void writePoint(MCGAL::Point& p, int* offset);
    MCGAL::Point readPoint(int* offset);

    float readFloatByOffset(int offset);
    int16_t readInt16ByOffset(int offset);
    uint16_t readuInt16ByOffset(int offset);
    int readIntByOffset(int offset);
    unsigned char readCharByOffset(int offset);
    MCGAL::Point readPointByOffset(int offset);

  public:
    DeCompressTool(char** paths, int bacth_size, bool is_base);
    void decode(int lod);
    ~DeCompressTool();
    void dumpto(std::string prefix);
};
#endif