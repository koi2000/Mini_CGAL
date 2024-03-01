#ifndef DECOMPRESS_TOOL
#define DECOMPRESS_TOOL
#include "../MCGAL/Core_CUDA/include/core.cuh"
#include "cuda_functions.cuh"
#include "thrust_struct.cuh"
#include "util.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <omp.h>
#include <queue>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#define BUFFER_SIZE 10 * 10 * 1024 * 1024
#define SPLITABLE_SIZE 10 * 10 * 1024
/**
 * @brief 用于对一个batch里的数据进行decompress，尝试full gpu处理
 * 可能存在的需求，让某个mesh多decode一轮，而不是从baseMesh开始decode，这种情况需要提前存储mesh的信息
 * 需要维护的东西太多了，分阶段讨论，首先就是第一个，每个mesh的vertex和facet
 *
 * 在RemovedVerticesDecodingStep阶段 需要知道每个mesh的vh_departureConquest
 * 在insertRemovedVerticesOnCuda阶段，需要为每个mesh进行preAlloc，在此时有一个比较严重的问题
 * 不知道每个mesh有哪些facet，此时阶段可以分为
 * 1. readBaseMesh 在此阶段，记录每个mesh的vh_departureConquest，
 *    剩下的压根就不管，只在pool里记录一个meshId。并行度为mesh的数量
 * 2. RemovedVerticesDecodingStep 这一步就是一个bfs，似乎啥也管不了
 * 3. InsertedEdgeDecodingStep 这一步也是一个简单的bfs，也是想不出来咋并行
 * 4. preAlloc阶段，这一步提前分配点，边，面信息，可以有较高的并行度
 * 5. insertRemovedVerticesOnCuda 这一步似乎也可以有较高的并行度
 * 6. joinFacet 这一步不好评价，等和老师商量一下后再看看咋处理
 */
class DeCompressTool {
  private:
    // cpu
    char* buffer;
    std::vector<int> stOffsets;
    std::vector<int> lods;
    std::vector<int> nbDecimations;
    int i_nbDecimations;
    std::vector<int> vh_departureConquest;
    std::vector<int> splitableCounts;
    std::vector<int> insertedCounts;
    int i_curDecimationId = 0;
    bool own_data = true;
    bool is_base = true;
    int batch_size = 0;
    int i_decompPercentage = 0;
    bool b_jobCompleted;
    // cuda
    char* dbuffer;
    int* dstOffsets;
    // 直接存一维数组，通过奇数位偶数位访问
    int* dvh_departureConquest;
    int* dfaceIndexes;
    int* dvertexIndexes;
    int* dstHalfedgeIndexes;
    int* dstFacetIndexes;
    int* dSplittabelCount;

  public:
    DeCompressTool(char* bufferPath, std::vector<int> stOffsets);
    DeCompressTool(char* data, size_t dsize, std::vector<int> stOffsets, bool is_base, bool own_data = true);
    DeCompressTool(char** paths, int number, bool is_base);
    ~DeCompressTool();

    // cpu
    void decode(int lod);
    void RemovedVerticesDecodingStep(int meshId);
    void InsertedEdgeDecodingStep(int meshId);
    void BatchRemovedVerticesDecodingStep();
    void BatchInsertedEdgeDecodingStep();
    // cuda
    void BatchInsertedEdgeDecodingStepOnCuda();

    void RemovedVerticesDecodingOnCuda();
    void InsertedEdgeDecodingOnCuda();

    void insertRemovedVertices();
    void insertRemovedVerticesOnCuda();
    void removeInsertedEdges(int meshId);
    void readBaseMesh(int meshId, int* offset);
    MCGAL::Halfedge* pushHehInit(int meshId);
    void buildFromBuffer(int meshId, std::deque<MCGAL::Point>* p_pointDeque, std::deque<uint32_t*>* p_faceDeque);
    MCGAL::Halfedge* join_facet(MCGAL::Halfedge* h);
    MCGAL::Halfedge* find_prev(MCGAL::Halfedge* h) const;
    inline void remove_tip(MCGAL::Halfedge* h) const;
    void startNextDecompresssionOp();

    // cuda
    void decodeOnCuda();
    __device__ void readBaseMeshOnCuda(char* buffer, int* stOffsets, int num);
    __device__ void pushHehInitOnCuda();
    __device__ void RemovedVerticesDecodingStepOnCuda(char* buffer, int* stOffsets, int num);
    __device__ void InsertedEdgeDecodingStepOnCuda(char* buffer, int* stOffsets, int num);
    void removeInsertedEdgesOnCuda();

    // IOs
    void dumpto(std::string prefix);
    void dumpto(std::vector<MCGAL::Vertex*> vertices, std::vector<MCGAL::Facet*> facets, char* path);
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
};
#endif