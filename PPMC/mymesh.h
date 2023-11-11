//
// Created by DELL on 2023/11/9.
//

#ifndef PROGRESSIVEPOLYGONS_MYMESH_H
#define PROGRESSIVEPOLYGONS_MYMESH_H
#include <fstream>
#include <iostream>
#include <stdint.h>

#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/bounding_box.h>
#include <CGAL/circulator.h>

#include <queue>

// Range coder includes.
#include "util.h"
#include <unordered_set>

typedef CGAL::Simple_cartesian<float> MyKernel;
typedef MyKernel::Point_3 Point;
typedef MyKernel::Vector_3 Vector;
typedef MyKernel::Triangle_3 Triangle;

typedef CGAL::Simple_cartesian<double> MyKernelDouble;
typedef MyKernelDouble::Vector_3 VectorDouble;

typedef CGAL::Simple_cartesian<int> MyKernelInt;
typedef MyKernelInt::Point_3 PointInt;
typedef MyKernelInt::Vector_3 VectorInt;

// Operation list.
enum Operation {
    Idle,
    DecimationConquest,
    RemovedVertexCoding,
    InsertedEdgeCoding,
    AdaptiveQuantization,  // Compression.
    UndecimationConquest,
    InsertedEdgeDecoding,
    AdaptiveUnquantization  // Decompression.
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

    unordered_set<Point> removed_vertices;
    // unordered_set<Triangle> removed_triangles;
    int id;
    int ref = 0;

    static int counter;
    static int alive;
};

// My face type has a vertex flag
template <class Refs> class MyFace : public CGAL::HalfedgeDS_face_base<Refs> {
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
    inline pair<float, float> getHausdorfDistance() {
        return pair<float, float>(proxy_hausdorff_distance, hausdorff_distance);
    }

    inline void resetHausdorff() {
        hausdorff_distance = 0.0;
        proxy_hausdorff_distance = 0.0;
    }

    inline void setProxyHausdorff(float prh) {
        proxy_hausdorff_distance = prh;
    }

    inline void setHausdorff(float hd) {
        hausdorff_distance = hd;
    }

    inline void updateProxyHausdorff(float prh) {
        proxy_hausdorff_distance = max(prh, proxy_hausdorff_distance);
    }

    inline void updateHausdorff(float hd) {
        hausdorff_distance = max(hd, hausdorff_distance);
    }

    inline float getProxyHausdorff() {
        return proxy_hausdorff_distance;
    }

    inline float getHausdorff() {
        return hausdorff_distance;
    }
    replacing_group* rg = NULL;
    vector<Triangle> triangles;
    MyTriangle* tri = NULL;
};

// My vertex type has a isConquered flag
template <class Refs> class MyVertex : public CGAL::HalfedgeDS_vertex_base<Refs, CGAL::Tag_true, Point> {
    enum Flag { Unconquered = 0, Conquered = 1 };

  public:
    MyVertex() : CGAL::HalfedgeDS_vertex_base<Refs, CGAL::Tag_true, Point>(), flag(Unconquered) {}

    MyVertex(const Point& p) : CGAL::HalfedgeDS_vertex_base<Refs, CGAL::Tag_true, Point>(p), flag(Unconquered) {}

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

    inline unsigned getQuantCellId() const {
        return i_quantCellId;
    }

    inline void setQuantCellId(unsigned nId) {
        i_quantCellId = nId;
    }

    inline Point getOldPos() const {
        return oldPos;
    }

    inline void setOldPos(Point pos) {
        oldPos = pos;
    }

  private:
    Flag flag;
    size_t id;
    unsigned i_quantCellId;
    Point oldPos;
};

// My vertex type has a isConquered flag
template <class Refs> class MyHalfedge : public CGAL::HalfedgeDS_halfedge_base<Refs> {
    enum Flag { NotYetInQueue = 0, InQueue = 1, InQueue2 = 2, NoLongerInQueue = 3 };
    enum Flag2 { Original, Added, New };
    enum ProcessedFlag { NotProcessed, Processed };

  public:
    MyHalfedge() : flag(NotYetInQueue), flag2(Original), processedFlag(NotProcessed) {}

    inline void resetState() {
        flag = NotYetInQueue;
        flag2 = Original;
        processedFlag = NotProcessed;
    }

    /* Flag 1 */

    inline void setInQueue() {
        flag = InQueue;
    }

    inline void setInProblematicQueue() {
        assert(flag == InQueue);
        flag = InQueue2;
    }

    inline void removeFromQueue() {
        assert(flag == InQueue || flag == InQueue2);
        flag = NoLongerInQueue;
    }

    inline bool isInNormalQueue() const {
        return flag == InQueue;
    }

    inline bool isInProblematicQueue() const {
        return flag == InQueue2;
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

  private:
    Flag flag;
    Flag2 flag2;
    ProcessedFlag processedFlag;
};

struct MyItems : public CGAL::Polyhedron_items_3
{
    template <class Refs, class Traits> struct Face_wrapper
    { typedef MyFace<Refs> Face; };

    template <class Refs, class Traits> struct Vertex_wrapper
    { typedef MyVertex<Refs> Vertex; };

    template <class Refs, class Traits> struct Halfedge_wrapper
    { typedef MyHalfedge<Refs> Halfedge; };
};

class MyMesh : public CGAL::Polyhedron_3<MyKernel, MyItems> {
    typedef CGAL::Polyhedron_3<MyKernel, MyItems> PolyhedronT;

  public:
    MyMesh(char filename[],
           std::string filePathOutput,
           unsigned i_decompPercentage,
           const int i_mode,
           unsigned i_quantBits,
           bool b_useAdaptiveQuantization,
           bool b_useLiftingScheme,
           bool b_useCurvaturePrediction,
           bool b_useConnectivityPredictionFaces,
           bool b_useConnectivityPredictionEdges,
           bool b_allowConcaveFaces,
           bool b_useTriangleMeshConnectivityPredictionFaces);

    ~MyMesh();

    // simple
    void encode(int lod);

    void stepOperation();

    void batchOperation();

    void completeOperation();

    Vector computeNormal(Facet_const_handle f) const;

    Vector computeVertexNormal(Halfedge_const_handle heh) const;

    Point barycenter(Facet_const_handle f) const;

    float getBBoxDiagonal() const;

    Vector getBBoxCenter() const;

    void startCompress();

  private:
    // General
    void computeBoundingBox();

    void determineQuantStep();

    void quantizeVertexPositions();

    PointInt getQuantizedPos(Point p) const;

    Point getPos(PointInt p) const;

    // Compression
    void startNextCompresssionOp();

    void beginDecimationConquest();

    void beginInsertedEdgeCoding();

    void decimationStep();

    void RemovedVertexCodingStep();

    void InsertedEdgeCodingStep();

    Halfedge_handle vertexCut(Halfedge_handle startH);

    void determineResiduals();

    void encodeInsertedEdges(unsigned i_operationId);

    void encodeRemovedVertices(unsigned i_operationId);

    void beginAdaptiveQuantization();

    void adaptiveQuantizationStep();

    void encodeAdaptiveQuantization(std::deque<unsigned>& symbols);

    void lift();

    // Compression geometry and connectivity tests;
    bool isRemovable(Vertex_handle v) const;

    bool isConvex(const std::vector<Vertex_const_handle>& polygon) const;

    bool isPlanar(const std::vector<Vertex_const_handle>& polygon, float epsilon) const;

    bool willViolateManifold(const std::vector<Halfedge_const_handle>& polygon) const;

    float removalError(Vertex_const_handle v, const std::vector<Vertex_const_handle>& polygon) const;

    // Decompression
    void startNextDecompressionOp();

    void beginUndecimationConquest();

    void beginInsertedEdgeDecoding();

    void undecimationStep();

    void InsertedEdgeDecodingStep();

    void insertRemovedVertices();

    void removeInsertedEdges();

    void decodeGeometrySym(Halfedge_handle heh_gate, Face_handle fh);

    void beginRemovedVertexCodingConquest();

    void determineGeometrySym(Halfedge_handle heh_gate, Face_handle fh);

    void beginAdaptiveUnquantization();

    void adaptiveUnquantizationStep();

    // Lifting
    void lift(bool b_unlift);

    // Adaptive quantization
    float determineKg();

    std::map<unsigned, unsigned> determineCellSymbols(Halfedge_handle heh_v, bool b_compression);

    // Utils
    Vector computeNormal(Halfedge_const_handle heh_gate) const;

    Vector computeNormal(const std::vector<Vertex_const_handle>& polygon) const;

    Vector computeNormal(Point p[3]) const;

    Point barycenter(Halfedge_handle heh_gate) const;

    Point barycenter(const std::vector<Vertex_const_handle>& polygon) const;

    unsigned vertexDegreeNotNew(Vertex_const_handle vh) const;

    VectorInt avgLaplacianVect(Halfedge_handle heh_gate) const;

    float triangleSurface(const Point p[]) const;

    float edgeLen(Halfedge_const_handle heh) const;

    float facePerimeter(const Face_handle fh) const;

    float faceSurface(Halfedge_handle heh) const;

    void pushHehInit();

    void updateAvgSurfaces(bool b_split, float f_faceSurface);

    void updateAvgEdgeLen(bool b_original, float f_edgeLen);

    // IOs
    void writeCompressedData();

    void readCompressedData();

    void writeFloat();

    float readFloat();

    void writeInt16(int16_t i);

    int16_t readInt16();

    void writeBaseMesh();

    void readBaseMesh();

    int writeCompressedFile() const;

    int readCompressedFile(char psz_filePath[]);

    void writeMeshOff(const char psz_filePath[]);

    void writeCurrentOperationMesh(std::string pathPrefix, unsigned i_id) const;

    // Gate queues
    std::queue<Halfedge_handle> gateQueue;
    std::queue<Halfedge_handle> problematicGateQueue;

    // Processing mode: 0 for compression and 1 for decompression.
    int i_mode;
    bool b_jobCompleted;  // True if the job has been completed.

    Operation operation;
    unsigned i_curDecimationId;
    unsigned i_nbDecimations;
    unsigned i_curQuantizationId;
    unsigned i_nbQuantizations;
    unsigned i_curOperationId;

    unsigned i_levelNotConvexId;
    bool b_testConvexity;

    // The vertices of the edge that is the departure of the coding and decoding
    // conquests.
    Vertex_handle vh_departureConquest[2];

    std::deque<unsigned> typeOfOperation;  // O - decimation, 1 - adaptive quantization.

    // Geometry symbol list.
    std::deque<std::deque<VectorInt>> geometrySym;
    std::deque<std::deque<unsigned>> adaptiveQuantSym;

    // Connectivity symbol list.
    std::deque<std::deque<std::pair<unsigned, unsigned>>> connectFaceSym;
    std::deque<std::deque<std::pair<unsigned, unsigned>>> connectEdgeSym;
    std::deque<unsigned> facesConnectPredictionUsed;
    std::deque<unsigned> edgesConnectPredictionUsed;

    // Size used for the encoding.
    size_t connectivitySize;
    size_t geometrySize;

    // Number of vertices removed during current conquest.
    unsigned i_nbRemovedVertices;

    Point bbMin;
    Point bbMax;
    float f_bbVolume;

    unsigned i_quantBits;
    float f_quantStep;
    float f_adaptQuantRescaling;

    // Initial number of vertices and faces.
    size_t i_nbVerticesInit;
    size_t i_nbFacetsInit;

    // The compressed data;
    char* p_data;
    size_t dataOffset;  // the offset to read and write.

    std::string filePathOutput;
    unsigned i_decompPercentage;

    // Compression and decompression variables.
    rangecoder rangeCoder;

    // Range coder data model.
    qsmodel alphaBetaModel, gammaModel, quantModel, connectModel;

    int alphaBetaMin, gammaMin;

    // Variable for connectivity prediction.
    float f_avgSurfaceFaceWithoutCenterRemoved;
    float f_avgSurfaceFaceWithCenterRemoved;

    float f_avgInsertedEdgesLength;
    float f_avgOriginalEdgesLength;

    unsigned i_nbFacesWithoutCenterRemoved;
    unsigned i_nbFacesWithCenterRemoved;

    unsigned i_nbInsertedEdges;
    unsigned i_nbOriginalEdges;

    unsigned i_nbGoodPredictions;
    bool b_predictionUsed;

    std::filebuf fbDebug;
    std::ostream osDebug;

    // Codec features status.
    bool b_useAdaptiveQuantization;
    bool b_useLiftingScheme;
    bool b_useCurvaturePrediction;
    bool b_useConnectivityPredictionFaces;
    bool b_useConnectivityPredictionEdges;
    bool b_useTriangleMeshConnectivityPredictionFaces;
};

#endif  // PROGRESSIVEPOLYGONS_MYMESH_H
