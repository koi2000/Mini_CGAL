//
// Created by DELL on 2023/11/9.
//
#include "mymesh.h"
#include "configuration.h"
#include "frenetRotation.h"
#include <algorithm>

MyMesh::MyMesh(char filename[],
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
               bool b_useTriangleMeshConnectivityPredictionFaces)
    : CGAL::Polyhedron_3<CGAL::Simple_cartesian<float>, MyItems>(), i_mode(i_mode), b_jobCompleted(false),
      operation(Idle), i_curDecimationId(0), i_curQuantizationId(0), i_curOperationId(0), i_levelNotConvexId(0),
      b_testConvexity(false), connectivitySize(0), geometrySize(0), i_quantBits(i_quantBits), dataOffset(0),
      filePathOutput(filePathOutput), i_decompPercentage(i_decompPercentage), osDebug(&fbDebug),
      b_useAdaptiveQuantization(b_useAdaptiveQuantization), b_useLiftingScheme(b_useLiftingScheme),
      b_useCurvaturePrediction(b_useCurvaturePrediction),
      b_useConnectivityPredictionFaces(b_useConnectivityPredictionFaces),
      b_useConnectivityPredictionEdges(b_useConnectivityPredictionEdges),
      b_useTriangleMeshConnectivityPredictionFaces(b_useTriangleMeshConnectivityPredictionFaces) {
    p_data = new char[BUFFER_SIZE];
    memset(p_data, 0, BUFFER_SIZE);
    // 初始化range coder
    rangeCoder.p_data = p_data;
    rangeCoder.p_dataOffset = &dataOffset;

    if (i_mode == COMPRESSION_MODE_ID) {
        std::filebuf fb;
        fb.open(filename, std::ios::in);
        if (fb.is_open()) {
            std::istream is(&fb);
            // 将文件读取为对象
            is >> *this;

            fb.close();
            //
            if (keep_largest_connected_components(1) != 0) {
                std::cout << "Can't compress the mesh." << std::endl;
                std::cout << "The codec doesn't handle meshes with several connected components." << std::endl;
                exit(EXIT_FAILURE);
            }

            if (!is_closed()) {
                std::cout << "Can't compress the mesh." << std::endl;
                std::cout << "The codec doesn't handle meshes with borders." << std::endl;
                exit(EXIT_FAILURE);
            }

            /* The special connectivity prediction scheme for triangle
               mesh is not used if the current mesh is not a pure triangle mesh. */
            if (!is_pure_triangle())
                this->b_useTriangleMeshConnectivityPredictionFaces = false;

            // 计算boundingBox
            computeBoundingBox();
            // 计算quantStep
            determineQuantStep();
            // 对每个点进行量化
            quantizeVertexPositions();
        }
        vh_departureConquest[0] = halfedges_begin()->opposite()->vertex();
        vh_departureConquest[1] = halfedges_begin()->vertex();
        if (!b_allowConcaveFaces)
            b_testConvexity = true;
    } else {

    }
    i_nbVerticesInit = size_of_vertices();
    i_nbFacetsInit = size_of_facets();
}

MyMesh::~MyMesh() {
    delete[] p_data;
    fbDebug.close();
}

/**
 * Finish completely the current operation.
 */
void MyMesh::completeOperation() {
    while (!b_jobCompleted)
        batchOperation();
}

void MyMesh::startCompress() {
    startNextCompresssionOp();
}

void MyMesh::computeBoundingBox() {
    std::list<Point> vList;
    for (MyMesh::Vertex_iterator vit = vertices_begin(); vit != vertices_end(); ++vit) {
        vList.push_back(vit->point());
    }
    MyKernel::Iso_cuboid_3 bBox = CGAL::bounding_box(vList.begin(), vList.end());
    bbMin = bBox.min();
    bbMax = bBox.max();
    Vector bbVect = bbMax - bbMin;

    f_bbVolume = bbVect.x() * bbVect.y() * bbVect.z();
}

// 获取bounding box的对角线长度
float MyMesh::getBBoxDiagonal() const {
    return sqrt(Vector(bbMin, bbMax).squared_length());
}

// 获取bounding box的center
Vector MyMesh::getBBoxCenter() const {
    return ((bbMax - CGAL::ORIGIN) + (bbMin - CGAL::ORIGIN)) / 2;
}

// 根据boundingbox的大小确定量化的步大小
void MyMesh::determineQuantStep() {
    float f_maxRange = 0;
    for (unsigned i = 0; i < 3; ++i) {
        float range = bbMax[i] - bbMin[i];
        if (range > f_maxRange)
            f_maxRange = range;
    }
    f_quantStep = f_maxRange / (1 << i_quantBits);

    if (b_useLiftingScheme)
        bbMin = bbMin - Vector(1, 1, 1) * (1 << (i_quantBits - 1)) * f_quantStep;

    if (b_useAdaptiveQuantization)
        f_adaptQuantRescaling = 10 / f_maxRange;
}

// 计算存储量化后的信息
void MyMesh::quantizeVertexPositions() {
    printf("Quantize the vertex positions.\n");
    unsigned i_maxCoord = 1 << i_quantBits;

    // Update the positions to fit the quantization.
    for (MyMesh::Vertex_iterator vit = vertices_begin(); vit != vertices_end(); ++vit) {
        Point p = vit->point();
        PointInt quantPoint = getQuantizedPos(p);

        if (!b_useLiftingScheme) {
            // Make sure the last value is in the range.
            assert(quantPoint.x() <= i_maxCoord);
            assert(quantPoint.y() <= i_maxCoord);
            assert(quantPoint.z() <= i_maxCoord);
            /* The max value is the unique that have to to be reassigned
               because else it would get out the range. */
            quantPoint = PointInt(quantPoint.x() == i_maxCoord ? i_maxCoord - 1 : quantPoint.x(),
                                  quantPoint.y() == i_maxCoord ? i_maxCoord - 1 : quantPoint.y(),
                                  quantPoint.z() == i_maxCoord ? i_maxCoord - 1 : quantPoint.z());
        }

        Point newPos = getPos(quantPoint);
        vit->point() = newPos;
    }
}

/**
 * Quantize a position
 */
PointInt MyMesh::getQuantizedPos(Point p) const {
    return PointInt((p.x() - bbMin.x()) / (f_quantStep * (1 << i_curQuantizationId)),
                    (p.y() - bbMin.y()) / (f_quantStep * (1 << i_curQuantizationId)),
                    (p.z() - bbMin.z()) / (f_quantStep * (1 << i_curQuantizationId)));
}

/**
 * Get a position from the quantized coordinates.
 */
Point MyMesh::getPos(PointInt p) const {
    return Point((p.x() + 0.5) * f_quantStep * (1 << i_curQuantizationId) + bbMin.x(),
                 (p.y() + 0.5) * f_quantStep * (1 << i_curQuantizationId) + bbMin.y(),
                 (p.z() + 0.5) * f_quantStep * (1 << i_curQuantizationId) + bbMin.z());
}
