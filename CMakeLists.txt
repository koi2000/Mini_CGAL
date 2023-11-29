cmake_minimum_required(VERSION 3.16)
project(Mini_CGAL)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wno-unused-result")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")


set(CMAKE_BUILD_TYPE "Debug")

include_directories(
    ./include
    /usr/include
    /usr/local/include
    /usr/local/cgal/include
    # /usr/local/include/eigen3
    /usr/include/eigen3
)

# Add library paths
link_directories(
    /usr/local/cgal/lib
    /usr/lib/x86_64-linux-gnu/
)

set(PPMC_SOURCE_DIR "PPMC")
set(MINICGAL_SOURCE_DIR "MCGAL/Core")
set(MINICGAL_PPMC_SOURCE_DIR "MCGAL_PPMC")

# set(RANGECODER_SOURCE_DIR "PPMC/rangeCoder")
find_package(CGAL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include(${CGAL_USE_FILE})

# 获取源文件列表
file(GLOB_RECURSE PPMC_FILES
    "${PPMC_SOURCE_DIR}/*.h"
    "${PPMC_SOURCE_DIR}/*.cpp"
)

file(GLOB_RECURSE MINI_CGAL_FILES
    "${MINICGAL_SOURCE_DIR}/*.h"
    "${MINICGAL_SOURCE_DIR}/*.cpp"
)

file(GLOB_RECURSE MINICGAL_PPMC_FILES
    "${MINICGAL_SOURCE_DIR}/*.h"
    "${MINICGAL_SOURCE_DIR}/*.cpp"
    "${MINICGAL_PPMC_SOURCE_DIR}/*.h"
    "${MINICGAL_PPMC_SOURCE_DIR}/*.cpp"
)

add_subdirectory(CUDA_new)

add_executable(Mini_CGAL main.cpp ${PPMC_FILES})
set_target_properties(Mini_CGAL PROPERTIES COMPILE_FLAGS -frounding-math)
target_link_libraries(Mini_CGAL ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

add_executable(PPMC_compress tools/PPMC_test.cpp ${PPMC_FILES})
set_target_properties(PPMC_compress PROPERTIES COMPILE_FLAGS -frounding-math)
target_link_libraries(PPMC_compress ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

add_executable(Mini_CGAL_test tools/MiniCGAL_test.cpp ${MINI_CGAL_FILES})
set_target_properties(Mini_CGAL_test PROPERTIES COMPILE_FLAGS -frounding-math)
target_link_libraries(Mini_CGAL_test ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

add_executable(MiniCGAL_Compress tools/MCompress_test.cpp ${MINICGAL_PPMC_FILES})
set_target_properties(MiniCGAL_Compress PROPERTIES COMPILE_FLAGS -frounding-math)
target_link_libraries(MiniCGAL_Compress ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

add_executable(MiniCGAL_Decompress tools/MDecompress_test.cpp ${MINICGAL_PPMC_FILES})
set_target_properties(MiniCGAL_Decompress PROPERTIES COMPILE_FLAGS -frounding-math)
target_link_libraries(MiniCGAL_Decompress ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)