cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12/bin/nvcc")
project(Mini_CGAL LANGUAGES  CXX C CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89) 

set(CMAKE_CUDA_FLAGS "-arch=compute_89 -code=sm_89 -rdc=true -g -G -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -std=c++14")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wno-unused-result")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")


# add_definitions(-DGRID_SIZE=${GRID_SIZE})
# add_definitions(-DBLOCK_SIZE=${BLOCK_SIZE})
# set(CMAKE_BUILD_TYPE "Debug")
# set(CUDA_DEBUG ON)
# set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0")
option(GRID "Description" 1)
option(BLOCK "Description" 512)

include_directories(
    ./include
    /usr/include
    /usr/local/include
    /usr/local/cgal/include
    # /usr/local/include/eigen3
    /usr/include/eigen3
    ${CUDA_INCLUDE_DIRS}
)
find_package(CGAL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(CUDA REQUIRED)

# Add library paths
link_directories(
    /usr/local/cgal/lib
    /usr/lib/x86_64-linux-gnu/
    # /usr/lib/x86_64-linux-gnu/
)

set(PPMC_SOURCE_DIR "PPMC")
set(MINICGAL_SOURCE_DIR "MCGAL/Core")
set(MINICGAL_CUDA_SOURCE_DIR "MCGAL/Core_CUDA")
set(MINICGAL_PPMC_SOURCE_DIR "MCGAL_PPMC")
set(PPMC_CUDA_SOURCE_DIR "PPMC_CUDA")

include(${CGAL_USE_FILE})

# get file list
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

# file(GLOB_RECURSE MINICGAL_CUDA_PPMC_FILES
#     "${MINICGAL_CUDA_SOURCE_DIR}/*.h"
#     "${MINICGAL_CUDA_SOURCE_DIR}/include/*.h"
#     "${MINICGAL_CUDA_SOURCE_DIR}/include/*.cuh"
#     "${MINICGAL_CUDA_SOURCE_DIR}/*.cpp"
#     "${MINICGAL_CUDA_SOURCE_DIR}/*.cu"
#     "${MINICGAL_CUDA_SOURCE_DIR}/*.cuh"
#     "${PPMC_CUDA_SOURCE_DIR}/*.h"
#     "${PPMC_CUDA_SOURCE_DIR}/*.cpp"
#     "${PPMC_CUDA_SOURCE_DIR}/*.cu"
#     "${PPMC_CUDA_SOURCE_DIR}/*.cuh"
# )

file(GLOB_RECURSE CUDA_PPMC
    "${PPMC_CUDA_SOURCE_DIR}/*.cu"
    "${PPMC_CUDA_SOURCE_DIR}/*.cuh"
)

file(GLOB_RECURSE MINICGAL_CUDA
    "${MINICGAL_CUDA_SOURCE_DIR}/include/*.cu"
    "${MINICGAL_CUDA_SOURCE_DIR}/include/*.cuh"
    "${MINICGAL_CUDA_SOURCE_DIR}/*.cu"
    "${MINICGAL_CUDA_SOURCE_DIR}/*.cuh"
)

add_subdirectory(CUDA_new)

# add_executable(Mini_CGAL main.cpp ${PPMC_FILES})
# set_target_properties(Mini_CGAL PROPERTIES COMPILE_FLAGS -frounding-math)
# target_link_libraries(Mini_CGAL ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

# add_executable(PPMC_compress tools/PPMC_test.cpp ${PPMC_FILES})
# set_target_properties(PPMC_compress PROPERTIES COMPILE_FLAGS -frounding-math)
# target_link_libraries(PPMC_compress ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

# add_executable(Mini_CGAL_test tools/MiniCGAL_test.cpp ${MINI_CGAL_FILES})
# set_target_properties(Mini_CGAL_test PROPERTIES COMPILE_FLAGS -frounding-math)
# target_link_libraries(Mini_CGAL_test ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

# add_executable(MiniCGAL_Compress tools/MCompress_test.cpp ${MINICGAL_PPMC_FILES})
# set_target_properties(MiniCGAL_Compress PROPERTIES COMPILE_FLAGS -frounding-math)
# target_link_libraries(MiniCGAL_Compress ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

# add_executable(MiniCGAL_Decompress tools/MDecompress_test.cpp ${MINICGAL_PPMC_FILES})
# set_target_properties(MiniCGAL_Decompress PROPERTIES COMPILE_FLAGS -frounding-math)
# target_link_libraries(MiniCGAL_Decompress ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

# add_executable(MiniCGAL_CudaDecompress tools/MCudaDecompress_test.cu ${MINICGAL_CUDA_PPMC_FILES})
add_executable(MiniCGAL_CudaDecompress tools/MCudaDecompress_test.cu  ${CUDA_PPMC} ${MINICGAL_CUDA})
set_target_properties(MiniCGAL_CudaDecompress PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# target_compile_definitions(MiniCGAL_CudaDecompress PUBLIC BLOCK_SIZE=${BLOCK} GRID_SIZE=${GRID} PROHIBIT_LOG)
target_compile_definitions(MiniCGAL_CudaDecompress PRIVATE PROHIBIT_LOG)
target_compile_definitions(MiniCGAL_CudaDecompress PRIVATE BLOCK_SIZE=${BLOCK})
target_compile_definitions(MiniCGAL_CudaDecompress PRIVATE GRID_SIZE=${GRID})
target_link_libraries(MiniCGAL_CudaDecompress ${CUDA_LIBRARIES})

message(STATUS "list = ${CUDA_PPMC}" )