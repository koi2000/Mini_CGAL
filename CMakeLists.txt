cmake_minimum_required(VERSION 3.16)
project(Mini_CGAL)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wno-unused-result")
set(CMAKE_BUILD_TYPE "Debug")

include_directories(
    ./include
    /usr/include
    /usr/local/include
    /usr/local/cgal/include
    /usr/local/include/eigen3
)

# Add library paths
link_directories(
    /usr/local/cgal/lib
    /usr/lib/x86_64-linux-gnu/
)

set(PPMC_SOURCE_DIR "PPMC")
set(MINICGAL_SOURCE_DIR "MCGAL")
# set(RANGECODER_SOURCE_DIR "PPMC/rangeCoder")
find_package(CGAL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include(${CGAL_USE_FILE})

# 获取源文件列表
file(GLOB_RECURSE SOURCE_FILES
        "${PPMC_SOURCE_DIR}/*.h"
        "${PPMC_SOURCE_DIR}/*.cpp"
        "${MINICGAL_SOURCE_DIR}/*.h"
        "${MINICGAL_SOURCE_DIR}/*.cpp"
        )

# add_executable(Mini_CGAL main.cpp ${SOURCE_FILES})
# set_target_properties(Mini_CGAL PROPERTIES COMPILE_FLAGS -frounding-math)
# target_link_libraries(Mini_CGAL ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

# add_executable(compress tools/compress.cpp ${SOURCE_FILES})
# set_target_properties(Mini_CGAL PROPERTIES COMPILE_FLAGS -frounding-math)
# target_link_libraries(compress ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)

add_executable(Mini_CGAL_test tools/MiniCGAL_test.cpp ${SOURCE_FILES})
set_target_properties(Mini_CGAL_test PROPERTIES COMPILE_FLAGS -frounding-math)
target_link_libraries(Mini_CGAL_test ${CGAL_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} m)