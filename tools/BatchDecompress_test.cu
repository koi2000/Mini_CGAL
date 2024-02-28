#include "../PPMC/BATCH_PPMC/DecompressTool.cuh"
#include <iostream>
using namespace std;

__global__ void kernel() {
    printf("hello world\n");
}

void compress(int argc, char** argv) {
    // const char* strings[] = {
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer1", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer2",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer3", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer4",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer5", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer6",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer7", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer8",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer9", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer10",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer11", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer12",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer13", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer14",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer15", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer16",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer17", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer18",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer19", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer20"};
    const char* strings[] = {
        "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer1", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer2",
        "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer3", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer4",
        "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer5", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer6",
        "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer7", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer8",
        "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer9", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer10"};
    // const char* strings[] = {
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer1", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer2",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer3"};
    char** paths = new char*[sizeof(strings) / sizeof(strings[0])];
    for (size_t i = 0; i < sizeof(strings) / sizeof(strings[0]); ++i) {
        paths[i] = new char[strlen(strings[i]) + 1];
        strcpy(paths[i], strings[i]);
    }
    DeCompressTool* deCompressTool = new DeCompressTool(paths, 10, true);
    int lod = 100;
    char path[256];
    sprintf(path, "%s", "./gisdata/compressed_0_mesh_%d_mesh.off");
    deCompressTool->dumpto(path);
    struct timeval start = get_cur_time();
    for (uint i = 10; i <= lod; i += 10) {
        deCompressTool->decode(i);
        logt("decode to %d", start, i);
        sprintf(path, "./gisdata/compressed_%d%s", i, "_mesh_%d_mesh.off");
        // std::cout << path << std::endl;
        // printf("%s", path);
        // deCompressTool->dumpto(path);
    }
    delete deCompressTool;
}

int main(int argc, char** argv) {
    // kernel<<<1, 1>>>();
    compress(argc, argv);
    return 0;
}