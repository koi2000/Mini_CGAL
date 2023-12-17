#include "../PPMC/BATCH_PPMC/DecompressTool.cuh"

using namespace std;

__global__ void kernel() {
    printf("hello world\n");
}

void compress(int argc, char** argv) {
    const char* strings[] = {"/home/koi/mastercode/Mini_CGAL/buffers/buffer1",
                             "/home/koi/mastercode/Mini_CGAL/buffers/buffer2",
                             "/home/koi/mastercode/Mini_CGAL/buffers/buffer3",
                             "/home/koi/mastercode/Mini_CGAL/buffers/buffer4",
                             "/home/koi/mastercode/Mini_CGAL/buffers/buffer5"};
    char** paths = new char*[sizeof(strings) / sizeof(strings[0])];
    for (size_t i = 0; i < sizeof(strings) / sizeof(strings[0]); ++i) {
        paths[i] = new char[strlen(strings[i]) + 1];
        strcpy(paths[i], strings[i]);
    }
    DeCompressTool* deCompressTool = new DeCompressTool(paths, 5, true);
    int lod = 100;
    char path[256];
    sprintf(path, "./gisdata/compressed_0_mesh_%d.mesh.off");
    deCompressTool->dumpto(path);
    for (uint i = 10; i <= lod; i += 10) {
        deCompressTool->decode(i);
        sprintf(path, "./gisdata/compressed_0_mesh_%d.mesh.off", i);
        deCompressTool->dumpto(path);
    }
    delete deCompressTool;
}

int main(int argc, char** argv) {
    kernel<<<1, 1>>>();
    compress(argc, argv);
    return 0;
}