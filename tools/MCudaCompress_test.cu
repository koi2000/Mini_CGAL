#include "../PPMC/CUDA_PPMC/mymesh.cuh"
#include "../PPMC/CUDA_PPMC/util.h"
using namespace std;
MyMesh* read_mesh(char* path, bool complete_compression) {
    string mesh_str = read_file(path);
    MyMesh* mesh = new MyMesh(mesh_str, complete_compression);
    return mesh;
}

void compress(int argc, char** argv) {
    struct timeval start = get_cur_time();
    MyMesh* mesh = read_mesh(argv[1], true);
    logt("compress", start);
    char path[256];
    MyMesh* hm = new MyMesh(mesh);
    sprintf(path, "./gisdata2/compressed_0.mesh.off");
    hm->write_to_off(path);
    int lod = 100;
    for (uint i = 20; i <= lod; i += 20) {
        hm->decode(i);
        logt("decode to %d", start, i);
        // log("%d %f", i, MyMesh->getHausdorfDistance());
        sprintf(path, "./gisdata2/compressed_%d.mesh.off", i);
        hm->write_to_off(path);
    }

    delete mesh;
    delete hm;
}

int main(int argc, char** argv) {
    compress(argc, argv);
    return 0;
}