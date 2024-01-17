#include "../PPMC/MCGAL_PPMC/himesh.h"
#include "../PPMC/MCGAL_PPMC/util.h"
#include "omp.h"
using namespace std;
HiMesh* read_mesh(char* path, bool complete_compression) {
    string mesh_str = read_file(path);
    HiMesh* mesh = new HiMesh(mesh_str, complete_compression);
    return mesh;
}

void compress(int argc, char** argv) {
    struct timeval start = get_cur_time();
    HiMesh* mesh = read_mesh(argv[1], true);
    logt("compress", start);
    HiMesh* hm = new HiMesh(mesh);
    int lod = 100;
    omp_set_nested(1);
    char path[256];
    sprintf(path, "./gisdata1/compressed_0.mesh.off");
    hm->write_to_off(path);
    for (uint i = 10; i <= lod; i += 10) {
        hm->decode(i);
        logt("decode to %d", start, i);
        // log("%d %f", i, HiMesh->getHausdorfDistance());
        sprintf(path, "./gisdata1/compressed_%d.mesh.off", i);
        hm->write_to_off(path);
    }

    delete mesh;
    delete hm;
}

int main(int argc, char** argv) {
    compress(argc, argv);
    return 0;
}