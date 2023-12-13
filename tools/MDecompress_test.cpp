#include "../PPMC/MCGAL_PPMC/himesh.h"
#include "../PPMC/MCGAL_PPMC/util.h"

using namespace std;
MyMesh* read_mesh(char* path, bool complete_compression) {
    string mesh_str = read_file(path);
    MyMesh* mesh = new MyMesh(mesh_str, complete_compression);
    return mesh;
}

void compress(int argc, char** argv) {
    MyMesh* hm = new MyMesh(argv[1]);
    int lod = 100;

    char path[256];
    sprintf(path, "./gisdata/compressed_0.mesh.off");
    struct timeval start = get_cur_time();
    hm->write_to_off(path);
    for (uint i = 10; i <= lod; i += 10) {
        hm->decode(i);
        logt("decode to %d", start, i);
        sprintf(path, "./gisdata/compressed_%d.mesh.off", i);
        hm->write_to_off(path);
    }
    delete hm;
}

int main(int argc, char** argv) {
    compress(argc, argv);
    return 0;
}