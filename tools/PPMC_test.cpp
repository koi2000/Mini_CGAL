#include "../PPMC/mymesh.h"

MyMesh* read_mesh(char* path, bool complete_compression) {
    string mesh_str = hispeed::read_file(path);
    MyMesh* mesh = new MyMesh(mesh_str, complete_compression);
    return mesh;
}

void compress(int argc, char** argv) {
    struct timeval start = hispeed::get_cur_time();
    MyMesh* mesh = read_mesh(argv[1], true);
    hispeed::logt("compress", start);
    MyMesh* hm = new MyMesh(mesh);
    int lod = 100;

    char path[256];
    for (uint i = 20; i <= lod; i += 20) {
        hm->decode(i);
        hispeed::logt("decode to %d", start, i);
        // log("%d %f", i, MyMesh->getHausdorfDistance());
        sprintf(path, "./gisdata/compressed_%d.mesh.off", i);
        hm->write_to_off(path);
    }

    delete mesh;
    delete hm;
}

int main(int argc, char** argv) {
    compress(argc, argv);
    return 0;
}