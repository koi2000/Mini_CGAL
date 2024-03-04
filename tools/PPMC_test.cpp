#include "../PPMC/CGAL_PPMC/himesh.h"

MyMesh* read_mesh(char* path, bool complete_compression) {
    string mesh_str = hispeed::read_file(path);
    MyMesh* mesh = new MyMesh(mesh_str, complete_compression);
    return mesh;
}

void compress(int argc, char** argv) {
    struct timeval start = hispeed::get_cur_time();
    MyMesh* mesh = read_mesh(argv[1], true);
    mesh->dumpBuffer(argv[2]);
    hispeed::logt("compress", start);
    MyMesh* hm = new MyMesh(mesh);
    int lod = 100;

    char path[256];
    sprintf(path, "./gisdata1/compressed_0.mesh.off");
    hm->write_to_off(path);
    for (uint i = 10; i <= lod; i += 10) {
        hm->decode(i);
        // hispeed::logt("decode to %d", start, i);
        // log("%d %f", i, MyMesh->getHausdorfDistance());
        sprintf(path, "./gisdata1/compressed_%d.mesh.off", i);
        hm->write_to_off(path);
    }
    hispeed::logt("decode", start);
    delete mesh;
    delete hm;
}

int main(int argc, char** argv) {
    compress(argc, argv);
    return 0;
}