#include "../PPMC/BATCH_PPMC_CPU/DecompressTool.h"
#include <iostream>
using namespace std;

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
    // const char* strings[] = {
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer1", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer2",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer3", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer4",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer5", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer6",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer7", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer8",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer9", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer10"};
    // const char* strings[] = {
    //     "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer1", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer2",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer3", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer4",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer5", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer6",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer7", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer8",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer9", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer10"
    //     // "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer11", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer12",
    //     // "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer13", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer14",
    //     // "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer15", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer16",
    //     // "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer17", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer18",
    //     // "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer19", "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer20"
    //     };
    // const char* strings[] = {
    //     "/home/koi/mastercode/Mini_CGAL/buffers/avoidCompetitionBuffer11"};
    // const char* strings[] = {
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer1", "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer2",
    //     "/home/koi/mastercode/Mini_CGAL/buffers/newbuffer3"};
    const char* strings[] = {
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer1", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer2",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer3", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer4",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer5", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer6",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer7", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer8",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer9", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer10",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer11", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer12",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer13", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer14",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer15", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer16",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer17", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer18",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer19", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer20",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer21", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer22",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer23", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer24",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer25", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer26",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer27", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer28",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer29", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer30",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer31", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer32",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer33", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer34",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer35", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer36",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer37", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer38",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer39", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer40",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer41", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer42",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer43", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer44",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer45", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer46",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer47", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer48",
        "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer49", "/home/koi/mastercode/Mini_CGAL/buffers/oldbuffer50",
        };
    char** paths = new char*[sizeof(strings) / sizeof(strings[0])];
    for (size_t i = 0; i < sizeof(strings) / sizeof(strings[0]); ++i) {
        paths[i] = new char[strlen(strings[i]) + 1];
        strcpy(paths[i], strings[i]);
    }
    DeCompressTool* deCompressTool = new DeCompressTool(paths, 1, true);
    int lod = 100;
    char path[256];
    sprintf(path, "%s", "./gisdata/compressed_0_mesh_%d_mesh.off");
    deCompressTool->dumpto(path);
    struct timeval start = get_cur_time();
    for (uint i = 10; i <= lod; i += 10) {
        deCompressTool->decode(i);
        // logt("decode to %d", start, i);
        sprintf(path, "./gisdata/compressed_%d%s", i, "_mesh_%d_mesh.off");
        // std::cout << path << std::endl;
        // printf("%s", path);
        // deCompressTool->dumpto(path);
    }
    logt("decode", start);
    delete deCompressTool;
}

int main(int argc, char** argv) {
    // kernel<<<1, 1>>>();
    compress(argc, argv);
    return 0;
}