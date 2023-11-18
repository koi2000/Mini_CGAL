#include "../MCGAL/Core/core.h"
using namespace MCGAL;
int main() {
    Mesh* mesh = new Mesh();
    mesh->loadOFF("/home/koi/mastercode/Mini_CGAL/static/compressed_20.mesh.off");
    delete mesh;
    mesh = nullptr;
    return 0;
}