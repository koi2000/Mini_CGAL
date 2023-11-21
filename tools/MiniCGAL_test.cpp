#include "../MCGAL/Core/core.h"
#include <iostream>
#include <unordered_set>
using namespace MCGAL;
using namespace std;
int main() {
    // Mesh* mesh = new Mesh();
    // mesh->loadOFF("/home/koi/mastercode/Mini_CGAL/static/compressed_20.mesh.off");
    // delete mesh;
    // mesh = nullptr;
    Vertex* v1 = new Vertex(1, 1, 1);
    Vertex* v2 = new Vertex(2, 2, 2);
    Halfedge* h1 = new Halfedge(v1, v2);
    Halfedge* h2 = new Halfedge(v1, v2);
    unordered_set<Halfedge*, Halfedge::Hash, Halfedge::Equal> st;
    auto fn = st.hash_function();
    cout << fn(h1) << endl;
    cout << fn(h2) << endl;
    return 0;
}