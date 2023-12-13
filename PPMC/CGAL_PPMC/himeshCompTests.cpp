//
// Created by DELL on 2023/11/9.
//
#include "himesh.h"
//

bool MyMesh::willViolateManifold(const std::vector<Halfedge_const_handle>& polygon) const {
    unsigned i_degree = polygon.size();

    // Test if a patch vertex is not connected to one vertex
    // that is not one of its direct neighbor.
    // Test also that two vertices of the patch will not be doubly connected
    // after the vertex cut opeation.
    for (unsigned i = 0; i < i_degree; ++i) {
        Halfedge_around_vertex_const_circulator Hvc = polygon[i]->vertex()->vertex_begin();
        Halfedge_around_vertex_const_circulator Hvc_end = Hvc;
        CGAL_For_all(Hvc, Hvc_end) {
            // Look if the current vertex belongs to the patch.
            Vertex_const_handle vh = Hvc->opposite()->vertex();
            for (unsigned j = 0; j < i_degree; ++j) {
                if (vh == polygon[j]->vertex()) {
                    unsigned i_prev = i == 0 ? i_degree - 1 : i - 1;
                    unsigned i_next = i == i_degree - 1 ? 0 : i + 1;

                    if ((j == i_prev && polygon[i]->facet_degree() != 3)  // The vertex cut operation is forbidden.
                        || (j == i_next &&
                            polygon[i]->opposite()->facet_degree() != 3))  // The vertex cut operation is forbidden.
                        return true;
                }
            }
        }
    }

    return false;
}

Vector MyMesh::computeNormal(const std::vector<Vertex_const_handle>& polygon) const {
    Vector n(0, 0, 0);
    int s = polygon.size();
    for (int i = 0; i < s; ++i) {
        Vector op(CGAL::ORIGIN, polygon[i]->point());
        Vector op2(CGAL::ORIGIN, polygon[(i + 1) % s]->point());
        n = n + CGAL::cross_product(op, op2);
    }

    float f_sqLen = n.squared_length();
    return f_sqLen == 0 ? CGAL::NULL_VECTOR : n / sqrt(f_sqLen);
}

bool MyMesh::isConvex(const std::vector<Vertex_const_handle>& polygon) const {
    // project all points on a plane, taking the first point as origin
    Vector n = computeNormal(polygon);
    int s = polygon.size();
    std::vector<Point> projPoints(s);
    //   printf("s: %i\n", s);
    for (int i = 0; i < s; ++i) {
        // project polygon[i]->point() on the plane with normal n
        projPoints[i] = polygon[i]->point() - n * (Vector(polygon[0]->point(), polygon[i]->point()) * n);
        // 	printf("%f %f %f\n", projPoints[i][0], projPoints[i][1], projPoints[i][2]);
    }

    // now use the following test: a polygon is concave if for each edge, all the other points lie on the same side of
    // the edge
    for (int i = 0; i < s; ++i) {
        Vector ev(projPoints[i], projPoints[(i + 1) % s]);
        int globalSide = 0;
        int comp[9] = {0, 1, 2, 1, 1, 3, 2, 3, 2};
        //(0,0) -> 0
        //(0,+) -> +
        //(0,-) -> -
        //(+,0) -> +
        //(+,+) -> +
        //(+,-) -> 3
        //(-,0) -> -
        //(-,+) -> 3
        //(-,-) -> -
        for (int j = 0; j < s; ++j) {
            if (j == i || j == (i + 1))
                continue;
            Vector dv(projPoints[i], projPoints[j]);
            Vector evxn = CGAL::cross_product(ev, n);
            double cp = evxn * dv;
            int side = (fabs(cp) > 0.000001) * (cp > 0 ? 1 : 2);
            globalSide = comp[globalSide * 3 + side];
            if (globalSide == 3) {
                // 		printf("non convex\n");
                return false;
            }
        }
    }
    //   printf("convex\n");
    return true;
}

bool MyMesh::isRemovable(Vertex_handle v) const {
    //	if(size_of_vertices()<10){
    //		return false;
    //	}
    if (v != vh_departureConquest[0] && v != vh_departureConquest[1] && !v->isConquered() && v->vertex_degree() > 2 &&
        v->vertex_degree() <= 8) {
        // test convexity
        std::vector<Vertex_const_handle> vh_oneRing;
        std::vector<Halfedge_const_handle> heh_oneRing;

        vh_oneRing.reserve(v->vertex_degree());
        heh_oneRing.reserve(v->vertex_degree());
        // vh_oneRing.push_back(v);
        Halfedge_around_vertex_const_circulator hit(v->vertex_begin()), end(hit);
        do {
            vh_oneRing.push_back(hit->opposite()->vertex());
            heh_oneRing.push_back(hit->opposite());
        } while (++hit != end);
        //
        bool removable = !willViolateManifold(heh_oneRing)
        // && isProtruding(heh_oneRing);
        && isConvex(vh_oneRing);
        return removable;
    }
    return false;
}