#include <assert.h>
#include <string>
#include <unordered_set>
#include <vector>
namespace MCGAL {

class Point {
  public:
    Point() {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
    }

    Point(float x, float y, float z) {
        v[0] = x;
        v[1] = y;
        v[2] = z;
    }

    Point(Point* pt) {
        assert(pt);
        for (int i = 0; i < 3; i++) {
            v[i] = pt->v[i];
        }
    };

    float x() const {
        return v[0];
    }

    float y() const {
        return v[1];
    }

    float z() const {
        return v[2];
    }

    // Hash function for Point
    struct Hash {
        size_t operator()(const Point point) const {
            // Use a combination of hash functions for each member
            size_t hash = std::hash<float>{}(point.x());
            hash_combine(hash, std::hash<float>{}(point.y()));
            hash_combine(hash, std::hash<float>{}(point.z()));
            return hash;
        }
    };

    // Equality comparison for Vertex
    struct Equal {
        bool operator()(const Point p1, const Point p2) const {
            // Compare each member for equality
            return p1.x() == p2.x() && p1.y() == p2.y() && p1.z() == p2.z();
            // Add other members if needed
        }
    };

    static void hash_combine(size_t& seed, size_t hash) {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

  protected:
    float v[3];
};

class Halfedge;
class Face;
class Vertex;

class Vertex : public Point {
  public:
    Vertex() : Point() {}
    Vertex(const Point& p) : Point(p) {}
    Vertex(float v1, float v2, float v3) : Point(v1, v2, v3) {}

    int vid_ = 0;
    std::unordered_set<Halfedge*> halfedges;
    std::unordered_set<Halfedge*> opposite_half_edges;

    int degree() {
        return halfedges.size();
    }

    void print() {
        printf("%f %f %f\n", v[0], v[1], v[2]);
    }

    void setVid(int id) {
        this->vid_ = id;
    }

    float x() const {
        return v[0];
    }

    float y() const {
        return v[1];
    }

    float z() const {
        return v[2];
    }

    int vid() const {
        return vid_;
    }

    // Hash function for Vertex
    struct Hash {
        size_t operator()(const Vertex* vertex) const {
            // Use a combination of hash functions for each member
            size_t hash = std::hash<float>{}(vertex->x());
            hash_combine(hash, std::hash<float>{}(vertex->y()));
            hash_combine(hash, std::hash<float>{}(vertex->z()));
            hash_combine(hash, std::hash<int>{}(vertex->vid()));
            return hash;
        }
    };

    // Equality comparison for Vertex
    struct Equal {
        bool operator()(const Vertex* v1, const Vertex* v2) const {
            // Compare each member for equality
            return v1->x() == v2->x() && v1->y() == v2->y() && v1->z() == v2->z();
            // Add other members if needed
        }
    };

    static void hash_combine(size_t& seed, size_t hash) {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

class Halfedge {
  public:
    Halfedge() {}
    Vertex* vertex = nullptr;
    Vertex* end_vertex = nullptr;
    Face* face = nullptr;
    Halfedge* next = nullptr;
    Halfedge* opposite = nullptr;
    Halfedge(Vertex* v1, Vertex* v2);
    ~Halfedge();
};

class Face {
  public:
    std::vector<Vertex*> vertices;
    std::unordered_set<Halfedge*> halfedges;

  public:
    Face(){};
    ~Face() {
        for (Halfedge* h : halfedges) {
            delete h;
        }
        halfedges.clear();
        vertices.clear();
    }

    Face(const Face& face) {
        this->vertices = face.vertices;
        this->halfedges = face.halfedges;
    }

    Face(Vertex* v1, Vertex* v2, Vertex* v3) {
        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v3);
    }

    Face* clone() {
        return new Face(*this);
    }

    Face(std::vector<Vertex*>& vs) {
        Halfedge* prev = nullptr;
        Halfedge* head = nullptr;
        for (int i = 0; i < vs.size(); i++) {
            vertices.push_back(vs[i]);
            Vertex* nextv = vs[(i + 1) % vs.size()];
            Halfedge* hf = new Halfedge(vs[i], nextv);
            halfedges.insert(hf);
            vs[i]->halfedges.insert(hf);
            hf->face = this;
            if (prev != NULL) {
                prev->next = hf;
            } else {
                head = hf;
            }
            if (i == vs.size() - 1) {
                hf->next = head;
            }
            prev = hf;
        }
    }

    void print() {
        printf("totally %ld vertices:\n", vertices.size());
        int idx = 0;
        for (Vertex* v : vertices) {
            printf("%d:\t", idx++);
            v->print();
        }
    }

    void print_off() {
        printf("OFF\n%ld 1 0\n", vertices.size());
        for (Vertex* v : vertices) {
            v->print();
        }
        printf("%ld\t", vertices.size());
        for (int i = 0; i < vertices.size(); i++) {
            printf("%d ", i);
        }
        printf("\n");
    }

    bool equal(const Face& rhs) const {
        if (vertices.size() != rhs.vertices.size()) {
            return false;
        }
        for (int i = 0; i < vertices.size(); i++) {
            if (vertices[i] != rhs.vertices[i]) {
                return false;
            }
        }
        return true;
    }
    bool operator==(const Face& rhs) const {
        return this->equal(rhs);
    }

    int facet_degree() {
        return vertices.size();
    }
    // split the face and make sure the one without v as the new
    Face* split(Vertex* v);
    void remove(Halfedge* h);
};

class Mesh {
  public:
    std::unordered_set<Vertex*, Vertex::Hash, Vertex::Equal> vertices;
    std::vector<Halfedge*> halfedges;
    std::unordered_set<Face*> faces;
    // 用于dump OFF文件
    std::vector<std::vector<int>> face_index;
    int nb_vertices = 0;
    int nb_faces = 0;
    int nb_edges = 0;

  public:
    Mesh() {}
    ~Mesh();

    // IOS
    bool loadOFF(std::string path);
    void dumpto(std::string path);
    void print();
    std::string to_string();
    Vertex* get_vertex(int vseq = 0);

    // element operating
    Face* add_face(std::vector<Vertex*>& vs);
    Face* add_face(Face* face);
    Face* remove_vertex(Vertex* v);
    Halfedge* merge_edges(Vertex* v);

    /*
     * statistics
     *
     * */
    size_t size_of_vertices() {
        return vertices.size();
    }

    size_t size_of_facets() {
        return faces.size();
    }

    size_t size_of_halfedges() {
        return halfedges.size();
    }

    Halfedge* create_center_vertex(Halfedge* h) {
        Halfedge* hnew = new Halfedge();
        Vertex* vnew = new Vertex();
        close_tip(hnew, vnew);
        insert_tip(hnew->opposite, h);
        hnew->face = h->face;
        // set_face_halfedge(h);
        Halfedge* g = hnew->opposite->next;
        while (g->next != hnew) {
            Halfedge* gnew = new Halfedge();
            insert_tip(gnew, hnew);
            insert_tip(gnew->opposite, g);
            Face* fnew = hnew->face->clone();
            g->face = fnew;
            gnew->face = fnew;
            gnew->next->face = fnew;
            g = gnew->opposite->next;
        }
        hnew->next->face = hnew->face;
        return hnew;
    }

    void close_tip(Halfedge* h, Vertex* v) const {
        // makes `h->opposite()' the successor of h and sets the incident
        // vertex of h to v.
        h->next = h->opposite;
        // set_prev(h->opposite(), h);
        h->vertex = v;
    }

    void insert_tip(Halfedge* h, Halfedge* v) const {
        // inserts the tip of the edge h into the halfedges around the
        // vertex pointed to by v. Halfedge `h->opposite()' is the new
        // successor of v and `h->next()' will be set to `v->next()'. The
        // vertex of h will be set to the vertex v refers to if vertices
        // are supported.
        h->next = v->next;
        v->next = h->opposite;
        // set_prev(h->next(), h);
        // set_prev(h->opposite(), v);
        h->vertex = v->vertex;
    }

    Halfedge* find_prev(Halfedge* h) const {
        Halfedge* g = h;
        while (g->next != h)
            g = g->next;
        return g;
    }

    Halfedge* erase_center_vertex(Halfedge* h) {
        // h points to the vertex that gets removed
        Halfedge* g = h->next->opposite;
        Halfedge* hret = find_prev(h);
        Face* face = new Face();
        while (g != h) {
            Halfedge* gprev = find_prev(g);
            // set_vertex_halfedge(gprev);
            remove_tip(gprev);
            if (g->face != face) {
                faces.erase(g->face);
            }
            Halfedge* gnext = g->next->opposite;
            // hds->edges_erase(g);
            g = gnext;
        }
        // set_vertex_halfedge(hret);
        remove_tip(hret);
        // vertices_erase(get_vertex(h));
        vertices.erase(h->end_vertex);
        // hds->edges_erase(h);
        set_face_in_face_loop(hret, face);
        // set_face_halfedge(hret);
        return hret;
    }

    void set_face_in_face_loop(Halfedge* h, Face* f) const {
        Halfedge* end = h;
        do {
            h->face = f;
            h = h->next;
            f->halfedges.insert(h);
            f->vertices.push_back(h->vertex);
        } while (h != end);
    }

    void remove_tip(Halfedge* h) const {
        // removes the edge `h->next()->opposite()' from the halfedge
        // circle around the vertex referred to by h. The new successor
        // halfedge of h will be `h->next()->opposite()->next()'.
        // Halfedge* prev = find_prev(h);
        // prev->next = h;
        h->next = h->next->opposite->next;
        // set_prev(h->next, h);
    }
};

}  // namespace MCGAL