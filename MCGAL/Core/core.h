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

  protected:
    float v[3];
};

class Halfedge;
class Face;
class Vertex;

class Vertex : public Point {
  public:
    Vertex() : Point() {}
    Vertex(float v1, float v2, float v3) : Point(v1, v2, v3) {}

    int id_ = 0;
    std::unordered_set<Halfedge*> halfedges;
    std::unordered_set<Halfedge*> opposite_half_edges;

    int degree() {
        return halfedges.size();
    }

    void print() {
        printf("%f %f %f\n", v[0], v[1], v[2]);
    }

    void setId(int id) {
        this->id_ = id;
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

    int id() const {
        return id_;
    }

    // Hash function for Vertex
    struct Hash {
        size_t operator()(const Vertex* vertex) const {
            // Use a combination of hash functions for each member
            size_t hash = std::hash<float>{}(vertex->x());
            hash_combine(hash, std::hash<float>{}(vertex->y()));
            hash_combine(hash, std::hash<float>{}(vertex->z()));
            hash_combine(hash, std::hash<int>{}(vertex->id()));
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

    Face(Vertex* v1, Vertex* v2, Vertex* v3) {
        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v3);
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
    // bool loadOFF(char* data, bool owned = false);
    bool loadOFF(std::string path);
    // bool loadOFF(char* path);
    bool parse(std::string str);
    bool parse(char* str, size_t size);
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
};

}  // namespace MCGAL