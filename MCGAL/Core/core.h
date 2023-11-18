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
        }
    };

    static void hash_combine(size_t& seed, size_t hash) {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    float& operator[](int index) {
        if (index >= 0 && index < 3) {
            return v[index];
        } else {
            throw std::out_of_range("Index out of range");
        }
    }

  protected:
    float v[3];
};

class Halfedge;
class Face;
class Vertex;

class Vertex : public Point {
    enum Flag { Unconquered = 0, Conquered = 1 };
    Flag flag = Unconquered;
    unsigned int id = 0;

  public:
    Vertex() : Point() {}
    Vertex(const Point& p) : Point(p) {}
    Vertex(float v1, float v2, float v3) : Point(v1, v2, v3) {}

    int vid_ = 0;
    std::unordered_set<Halfedge*> halfedges;
    std::unordered_set<Halfedge*> opposite_half_edges;

    int vertex_degree() {
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

    Point point() {
        return Point(this);
    }

    void setPoint(const Point& p) {
        this->v[0] = p.x();
        this->v[1] = p.y();
        this->v[2] = p.z();
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

    inline void resetState() {
        flag = Unconquered;
    }

    inline bool isConquered() const {
        return flag == Conquered;
    }

    inline void setConquered() {
        flag = Conquered;
    }

    inline size_t getId() const {
        return id;
    }

    inline void setId(size_t nId) {
        id = nId;
    }
};

class Halfedge {
    enum Flag { NotYetInQueue = 0, InQueue = 1, NoLongerInQueue = 2 };
    enum Flag2 { Original, Added, New };
    enum ProcessedFlag { NotProcessed, Processed };

    Flag flag = NotYetInQueue;
    Flag2 flag2 = Original;
    ProcessedFlag processedFlag = NotProcessed;

  public:
    Halfedge() {}
    Vertex* vertex = nullptr;
    Vertex* end_vertex = nullptr;
    Face* face = nullptr;
    Halfedge* next = nullptr;
    Halfedge* opposite = nullptr;
    Halfedge(Vertex* v1, Vertex* v2);
    ~Halfedge();

    inline void resetState() {
        flag = NotYetInQueue;
        flag2 = Original;
        processedFlag = NotProcessed;
    }

    /* Flag 1 */

    inline void setInQueue() {
        flag = InQueue;
    }

    inline void removeFromQueue() {
        assert(flag == InQueue);
        flag = NoLongerInQueue;
    }

    /* Processed flag */

    inline void resetProcessedFlag() {
        processedFlag = NotProcessed;
    }

    inline void setProcessed() {
        processedFlag = Processed;
    }

    inline bool isProcessed() const {
        return (processedFlag == Processed);
    }

    /* Flag 2 */

    inline void setAdded() {
        assert(flag2 == Original);
        flag2 = Added;
    }

    inline void setNew() {
        assert(flag2 == Original);
        flag2 = New;
    }

    inline bool isAdded() const {
        return flag2 == Added;
    }

    inline bool isOriginal() const {
        return flag2 == Original;
    }

    inline bool isNew() const {
        return flag2 == New;
    }
};

class replacing_group;

class Face {
    typedef MCGAL::Point Point;
    enum Flag { Unknown = 0, Splittable = 1, Unsplittable = 2 };
    enum ProcessedFlag { NotProcessed, Processed };

    Flag flag = Unknown;
    ProcessedFlag processedFlag = NotProcessed;

    Point removedVertexPos;

  public:
    std::unordered_set<Vertex*> vertices;
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

    Face(Halfedge* hit) {
        Halfedge* st(hit);
        Halfedge* ed(hit);
        std::vector<Halfedge*> edges;
        do {
            edges.push_back(st);
            st = st->next;
        } while (st != ed);
        this->reset(edges);
    }

    Face(Vertex* v1, Vertex* v2, Vertex* v3) {
        vertices.insert(v1);
        vertices.insert(v2);
        vertices.insert(v3);
    }

    Face* clone() {
        return new Face(*this);
    }

    Face(std::vector<Vertex*>& vs) {
        Halfedge* prev = nullptr;
        Halfedge* head = nullptr;
        for (int i = 0; i < vs.size(); i++) {
            vertices.insert(vs[i]);
            Vertex* nextv = vs[(i + 1) % vs.size()];
            Halfedge* hf = new Halfedge(vs[i], nextv);
            halfedges.insert(hf);
            // vs[i]->halfedges.insert(hf);
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

    void reset(Halfedge* h) {
        Halfedge* st = h;
        Halfedge* ed = h;
        std::vector<Halfedge*> edges;
        do {
            edges.push_back(st);
            st = st->next;
        } while (st != ed);
        reset(edges);
    }

    void reset(std::vector<Halfedge*>& hs) {
        this->halfedges.clear();
        this->vertices.clear();
        for (int i = 0; i < hs.size(); i++) {
            // hs[i]->next = hs[(i + 1) % hs.size()];
            // hs[i]->face = this;
            halfedges.insert(hs[i]);
            vertices.insert(hs[i]->vertex);
            hs[i]->face = this;
        }
    }

    void erase_vertices(std::vector<Vertex*> rvertices) {
        // eunmerate to delete element
        for (Vertex* vt : rvertices) {
            for (auto iter = vertices.begin(); iter != vertices.end();) {
                if (*iter == vt) {
                    iter = vertices.erase(iter);
                } else {
                    iter++;
                }
            }
            for (auto iter = halfedges.begin(); iter != halfedges.end();) {
                if ((*iter)->end_vertex == vt || (*iter)->vertex == vt) {
                    (*iter)->face = nullptr;
                    iter = halfedges.erase(iter);
                } else {
                    iter++;
                }
            }
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
        for (Vertex* vertix : vertices) {
            if (!rhs.vertices.count(vertix)) {
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

    inline void resetState() {
        flag = Unknown;
        processedFlag = NotProcessed;
    }

    inline void resetProcessedFlag() {
        processedFlag = NotProcessed;
    }

    inline bool isConquered() const {
        return (flag == Splittable || flag == Unsplittable);
    }

    inline bool isSplittable() const {
        return (flag == Splittable);
    }

    inline bool isUnsplittable() const {
        return (flag == Unsplittable);
    }

    inline void setSplittable() {
        assert(flag == Unknown);
        flag = Splittable;
    }

    inline void setUnsplittable() {
        assert(flag == Unknown);
        flag = Unsplittable;
    }

    inline void setProcessedFlag() {
        processedFlag = Processed;
    }

    inline bool isProcessed() const {
        return (processedFlag == Processed);
    }

    inline Point getRemovedVertexPos() const {
        return removedVertexPos;
    }

    inline void setRemovedVertexPos(Point p) {
        removedVertexPos = p;
    }

  public:
    replacing_group* rg = NULL;
};

class Mesh {
  public:
    std::unordered_set<Vertex*, Vertex::Hash, Vertex::Equal> vertices;
    std::unordered_set<Halfedge*> halfedges;
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

    Halfedge* split_facet(Halfedge* h, Halfedge* g) {
        Face* origin = h->face;
        // early expose
        Face* fnew = new Face();
        // create new halfedge
        Halfedge* hnew = new Halfedge(h->end_vertex, g->end_vertex);
        Halfedge* oppo_hnew = new Halfedge(g->end_vertex, h->end_vertex);
        // set the opposite
        hnew->opposite = oppo_hnew;
        oppo_hnew->opposite = hnew;
        // set the connect information
        hnew->next = g->next;
        oppo_hnew->next = h->next;
        h->next = hnew;
        g->next = oppo_hnew;
        // add the new halfedge to origin face

        // delete old halfedge from origin face

        // create new face depend on vertexs
        Halfedge* gst = g->next;
        Halfedge* ged = gst;
        Halfedge* hst = h->next;
        Halfedge* hed = gst;
        std::vector<Halfedge*> origin_face;
        std::vector<Halfedge*> new_face;
        do {
            new_face.push_back(hst);
            hst = hst->next;
        } while (hst != hed);
        // new_face.push_back(hnew);
        // hst = h->next;
        // hed = g->next;
        do {
            origin_face.push_back(gst);
            gst = gst->next;
        } while (gst != ged);
        // origin_face.push_back(oppo_hnew);
        origin->reset(origin_face);
        fnew->reset(new_face);
        // add halfedge and face to mesh
        this->halfedges.insert(hnew);
        this->halfedges.insert(oppo_hnew);
        this->faces.insert(fnew);
        return hnew;
    }

    Halfedge* create_center_vertex(Halfedge* h) {
        this->faces.erase(h->face);
        Vertex* vnew = new Vertex();
        Halfedge* hnew = new Halfedge(h->end_vertex, vnew);
        // close_tip(hnew, vnew);
        Halfedge* oppo_new = new Halfedge(vnew, h->end_vertex);
        this->halfedges.insert(hnew);
        this->halfedges.insert(oppo_new);
        hnew->opposite = oppo_new;
        oppo_new->opposite = hnew;
        insert_tip(hnew->opposite, h);
        // hnew->face = new Face(hnew);
        // this->faces.insert(hnew->face);
        Halfedge* g = hnew->opposite->next;
        std::vector<Halfedge*> origin_around_halfedge;
        origin_around_halfedge.push_back(h);
        while (g->next != hnew) {
            Halfedge* gnew = new Halfedge(g->end_vertex, vnew);
            Halfedge* oppo_gnew = new Halfedge(vnew, g->end_vertex);
            this->halfedges.insert(gnew);
            this->halfedges.insert(oppo_gnew);
            gnew->opposite = oppo_gnew;
            oppo_gnew->opposite = gnew;
            origin_around_halfedge.push_back(g);
            // insert_tip(gnew, hnew);
            gnew->next = hnew->opposite;
            insert_tip(gnew->opposite, g);
            // Face* fnew = new Face(gnew);
            // this->faces.insert(fnew);
            g = gnew->opposite->next;
            hnew = gnew;
        }
        // collect all the halfedge
        for (Halfedge* hit : origin_around_halfedge) {
            Face* face = new Face(hit);
            this->faces.insert(face);
        }
        // create new face and set new face

        return hnew;
    }

    void close_tip(Halfedge* h, Vertex* v) const {
        h->next = h->opposite;
        h->vertex = v;
    }

    void insert_tip(Halfedge* h, Halfedge* v) const {
        h->next = v->next;
        v->next = h->opposite;
    }

    Halfedge* find_prev(Halfedge* h) const {
        Halfedge* g = h;
        while (g->next != h)
            g = g->next;
        return g;
    }

    Halfedge* erase_center_vertex(Halfedge* h) {
        Halfedge* g = h->next->opposite;
        Halfedge* hret = find_prev(h);
        Face* face = new Face();
        faces.insert(face);
        while (g != h) {
            Halfedge* gprev = find_prev(g);
            remove_tip(gprev);
            // if (g->face != face) {
            faces.erase(g->face);
            // }
            Halfedge* gnext = g->next->opposite;
            this->halfedges.erase(g);
            g = gnext;
        }
        faces.erase(h->face);
        remove_tip(hret);
        vertices.erase(h->end_vertex);
        this->halfedges.erase(h);
        set_face_in_face_loop(hret, face);
        return hret;
    }

    void set_face_in_face_loop(Halfedge* h, Face* f) const {
        Halfedge* end = h;
        do {
            h->face = f;
            f->halfedges.insert(h);
            f->vertices.insert(h->vertex);
            h = h->next;
        } while (h != end);
    }

    void remove_tip(Halfedge* h) const {
        h->next = h->next->opposite->next;
    }

    Halfedge* join_face(Halfedge* h) {
        Halfedge* hprev = find_prev(h);
        Halfedge* gprev = find_prev(h->opposite);
        remove_tip(hprev);
        remove_tip(gprev);
        // hds->edges_erase(h);
        // halfedges.erase(h);
        for (auto it = halfedges.begin(); it != halfedges.end(); it++) {
            if (*it == h) {
                halfedges.erase(it);
                break;
            }
        }

        for (auto it = halfedges.begin(); it != halfedges.end(); it++) {
            if (*it == h->opposite) {
                halfedges.erase(it);
                break;
            }
        }
        if (gprev->face != h->face)
            faces.erase(gprev->face);

        h = hprev;
        h->face->reset(h);
        return hprev;
    }
};

class replacing_group {
  public:
    replacing_group() {
        // cout<<this<<" is constructed"<<endl;
        id = counter++;
        alive++;
    }
    ~replacing_group() {
        removed_vertices.clear();
        alive--;
    }

    void print() {
        // log("%5d (%2d refs %4d alive) - removed_vertices: %ld", id, ref, alive, removed_vertices.size());
    }

    std::unordered_set<MCGAL::Point, Point::Hash, Point::Equal> removed_vertices;
    // unordered_set<Triangle> removed_triangles;
    int id;
    int ref = 0;

    static int counter;
    static int alive;
};

}  // namespace MCGAL