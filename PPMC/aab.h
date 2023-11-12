#ifndef AAB_H
#define AAB_H

#include <assert.h>
#include <float.h>
#include <immintrin.h>
#include <iostream>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <vector>

typedef struct range
{
  public:
    float mindist = 0;
    float maxdist = 0;

    bool operator>(range& d) {
        return mindist > d.maxdist;
    }
    bool operator>=(range& d) {
        return mindist >= d.maxdist;
    }
    bool operator<(range& d) {
        return maxdist < d.mindist;
    }
    bool operator<=(range& d) {
        return maxdist <= d.mindist;
    }
    bool operator==(range& d) {
        return !(mindist > d.maxdist || maxdist < d.mindist);
    }
    friend std::ostream& operator<<(std::ostream& os, const range& d) {
        os << d.mindist << "->" << d.maxdist;
        return os;
    }
    void print() {
        printf("[%f,%f]\n", mindist, maxdist);
    }
    bool valid() {
        return mindist <= maxdist;
    }
} range;

class aab {
  public:
    float low[3];
    float high[3];

  public:
    aab();
    aab(const aab& b);
    aab(float min_x, float min_y, float min_z, float max_x, float max_y, float max_z);
    void set_box(float l0, float l1, float l2, float h0, float h1, float h2);
    void reset();
    void update(float x, float y, float z);
    void update(const aab& p);
    void set_box(const aab& b);
    bool intersect(aab& object);
    bool contains(aab* object);
    bool contains(float* point);
    void print();

    friend std::ostream& operator<<(std::ostream& os, const aab& p) {
        os << "(";
        os << p.low[0] << ",";
        os << p.low[1] << ",";
        os << p.low[2] << ")";
        os << " -> (";
        os << p.high[0] << ",";
        os << p.high[1] << ",";
        os << p.high[2] << ")";
        return os;
    }
    float diagonal_length();
    float volume();
    range distance(const aab& b);
};

class weighted_aab : public aab {
  public:
    int id;
    uint size = 1;
};
#endif // AAB_H