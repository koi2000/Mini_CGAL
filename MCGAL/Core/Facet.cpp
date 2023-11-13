#include "core.h"

namespace MCGAL {
void Face::remove(Halfedge* rh) {
    halfedges.erase(rh);
    for (Halfedge* h : halfedges) {
        if (h->next == rh) {
            h->next = NULL;
        }
    }
    delete rh;
}



}  // namespace MCGAL
