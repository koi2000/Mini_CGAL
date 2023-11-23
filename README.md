# Mini-CGAL

## Overview

## DataStructure

Point

Vertex

Face

Halfedge

Mesh

## TODO:
check bfs count

using perf to find out most time consming block

using pool to replace new object

## Performance analysis
the performance of MCGAL is poor. 

possible reason: find_prev

```
15:24:41.0354 thread 11830:	1 RemovedVerticesDecodingStep takes 0.080000 ms
15:24:41.0354 thread 11830:	1 InsertedEdgeDecodingStep takes 0.120000 ms
15:24:41.0356 thread 11830:	1 insertRemovedVertices takes 2.188000 ms
15:24:41.0357 thread 11830:	1 removeInsertedEdges takes 0.978000 ms
15:24:41.0357 thread 11830:	2 RemovedVerticesDecodingStep takes 0.143000 ms
15:24:41.0358 thread 11830:	2 InsertedEdgeDecodingStep takes 0.254000 ms
15:24:41.0363 thread 11830:	2 insertRemovedVertices takes 5.622000 ms
15:24:41.0365 thread 11830:	2 removeInsertedEdges takes 1.501000 ms
15:24:41.0365 thread 11830:	decode to 20 takes 13.572000 ms

15:24:41.0365 thread 11830:	3 RemovedVerticesDecodingStep takes 0.116000 ms
15:24:41.0365 thread 11830:	3 InsertedEdgeDecodingStep takes 0.265000 ms
15:24:41.0382 thread 11830:	3 insertRemovedVertices takes 16.597000 ms
15:24:41.0386 thread 11830:	3 removeInsertedEdges takes 4.670000 ms
15:24:41.0387 thread 11830:	4 RemovedVerticesDecodingStep takes 0.192000 ms
15:24:41.0387 thread 11830:	4 InsertedEdgeDecodingStep takes 0.374000 ms
15:24:41.0429 thread 11830:	4 insertRemovedVertices takes 41.541000 ms
15:24:41.0434 thread 11830:	4 removeInsertedEdges takes 5.012000 ms
15:24:41.0434 thread 11830:	decode to 40 takes 69.344000 ms

15:24:41.0435 thread 11830:	5 RemovedVerticesDecodingStep takes 0.290000 ms
15:24:41.0436 thread 11830:	5 InsertedEdgeDecodingStep takes 0.566000 ms
15:24:41.0564 thread 11830:	5 insertRemovedVertices takes 128.513000 ms
15:24:41.0572 thread 11830:	5 removeInsertedEdges takes 8.225000 ms
15:24:41.0574 thread 11830:	6 RemovedVerticesDecodingStep takes 0.500000 ms
15:24:41.0575 thread 11830:	6 InsertedEdgeDecodingStep takes 0.916000 ms
15:24:41.0930 thread 11830:	6 insertRemovedVertices takes 355.065000 ms
15:24:41.0944 thread 11830:	6 removeInsertedEdges takes 13.494000 ms
15:24:41.0944 thread 11830:	decode to 60 takes 509.652000 ms

15:24:41.0949 thread 11830:	7 RemovedVerticesDecodingStep takes 0.979000 ms
15:24:41.0951 thread 11830:	7 InsertedEdgeDecodingStep takes 1.442000 ms
15:24:42.0778 thread 11830:	7 insertRemovedVertices takes 827.670000 ms
15:24:42.0804 thread 11830:	7 removeInsertedEdges takes 26.048000 ms
15:24:42.0811 thread 11830:	8 RemovedVerticesDecodingStep takes 1.733000 ms
15:24:42.0814 thread 11830:	8 InsertedEdgeDecodingStep takes 2.616000 ms
15:24:45.0006 thread 11830:	8 insertRemovedVertices takes 2.192264 s
15:24:45.0043 thread 11830:	8 removeInsertedEdges takes 36.779000 ms
15:24:45.0043 thread 11830:	decode to 80 takes 3.099103 s

15:24:45.0054 thread 11830:	9 RemovedVerticesDecodingStep takes 2.966000 ms
15:24:45.0071 thread 11830:	9 InsertedEdgeDecodingStep takes 16.931000 ms
15:24:53.0804 thread 11830:	9 insertRemovedVertices takes 8.732907 s
15:24:53.0899 thread 11830:	9 removeInsertedEdges takes 95.220000 ms
15:24:53.0946 thread 11830:	10 RemovedVerticesDecodingStep takes 33.044000 ms
15:24:53.0966 thread 11830:	10 InsertedEdgeDecodingStep takes 20.319000 ms
15:25:19.0051 thread 11830:	10 insertRemovedVertices takes 25.084396 s
15:25:19.0077 thread 11830:	10 removeInsertedEdges takes 25.736000 ms
15:25:19.0077 thread 11830:	decode to 100 takes 34.033891 s
```

```
15:54:53.0761 thread 18464:     compress takes 553.501000 ms
15:54:53.0764 thread 18464:     1 RemovedVerticesDecodingStep takes 0.148000 ms
15:54:53.0764 thread 18464:     1 InsertedEdgeDecodingStep takes 0.156000 ms
15:54:53.0764 thread 18464:     1 insertRemovedVertices takes 0.315000 ms
15:54:53.0765 thread 18464:     1 removeInsertedEdges takes 0.332000 ms
15:54:53.0765 thread 18464:     2 RemovedVerticesDecodingStep takes 0.111000 ms
15:54:53.0765 thread 18464:     2 InsertedEdgeDecodingStep takes 0.165000 ms
15:54:53.0766 thread 18464:     2 insertRemovedVertices takes 0.460000 ms
15:54:53.0766 thread 18464:     2 removeInsertedEdges takes 0.356000 ms
15:54:53.0766 thread 18464:     decode to 20 takes 4.850000 ms

15:54:53.0769 thread 18464:     3 RemovedVerticesDecodingStep takes 0.111000 ms
15:54:53.0769 thread 18464:     3 InsertedEdgeDecodingStep takes 0.242000 ms
15:54:53.0770 thread 18464:     3 insertRemovedVertices takes 0.408000 ms
15:54:53.0770 thread 18464:     3 removeInsertedEdges takes 0.408000 ms
15:54:53.0770 thread 18464:     4 RemovedVerticesDecodingStep takes 0.145000 ms
15:54:53.0771 thread 18464:     4 InsertedEdgeDecodingStep takes 0.355000 ms
15:54:53.0772 thread 18464:     4 insertRemovedVertices takes 1.183000 ms
15:54:53.0773 thread 18464:     4 removeInsertedEdges takes 1.189000 ms
15:54:53.0773 thread 18464:     decode to 40 takes 7.024000 ms

15:54:53.0781 thread 18464:     5 RemovedVerticesDecodingStep takes 0.336000 ms
15:54:53.0782 thread 18464:     5 InsertedEdgeDecodingStep takes 0.643000 ms
15:54:53.0783 thread 18464:     5 insertRemovedVertices takes 1.305000 ms
15:54:53.0785 thread 18464:     5 removeInsertedEdges takes 1.386000 ms
15:54:53.0786 thread 18464:     6 RemovedVerticesDecodingStep takes 0.376000 ms
15:54:53.0786 thread 18464:     6 InsertedEdgeDecodingStep takes 0.782000 ms
15:54:53.0788 thread 18464:     6 insertRemovedVertices takes 1.360000 ms
15:54:53.0790 thread 18464:     6 removeInsertedEdges takes 1.952000 ms
15:54:53.0790 thread 18464:     decode to 60 takes 16.603000 ms

15:54:53.0809 thread 18464:     7 RemovedVerticesDecodingStep takes 0.991000 ms
15:54:53.0811 thread 18464:     7 InsertedEdgeDecodingStep takes 1.488000 ms
15:54:53.0813 thread 18464:     7 insertRemovedVertices takes 2.478000 ms
15:54:53.0817 thread 18464:     7 removeInsertedEdges takes 3.691000 ms
15:54:53.0822 thread 18464:     8 RemovedVerticesDecodingStep takes 1.679000 ms
15:54:53.0825 thread 18464:     8 InsertedEdgeDecodingStep takes 2.932000 ms
15:54:53.0830 thread 18464:     8 insertRemovedVertices takes 4.874000 ms
15:54:53.0838 thread 18464:     8 removeInsertedEdges takes 7.538000 ms
15:54:53.0838 thread 18464:     decode to 80 takes 48.218000 ms

15:54:53.0893 thread 18464:     9 RemovedVerticesDecodingStep takes 1.951000 ms
15:54:53.0896 thread 18464:     9 InsertedEdgeDecodingStep takes 3.209000 ms
15:54:53.0905 thread 18464:     9 insertRemovedVertices takes 9.211000 ms
15:54:53.0918 thread 18464:     9 removeInsertedEdges takes 13.319000 ms
15:54:53.0928 thread 18464:     10 RemovedVerticesDecodingStep takes 5.117000 ms
15:54:53.0932 thread 18464:     10 InsertedEdgeDecodingStep takes 3.955000 ms
15:54:53.0947 thread 18464:     10 insertRemovedVertices takes 15.313000 ms
15:54:53.0954 thread 18464:     10 removeInsertedEdges takes 7.353000 ms
15:54:53.0955 thread 18464:     decode to 100 takes 116.607000 ms
```