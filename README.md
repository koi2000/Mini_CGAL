# Mini-CGAL

## Overview

## DataStructure

Point

Vertex

Facet

Halfedge

Mesh

## TODO:
check bfs count

using perf to find out most time consming block

using pool to replace new object

## Performance analysis
the performance of MCGAL is poor. 

vector version
```
15:23:07.0868 thread 33504:     1 RemovedVerticesDecodingStep takes 0.107000 ms
15:23:07.0868 thread 33504:     1 InsertedEdgeDecodingStep takes 0.115000 ms
15:23:07.0869 thread 33504:     1 insertRemovedVertices takes 0.648000 ms
15:23:07.0869 thread 33504:     1 removeInsertedEdges takes 0.321000 ms
15:23:07.0869 thread 33504:     2 RemovedVerticesDecodingStep takes 0.097000 ms
15:23:07.0869 thread 33504:     2 InsertedEdgeDecodingStep takes 0.154000 ms
15:23:07.0870 thread 33504:     2 insertRemovedVertices takes 0.946000 ms
15:23:07.0871 thread 33504:     2 removeInsertedEdges takes 0.465000 ms
15:23:07.0871 thread 33504:     decode to 20 takes 4.261000 ms
15:23:07.0874 thread 33504:     3 RemovedVerticesDecodingStep takes 0.130000 ms
15:23:07.0874 thread 33504:     3 InsertedEdgeDecodingStep takes 0.205000 ms
15:23:07.0875 thread 33504:     3 insertRemovedVertices takes 1.391000 ms
15:23:07.0876 thread 33504:     3 removeInsertedEdges takes 0.668000 ms
15:23:07.0877 thread 33504:     4 RemovedVerticesDecodingStep takes 0.172000 ms
15:23:07.0877 thread 33504:     4 InsertedEdgeDecodingStep takes 0.301000 ms
15:23:07.0879 thread 33504:     4 insertRemovedVertices takes 2.044000 ms
15:23:07.0880 thread 33504:     4 removeInsertedEdges takes 1.049000 ms
15:23:07.0880 thread 33504:     decode to 40 takes 9.244000 ms
15:23:07.0887 thread 33504:     5 RemovedVerticesDecodingStep takes 0.283000 ms
15:23:07.0887 thread 33504:     5 InsertedEdgeDecodingStep takes 0.394000 ms
15:23:07.0890 thread 33504:     5 insertRemovedVertices takes 3.094000 ms
15:23:07.0892 thread 33504:     5 removeInsertedEdges takes 1.421000 ms
15:23:07.0894 thread 33504:     6 RemovedVerticesDecodingStep takes 0.360000 ms
15:23:07.0895 thread 33504:     6 InsertedEdgeDecodingStep takes 0.586000 ms
15:23:07.0900 thread 33504:     6 insertRemovedVertices takes 4.961000 ms
15:23:07.0902 thread 33504:     6 removeInsertedEdges takes 2.160000 ms
15:23:07.0902 thread 33504:     decode to 60 takes 21.730000 ms
15:23:07.0922 thread 33504:     7 RemovedVerticesDecodingStep takes 0.554000 ms
15:23:07.0922 thread 33504:     7 InsertedEdgeDecodingStep takes 0.809000 ms
15:23:07.0930 thread 33504:     7 insertRemovedVertices takes 7.559000 ms
15:23:07.0934 thread 33504:     7 removeInsertedEdges takes 3.980000 ms
15:23:07.0945 thread 33504:     8 RemovedVerticesDecodingStep takes 1.163000 ms
15:23:07.0947 thread 33504:     8 InsertedEdgeDecodingStep takes 1.244000 ms
15:23:07.0960 thread 33504:     8 insertRemovedVertices takes 13.363000 ms
15:23:07.0968 thread 33504:     8 removeInsertedEdges takes 7.917000 ms
15:23:07.0968 thread 33504:     decode to 80 takes 66.196000 ms
15:23:08.0029 thread 33504:     9 RemovedVerticesDecodingStep takes 1.551000 ms
15:23:08.0031 thread 33504:     9 InsertedEdgeDecodingStep takes 2.354000 ms
15:23:08.0053 thread 33504:     9 insertRemovedVertices takes 21.749000 ms
15:23:08.0070 thread 33504:     9 removeInsertedEdges takes 17.196000 ms
15:23:08.0149 thread 33504:     10 RemovedVerticesDecodingStep takes 3.566000 ms
15:23:08.0152 thread 33504:     10 InsertedEdgeDecodingStep takes 3.378000 ms
15:23:08.0188 thread 33504:     10 insertRemovedVertices takes 36.017000 ms
15:23:08.0195 thread 33504:     10 removeInsertedEdges takes 7.081000 ms
15:23:08.0196 thread 33504:     decode to 100 takes 227.538000 ms
```

CGAL
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