# Mini-CGAL

## Overview

## DataStructure

Point

Vertex

Facet

Halfedge

Mesh

## TODO:
分析一下现在性能慢的主要原因
第一个 compcat的次数太多，sort的次数太多，修改方法：通过多源点的方法提升并行度，降低bfs的level
第二个 prealloc on cuda中atomic的次数太多，竞争比较激烈
第三个 joinFacet 目前还没有迁移至cuda上

## Performance analysis
the performance of MCGAL is poor. 

cuda version
```
18:03:50.0884 thread 14188:     9 thrust init in remove vertex takes 4.526000 ms
18:03:50.0886 thread 14188:     9 bfs init in remove vertex takes 1.042000 ms
18:03:50.0900 thread 14188:     9 15 compact takes 9.014000 ms
18:03:50.0913 thread 14188:     9 26 compact takes 7.979000 ms
18:03:50.0926 thread 14188:     9 37 compact takes 7.870000 ms
18:03:50.0940 thread 14188:     9 48 compact takes 8.859000 ms
18:03:50.0953 thread 14188:     9 58 compact takes 8.964000 ms
18:03:50.0966 thread 14188:     9 68 compact takes 8.864000 ms
18:03:50.0979 thread 14188:     9 78 compact takes 8.683000 ms
18:03:50.0993 thread 14188:     9 88 compact takes 8.701000 ms
18:03:51.0006 thread 14188:     9 98 compact takes 9.507000 ms
18:03:51.0008 thread 14188:     9 bfs takes 122.531000 ms
18:03:51.0015 thread 14188:     9 sort takes 6.533000 ms
18:03:51.0018 thread 14188:     9 RemovedVerticesDecodingStep takes 137.881000 ms
18:03:51.0175 thread 14188:     9 InsertedEdgeDecodingStep takes 156.766000 ms
18:03:51.0181 thread 14188:     index:94340,splittable:94340
18:03:51.0573 thread 14188:     9 insertRemovedVertices takes 397.935000 ms
18:03:51.0774 thread 14188:     9 removeInsertedEdges takes 201.432000 ms
18:03:51.0774 thread 14188:     decode to 90 takes 983.135000 ms
18:03:51.0910 thread 14188:     10 thrust init in remove vertex takes 5.484000 ms
18:03:51.0911 thread 14188:     10 bfs init in remove vertex takes 1.512000 ms
18:03:51.0925 thread 14188:     10 15 compact takes 7.092000 ms
18:03:51.0938 thread 14188:     10 26 compact takes 8.185000 ms
18:03:51.0951 thread 14188:     10 37 compact takes 8.105000 ms
18:03:51.0963 thread 14188:     10 47 compact takes 7.231000 ms
18:03:51.0975 thread 14188:     10 57 compact takes 7.597000 ms
18:03:51.0989 thread 14188:     10 67 compact takes 8.760000 ms
18:03:51.0992 thread 14188:     10 bfs takes 80.981000 ms
18:03:51.0999 thread 14188:     10 sort takes 6.629000 ms
18:03:52.0001 thread 14188:     10 RemovedVerticesDecodingStep takes 97.208000 ms
18:03:52.0271 thread 14188:     10 InsertedEdgeDecodingStep takes 269.329000 ms
18:03:52.0278 thread 14188:     index:97060,splittable:97060
18:03:52.0792 thread 14188:     10 insertRemovedVertices takes 521.677000 ms
18:03:52.0908 thread 14188:     10 removeInsertedEdges takes 115.345000 ms
18:03:52.0908 thread 14188:     decode to 100 takes 1.133752 s
```

vector version
```
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