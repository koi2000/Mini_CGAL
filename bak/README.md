# Mini-CGAL

## Overview

## DataStructure

Point

Vertex

Facet

Halfedge

Mesh

## TODO:

multi object

bfs on cuda

joinFacetOnCuda

fix encode

join facet 无锁化思路
分段，分成若干个block，通过bfs算法，将有多个面连接的一系列的facet组成一个facet block，每个线程处理一个block
大部分线程中的block可能只有一个，一部分需要处理多个


首先理清楚decode的整体流程，传入的是一个buffer，首先需要从这个buffer中读取出数据转化为三维的真实数据信息
第一步是读信息，分别是 每个面是否有点 以及每个边是否是新加入的，

可以尝试full gpu

每一轮decode的逻辑全都是一样的，以后可以每次只需要对mesh进行decode即可，不需要管哪个面是第几轮，以及

目前整体的解压缩过程有一部分在


先用中文写一段思路，后面再删除，我现在需要解决的问题，一是进一步提高decode的性能，把能在cuda上进行运算的尽量全搞到cuda上去，
二是对其进行模块化的封装，想办法让他能够同时处理一个batch的数据，cuda处理每个面，但不清楚具体某个面属于哪个mesh

针对第一个点，第一个问题是如何读取，bfs这一步大概率还是只能由cpu去做，因为涉及到的一个点是要从buffer里读取数据，除非修改encode结构，
或者说干脆把buffer也迁移到cuda中去，mesh中始终有一部分数据再cpu里面

pool结构对于这种并发有很多的优势，现在需要想办法让joinfacet也可以支持并发，能在cpu上做的直接在cpu上做，解决一些资源竞争的问题

我现在这套设计 可以很好的支持扩展，到时候唯一需要担心的问题是显存不够，就需要多个batch的去执行



## Performance analysis
the performance of MCGAL is poor. 

cuda version
```
15:39:41.0271 thread 60536:     1 RemovedVerticesDecodingStep takes 1.425000 ms
15:39:41.0274 thread 60536:     1 InsertedEdgeDecodingStep takes 2.680000 ms
15:39:41.0278 thread 60536:     1 collect face information takes 3.808000 ms
15:39:41.0288 thread 60536:     1 cuda memory copy takes 9.913000 ms
15:39:41.0302 thread 60536:     1 kernel function takes 13.924000 ms
15:39:41.0313 thread 60536:     1 cuda memory copy back takes 11.572000 ms
15:39:41.0313 thread 60536:     1 insertRemovedVertices takes 39.265000 ms
15:39:41.0330 thread 60536:     1 removeInsertedEdges takes 16.444000 ms
15:39:41.0330 thread 60536:     decode to 50 takes 91.012000 ms
15:39:41.0456 thread 60536:     2 RemovedVerticesDecodingStep takes 2.136000 ms
15:39:41.0458 thread 60536:     2 InsertedEdgeDecodingStep takes 2.609000 ms
15:39:41.0465 thread 60536:     2 collect face information takes 6.502000 ms
15:39:41.0482 thread 60536:     2 cuda memory copy takes 17.172000 ms
15:39:41.0482 thread 60536:     2 kernel function takes 0.530000 ms
15:39:41.0503 thread 60536:     2 cuda memory copy back takes 20.881000 ms
15:39:41.0503 thread 60536:     2 insertRemovedVertices takes 45.133000 ms
15:39:41.0511 thread 60536:     2 removeInsertedEdges takes 7.600000 ms
15:39:41.0511 thread 60536:     decode to 100 takes 181.329000 ms
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