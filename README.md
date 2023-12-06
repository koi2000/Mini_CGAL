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

cuda version
```
15:22:41.0826 thread 59149:     1 RemovedVerticesDecodingStep takes 0.110000 ms
15:22:41.0826 thread 59149:     1 InsertedEdgeDecodingStep takes 0.143000 ms
15:22:41.0826 thread 59149:     1 collect face information takes 0.067000 ms
15:22:41.0826 thread 59149:     1 cuda memory copy takes 0.357000 ms
get from pool execution time: 0.001770 ms
first execution time: 0.122306 ms
main loop execution time: 0.458536 ms
facet reset execution time: 0.838345 ms
15:22:41.0844 thread 59149:     1 kernel function takes 17.605000 ms
15:22:41.0844 thread 59149:     1 insertRemovedVertices takes 18.067000 ms
15:22:41.0845 thread 59149:     1 removeInsertedEdges takes 0.528000 ms
15:22:41.0845 thread 59149:     decode to 10 takes 20.236000 ms
15:22:41.0847 thread 59149:     2 RemovedVerticesDecodingStep takes 0.150000 ms
15:22:41.0847 thread 59149:     2 InsertedEdgeDecodingStep takes 0.175000 ms
15:22:41.0847 thread 59149:     2 collect face information takes 0.120000 ms
15:22:41.0847 thread 59149:     2 cuda memory copy takes 0.371000 ms
get from pool execution time: 0.001481 ms
first execution time: 0.165103 ms
main loop execution time: 0.697919 ms
facet reset execution time: 1.190826 ms
15:22:41.0850 thread 59149:     2 kernel function takes 2.453000 ms
15:22:41.0850 thread 59149:     2 insertRemovedVertices takes 2.982000 ms
15:22:41.0851 thread 59149:     2 removeInsertedEdges takes 0.785000 ms
15:22:41.0851 thread 59149:     decode to 20 takes 6.112000 ms
15:22:41.0854 thread 59149:     3 RemovedVerticesDecodingStep takes 0.135000 ms
15:22:41.0854 thread 59149:     3 InsertedEdgeDecodingStep takes 0.272000 ms
15:22:41.0854 thread 59149:     3 collect face information takes 0.167000 ms
15:22:41.0855 thread 59149:     3 cuda memory copy takes 0.470000 ms
get from pool execution time: 0.001481 ms
first execution time: 0.483127 ms
main loop execution time: 1.514941 ms
facet reset execution time: 2.486399 ms
15:22:41.0859 thread 59149:     3 kernel function takes 3.741000 ms
15:22:41.0859 thread 59149:     3 insertRemovedVertices takes 4.395000 ms
15:22:41.0860 thread 59149:     3 removeInsertedEdges takes 1.154000 ms
15:22:41.0860 thread 59149:     decode to 30 takes 9.038000 ms
15:22:41.0864 thread 59149:     4 RemovedVerticesDecodingStep takes 0.193000 ms
15:22:41.0865 thread 59149:     4 InsertedEdgeDecodingStep takes 0.409000 ms
15:22:41.0865 thread 59149:     4 collect face information takes 0.254000 ms
15:22:41.0865 thread 59149:     4 cuda memory copy takes 0.454000 ms
get from pool execution time: 0.001482 ms
first execution time: 0.375336 ms
main loop execution time: 1.408609 ms
facet reset execution time: 2.553312 ms
15:22:41.0869 thread 59149:     4 kernel function takes 3.796000 ms
15:22:41.0869 thread 59149:     4 insertRemovedVertices takes 4.520000 ms
15:22:41.0871 thread 59149:     4 removeInsertedEdges takes 1.602000 ms
15:22:41.0871 thread 59149:     decode to 40 takes 11.092000 ms
15:22:41.0878 thread 59149:     5 RemovedVerticesDecodingStep takes 0.271000 ms
15:22:41.0879 thread 59149:     5 InsertedEdgeDecodingStep takes 0.544000 ms
15:22:41.0879 thread 59149:     5 collect face information takes 0.382000 ms
15:22:41.0879 thread 59149:     5 cuda memory copy takes 0.427000 ms
get from pool execution time: 0.001525 ms
first execution time: 0.542259 ms
main loop execution time: 1.826587 ms
facet reset execution time: 3.737960 ms
15:22:41.0884 thread 59149:     5 kernel function takes 4.989000 ms
15:22:41.0884 thread 59149:     5 insertRemovedVertices takes 5.813000 ms
15:22:41.0887 thread 59149:     5 removeInsertedEdges takes 2.294000 ms
15:22:41.0887 thread 59149:     decode to 50 takes 15.821000 ms
15:22:41.0898 thread 59149:     6 RemovedVerticesDecodingStep takes 0.408000 ms
15:22:41.0899 thread 59149:     6 InsertedEdgeDecodingStep takes 0.841000 ms
15:22:41.0899 thread 59149:     6 collect face information takes 0.516000 ms
15:22:41.0900 thread 59149:     6 cuda memory copy takes 0.491000 ms
get from pool execution time: 0.001530 ms
first execution time: 1.081520 ms
main loop execution time: 3.751541 ms
facet reset execution time: 7.921517 ms
15:22:41.0909 thread 59149:     6 kernel function takes 9.216000 ms
15:22:41.0909 thread 59149:     6 insertRemovedVertices takes 10.240000 ms
15:22:41.0913 thread 59149:     6 removeInsertedEdges takes 3.753000 ms
15:22:41.0913 thread 59149:     decode to 60 takes 26.085000 ms
15:22:41.0932 thread 59149:     7 RemovedVerticesDecodingStep takes 0.595000 ms
15:22:41.0933 thread 59149:     7 InsertedEdgeDecodingStep takes 1.218000 ms
15:22:41.0934 thread 59149:     7 collect face information takes 0.927000 ms
15:22:41.0935 thread 59149:     7 cuda memory copy takes 0.512000 ms
get from pool execution time: 0.001477 ms
first execution time: 1.731573 ms
main loop execution time: 7.291492 ms
facet reset execution time: 11.283357 ms
15:22:41.0947 thread 59149:     7 kernel function takes 12.640000 ms
15:22:41.0947 thread 59149:     7 insertRemovedVertices takes 14.095000 ms
15:22:41.0953 thread 59149:     7 removeInsertedEdges takes 5.396000 ms
15:22:41.0953 thread 59149:     decode to 70 takes 39.790000 ms
15:22:41.0985 thread 59149:     8 RemovedVerticesDecodingStep takes 0.917000 ms
15:22:41.0987 thread 59149:     8 InsertedEdgeDecodingStep takes 1.757000 ms
15:22:41.0988 thread 59149:     8 collect face information takes 1.817000 ms
15:22:41.0989 thread 59149:     8 cuda memory copy takes 0.817000 ms
get from pool execution time: 0.001620 ms
first execution time: 2.753401 ms
main loop execution time: 10.016661 ms
facet reset execution time: 17.440144 ms
15:22:42.0008 thread 59149:     8 kernel function takes 18.927000 ms
15:22:42.0008 thread 59149:     8 insertRemovedVertices takes 21.577000 ms
15:22:42.0016 thread 59149:     8 removeInsertedEdges takes 8.373000 ms
15:22:42.0017 thread 59149:     decode to 80 takes 63.894000 ms
15:22:42.0075 thread 59149:     9 RemovedVerticesDecodingStep takes 1.591000 ms
15:22:42.0078 thread 59149:     9 InsertedEdgeDecodingStep takes 3.178000 ms
15:22:42.0082 thread 59149:     9 collect face information takes 3.494000 ms
15:22:42.0083 thread 59149:     9 cuda memory copy takes 0.796000 ms
get from pool execution time: 0.001576 ms
first execution time: 7.336031 ms
main loop execution time: 20.122690 ms
facet reset execution time: 25.883993 ms
15:22:42.0110 thread 59149:     9 kernel function takes 27.273000 ms
15:22:42.0110 thread 59149:     9 insertRemovedVertices takes 31.580000 ms
15:22:42.0128 thread 59149:     9 removeInsertedEdges takes 18.061000 ms
15:22:42.0128 thread 59149:     decode to 90 takes 111.628000 ms
15:22:42.0257 thread 59149:     10 RemovedVerticesDecodingStep takes 2.424000 ms
15:22:42.0261 thread 59149:     10 InsertedEdgeDecodingStep takes 3.205000 ms
15:22:42.0267 thread 59149:     10 collect face information takes 6.793000 ms
15:22:42.0268 thread 59149:     10 cuda memory copy takes 0.813000 ms
get from pool execution time: 0.001538 ms
first execution time: 6.135806 ms
main loop execution time: 30.399309 ms
facet reset execution time: 44.664577 ms
15:22:42.0314 thread 59149:     10 kernel function takes 46.106000 ms
15:22:42.0314 thread 59149:     10 insertRemovedVertices takes 53.729000 ms
15:22:42.0322 thread 59149:     10 removeInsertedEdges takes 7.261000 ms
15:22:42.0322 thread 59149:     decode to 100 takes 193.502000 ms
```

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