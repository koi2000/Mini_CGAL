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
10:28:35.0423 thread 14365:     1 RemovedVerticesDecodingStep takes 0.110000 ms
10:28:35.0423 thread 14365:     1 InsertedEdgeDecodingStep takes 0.121000 ms
10:28:35.0425 thread 14365:     1 insertRemovedVertices takes 1.849000 ms
10:28:35.0426 thread 14365:     1 removeInsertedEdges takes 0.668000 ms
10:28:35.0426 thread 14365:     2 RemovedVerticesDecodingStep takes 0.117000 ms
10:28:35.0426 thread 14365:     2 InsertedEdgeDecodingStep takes 0.210000 ms
10:28:35.0429 thread 14365:     2 insertRemovedVertices takes 2.683000 ms
10:28:35.0430 thread 14365:     2 removeInsertedEdges takes 0.951000 ms
10:28:35.0430 thread 14365:     decode to 20 takes 8.151000 ms
10:28:35.0464 thread 14365:     3 RemovedVerticesDecodingStep takes 0.171000 ms
10:28:35.0464 thread 14365:     3 InsertedEdgeDecodingStep takes 0.209000 ms
10:28:35.0469 thread 14365:     3 insertRemovedVertices takes 4.530000 ms
10:28:35.0470 thread 14365:     3 removeInsertedEdges takes 1.435000 ms
10:28:35.0470 thread 14365:     4 RemovedVerticesDecodingStep takes 0.176000 ms
10:28:35.0471 thread 14365:     4 InsertedEdgeDecodingStep takes 0.282000 ms
10:28:35.0477 thread 14365:     4 insertRemovedVertices takes 6.600000 ms
10:28:35.0479 thread 14365:     4 removeInsertedEdges takes 2.292000 ms
10:28:35.0480 thread 14365:     decode to 40 takes 49.686000 ms
10:28:35.0486 thread 14365:     5 RemovedVerticesDecodingStep takes 0.250000 ms
10:28:35.0486 thread 14365:     5 InsertedEdgeDecodingStep takes 0.414000 ms
10:28:35.0496 thread 14365:     5 insertRemovedVertices takes 10.196000 ms
10:28:35.0501 thread 14365:     5 removeInsertedEdges takes 4.163000 ms
10:28:35.0502 thread 14365:     6 RemovedVerticesDecodingStep takes 0.474000 ms
10:28:35.0502 thread 14365:     6 InsertedEdgeDecodingStep takes 0.665000 ms
10:28:35.0520 thread 14365:     6 insertRemovedVertices takes 18.003000 ms
10:28:35.0529 thread 14365:     6 removeInsertedEdges takes 9.001000 ms
10:28:35.0529 thread 14365:     decode to 60 takes 49.809000 ms
10:28:35.0546 thread 14365:     7 RemovedVerticesDecodingStep takes 0.567000 ms
10:28:35.0547 thread 14365:     7 InsertedEdgeDecodingStep takes 0.848000 ms
10:28:35.0573 thread 14365:     7 insertRemovedVertices takes 26.329000 ms
10:28:35.0590 thread 14365:     7 removeInsertedEdges takes 16.884000 ms
10:28:35.0593 thread 14365:     8 RemovedVerticesDecodingStep takes 0.978000 ms
10:28:35.0595 thread 14365:     8 InsertedEdgeDecodingStep takes 1.215000 ms
10:28:35.0640 thread 14365:     8 insertRemovedVertices takes 45.319000 ms
10:28:35.0662 thread 14365:     8 removeInsertedEdges takes 22.157000 ms
10:28:35.0662 thread 14365:     decode to 80 takes 132.729000 ms
10:28:35.0708 thread 14365:     9 RemovedVerticesDecodingStep takes 2.245000 ms
10:28:35.0711 thread 14365:     9 InsertedEdgeDecodingStep takes 2.145000 ms
10:28:35.0779 thread 14365:     9 insertRemovedVertices takes 68.941000 ms
10:28:35.0825 thread 14365:     9 removeInsertedEdges takes 45.140000 ms
10:28:35.0834 thread 14365:     10 RemovedVerticesDecodingStep takes 3.647000 ms
10:28:35.0838 thread 14365:     10 InsertedEdgeDecodingStep takes 4.001000 ms
10:28:35.0938 thread 14365:     10 insertRemovedVertices takes 99.443000 ms
10:28:35.0945 thread 14365:     10 removeInsertedEdges takes 7.280000 ms
10:28:35.0945 thread 14365:     decode to 100 takes 283.142000 ms
koi@DESKTOP-RFUIFBN:~/mastercode/Mini_CGAL/build$ ./MiniCGAL_Decompress /home/koi/mastercode/Mini_CGAL/static/buffer
10:29:11.0715 thread 14468:     1 RemovedVerticesDecodingStep takes 0.110000 ms
10:29:11.0716 thread 14468:     1 InsertedEdgeDecodingStep takes 0.145000 ms
10:29:11.0717 thread 14468:     1 insertRemovedVertices takes 1.530000 ms
10:29:11.0718 thread 14468:     1 removeInsertedEdges takes 0.672000 ms
10:29:11.0718 thread 14468:     2 RemovedVerticesDecodingStep takes 0.101000 ms
10:29:11.0718 thread 14468:     2 InsertedEdgeDecodingStep takes 0.155000 ms
10:29:11.0720 thread 14468:     2 insertRemovedVertices takes 2.189000 ms
10:29:11.0721 thread 14468:     2 removeInsertedEdges takes 0.973000 ms
10:29:11.0721 thread 14468:     decode to 20 takes 7.283000 ms
10:29:11.0724 thread 14468:     3 RemovedVerticesDecodingStep takes 0.130000 ms
10:29:11.0724 thread 14468:     3 InsertedEdgeDecodingStep takes 0.207000 ms
10:29:11.0728 thread 14468:     3 insertRemovedVertices takes 3.354000 ms
10:29:11.0729 thread 14468:     3 removeInsertedEdges takes 1.451000 ms
10:29:11.0729 thread 14468:     4 RemovedVerticesDecodingStep takes 0.186000 ms
10:29:11.0730 thread 14468:     4 InsertedEdgeDecodingStep takes 0.305000 ms
10:29:11.0735 thread 14468:     4 insertRemovedVertices takes 5.165000 ms
10:29:11.0737 thread 14468:     4 removeInsertedEdges takes 2.253000 ms
10:29:11.0737 thread 14468:     decode to 40 takes 15.912000 ms
10:29:11.0744 thread 14468:     5 RemovedVerticesDecodingStep takes 0.264000 ms
10:29:11.0744 thread 14468:     5 InsertedEdgeDecodingStep takes 0.393000 ms
10:29:11.0752 thread 14468:     5 insertRemovedVertices takes 8.209000 ms
10:29:11.0757 thread 14468:     5 removeInsertedEdges takes 4.361000 ms
10:29:11.0757 thread 14468:     6 RemovedVerticesDecodingStep takes 0.370000 ms
10:29:11.0758 thread 14468:     6 InsertedEdgeDecodingStep takes 0.582000 ms
10:29:11.0772 thread 14468:     6 insertRemovedVertices takes 14.370000 ms
10:29:11.0781 thread 14468:     6 removeInsertedEdges takes 8.183000 ms
10:29:11.0781 thread 14468:     decode to 60 takes 43.346000 ms
10:29:11.0797 thread 14468:     7 RemovedVerticesDecodingStep takes 0.578000 ms
10:29:11.0798 thread 14468:     7 InsertedEdgeDecodingStep takes 0.836000 ms
10:29:11.0820 thread 14468:     7 insertRemovedVertices takes 21.966000 ms
10:29:11.0834 thread 14468:     7 removeInsertedEdges takes 14.471000 ms
10:29:11.0837 thread 14468:     8 RemovedVerticesDecodingStep takes 1.115000 ms
10:29:11.0838 thread 14468:     8 InsertedEdgeDecodingStep takes 1.282000 ms
10:29:11.0875 thread 14468:     8 insertRemovedVertices takes 36.893000 ms
10:29:11.0897 thread 14468:     8 removeInsertedEdges takes 22.155000 ms
10:29:11.0897 thread 14468:     decode to 80 takes 116.910000 ms
10:29:11.0944 thread 14468:     9 RemovedVerticesDecodingStep takes 2.231000 ms
10:29:11.0947 thread 14468:     9 InsertedEdgeDecodingStep takes 2.424000 ms
10:29:12.0007 thread 14468:     9 insertRemovedVertices takes 60.093000 ms
10:29:12.0052 thread 14468:     9 removeInsertedEdges takes 45.300000 ms
10:29:12.0062 thread 14468:     10 RemovedVerticesDecodingStep takes 3.406000 ms
10:29:12.0065 thread 14468:     10 InsertedEdgeDecodingStep takes 3.007000 ms
10:29:12.0146 thread 14468:     10 insertRemovedVertices takes 81.634000 ms
10:29:12.0153 thread 14468:     10 removeInsertedEdges takes 7.255000 ms
10:29:12.0154 thread 14468:     decode to 100 takes 256.019000 ms
```


```
17:34:17.0382 thread 2824:      compress takes 0.055000 ms
Error opening file: ./gisdata/compressed_0.mesh.off
17:34:17.0384 thread 2824:      1 RemovedVerticesDecodingStep takes 0.085000 ms
17:34:17.0385 thread 2824:      1 InsertedEdgeDecodingStep takes 0.124000 ms
17:34:17.0387 thread 2824:      1 insertRemovedVertices takes 2.069000 ms
17:34:17.0388 thread 2824:      1 removeInsertedEdges takes 0.987000 ms
17:34:17.0388 thread 2824:      2 RemovedVerticesDecodingStep takes 0.150000 ms
17:34:17.0388 thread 2824:      2 InsertedEdgeDecodingStep takes 0.230000 ms
17:34:17.0392 thread 2824:      2 insertRemovedVertices takes 3.890000 ms
17:34:17.0394 thread 2824:      2 removeInsertedEdges takes 1.612000 ms
17:34:17.0394 thread 2824:      decode to 20 takes 11.318000 ms
Error opening file: ./gisdata/compressed_20.mesh.off
17:34:17.0394 thread 2824:      3 RemovedVerticesDecodingStep takes 0.155000 ms
17:34:17.0394 thread 2824:      3 InsertedEdgeDecodingStep takes 0.293000 ms
17:34:17.0398 thread 2824:      3 insertRemovedVertices takes 4.160000 ms
17:34:17.0404 thread 2824:      3 removeInsertedEdges takes 6.184000 ms
17:34:17.0406 thread 2824:      4 RemovedVerticesDecodingStep takes 0.305000 ms
17:34:17.0406 thread 2824:      4 InsertedEdgeDecodingStep takes 0.624000 ms
17:34:17.0415 thread 2824:      4 insertRemovedVertices takes 9.177000 ms
17:34:17.0422 thread 2824:      4 removeInsertedEdges takes 7.008000 ms
17:34:17.0422 thread 2824:      decode to 40 takes 28.812000 ms
Error opening file: ./gisdata/compressed_40.mesh.off
17:34:17.0424 thread 2824:      5 RemovedVerticesDecodingStep takes 0.602000 ms
17:34:17.0425 thread 2824:      5 InsertedEdgeDecodingStep takes 0.874000 ms
17:34:17.0438 thread 2824:      5 insertRemovedVertices takes 13.603000 ms
17:34:17.0448 thread 2824:      5 removeInsertedEdges takes 10.038000 ms
17:34:17.0451 thread 2824:      6 RemovedVerticesDecodingStep takes 1.039000 ms
17:34:17.0453 thread 2824:      6 InsertedEdgeDecodingStep takes 2.399000 ms
17:34:17.0482 thread 2824:      6 insertRemovedVertices takes 28.376000 ms
17:34:17.0499 thread 2824:      6 removeInsertedEdges takes 16.828000 ms
17:34:17.0499 thread 2824:      decode to 60 takes 76.345000 ms
Error opening file: ./gisdata/compressed_60.mesh.off
17:34:17.0504 thread 2824:      7 RemovedVerticesDecodingStep takes 1.456000 ms
17:34:17.0506 thread 2824:      7 InsertedEdgeDecodingStep takes 2.192000 ms
17:34:17.0547 thread 2824:      7 insertRemovedVertices takes 41.345000 ms
17:34:17.0572 thread 2824:      7 removeInsertedEdges takes 25.004000 ms
17:34:17.0582 thread 2824:      8 RemovedVerticesDecodingStep takes 3.204000 ms
17:34:17.0587 thread 2824:      8 InsertedEdgeDecodingStep takes 5.092000 ms
17:34:17.0663 thread 2824:      8 insertRemovedVertices takes 75.882000 ms
17:34:17.0702 thread 2824:      8 removeInsertedEdges takes 39.349000 ms
17:34:17.0702 thread 2824:      decode to 80 takes 203.617000 ms
Error opening file: ./gisdata/compressed_80.mesh.off
17:34:17.0715 thread 2824:      9 RemovedVerticesDecodingStep takes 4.020000 ms
17:34:17.0732 thread 2824:      9 InsertedEdgeDecodingStep takes 16.729000 ms
17:34:17.0835 thread 2824:      9 insertRemovedVertices takes 103.238000 ms
17:34:17.0925 thread 2824:      9 removeInsertedEdges takes 89.978000 ms
17:34:17.0968 thread 2824:      10 RemovedVerticesDecodingStep takes 30.859000 ms
17:34:17.0989 thread 2824:      10 InsertedEdgeDecodingStep takes 20.321000 ms
17:34:18.0156 thread 2824:      10 insertRemovedVertices takes 167.224000 ms
17:34:18.0177 thread 2824:      10 removeInsertedEdges takes 21.207000 ms
17:34:18.0177 thread 2824:      decode to 100 takes 474.956000 ms
Error opening file: ./gisdata/compressed_100.mesh.off
```

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