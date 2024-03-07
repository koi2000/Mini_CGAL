# 性能分析

|                      | removeVertexDecoding | insertedEdgeDecoding | insertRemovedVertex | removeInsertedEdge | total  |
| -------------------- | -------------------- | -------------------- | ------------------- | ------------------ | ------ |
| CGAL                 | 4.650000             | 4.241000             | 16.434000           | 7.094000           | 76ms   |
| CPU并行（1个mesh）   | 8.188000             | 6.956000             | 12.039000           | 2.085000           | 159ms  |
| CPU并行（10个mesh）  | 12.191000            | 25.475000            | 60.068000           | 14.911000          | 648ms  |
| CPU并行（50个mesh）  | 20.893000            | 37.908000            | 267.368000          | 58.352000          | 1882ms |
| CUDA并行（1个mesh）  | 5.103000             | 5.218000             | 6.389000            | 4.424000           | 329ms  |
| CUDA并行（10个mesh） | 12.708000            | 27.727000            | 13.666000           | 10.995000          | 1089ms |
| CUDA并行（20个mesh） | 17.960000            | 43.223000            | 21.814000           | 21.815000          | 1809ms |

thrust sort
1000000 18ms
10000000 20ms
100000000 34ms
1000000000 1544ms

CGAL单个
```
17:22:10.0991 thread 18691:     1 RemovedVerticesDecodingStep takes 0.071000 ms
17:22:10.0991 thread 18691:     1 InsertedEdgeDecodingStep takes 0.183000 ms
17:22:10.0991 thread 18691:     1 insertRemovedVertices takes 0.252000 ms
17:22:10.0991 thread 18691:     1 removeInsertedEdges takes 0.145000 ms
17:22:10.0991 thread 18691:     decode to 10 takes 0.698000 ms
17:22:10.0991 thread 18691:     2 RemovedVerticesDecodingStep takes 0.086000 ms
17:22:10.0992 thread 18691:     2 InsertedEdgeDecodingStep takes 0.151000 ms
17:22:10.0992 thread 18691:     2 insertRemovedVertices takes 0.309000 ms
17:22:10.0992 thread 18691:     2 removeInsertedEdges takes 0.204000 ms
17:22:10.0992 thread 18691:     decode to 20 takes 0.809000 ms
17:22:10.0992 thread 18691:     3 RemovedVerticesDecodingStep takes 0.115000 ms
17:22:10.0992 thread 18691:     3 InsertedEdgeDecodingStep takes 0.202000 ms
17:22:10.0993 thread 18691:     3 insertRemovedVertices takes 0.466000 ms
17:22:10.0993 thread 18691:     3 removeInsertedEdges takes 0.366000 ms
17:22:10.0993 thread 18691:     decode to 30 takes 1.252000 ms
17:22:10.0994 thread 18691:     4 RemovedVerticesDecodingStep takes 0.153000 ms
17:22:10.0994 thread 18691:     4 InsertedEdgeDecodingStep takes 0.322000 ms
17:22:10.0995 thread 18691:     4 insertRemovedVertices takes 0.634000 ms
17:22:10.0995 thread 18691:     4 removeInsertedEdges takes 0.518000 ms
17:22:10.0995 thread 18691:     decode to 40 takes 1.779000 ms
17:22:10.0996 thread 18691:     5 RemovedVerticesDecodingStep takes 0.223000 ms
17:22:10.0996 thread 18691:     5 InsertedEdgeDecodingStep takes 0.466000 ms
17:22:10.0997 thread 18691:     5 insertRemovedVertices takes 0.962000 ms
17:22:10.0998 thread 18691:     5 removeInsertedEdges takes 0.769000 ms
17:22:10.0998 thread 18691:     decode to 50 takes 2.645000 ms
17:22:10.0998 thread 18691:     6 RemovedVerticesDecodingStep takes 0.333000 ms
17:22:10.0999 thread 18691:     6 InsertedEdgeDecodingStep takes 0.698000 ms
17:22:11.0001 thread 18691:     6 insertRemovedVertices takes 1.542000 ms
17:22:11.0002 thread 18691:     6 removeInsertedEdges takes 1.301000 ms
17:22:11.0002 thread 18691:     decode to 60 takes 4.239000 ms
17:22:11.0003 thread 18691:     7 RemovedVerticesDecodingStep takes 0.505000 ms
17:22:11.0004 thread 18691:     7 InsertedEdgeDecodingStep takes 1.011000 ms
17:22:11.0006 thread 18691:     7 insertRemovedVertices takes 2.112000 ms
17:22:11.0008 thread 18691:     7 removeInsertedEdges takes 2.000000 ms
17:22:11.0008 thread 18691:     decode to 70 takes 6.164000 ms
17:22:11.0010 thread 18691:     8 RemovedVerticesDecodingStep takes 0.770000 ms
17:22:11.0011 thread 18691:     8 InsertedEdgeDecodingStep takes 1.601000 ms
17:22:11.0015 thread 18691:     8 insertRemovedVertices takes 3.818000 ms
17:22:11.0019 thread 18691:     8 removeInsertedEdges takes 3.644000 ms
17:22:11.0019 thread 18691:     decode to 80 takes 10.702000 ms
17:22:11.0023 thread 18691:     9 RemovedVerticesDecodingStep takes 2.438000 ms
17:22:11.0026 thread 18691:     9 InsertedEdgeDecodingStep takes 3.500000 ms
17:22:11.0038 thread 18691:     9 insertRemovedVertices takes 11.265000 ms
17:22:11.0050 thread 18691:     9 removeInsertedEdges takes 12.646000 ms
17:22:11.0050 thread 18691:     decode to 90 takes 31.658000 ms
17:22:11.0060 thread 18691:     10 RemovedVerticesDecodingStep takes 4.650000 ms
17:22:11.0064 thread 18691:     10 InsertedEdgeDecodingStep takes 4.241000 ms
17:22:11.0080 thread 18691:     10 insertRemovedVertices takes 16.434000 ms
17:22:11.0087 thread 18691:     10 removeInsertedEdges takes 7.094000 ms
17:22:11.0087 thread 18691:     decode to 100 takes 36.974000 ms
```
1个mesh CUDA
```
18:07:38.0509 thread 36761:     removeVertexDecoding Step 0.000558
18:07:38.0509 thread 36761:     1 RemovedVerticesDecodingStep takes 0.686000 ms
18:07:38.0510 thread 36761:     1 InsertedEdgeDecodingStep takes 0.851000 ms
18:07:38.0510 thread 36761:     1 cuda memcpy takes 0.428000 ms
18:07:38.0536 thread 36761:     1 thrust init in real remove vertex takes 25.409000 ms
18:07:38.0540 thread 36761:     1 prealloc on cuda takes 3.913000 ms
18:07:38.0543 thread 36761:     1 core kernel takes 3.365000 ms
18:07:38.0543 thread 36761:     1 insertRemovedVertices takes 32.775000 ms
18:07:38.0578 thread 36761:     1 removeInsertedEdges takes 34.329000 ms
18:07:38.0580 thread 36761:     removeVertexDecoding Step 0.000766
18:07:38.0580 thread 36761:     2 RemovedVerticesDecodingStep takes 0.818000 ms
18:07:38.0581 thread 36761:     2 InsertedEdgeDecodingStep takes 1.099000 ms
18:07:38.0582 thread 36761:     2 cuda memcpy takes 0.590000 ms
18:07:38.0584 thread 36761:     2 thrust init in real remove vertex takes 2.120000 ms
18:07:38.0587 thread 36761:     2 prealloc on cuda takes 3.038000 ms
18:07:38.0587 thread 36761:     2 core kernel takes 0.366000 ms
18:07:38.0587 thread 36761:     2 insertRemovedVertices takes 5.557000 ms
18:07:38.0589 thread 36761:     2 removeInsertedEdges takes 1.741000 ms
18:07:38.0592 thread 36761:     removeVertexDecoding Step 0.001056
18:07:38.0592 thread 36761:     3 RemovedVerticesDecodingStep takes 1.102000 ms
18:07:38.0593 thread 36761:     3 InsertedEdgeDecodingStep takes 1.428000 ms
18:07:38.0594 thread 36761:     3 cuda memcpy takes 0.910000 ms
18:07:38.0596 thread 36761:     3 thrust init in real remove vertex takes 2.606000 ms
18:07:38.0600 thread 36761:     3 prealloc on cuda takes 3.317000 ms
18:07:38.0600 thread 36761:     3 core kernel takes 0.360000 ms
18:07:38.0600 thread 36761:     3 insertRemovedVertices takes 6.330000 ms
18:07:38.0602 thread 36761:     3 removeInsertedEdges takes 1.816000 ms
18:07:38.0605 thread 36761:     removeVertexDecoding Step 0.001385
18:07:38.0605 thread 36761:     4 RemovedVerticesDecodingStep takes 1.437000 ms
18:07:38.0607 thread 36761:     4 InsertedEdgeDecodingStep takes 1.852000 ms
18:07:38.0609 thread 36761:     4 cuda memcpy takes 1.304000 ms
18:07:38.0611 thread 36761:     4 thrust init in real remove vertex takes 2.124000 ms
18:07:38.0614 thread 36761:     4 prealloc on cuda takes 3.621000 ms
18:07:38.0615 thread 36761:     4 core kernel takes 0.358000 ms
18:07:38.0615 thread 36761:     4 insertRemovedVertices takes 6.166000 ms
18:07:38.0617 thread 36761:     4 removeInsertedEdges takes 1.857000 ms
18:07:38.0621 thread 36761:     removeVertexDecoding Step 0.001788
18:07:38.0621 thread 36761:     5 RemovedVerticesDecodingStep takes 1.840000 ms
18:07:38.0623 thread 36761:     5 InsertedEdgeDecodingStep takes 2.189000 ms
18:07:38.0625 thread 36761:     5 cuda memcpy takes 1.612000 ms
18:07:38.0627 thread 36761:     5 thrust init in real remove vertex takes 2.121000 ms
18:07:38.0630 thread 36761:     5 prealloc on cuda takes 3.246000 ms
18:07:38.0630 thread 36761:     5 core kernel takes 0.385000 ms
18:07:38.0631 thread 36761:     5 insertRemovedVertices takes 5.815000 ms
18:07:38.0632 thread 36761:     5 removeInsertedEdges takes 1.928000 ms
18:07:38.0638 thread 36761:     removeVertexDecoding Step 0.002296
18:07:38.0638 thread 36761:     6 RemovedVerticesDecodingStep takes 2.377000 ms
18:07:38.0641 thread 36761:     6 InsertedEdgeDecodingStep takes 3.068000 ms
18:07:38.0643 thread 36761:     6 cuda memcpy takes 2.200000 ms
18:07:38.0645 thread 36761:     6 thrust init in real remove vertex takes 2.264000 ms
18:07:38.0649 thread 36761:     6 prealloc on cuda takes 3.462000 ms
18:07:38.0649 thread 36761:     6 core kernel takes 0.357000 ms
18:07:38.0649 thread 36761:     6 insertRemovedVertices takes 6.121000 ms
18:07:38.0651 thread 36761:     6 removeInsertedEdges takes 1.977000 ms
18:07:38.0659 thread 36761:     removeVertexDecoding Step 0.003134
18:07:38.0659 thread 36761:     7 RemovedVerticesDecodingStep takes 3.205000 ms
18:07:38.0663 thread 36761:     7 InsertedEdgeDecodingStep takes 4.057000 ms
18:07:38.0666 thread 36761:     7 cuda memcpy takes 3.014000 ms
18:07:38.0668 thread 36761:     7 thrust init in real remove vertex takes 2.234000 ms
18:07:38.0671 thread 36761:     7 prealloc on cuda takes 3.237000 ms
18:07:38.0672 thread 36761:     7 core kernel takes 0.352000 ms
18:07:38.0672 thread 36761:     7 insertRemovedVertices takes 5.891000 ms
18:07:38.0675 thread 36761:     7 removeInsertedEdges takes 2.896000 ms
18:07:38.0684 thread 36761:     removeVertexDecoding Step 0.003897
18:07:38.0684 thread 36761:     8 RemovedVerticesDecodingStep takes 3.986000 ms
18:07:38.0688 thread 36761:     8 InsertedEdgeDecodingStep takes 4.218000 ms
18:07:38.0692 thread 36761:     8 cuda memcpy takes 3.977000 ms
18:07:38.0695 thread 36761:     8 thrust init in real remove vertex takes 2.295000 ms
18:07:38.0698 thread 36761:     8 prealloc on cuda takes 3.843000 ms
18:07:38.0699 thread 36761:     8 core kernel takes 0.350000 ms
18:07:38.0699 thread 36761:     8 insertRemovedVertices takes 6.531000 ms
18:07:38.0702 thread 36761:     8 removeInsertedEdges takes 2.769000 ms
18:07:38.0712 thread 36761:     removeVertexDecoding Step 0.004175
18:07:38.0712 thread 36761:     9 RemovedVerticesDecodingStep takes 4.254000 ms
18:07:38.0716 thread 36761:     9 InsertedEdgeDecodingStep takes 4.291000 ms
18:07:38.0721 thread 36761:     9 cuda memcpy takes 4.728000 ms
18:07:38.0723 thread 36761:     9 thrust init in real remove vertex takes 2.316000 ms
18:07:38.0727 thread 36761:     9 prealloc on cuda takes 3.138000 ms
18:07:38.0727 thread 36761:     9 core kernel takes 0.364000 ms
18:07:38.0727 thread 36761:     9 insertRemovedVertices takes 5.861000 ms
18:07:38.0731 thread 36761:     9 removeInsertedEdges takes 3.836000 ms
18:07:38.0743 thread 36761:     removeVertexDecoding Step 0.005017
18:07:38.0744 thread 36761:     10 RemovedVerticesDecodingStep takes 5.103000 ms
18:07:38.0749 thread 36761:     10 InsertedEdgeDecodingStep takes 5.218000 ms
18:07:38.0755 thread 36761:     10 cuda memcpy takes 5.900000 ms
18:07:38.0758 thread 36761:     10 thrust init in real remove vertex takes 2.869000 ms
18:07:38.0761 thread 36761:     10 prealloc on cuda takes 3.051000 ms
18:07:38.0761 thread 36761:     10 core kernel takes 0.403000 ms
18:07:38.0761 thread 36761:     10 insertRemovedVertices takes 6.389000 ms
18:07:38.0765 thread 36761:     10 removeInsertedEdges takes 4.424000 ms
18:07:38.0765 thread 36761:     decode takes 329.869000 ms
```
10个mesh CUDA
```
18:01:10.0562 thread 33876:     removeVertexDecoding Step 0.002380
18:01:10.0562 thread 33876:     1 RemovedVerticesDecodingStep takes 2.545000 ms
18:01:10.0567 thread 33876:     1 InsertedEdgeDecodingStep takes 4.462000 ms
18:01:10.0571 thread 33876:     1 cuda memcpy takes 4.399000 ms
18:01:10.0598 thread 33876:     1 thrust init in real remove vertex takes 27.235000 ms
18:01:10.0602 thread 33876:     1 prealloc on cuda takes 4.058000 ms
18:01:10.0606 thread 33876:     1 core kernel takes 3.919000 ms
18:01:10.0606 thread 33876:     1 insertRemovedVertices takes 35.281000 ms
18:01:10.0642 thread 33876:     1 removeInsertedEdges takes 35.491000 ms
18:01:10.0653 thread 33876:     removeVertexDecoding Step 0.002685
18:01:10.0653 thread 33876:     2 RemovedVerticesDecodingStep takes 2.761000 ms
18:01:10.0658 thread 33876:     2 InsertedEdgeDecodingStep takes 5.705000 ms
18:01:10.0666 thread 33876:     2 cuda memcpy takes 7.306000 ms
18:01:10.0668 thread 33876:     2 thrust init in real remove vertex takes 2.867000 ms
18:01:10.0672 thread 33876:     2 prealloc on cuda takes 3.773000 ms
18:01:10.0673 thread 33876:     2 core kernel takes 0.867000 ms
18:01:10.0673 thread 33876:     2 insertRemovedVertices takes 7.548000 ms
18:01:10.0678 thread 33876:     2 removeInsertedEdges takes 4.591000 ms
18:01:10.0694 thread 33876:     removeVertexDecoding Step 0.003657
18:01:10.0694 thread 33876:     3 RemovedVerticesDecodingStep takes 3.738000 ms
18:01:10.0701 thread 33876:     3 InsertedEdgeDecodingStep takes 7.697000 ms
18:01:10.0711 thread 33876:     3 cuda memcpy takes 9.338000 ms
18:01:10.0714 thread 33876:     3 thrust init in real remove vertex takes 3.548000 ms
18:01:10.0718 thread 33876:     3 prealloc on cuda takes 3.439000 ms
18:01:10.0718 thread 33876:     3 core kernel takes 0.432000 ms
18:01:10.0718 thread 33876:     3 insertRemovedVertices takes 7.745000 ms
18:01:10.0724 thread 33876:     3 removeInsertedEdges takes 5.257000 ms
18:01:10.0744 thread 33876:     removeVertexDecoding Step 0.004101
18:01:10.0744 thread 33876:     4 RemovedVerticesDecodingStep takes 4.185000 ms
18:01:10.0753 thread 33876:     4 InsertedEdgeDecodingStep takes 8.842000 ms
18:01:10.0766 thread 33876:     4 cuda memcpy takes 13.069000 ms
18:01:10.0769 thread 33876:     4 thrust init in real remove vertex takes 3.132000 ms
18:01:10.0772 thread 33876:     4 prealloc on cuda takes 3.179000 ms
18:01:10.0773 thread 33876:     4 core kernel takes 0.688000 ms
18:01:10.0773 thread 33876:     4 insertRemovedVertices takes 7.349000 ms
18:01:10.0778 thread 33876:     4 removeInsertedEdges takes 5.167000 ms
18:01:10.0805 thread 33876:     removeVertexDecoding Step 0.005271
18:01:10.0805 thread 33876:     5 RemovedVerticesDecodingStep takes 5.356000 ms
18:01:10.0816 thread 33876:     5 InsertedEdgeDecodingStep takes 11.189000 ms
18:01:10.0833 thread 33876:     5 cuda memcpy takes 17.026000 ms
18:01:10.0836 thread 33876:     5 thrust init in real remove vertex takes 3.225000 ms
18:01:10.0840 thread 33876:     5 prealloc on cuda takes 3.492000 ms
18:01:10.0840 thread 33876:     5 core kernel takes 0.613000 ms
18:01:10.0841 thread 33876:     5 insertRemovedVertices takes 7.670000 ms
18:01:10.0846 thread 33876:     5 removeInsertedEdges takes 5.183000 ms
18:01:10.0881 thread 33876:     removeVertexDecoding Step 0.006842
18:01:10.0881 thread 33876:     6 RemovedVerticesDecodingStep takes 6.932000 ms
18:01:10.0896 thread 33876:     6 InsertedEdgeDecodingStep takes 15.136000 ms
18:01:10.0921 thread 33876:     6 cuda memcpy takes 24.942000 ms
18:01:10.0926 thread 33876:     6 thrust init in real remove vertex takes 4.309000 ms
18:01:10.0929 thread 33876:     6 prealloc on cuda takes 3.464000 ms
18:01:10.0930 thread 33876:     6 core kernel takes 0.753000 ms
18:01:10.0930 thread 33876:     6 insertRemovedVertices takes 9.163000 ms
18:01:10.0937 thread 33876:     6 removeInsertedEdges takes 7.001000 ms
18:01:10.0988 thread 33876:     removeVertexDecoding Step 0.009055
18:01:10.0988 thread 33876:     7 RemovedVerticesDecodingStep takes 9.165000 ms
18:01:11.0009 thread 33876:     7 InsertedEdgeDecodingStep takes 21.042000 ms
18:01:11.0042 thread 33876:     7 cuda memcpy takes 33.359000 ms
18:01:11.0046 thread 33876:     7 thrust init in real remove vertex takes 4.141000 ms
18:01:11.0050 thread 33876:     7 prealloc on cuda takes 3.445000 ms
18:01:11.0050 thread 33876:     7 core kernel takes 0.736000 ms
18:01:11.0051 thread 33876:     7 insertRemovedVertices takes 8.639000 ms
18:01:11.0060 thread 33876:     7 removeInsertedEdges takes 8.845000 ms
18:01:11.0120 thread 33876:     removeVertexDecoding Step 0.009992
18:01:11.0120 thread 33876:     8 RemovedVerticesDecodingStep takes 10.087000 ms
18:01:11.0144 thread 33876:     8 InsertedEdgeDecodingStep takes 23.474000 ms
18:01:11.0183 thread 33876:     8 cuda memcpy takes 38.898000 ms
18:01:11.0188 thread 33876:     8 thrust init in real remove vertex takes 4.786000 ms
18:01:11.0191 thread 33876:     8 prealloc on cuda takes 3.357000 ms
18:01:11.0192 thread 33876:     8 core kernel takes 0.896000 ms
18:01:11.0192 thread 33876:     8 insertRemovedVertices takes 9.566000 ms
18:01:11.0201 thread 33876:     8 removeInsertedEdges takes 8.782000 ms
18:01:11.0273 thread 33876:     removeVertexDecoding Step 0.010986
18:01:11.0273 thread 33876:     9 RemovedVerticesDecodingStep takes 11.069000 ms
18:01:11.0298 thread 33876:     9 InsertedEdgeDecodingStep takes 24.187000 ms
18:01:11.0348 thread 33876:     9 cuda memcpy takes 50.401000 ms
18:01:11.0353 thread 33876:     9 thrust init in real remove vertex takes 5.412000 ms
18:01:11.0357 thread 33876:     9 prealloc on cuda takes 3.737000 ms
18:01:11.0358 thread 33876:     9 core kernel takes 0.843000 ms
18:01:11.0359 thread 33876:     9 insertRemovedVertices takes 10.667000 ms
18:01:11.0368 thread 33876:     9 removeInsertedEdges takes 9.771000 ms
18:01:11.0456 thread 33876:     removeVertexDecoding Step 0.012617
18:01:11.0456 thread 33876:     10 RemovedVerticesDecodingStep takes 12.708000 ms
18:01:11.0484 thread 33876:     10 InsertedEdgeDecodingStep takes 27.727000 ms
18:01:11.0543 thread 33876:     10 cuda memcpy takes 59.309000 ms
18:01:11.0549 thread 33876:     10 thrust init in real remove vertex takes 5.835000 ms
18:01:11.0553 thread 33876:     10 prealloc on cuda takes 3.607000 ms
18:01:11.0556 thread 33876:     10 core kernel takes 3.532000 ms
18:01:11.0557 thread 33876:     10 insertRemovedVertices takes 13.666000 ms
18:01:11.0568 thread 33876:     10 removeInsertedEdges takes 10.995000 ms
18:01:11.0568 thread 33876:     decode takes 1.086840 s
```
20个mesh CUDA
```
18:04:28.0231 thread 35689:     removeVertexDecoding Step 0.004430
18:04:28.0232 thread 35689:     1 RemovedVerticesDecodingStep takes 4.567000 ms
18:04:28.0239 thread 35689:     1 InsertedEdgeDecodingStep takes 7.134000 ms
18:04:28.0247 thread 35689:     1 cuda memcpy takes 8.682000 ms
18:04:28.0276 thread 35689:     1 thrust init in real remove vertex takes 28.612000 ms
18:04:28.0280 thread 35689:     1 prealloc on cuda takes 4.223000 ms
18:04:28.0285 thread 35689:     1 core kernel takes 5.103000 ms
18:04:28.0286 thread 35689:     1 insertRemovedVertices takes 38.259000 ms
18:04:28.0321 thread 35689:     1 removeInsertedEdges takes 35.802000 ms
18:04:28.0340 thread 35689:     removeVertexDecoding Step 0.003855
18:04:28.0340 thread 35689:     2 RemovedVerticesDecodingStep takes 3.931000 ms
18:04:28.0349 thread 35689:     2 InsertedEdgeDecodingStep takes 8.837000 ms
18:04:28.0363 thread 35689:     2 cuda memcpy takes 13.829000 ms
18:04:28.0366 thread 35689:     2 thrust init in real remove vertex takes 2.997000 ms
18:04:28.0370 thread 35689:     2 prealloc on cuda takes 3.742000 ms
18:04:28.0370 thread 35689:     2 core kernel takes 0.509000 ms
18:04:28.0370 thread 35689:     2 insertRemovedVertices takes 7.595000 ms
18:04:28.0375 thread 35689:     2 removeInsertedEdges takes 4.920000 ms
18:04:28.0402 thread 35689:     removeVertexDecoding Step 0.004820
18:04:28.0402 thread 35689:     3 RemovedVerticesDecodingStep takes 4.895000 ms
18:04:28.0413 thread 35689:     3 InsertedEdgeDecodingStep takes 10.634000 ms
18:04:28.0430 thread 35689:     3 cuda memcpy takes 17.206000 ms
18:04:28.0434 thread 35689:     3 thrust init in real remove vertex takes 3.669000 ms
18:04:28.0438 thread 35689:     3 prealloc on cuda takes 3.829000 ms
18:04:28.0438 thread 35689:     3 core kernel takes 0.547000 ms
18:04:28.0439 thread 35689:     3 insertRemovedVertices takes 8.359000 ms
18:04:28.0444 thread 35689:     3 removeInsertedEdges takes 5.624000 ms
18:04:28.0481 thread 35689:     removeVertexDecoding Step 0.005917
18:04:28.0481 thread 35689:     4 RemovedVerticesDecodingStep takes 6.000000 ms
18:04:28.0494 thread 35689:     4 InsertedEdgeDecodingStep takes 13.539000 ms
18:04:28.0520 thread 35689:     4 cuda memcpy takes 25.663000 ms
18:04:28.0524 thread 35689:     4 thrust init in real remove vertex takes 3.966000 ms
18:04:28.0527 thread 35689:     4 prealloc on cuda takes 3.461000 ms
18:04:28.0528 thread 35689:     4 core kernel takes 0.698000 ms
18:04:28.0528 thread 35689:     4 insertRemovedVertices takes 8.459000 ms
18:04:28.0535 thread 35689:     4 removeInsertedEdges takes 7.231000 ms
18:04:28.0584 thread 35689:     removeVertexDecoding Step 0.007742
18:04:28.0584 thread 35689:     5 RemovedVerticesDecodingStep takes 7.822000 ms
18:04:28.0602 thread 35689:     5 InsertedEdgeDecodingStep takes 17.477000 ms
18:04:28.0636 thread 35689:     5 cuda memcpy takes 34.807000 ms
18:04:28.0641 thread 35689:     5 thrust init in real remove vertex takes 4.508000 ms
18:04:28.0644 thread 35689:     5 prealloc on cuda takes 3.379000 ms
18:04:28.0645 thread 35689:     5 core kernel takes 1.037000 ms
18:04:28.0646 thread 35689:     5 insertRemovedVertices takes 9.340000 ms
18:04:28.0655 thread 35689:     5 removeInsertedEdges takes 9.552000 ms
18:04:28.0722 thread 35689:     removeVertexDecoding Step 0.010272
18:04:28.0722 thread 35689:     6 RemovedVerticesDecodingStep takes 10.374000 ms
18:04:28.0747 thread 35689:     6 InsertedEdgeDecodingStep takes 24.251000 ms
18:04:28.0791 thread 35689:     6 cuda memcpy takes 44.261000 ms
18:04:28.0796 thread 35689:     6 thrust init in real remove vertex takes 5.366000 ms
18:04:28.0800 thread 35689:     6 prealloc on cuda takes 3.271000 ms
18:04:28.0802 thread 35689:     6 core kernel takes 2.697000 ms
18:04:28.0803 thread 35689:     6 insertRemovedVertices takes 11.900000 ms
18:04:28.0813 thread 35689:     6 removeInsertedEdges takes 10.408000 ms
18:04:28.0904 thread 35689:     removeVertexDecoding Step 0.013658
18:04:28.0904 thread 35689:     7 RemovedVerticesDecodingStep takes 13.746000 ms
18:04:28.0936 thread 35689:     7 InsertedEdgeDecodingStep takes 31.849000 ms
18:04:28.0998 thread 35689:     7 cuda memcpy takes 61.919000 ms
18:04:29.0005 thread 35689:     7 thrust init in real remove vertex takes 6.426000 ms
18:04:29.0008 thread 35689:     7 prealloc on cuda takes 3.397000 ms
18:04:29.0010 thread 35689:     7 core kernel takes 1.822000 ms
18:04:29.0011 thread 35689:     7 insertRemovedVertices takes 12.357000 ms
18:04:29.0026 thread 35689:     7 removeInsertedEdges takes 15.104000 ms
18:04:29.0146 thread 35689:     removeVertexDecoding Step 0.014899
18:04:29.0146 thread 35689:     8 RemovedVerticesDecodingStep takes 15.009000 ms
18:04:29.0180 thread 35689:     8 InsertedEdgeDecodingStep takes 34.353000 ms
18:04:29.0260 thread 35689:     8 cuda memcpy takes 79.207000 ms
18:04:29.0267 thread 35689:     8 thrust init in real remove vertex takes 6.937000 ms
18:04:29.0270 thread 35689:     8 prealloc on cuda takes 3.695000 ms
18:04:29.0272 thread 35689:     8 core kernel takes 1.862000 ms
18:04:29.0273 thread 35689:     8 insertRemovedVertices takes 13.022000 ms
18:04:29.0289 thread 35689:     8 removeInsertedEdges takes 16.033000 ms
18:04:29.0424 thread 35689:     removeVertexDecoding Step 0.014995
18:04:29.0424 thread 35689:     9 RemovedVerticesDecodingStep takes 15.084000 ms
18:04:29.0461 thread 35689:     9 InsertedEdgeDecodingStep takes 37.274000 ms
18:04:29.0555 thread 35689:     9 cuda memcpy takes 94.012000 ms
18:04:29.0563 thread 35689:     9 thrust init in real remove vertex takes 7.466000 ms
18:04:29.0567 thread 35689:     9 prealloc on cuda takes 4.343000 ms
18:04:29.0570 thread 35689:     9 core kernel takes 2.997000 ms
18:04:29.0571 thread 35689:     9 insertRemovedVertices takes 15.366000 ms
18:04:29.0589 thread 35689:     9 removeInsertedEdges takes 18.421000 ms
18:04:29.0752 thread 35689:     removeVertexDecoding Step 0.017873
18:04:29.0752 thread 35689:     10 RemovedVerticesDecodingStep takes 17.960000 ms
18:04:29.0795 thread 35689:     10 InsertedEdgeDecodingStep takes 43.223000 ms
18:04:29.0907 thread 35689:     10 cuda memcpy takes 112.259000 ms
18:04:29.0916 thread 35689:     10 thrust init in real remove vertex takes 8.775000 ms
18:04:29.0921 thread 35689:     10 prealloc on cuda takes 5.401000 ms
18:04:29.0928 thread 35689:     10 core kernel takes 6.880000 ms
18:04:29.0929 thread 35689:     10 insertRemovedVertices takes 21.814000 ms
18:04:29.0951 thread 35689:     10 removeInsertedEdges takes 21.815000 ms
18:04:29.0951 thread 35689:     decode takes 1.809978 s
```
1个mesh CPU
```
18:08:14.0835 thread 36911:     0 resetState takes 3.456000 ms
18:08:14.0836 thread 36911:     1 RemovedVerticesDecodingStep takes 0.878000 ms
18:08:14.0837 thread 36911:     1 InsertedEdgeDecodingStep takes 0.854000 ms
18:08:14.0840 thread 36911:     1 insertRemovedVertices takes 3.256000 ms
18:08:14.0842 thread 36911:     1 removedInsertedEdges takes 2.094000 ms
18:08:14.0842 thread 36911:     1 resetState takes 0.261000 ms
18:08:14.0843 thread 36911:     2 RemovedVerticesDecodingStep takes 1.169000 ms
18:08:14.0844 thread 36911:     2 InsertedEdgeDecodingStep takes 1.159000 ms
18:08:14.0849 thread 36911:     2 insertRemovedVertices takes 4.339000 ms
18:08:14.0853 thread 36911:     2 removedInsertedEdges takes 4.231000 ms
18:08:14.0854 thread 36911:     2 resetState takes 0.429000 ms
18:08:14.0856 thread 36911:     3 RemovedVerticesDecodingStep takes 2.196000 ms
18:08:14.0857 thread 36911:     3 InsertedEdgeDecodingStep takes 1.722000 ms
18:08:14.0858 thread 36911:     3 insertRemovedVertices takes 0.728000 ms
18:08:14.0859 thread 36911:     3 removedInsertedEdges takes 0.973000 ms
18:08:14.0860 thread 36911:     3 resetState takes 0.523000 ms
18:08:14.0862 thread 36911:     4 RemovedVerticesDecodingStep takes 2.738000 ms
18:08:14.0865 thread 36911:     4 InsertedEdgeDecodingStep takes 2.161000 ms
18:08:14.0870 thread 36911:     4 insertRemovedVertices takes 5.136000 ms
18:08:14.0871 thread 36911:     4 removedInsertedEdges takes 1.230000 ms
18:08:14.0872 thread 36911:     4 resetState takes 0.556000 ms
18:08:14.0875 thread 36911:     5 RemovedVerticesDecodingStep takes 3.071000 ms
18:08:14.0878 thread 36911:     5 InsertedEdgeDecodingStep takes 2.959000 ms
18:08:14.0881 thread 36911:     5 insertRemovedVertices takes 3.404000 ms
18:08:14.0882 thread 36911:     5 removedInsertedEdges takes 1.360000 ms
18:08:14.0883 thread 36911:     5 resetState takes 0.884000 ms
18:08:14.0887 thread 36911:     6 RemovedVerticesDecodingStep takes 4.306000 ms
18:08:14.0891 thread 36911:     6 InsertedEdgeDecodingStep takes 3.443000 ms
18:08:14.0892 thread 36911:     6 insertRemovedVertices takes 1.387000 ms
18:08:14.0895 thread 36911:     6 removedInsertedEdges takes 2.885000 ms
18:08:14.0897 thread 36911:     6 resetState takes 1.312000 ms
18:08:14.0902 thread 36911:     7 RemovedVerticesDecodingStep takes 5.170000 ms
18:08:14.0906 thread 36911:     7 InsertedEdgeDecodingStep takes 4.587000 ms
18:08:14.0909 thread 36911:     7 insertRemovedVertices takes 2.624000 ms
18:08:14.0913 thread 36911:     7 removedInsertedEdges takes 3.992000 ms
18:08:14.0915 thread 36911:     7 resetState takes 1.529000 ms
18:08:14.0921 thread 36911:     8 RemovedVerticesDecodingStep takes 6.017000 ms
18:08:14.0926 thread 36911:     8 InsertedEdgeDecodingStep takes 5.972000 ms
18:08:14.0934 thread 36911:     8 insertRemovedVertices takes 7.595000 ms
18:08:14.0936 thread 36911:     8 removedInsertedEdges takes 2.171000 ms
18:08:14.0938 thread 36911:     8 resetState takes 1.468000 ms
18:08:14.0945 thread 36911:     9 RemovedVerticesDecodingStep takes 7.094000 ms
18:08:14.0950 thread 36911:     9 InsertedEdgeDecodingStep takes 5.430000 ms
18:08:14.0954 thread 36911:     9 insertRemovedVertices takes 3.515000 ms
18:08:14.0960 thread 36911:     9 removedInsertedEdges takes 5.879000 ms
18:08:14.0962 thread 36911:     9 resetState takes 1.990000 ms
18:08:14.0970 thread 36911:     10 RemovedVerticesDecodingStep takes 8.188000 ms
18:08:14.0977 thread 36911:     10 InsertedEdgeDecodingStep takes 6.956000 ms
18:08:14.0989 thread 36911:     10 insertRemovedVertices takes 12.039000 ms
18:08:14.0991 thread 36911:     10 removedInsertedEdges takes 2.085000 ms
18:08:14.0991 thread 36911:     decode takes 159.672000 ms
```

10个mesh CPU
```
18:02:50.0062 thread 34481:     0 resetState takes 4.413000 ms
18:02:50.0068 thread 34481:     1 RemovedVerticesDecodingStep takes 6.009000 ms
18:02:50.0072 thread 34481:     1 InsertedEdgeDecodingStep takes 4.205000 ms
18:02:50.0081 thread 34481:     1 insertRemovedVertices takes 8.508000 ms
18:02:50.0087 thread 34481:     1 removedInsertedEdges takes 6.529000 ms
18:02:50.0089 thread 34481:     1 resetState takes 1.836000 ms
18:02:50.0092 thread 34481:     2 RemovedVerticesDecodingStep takes 2.829000 ms
18:02:50.0098 thread 34481:     2 InsertedEdgeDecodingStep takes 5.525000 ms
18:02:50.0110 thread 34481:     2 insertRemovedVertices takes 12.098000 ms
18:02:50.0115 thread 34481:     2 removedInsertedEdges takes 5.686000 ms
18:02:50.0118 thread 34481:     2 resetState takes 2.603000 ms
18:02:50.0122 thread 34481:     3 RemovedVerticesDecodingStep takes 3.618000 ms
18:02:50.0130 thread 34481:     3 InsertedEdgeDecodingStep takes 8.716000 ms
18:02:50.0141 thread 34481:     3 insertRemovedVertices takes 11.113000 ms
18:02:50.0149 thread 34481:     3 removedInsertedEdges takes 7.461000 ms
18:02:50.0153 thread 34481:     3 resetState takes 3.812000 ms
18:02:50.0157 thread 34481:     4 RemovedVerticesDecodingStep takes 4.007000 ms
18:02:50.0166 thread 34481:     4 InsertedEdgeDecodingStep takes 9.582000 ms
18:02:50.0183 thread 34481:     4 insertRemovedVertices takes 16.852000 ms
18:02:50.0197 thread 34481:     4 removedInsertedEdges takes 14.045000 ms
18:02:50.0202 thread 34481:     4 resetState takes 5.509000 ms
18:02:50.0208 thread 34481:     5 RemovedVerticesDecodingStep takes 5.248000 ms
18:02:50.0219 thread 34481:     5 InsertedEdgeDecodingStep takes 10.719000 ms
18:02:50.0234 thread 34481:     5 insertRemovedVertices takes 15.493000 ms
18:02:50.0253 thread 34481:     5 removedInsertedEdges takes 19.094000 ms
18:02:50.0260 thread 34481:     5 resetState takes 6.189000 ms
18:02:50.0268 thread 34481:     6 RemovedVerticesDecodingStep takes 7.946000 ms
18:02:50.0284 thread 34481:     6 InsertedEdgeDecodingStep takes 16.818000 ms
18:02:50.0307 thread 34481:     6 insertRemovedVertices takes 22.447000 ms
18:02:50.0321 thread 34481:     6 removedInsertedEdges takes 14.197000 ms
18:02:50.0329 thread 34481:     6 resetState takes 8.425000 ms
18:02:50.0339 thread 34481:     7 RemovedVerticesDecodingStep takes 9.729000 ms
18:02:50.0360 thread 34481:     7 InsertedEdgeDecodingStep takes 21.025000 ms
18:02:50.0385 thread 34481:     7 insertRemovedVertices takes 25.056000 ms
18:02:50.0405 thread 34481:     7 removedInsertedEdges takes 19.901000 ms
18:02:50.0416 thread 34481:     7 resetState takes 11.090000 ms
18:02:50.0428 thread 34481:     8 RemovedVerticesDecodingStep takes 11.921000 ms
18:02:50.0448 thread 34481:     8 InsertedEdgeDecodingStep takes 20.110000 ms
18:02:50.0471 thread 34481:     8 insertRemovedVertices takes 23.042000 ms
18:02:50.0490 thread 34481:     8 removedInsertedEdges takes 18.654000 ms
18:02:50.0503 thread 34481:     8 resetState takes 13.037000 ms
18:02:50.0515 thread 34481:     9 RemovedVerticesDecodingStep takes 11.613000 ms
18:02:50.0536 thread 34481:     9 InsertedEdgeDecodingStep takes 21.185000 ms
18:02:50.0559 thread 34481:     9 insertRemovedVertices takes 23.194000 ms
18:02:50.0579 thread 34481:     9 removedInsertedEdges takes 19.536000 ms
18:02:50.0594 thread 34481:     9 resetState takes 15.143000 ms
18:02:50.0606 thread 34481:     10 RemovedVerticesDecodingStep takes 12.191000 ms
18:02:50.0632 thread 34481:     10 InsertedEdgeDecodingStep takes 25.475000 ms
18:02:50.0692 thread 34481:     10 insertRemovedVertices takes 60.068000 ms
18:02:50.0706 thread 34481:     10 removedInsertedEdges takes 14.911000 ms
18:02:50.0707 thread 34481:     decode takes 648.879000 ms
```

50个mesh CPU 
```
17:52:37.0086 thread 31449:     0 resetState takes 8.099000 ms
17:52:37.0091 thread 31449:     1 RemovedVerticesDecodingStep takes 5.239000 ms
17:52:37.0099 thread 31449:     1 InsertedEdgeDecodingStep takes 8.446000 ms
17:52:37.0131 thread 31449:     1 insertRemovedVertices takes 31.804000 ms
17:52:37.0151 thread 31449:     1 removedInsertedEdges takes 20.175000 ms
17:52:37.0159 thread 31449:     1 resetState takes 7.267000 ms
17:52:37.0162 thread 31449:     2 RemovedVerticesDecodingStep takes 3.704000 ms
17:52:37.0170 thread 31449:     2 InsertedEdgeDecodingStep takes 7.836000 ms
17:52:37.0198 thread 31449:     2 insertRemovedVertices takes 27.890000 ms
17:52:37.0223 thread 31449:     2 removedInsertedEdges takes 24.901000 ms
17:52:37.0234 thread 31449:     2 resetState takes 10.774000 ms
17:52:37.0240 thread 31449:     3 RemovedVerticesDecodingStep takes 5.806000 ms
17:52:37.0250 thread 31449:     3 InsertedEdgeDecodingStep takes 10.330000 ms
17:52:37.0287 thread 31449:     3 insertRemovedVertices takes 37.555000 ms
17:52:37.0313 thread 31449:     3 removedInsertedEdges takes 25.513000 ms
17:52:37.0329 thread 31449:     3 resetState takes 16.033000 ms
17:52:37.0336 thread 31449:     4 RemovedVerticesDecodingStep takes 7.243000 ms
17:52:37.0350 thread 31449:     4 InsertedEdgeDecodingStep takes 13.043000 ms
17:52:37.0392 thread 31449:     4 insertRemovedVertices takes 42.937000 ms
17:52:37.0425 thread 31449:     4 removedInsertedEdges takes 32.464000 ms
17:52:37.0447 thread 31449:     4 resetState takes 21.729000 ms
17:52:37.0456 thread 31449:     5 RemovedVerticesDecodingStep takes 9.227000 ms
17:52:37.0472 thread 31449:     5 InsertedEdgeDecodingStep takes 16.284000 ms
17:52:37.0526 thread 31449:     5 insertRemovedVertices takes 53.492000 ms
17:52:37.0562 thread 31449:     5 removedInsertedEdges takes 36.167000 ms
17:52:37.0591 thread 31449:     5 resetState takes 29.689000 ms
17:52:37.0603 thread 31449:     6 RemovedVerticesDecodingStep takes 11.492000 ms
17:52:37.0623 thread 31449:     6 InsertedEdgeDecodingStep takes 20.376000 ms
17:52:37.0702 thread 31449:     6 insertRemovedVertices takes 79.177000 ms
17:52:37.0748 thread 31449:     6 removedInsertedEdges takes 45.262000 ms
17:52:37.0789 thread 31449:     6 resetState takes 41.048000 ms
17:52:37.0804 thread 31449:     7 RemovedVerticesDecodingStep takes 14.946000 ms
17:52:37.0832 thread 31449:     7 InsertedEdgeDecodingStep takes 28.064000 ms
17:52:37.0913 thread 31449:     7 insertRemovedVertices takes 80.685000 ms
17:52:37.0981 thread 31449:     7 removedInsertedEdges takes 67.885000 ms
17:52:38.0033 thread 31449:     7 resetState takes 52.654000 ms
17:52:38.0050 thread 31449:     8 RemovedVerticesDecodingStep takes 16.673000 ms
17:52:38.0081 thread 31449:     8 InsertedEdgeDecodingStep takes 30.796000 ms
17:52:38.0151 thread 31449:     8 insertRemovedVertices takes 70.770000 ms
17:52:38.0228 thread 31449:     8 removedInsertedEdges takes 76.454000 ms
17:52:38.0293 thread 31449:     8 resetState takes 65.391000 ms
17:52:38.0310 thread 31449:     9 RemovedVerticesDecodingStep takes 17.060000 ms
17:52:38.0343 thread 31449:     9 InsertedEdgeDecodingStep takes 32.239000 ms
17:52:38.0423 thread 31449:     9 insertRemovedVertices takes 80.886000 ms
17:52:38.0495 thread 31449:     9 removedInsertedEdges takes 72.119000 ms
17:52:38.0575 thread 31449:     9 resetState takes 79.370000 ms
17:52:38.0596 thread 31449:     10 RemovedVerticesDecodingStep takes 20.893000 ms
17:52:38.0634 thread 31449:     10 InsertedEdgeDecodingStep takes 37.908000 ms
17:52:38.0901 thread 31449:     10 insertRemovedVertices takes 267.368000 ms
17:52:38.0959 thread 31449:     10 removedInsertedEdges takes 58.352000 ms
17:52:38.0960 thread 31449:     decode takes 1.882017 s
```