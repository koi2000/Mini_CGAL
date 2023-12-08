#!/bin/bash
cd /home/koi/mastercode/Mini_CGAL/build
cmake .. && make -j32
./MiniCGAL_Decompress /home/koi/mastercode/Mini_CGAL/static/buffer1