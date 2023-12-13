#!/bin/bash
round=${1:-2}
cd /home/koi/mastercode/Mini_CGAL/build
cmake .. && make -j32
case $round in
    1) 
        ./MiniCGAL_CudaDecompress /home/koi/mastercode/Mini_CGAL/static/buffer
        ;;
    2) 
        ./MiniCGAL_CudaCompress /home/koi/mastercode/Mini_CGAL/static/fixed.OFF
        ;;
esac