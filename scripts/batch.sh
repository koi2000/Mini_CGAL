#!/bin/bash

project_directory="../"
build_directory="../build"

output_file="output.log"

# blocks=("MACRO2A" "MACRO2B")
# grids=({1..19})
# blocks=({32..512..32})
grids=({1..19})
# grids=({1..2})
blocks=({32..512..32})
# blocks=({32..128..32})

rm -rf "${build_directory}"
mkdir -p "${build_directory}"
touch ${output_file}
cd "${build_directory}"
mkdir gisdata

for ((i=0; i<${#grids[@]}; i++)); do
    for ((j=0; j<${#blocks[@]}; j++)); do  
        cmake ${project_directory} -DTEST=ON -DGRID=${grids[i]} -DBLOCK=${blocks[j]} && make -j32
        # ./MiniCGAL_Decompress ../static/buffer2 > "${output_file}.${grids[i]}.${blocks[i]}"
        ./MiniCGAL_CudaDecompress ../static/buffer2 >> "/home/koi/mastercode/Mini_CGAL/scripts/output2.log" # "../scripts/${output_file}"
    done 
done

