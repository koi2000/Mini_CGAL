#!/bin/bash

build=true
run=true

while [ $# -gt 0 ]; do
    case "$1" in
        -build)
            run=false
            ;;
        -run)
            build=false
            ;;
        *)
            echo "invalid argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$build" = true ]; then
    cd ../build && cmake .. && make -j32
fi

if [ "$run" = true ]; then
    ./Batch_Decompress
fi
