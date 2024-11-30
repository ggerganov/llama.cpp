#!/bin/bash

name="$1"
args="${@:2}"

echo "Building $name with args: $args"

rm -fr build-cpu-$1
cmake -S . -B build-cpu-$1 -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF $args
cmake --build build-cpu-$1 --config Release -t ggml-cpu -j $(nproc)
cp build-cpu-$1/bin/libggml-cpu.so ./libggml-cpu-$1.so
rm -fr build-cpu-$1
