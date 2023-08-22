#!/bin/bash

cp -rpv ../ggml/src/ggml.c                ./ggml.c
cp -rpv ../ggml/src/ggml-alloc.c          ./ggml-alloc.c
cp -rpv ../ggml/src/ggml-cuda.h           ./ggml-cuda.h
cp -rpv ../ggml/src/ggml-cuda.cu          ./ggml-cuda.cu
cp -rpv ../ggml/src/ggml-opencl.h         ./ggml-opencl.h
cp -rpv ../ggml/src/ggml-opencl.cpp       ./ggml-opencl.cpp
cp -rpv ../ggml/src/ggml-metal.h          ./ggml-metal.h
cp -rpv ../ggml/src/ggml-metal.m          ./ggml-metal.m
cp -rpv ../ggml/src/ggml-metal.metal      ./ggml-metal.metal
cp -rpv ../ggml/include/ggml/ggml.h       ./ggml.h
cp -rpv ../ggml/include/ggml/ggml-alloc.h ./ggml-alloc.h

cp -rpv ../ggml/tests/test-opt.cpp    ./tests/test-opt.cpp
cp -rpv ../ggml/tests/test-grad0.cpp  ./tests/test-grad0.cpp
