#!/bin/bash

cp -rpv ../ggml/src/ggml.c                  ./ggml.c
cp -rpv ../ggml/src/ggml-alloc.c            ./ggml-alloc.c
cp -rpv ../ggml/src/ggml-backend-impl.h     ./ggml-backend-impl.h
cp -rpv ../ggml/src/ggml-backend.c          ./ggml-backend.c
cp -rpv ../ggml/src/ggml-common.h           ./ggml-common.h
cp -rpv ../ggml/src/ggml-cuda.cu            ./ggml-cuda.cu
cp -rpv ../ggml/src/ggml-cuda.h             ./ggml-cuda.h
cp -rpv ../ggml/src/ggml-impl.h             ./ggml-impl.h
cp -rpv ../ggml/src/ggml-kompute.cpp        ./ggml-kompute.cpp
cp -rpv ../ggml/src/ggml-kompute.h          ./ggml-kompute.h
cp -rpv ../ggml/src/ggml-metal.h            ./ggml-metal.h
cp -rpv ../ggml/src/ggml-metal.m            ./ggml-metal.m
cp -rpv ../ggml/src/ggml-metal.metal        ./ggml-metal.metal
cp -rpv ../ggml/src/ggml-mpi.h              ./ggml-mpi.h
cp -rpv ../ggml/src/ggml-mpi.c              ./ggml-mpi.c
cp -rpv ../ggml/src/ggml-opencl.cpp         ./ggml-opencl.cpp
cp -rpv ../ggml/src/ggml-opencl.h           ./ggml-opencl.h
cp -rpv ../ggml/src/ggml-quants.c           ./ggml-quants.c
cp -rpv ../ggml/src/ggml-quants.h           ./ggml-quants.h
cp -rpv ../ggml/src/ggml-sycl.cpp           ./ggml-sycl.cpp
cp -rpv ../ggml/src/ggml-sycl.h             ./ggml-sycl.h
cp -rpv ../ggml/src/ggml-vulkan.cpp         ./ggml-vulkan.cpp
cp -rpv ../ggml/src/ggml-vulkan.h           ./ggml-vulkan.h
cp -rpv ../ggml/include/ggml/ggml.h         ./ggml.h
cp -rpv ../ggml/include/ggml/ggml-alloc.h   ./ggml-alloc.h
cp -rpv ../ggml/include/ggml/ggml-backend.h ./ggml-backend.h

cp -rpv ../ggml/tests/test-opt.cpp         ./tests/test-opt.cpp
cp -rpv ../ggml/tests/test-grad0.cpp       ./tests/test-grad0.cpp
cp -rpv ../ggml/tests/test-backend-ops.cpp ./tests/test-backend-ops.cpp
