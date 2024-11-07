#!/bin/bash

cp -rpv ../ggml/CMakeLists.txt       ./ggml/CMakeLists.txt
cp -rpv ../ggml/src/CMakeLists.txt   ./ggml/src/CMakeLists.txt
cp -rpv ../ggml/cmake/FindSIMD.cmake ./ggml/cmake/FindSIMD.cmake

cp -rpv ../ggml/src/ggml*.c          ./ggml/src/
cp -rpv ../ggml/src/ggml*.cpp        ./ggml/src/
cp -rpv ../ggml/src/ggml*.h          ./ggml/src/
cp -rpv ../ggml/src/ggml*.cu         ./ggml/src/
cp -rpv ../ggml/src/ggml*.m          ./ggml/src/
cp -rpv ../ggml/src/ggml-amx/*       ./ggml/src/ggml-amx/
cp -rpv ../ggml/src/ggml-cann/*      ./ggml/src/ggml-cann/
cp -rpv ../ggml/src/ggml-cuda/*      ./ggml/src/ggml-cuda/
cp -rpv ../ggml/src/ggml-sycl/*      ./ggml/src/ggml-sycl/
cp -rpv ../ggml/src/vulkan-shaders/* ./ggml/src/vulkan-shaders/

cp -rpv ../ggml/include/ggml*.h ./ggml/include/

cp -rpv ../ggml/tests/test-opt.cpp           ./tests/test-opt.cpp
cp -rpv ../ggml/tests/test-grad0.cpp         ./tests/test-grad0.cpp
cp -rpv ../ggml/tests/test-quantize-fns.cpp  ./tests/test-quantize-fns.cpp
cp -rpv ../ggml/tests/test-quantize-perf.cpp ./tests/test-quantize-perf.cpp
cp -rpv ../ggml/tests/test-backend-ops.cpp   ./tests/test-backend-ops.cpp

cp -rpv ../LICENSE                     ./LICENSE
cp -rpv ../ggml/scripts/gen-authors.sh ./scripts/gen-authors.sh
