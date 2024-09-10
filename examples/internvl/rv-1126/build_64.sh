#!/bin/bash
cmake ../../../../ \
-DCMAKE_TOOLCHAIN_FILE=/home/qianlangyu/resource/toolchain/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/rv11xx_toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DGGML_NATIVE=OFF \

make -j4
