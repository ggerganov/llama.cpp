#!/bin/bash
cmake ../../../../ \
-DCMAKE_TOOLCHAIN_FILE=/home/qianlangyu/software/rk-3588/aarch64-linux-gnu.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release $1

make -j4
