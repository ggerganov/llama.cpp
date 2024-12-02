#!/bin/bash

make install_python_reqs

make clean_cmake

make mk_cmake \
    VCPKG_UNIX_PATH=/core/vcpkg/scripts/buildsystems/vcpkg.cmake

make -C build/ mk_run_tests \
    SCMP_BIN=/VulkanSDK/1.2.141.2/x86_64/bin/glslangValidator

# Copy output components
mkdir -p release/linux-amd64/lib/
mkdir -p release/linux-amd64/include/
cp build/src/libkompute.a release/linux-amd64/lib/
cp -r single_include/kompute release/linux-amd64/include/kompute

