#!/usr/bin/env bash
#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

print_usage() {
    echo "Usage: ./build.sh [fp32|fp16] [openssl|noopenssl] [--help]"
    echo ""
    echo "Options:"
    echo "  fp32    Build with FP32 precision (default)"
    echo "  fp16    Build with FP16 precision (faster for long-prompt inference)"
    echo "  openssl   Build with OpenSSL (default)"
    echo "  noopenssl Build without OpenSSL"
    echo "  --help  Print this help message"
}

PRECISION=fp32
OPENSSL=ON

for arg in "$@"; do
    case "$arg" in
        --help)
            print_usage
            exit 0
            ;;
        fp32|fp16)
            PRECISION="$arg"
            ;;
        openssl)
            OPENSSL=ON
            ;;
        noopenssl)
            OPENSSL=OFF
            ;;
        *)
            echo "Error: unknown option '$arg'"
            print_usage
            exit 1
            ;;
    esac
done

mkdir -p build
cd build
source /opt/intel/oneapi/setvars.sh

if [ "$PRECISION" = "fp16" ]; then
    #for FP16
    echo "cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DLLAMA_OPENSSL=$OPENSSL # faster for long-prompt inference
"
    cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DLLAMA_OPENSSL="$OPENSSL" # faster for long-prompt inference
else
    #for FP32
    echo "cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_OPENSSL=$OPENSSL"
    cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_OPENSSL="$OPENSSL"
fi

#build example/main
#cmake --build . --config Release --target main

#build example/llama-bench
#cmake --build . --config Release --target llama-bench

#build all binary
echo "cmake --build . --config Release -j$((($(nproc)+1)/2)) -v"
cmake --build . --config Release -j$((($(nproc)+1)/2)) -v
