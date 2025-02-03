#!/bin/bash

#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

source /opt/intel/oneapi/setvars.sh

# export GGML_SYCL_DEBUG=1

export ZES_ENABLE_SYSMAN=1
# Enable this to allow llama.cpp to check the free memory of the GPU by using:
# sycl::aspect::ext_intel_free_memory
#
# It's recommended to use this when using --split-mode=layer so that llama.cpp
# can better optimize the distribution of layers across the CPU and GPU.

INPUT_PROMPT="Building a website can be done in 10 simple steps:\nStep 1:"
MODEL_FILE="models/llama-2-7b.Q4_0.gguf"
NGL=33
CONTEXT=8192

if [ $# -gt 0 ]; then
    GGML_SYCL_DEVICE=$1
    echo "Using ${GGML_SYCL_DEVICE} as the main GPU"
    # Use on a single GPU
    EXTRA_ARGS="-mg ${GGML_SYCL_DEVICE} -sm none"
else
    # Use on multiple processors with the same max-compute units
    EXTRA_ARGS=""
fi

./build/bin/llama-cli -m "${MODEL_FILE}" -p "${INPUT_PROMPT}" -n 400 -no-cnv -e -ngl ${NGL} -s 0 -c ${CONTEXT} ${EXTRA_ARGS}

# The "-no-cnv" flag is to force non-base "instruct" models to continue.
# This way, we can automatically test this prompt without interference.
