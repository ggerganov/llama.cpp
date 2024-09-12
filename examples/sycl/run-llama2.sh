#!/bin/bash

#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

source /opt/intel/oneapi/setvars.sh

#export GGML_SYCL_DEBUG=1

#ZES_ENABLE_SYSMAN=1, Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory. Recommended to use when --split-mode = layer.

INPUT_PROMPT="Building a website can be done in 10 simple steps:\nStep 1:"
MODEL_FILE=llama-2-7b.Q4_0.gguf
NGL=33

if [ $# -gt 0 ]; then
    GGML_SYCL_DEVICE=$1
    echo "use $GGML_SYCL_DEVICE as main GPU"
    #use signle GPU only
    ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/${MODEL_FILE} -p "${INPUT_PROMPT}" -n 400 -e -ngl ${NGL} -s 0 -mg $GGML_SYCL_DEVICE -sm none

else
    #use multiple GPUs with same max compute units
    ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/${MODEL_FILE} -p "${INPUT_PROMPT}" -n 400 -e -ngl ${NGL} -s 0
fi
