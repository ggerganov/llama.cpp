#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 quant_type[fp16 or q4_k]"
    exit 1
fi
quant_type=$1

if [ "$quant_type" != "fp16" ] && [ "$quant_type" != "q4_k" ]; then
    echo "Usage: $0 quant_type[fp16 or q4_k]"
    exit 1
fi

SCRIPT_PATH=$(dirname "$(readlink -f "$0")")
ROOT_PATH=${SCRIPT_PATH}/../../

resource_root=/home/chenxiaotao03/model/llama.cpp/internvl-chat-2b-v1-5

llm_model_name=internlm2-1.8B-chat-q4_k.gguf
if [ "$quant_type" == "fp16" ]; then
    llm_model_name=internlm2-1.8B-chat-F16.gguf
fi

${ROOT_PATH}/build/bin/llama-internvl-cli \
    -m ${resource_root}/${llm_model_name} \
    --mmproj ${resource_root}/InternViT-300M-448px-f16.gguf \
    -t 4 \
    --image ${resource_root}/image1.jpg \
    -p "<image>\n请详细描述图片" \
    --gpu-layers 1000 \
    -b 4096 -c 4096 \
    -fa

