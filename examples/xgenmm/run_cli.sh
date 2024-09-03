#!/bin/bash

make xgenmm-cli

./xgenmm-cli -m /export/share/llamacpp_models/MiniCPM-Llama3-V-2_5/ggml-model-Q4_K_M.gguf \
    --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
    -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 \
    --image /export/home/llama.cpp/examples/xgenmm/imgs/image-1d100e9-1.jpg \
    -p "What is in the image?"