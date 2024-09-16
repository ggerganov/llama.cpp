#!/bin/bash

make xgenmm-cli

# ./xgenmm-cli -m /export/share/tawalgaonkar/llama.cpp/models/llm/xgenmm-phi-3-llm-Q4.gguf \
#     --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
#     -c 4096 --temp 0.01 --repeat-penalty 1.05 \
#     --image /export/home/llama.cpp/examples/xgenmm/imgs/image-1d100e9-1.jpg \
#     -p "<|system|>\nA chat between a curious user and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\nWhat is the color of this notebook?<|end|>\n<|assistant|>\n"


# ./xgenmm-cli -m /export/share/tawalgaonkar/llama.cpp/models/llm/xgenmm-phi-3-llm-Q4.gguf \
#     --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
#     -c 4096 --temp 0 --num_beams 1 \
#     --image /export/home/on-device-mm/notebooks/open-flamingo/imgs/receipt.jpg \
#     -p "<|system|>\nA chat between a curious user and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image> Describe this image.<|end|>\n<|assistant|>\n"


# ./xgenmm-cli -m /export/share/tawalgaonkar/llama.cpp/models/llm/xgenmm-phi-3-llm-Q4.gguf \
#     --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
#     -c 4096 --temp 0.01 --repeat-penalty 1.05 \
#     --image /export/home/llama.cpp/examples/xgenmm/imgs/image-1d100e9.jpg\
#     -p "<|system|>\nA chat between a curious user and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n How many objects are there in this image?<|end|>\n<|assistant|>\n"


./xgenmm-cli --model /export/share/tawalgaonkar/llama.cpp/models/llm/xgenmm-phi-3-llm-Q4.gguf \
    --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
    --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg\
    --prompt "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n Describe this image.<|end|>\n<|assistant|>\n" \
    --seed 42 --ctx-size 4096 --predict 1024 \
    --temp 0 --verbose-prompt

    # 


# ./xgenmm-cli --model /export/share/tawalgaonkar/llama.cpp/models/llm/xgenmm-phi-3-llm-Q4.gguf \
#     --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
#     --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg\
#     --prompt "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n What is the address of this restirant?<|end|>\n<|assistant|>\n" \
#     --seed 42 --ctx-size 4096 --predict 1024 \
#     --temp 0 --verbose-prompt
