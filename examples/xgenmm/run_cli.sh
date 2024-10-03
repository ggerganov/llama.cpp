#!/bin/bash

make xgenmm-cli




# ./xgenmm-cli --model /export/share/tawalgaonkar/llama.cpp/models/llm/xgenmm-phi-3-llm-Q4.gguf \
#     --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
#     --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg\
#     --prompt "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n Describe this image.<|end|>\n<|assistant|>\n" \
#     --seed 42 --ctx-size 4096 --predict 1024 \
#     --temp 0 --verbose-prompt


# ./xgenmm-cli --model /export/share/tawalgaonkar/llama.cpp/models/llm/xgenmm-phi-3-llm-Q4.gguf \
#     --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
#     --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg\
#     --prompt "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n What is the address of this restirant?<|end|>\n<|assistant|>\n" \
#     --seed 42 --ctx-size 4096 --predict 1024 \
#     --temp 0 --verbose-prompt


# ./xgenmm-cli --model /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf/phi3_mini_4k_instruct_f16.gguf \
#     --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
#     --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg\
#     --prompt "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n Describe this image.<|end|>\n<|assistant|>\n" \
#     --seed 42 --ctx-size 4096 --predict 1024 \
#     --temp 0 --verbose-prompt

# ./xgenmm-cli --model /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf/phi3_mini_4k_instruct_f32.gguf \
#     --mmproj /export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf_test/mmproj-model-f32.gguf \
#     --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg\
#     --prompt "<unk><s></s><|endoftext|><|assistant|><pad><|end|><image><image placeholder><|endofchunk|>" \
#     --seed 42 --ctx-size 4096 --predict 1024 \
#     --temp 0 --verbose-prompt

Q="What is the address of this resturant?"
# Q="Is this dine in or dine out receipt?"
# Q="What is the total amount paid?"
# Q="What is card holder's name?"
# Q="What is the transaction date?"
# Q="What is the phone number of this resturant?"
# Q="Who is the attendant?"
# Q="Who is the cashier?"
# Q="Briefly describe this image."
prompt="<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n $Q<|end|>\n<|assistant|>\n"
echo $prompt

# base_path=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf
# # model=$base_path/phi3_mini_4k_instruct_f32.gguf
# model=$base_path/phi3_mini_4k_instruct_f16.gguf
# mmproj=$base_path/mmproj-model-f32.gguf

base_path=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct_bf16_patch128/gguf
model=$base_path/phi3_mini_4k_instruct_f16.gguf
mmproj=$base_path/mmproj-model-f32.gguf

./xgenmm-cli --model $model\
    --mmproj $mmproj \
    --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg\
    --prompt "$prompt" \
    --seed 42 --ctx-size 4096 --predict 1024 \
    --temp 0.0 --verbose-prompt --color --ubatch-size 1280