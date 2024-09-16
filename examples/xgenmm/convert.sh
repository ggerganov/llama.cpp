source /export/share/yutong/miniconda3/bin/activate
conda activate xgenmm-flamingo
# which python
# # step 1: surgery
# python xgenmm_surgery.py

# # step 2: convert vit + projector to gguf 

# python xgenmm_convert_image_encoder_to_gguf.py \
#     --surgery_dir /export/share/yutong/xgenmm/llamacpp_wd \
#     --output_dirname gguf_test \
#     --version siglip_kosmos_phi3_4k_instruct \
#     --use_f32 

# step 3:  convert llm to gguf
# https://github.com/ggerganov/llama.cpp/discussions/7927
HF_TOKEN=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
LLM_PATH=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/llm
# LLM_OUTPUT_FILE=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf/phi3_.gguf
# downloads the tokenizer models of the specified models from Huggingface; generates the get_vocab_base_pre() function for convert_hf_to_gguf.py
cd ../..
# python convert_hf_to_gguf_update.py $HF_TOKEN
python convert_hf_to_gguf.py $LLM_PATH