source /export/share/yutong/miniconda3/bin/activate
conda activate xgenmm-flamingo
which python

# ======= siglip_kosmos_phi3_4k_instruct =======

# # # step 1: surgery
# # python xgenmm_surgery.py

# # # step 2: convert vit + projector to gguf 

# # python xgenmm_convert_image_encoder_to_gguf.py \
# #     --surgery_dir /export/share/yutong/xgenmm/llamacpp_wd \
# #     --output_dirname gguf \
# #     --version siglip_kosmos_phi3_4k_instruct \
# #     --use_f32 

# # step 3:  convert llm to gguf
# # https://github.com/ggerganov/llama.cpp/discussions/7927
# cd ../../
# # HF_TOKEN=<PUT YOUR TOKEN HERE>
# # downloads the tokenizer models of the specified models from Huggingface; generates the get_vocab_base_pre() function for convert_hf_to_gguf.py
# # python convert_hf_to_gguf_update.py $HF_TOKEN


# LLM_PATH=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/llm
# outtype=f32
# LLM_OUTPUT_FILE=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf/phi3_mini_4k_instruct_$outtype.gguf
# echo $LLM_OUTPUT_FILE
# python convert_hf_to_gguf.py $LLM_PATH --outfile $LLM_OUTPUT_FILE --outtype $outtype


# ======= siglip_kosmos_phi3_4k_instruct_bf16_patch128 =======

CKPT_PATH=/export/share/manli_shu/models/open-flamingo-dev/fixed_offset-bf16-maxlen2048-newsamplerv1-anyres_patch128-kosmos_non_instruct-phi3_4k_instruct_nq128_pre_V3_6-SFT_v3.6.1.v2-mantis-mix-v0.3.5-continue-8x16-ckpt0/checkpoint_0.pt
VERSION=siglip_kosmos_phi3_4k_instruct_bf16_patch128
SAVE_PATH=/export/share/yutong/xgenmm/llamacpp_wd
# step 1: surgery
python xgenmm_surgery.py --ckpt_pth $CKPT_PATH --save_pth $SAVE_PATH --version $VERSION
# step 2: convert vit + projector to gguf 
python xgenmm_convert_image_encoder_to_gguf.py \
    --surgery_dir  $SAVE_PATH\
    --output_dirname gguf \
    --version $VERSION \
    --use_f32 

# step 3:  convert llm to gguf
# https://github.com/ggerganov/llama.cpp/discussions/7927
cd ../../
# HF_TOKEN=<PUT YOUR TOKEN HERE>
# downloads the tokenizer models of the specified models from Huggingface; generates the get_vocab_base_pre() function for convert_hf_to_gguf.py
# python convert_hf_to_gguf_update.py $HF_TOKEN

# go to llm folder and nano config.json change vocab_size to 32064
LLM_PATH=$SAVE_PATH/$VERSION/llm
OUTTYPE=f16
LLM_OUTPUT_FILE=$SAVE_PATH/$VERSION/gguf/phi3_mini_4k_instruct_$OUTTYPE.gguf
python convert_hf_to_gguf.py $LLM_PATH --outfile $LLM_OUTPUT_FILE --outtype $OUTTYPE