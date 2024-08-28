# source /export/share/yutong/miniconda3/bin/activate
# conda activate xgenmm-flamingo
# which python
# # step 1: surgery
# python xgenmm_surgery.py

# step 2: convert to gguf (vit + projector)

python examples/xgenmm/xgenmm_convert_image_encoder_to_gguf.py \
    --surgery_dir /export/share/yutong/xgenmm/llamacpp_wd \
    --output_dirname gguf_test \
    --version siglip_kosmos_phi3_4k_instruct \
    --use_f32 