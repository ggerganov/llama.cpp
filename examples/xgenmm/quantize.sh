gguf_dir=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct_bf16_patch128/gguf
model_name=phi3_mini_4k_instruct_f16
quantize_method=Q4_K_M

outname=${model_name}_${quantize_method}
input_model_path=$gguf_dir/$model_name.gguf
output_model_path=$gguf_dir/$outname.gguf
echo $outname
cd ../../
./llama-quantize $input_model_path $output_model_path $quantize_method