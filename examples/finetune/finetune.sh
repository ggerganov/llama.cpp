# mistral arch  = llama

# data_dir="../"  \
# model_dir="../"  \
# model_name="openbuddy-mistral-7b-v13.1-q2_k" \
# dataset=slqm
# adam-iter 变量名不能带-
finetune="finetune"
if [ -n "$cmd" ]; then
    finetune=${cmd}
fi
./${finetune} \
  --train-data ${data_dir}/${dataset}.txt \
  --model-base ${model_dir}/${model_name}.gguf \
  --checkpoint-in  ${model_dir}/chk/chk-${dataset}-${model_name}-LATEST.gguf \
  --checkpoint-out ${model_dir}/chk/chk-${dataset}-${model_name}-ITERATION.gguf \
  --lora-out ${model_dir}/lora/lora-${dataset}-${model_name}-ITERATION.bin \
  --threads 4 --ctx 64 --batch 4  --adam-iter 1 --save-every 5 \
  --lora-r 8  --lora-alpha 16 \
  --grad-acc 1 \
  --escape \
  --epochs 3 \
  --use-checkpointing

#   # --seed 1

./export-lora \
      --model-base ${model_dir}/${model_name}.gguf \
      --model-out ${model_dir}/${dataset}-${model_name}.gguf  \
      --lora-scaled  ${model_dir}/lora/lora-${dataset}-${model_name}-LATEST.bin 1.0

./main \
    -m ${model_dir}/${dataset}-${model_name}.gguf \
    -n 512 \
    -p "雨对风"