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

adamcnt=1
if [ -n "$adamiter" ]; then
    echo ${adamiter}
    adamcnt=${adamiter}
fi

echo ${finetune} ${adamcnt}

if [ ! -n "$inter" ]; then
  echo "start finetune ......"
  ./${finetune} \
    --train-data ${data_dir}/${dataset}.txt \
    --model-base ${model_dir}/${model_name}.gguf \
    --checkpoint-in  ${model_dir}/chk/chk-${dataset}-${model_name}-LATEST.gguf \
    --checkpoint-out ${model_dir}/chk/chk-${dataset}-${model_name}-ITERATION.gguf \
    --lora-out ${model_dir}/lora/lora-${dataset}-${model_name}-ITERATION.bin \
    --threads 4 --ctx 64 --batch 4  --adam-iter ${adamcnt} --save-every 5 \
    --lora-r 8  --lora-alpha 16 --adam-alpha 3e-4\
    --epochs 3 \
    --use-checkpointing
    # --escape \
    # --grad-acc 1 \
  #   # --seed 1
fi

if [ -f "${model_dir}/lora/lora-${dataset}-${model_name}-LATEST.bin" ]; then
  echo "merge lora to model ......"
  ./export-lora \
        --model-base ${model_dir}/${model_name}.gguf \
        --model-out ${model_dir}/${dataset}-${model_name}.gguf  \
        --lora-scaled  ${model_dir}/lora/lora-${dataset}-${model_name}-LATEST.bin 1.0

fi

prompt='"人间清暑殿，天上广寒宫。"的下一句'
if [ -n "$prompt" ]; then
    prompt=${prompt}
fi

echo ${prompt}

./main \
    -m ${model_dir}/${dataset}-${model_name}.gguf \
    -n 512 \
    -p ${prompt}