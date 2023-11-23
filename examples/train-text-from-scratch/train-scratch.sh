

# train
#main: tokenize training data
# vocab_name="ggml-vocab-llama"
# vocab_dir="models"
# model_dir="../models/ggmls/"
# dataset="shakespeare"
# data_dir="../models"

./train-text-from-scratch \
        --train-data ${data_dir}/${dataset}.txt \
        --vocab-model ${vocab_dir}/${vocab_name}.gguf \
        --ctx 64 --embd 256 --head 8 --layer 16 \
        --checkpoint-in  ${model_dir}/chk/chk-${vocab_name}-LATEST.gguf \
        --checkpoint-out ${model_dir}/chk/chk-${vocab_name}-ITERATION.gguf \
        --model-out ${dataset}-${vocab_name}-f32-ITERATION.gguf \
        -t 6 -b 16 --seed 1 --adam-iter 256 \
        --no-checkpointing

./main -m ${dataset}-${vocab_name}-f32-LATEST.gguf
