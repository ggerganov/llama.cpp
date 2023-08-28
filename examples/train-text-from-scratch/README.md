# train-text-from-scratch

Basic usage instructions:

```bash
# get training data
wget https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

# train
./bin/train-text-from-scratch \
        --vocab-model ../models/ggml-vocab-llama.gguf \
        --ctx 64 --embd 256 --head 8 --layer 16 \
        --checkpoint-in  chk-shakespeare-256x16.gguf \
        --checkpoint-out chk-shakespeare-256x16.gguf \
        --model-out ggml-shakespeare-256x16-f32.gguf \
        --train-data "shakespeare.txt" \
        -t 6 -b 16 --seed 1 --adam-iter 256 \
        --no-checkpointing

# predict
./bin/main -m ggml-shakespeare-256x16-f32.gguf
```
