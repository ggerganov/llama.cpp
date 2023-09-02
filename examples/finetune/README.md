# finetune

Basic usage instructions:

```bash
# get training data
wget https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

# finetune LORA adapter
./bin/finetune \
        --model-base open-llama-3b-v2-q8_0.gguf \
        --checkpoint-in  chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf \
        --checkpoint-out chk-lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.gguf \
        --lora-out lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.bin \
        --train-data "shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing

# predict
./bin/main -m open-llama-3b-v2-q8_0.gguf --lora lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin
```

Finetune output files will be saved every N iterations (config with `--save-every N`).
The pattern "ITERATION" in the output filenames will be replaced with the iteration number and "LATEST" for the latest output.

Gradient checkpointing reduces the memory requirements by ~50% but increases the runtime.
If you have enough RAM, you can make finetuning a bit faster by disabling checkpointing with `--no-checkpointing`.

The LORA rank is configured for each model tensor type separately with these command line options:

```bash
  --rank-att-norm N          LORA rank for attention norm tensor (default 1)
  --rank-ffn-norm N          LORA rank for feed-forward norm tensor (default 1)
  --rank-out-norm N          LORA rank for output norm tensor (default 1)
  --rank-tok-embd N          LORA rank for token embeddings tensor (default 4)
  --rank-out N               LORA rank for output tensor (default 4)
  --rank-wq N                LORA rank for wq tensor (default 4)
  --rank-wk N                LORA rank for wk tensor (default 4)
  --rank-wv N                LORA rank for wv tensor (default 4)
  --rank-wo N                LORA rank for wo tensor (default 4)
  --rank-w1 N                LORA rank for w1 tensor (default 4)
  --rank-w2 N                LORA rank for w2 tensor (default 4)
  --rank-w3 N                LORA rank for w3 tensor (default 4)
```

To see all available options use `finetune --help`.
