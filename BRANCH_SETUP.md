# Setup this branch

## Create a lora adpter bin file

0. `mkdir models/open-llama` and download [Open-llama  (all files)](https://huggingface.co/openlm-research/open_llama_3b_v2/tree/main) in the folder `./models/open-llama`

2. `mkdir data && touch data/hot-lora.txt` and write a couple of words in it.

3. Run:
    ```bash
    # Convert base model to gguf
    python3 convert-hf-to-gguf.py models/open-llama/
    # Quantize base model
    ./quantize ./models/open-llama/ggml-model-f16.gguf ./models/open-llama/ggml-model-q8_0.gguf Q8_0
    # Obtain Lora adapter
    ./finetune  --model-base models/open-llama/ggml-model-q8_0.gguf \
    --checkpoint-in models/open-llama/chk-lora-ggml-model-q8_0-hot-lora-LATEST.gguf \
    --checkpoint-out models/open-llama/chk-lora-ggml-model-q8_0-hot-lora-ITERATION.gguf \
    --lora-out models/open-llama/lora-ggml-model-q8_0-hot-lora-ITERATION.bin \
    --train-data "data/hot-lora.txt" \
    --save-every 1 \
    --threads 1 \
    --adam-iter 1 \
    --batch 1 \
    --ctx 16 \
    --use-checkpointing
    ```

## Run main with adapter

Run main with base model and lora adapter to hot-swap
```bash
./main -m ./models/open-llama/ggml-model-f16.gguf \
--hot-lora models/open-llama/lora-ggml-model-q8_0-hot-lora-LATEST.bin \
-ngl 0 \
-n 128
```

With `ngl > 0` the code breaks. Probably because the Lora tensors try to interact with the base tensors (as in `lora_mul_mat`), but the lora tensors are not moved to the gpu buffer of the base tensors.

# Logic




# Current status

- Only one Lora adapter can be passed. 
- Applying only adapter to Q, K, V matrices to keep the code contained (fintuning trained lora tensors for all linear layers)
- GPU not supported