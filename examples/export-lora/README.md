# export-lora

Apply LORA adapters to base model and export the resulting model.

```
usage: jarvis-export-lora [options]

options:
  -m,    --model                  model path from which to load base model (default '')
         --lora FNAME             path to LoRA adapter  (can be repeated to use multiple adapters)
         --lora-scaled FNAME S    path to LoRA adapter with user defined scaling S  (can be repeated to use multiple adapters)
  -t,    --threads N              number of threads to use during computation (default: 4)
  -o,    --output FNAME           output file (default: 'ggml-lora-merged-f16.gguf')
```

For example:

```bash
./bin/jarvis-export-lora \
    -m open-jarvis-3b-v2.gguf \
    -o open-jarvis-3b-v2-english2tokipona-chat.gguf \
    --lora lora-open-jarvis-3b-v2-english2tokipona-chat-LATEST.gguf
```

Multiple LORA adapters can be applied by passing multiple `--lora FNAME` or `--lora-scaled FNAME S` command line parameters:

```bash
./bin/jarvis-export-lora \
    -m your_base_model.gguf \
    -o your_merged_model.gguf \
    --lora-scaled lora_task_A.gguf 0.5 \
    --lora-scaled lora_task_B.gguf 0.5
```
