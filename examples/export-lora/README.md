# export-lora

Apply LORA adapters to base model and export the resulting model.

```
usage: export-lora [options]

options:
  -h, --help                         show this help message and exit
  -m FNAME, --model-base FNAME       model path from which to load base model (default '')
  -o FNAME, --model-out FNAME        path to save exported model (default '')
  -l FNAME, --lora FNAME             apply LoRA adapter
  -s FNAME S, --lora-scaled FNAME S  apply LoRA adapter with user defined scaling S
  -t N, --threads N                  number of threads to use during computation (default: 4)
```

For example:

```bash
./bin/export-lora \
    -m open-llama-3b-v2-q8_0.gguf \
    -o open-llama-3b-v2-q8_0-english2tokipona-chat.gguf \
    -l lora-open-llama-3b-v2-q8_0-english2tokipona-chat-LATEST.bin
```

Multiple LORA adapters can be applied by passing multiple `-l FN` or `-s FN S` command line parameters.
