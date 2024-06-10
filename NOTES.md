# TODOs

1. How to debug mat_mul (run tests in cpp?)
2. How to wrap the suggestion from lauren on matmul (need to see how to find the llora info to pick up). Something about lora being loaded in the context? How to pick a specifi LoRA
3. check the PR "It was removed in [#7204](https://github.com/ggerganov/llama.cpp/pull/7204). `convert-lora-to-ggml.py` seems to write  loras to gguf witouth the model? Should check the train script and see how they match lora with base layers
4. https://github.com/ggerganov/llama.cpp/discussions/3489
5. check lora example in examples `examples/export-lora/export-lora.cpp`, ask gpt if can be used to extend applying multiple Loras, then ask back to lauren