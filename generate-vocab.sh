#!/usr/bin/env bash

python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-llama-spm.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-llama-bpe.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-phi-3.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-deepseek-llm.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-deepseek-coder.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-falcon.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-bert-bge.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-mpt.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-starcoder.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-gpt-2.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-phi.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-stablelm.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-qwen.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-mistral-bpe.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-mistral-spm.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-mixtral-bpe.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-mixtral-spm.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-refact.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tokenizers/command-r/ --outfile models/ggml-vocab-command-r.gguf --vocab-only
