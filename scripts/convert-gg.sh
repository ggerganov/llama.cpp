#!/bin/bash

set -e

# LLaMA v1
python3 convert.py ../llama1/7B  --outfile models/llama-7b/ggml-model-f16.gguf  --outtype f16
python3 convert.py ../llama1/13B --outfile models/llama-13b/ggml-model-f16.gguf --outtype f16
python3 convert.py ../llama1/30B --outfile models/llama-30b/ggml-model-f16.gguf --outtype f16
python3 convert.py ../llama1/65B --outfile models/llama-65b/ggml-model-f16.gguf --outtype f16

# LLaMA v2
python3 convert.py ../llama2/llama-2-7b  --outfile models/llama-7b-v2/ggml-model-f16.gguf  --outtype f16
python3 convert.py ../llama2/llama-2-13b --outfile models/llama-13b-v2/ggml-model-f16.gguf --outtype f16
python3 convert.py ../llama2/llama-2-70b --outfile models/llama-70b-v2/ggml-model-f16.gguf --outtype f16

# Code Llama
python3 convert.py ../codellama/CodeLlama-7b/  --outfile models/codellama-7b/ggml-model-f16.gguf  --outtype f16
python3 convert.py ../codellama/CodeLlama-13b/ --outfile models/codellama-13b/ggml-model-f16.gguf --outtype f16
python3 convert.py ../codellama/CodeLlama-34b/ --outfile models/codellama-34b/ggml-model-f16.gguf --outtype f16

# Falcon
python3 convert-falcon-hf-to-gguf.py ../falcon/falcon-7b  1
mv -v ../falcon/falcon-7b/ggml-model-f16.gguf models/falcon-7b/ggml-model-f16.gguf

python3 convert-falcon-hf-to-gguf.py ../falcon/falcon-40b 1
mv -v ../falcon/falcon-40b/ggml-model-f16.gguf models/falcon-40b/ggml-model-f16.gguf
