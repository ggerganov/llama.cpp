#!/usr/bin/env bash

make -j
cd models
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
cd ..
pip install -r requirements.txt
./convert.py models/Mistral-7B-Instruct-v0.1/
./quantize models/Mistral-7B-Instruct-v0.1/ggml-model-f16.gguf Q4_K_M
./server -m models/Mistral-7B-Instruct-v0.1/ggml-model-Q4_K_M.gguf
