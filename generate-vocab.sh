#!/usr/bin/env bash

python3 convert-hf-to-gguf.py models/meta-llama/Llama-2-7b-hf --outfile models/meta-llama/Llama-2-7b-hf/ggml-vocab-llama-2-7b-hf.gguf --vocab-only
python3 convert-hf-to-gguf.py models/meta-llama/Meta-Llama-3-8B --outfile models/meta-llama/Meta-Llama-3-8B/ggml-vocab-meta-llama-3-8b.gguf --vocab-only
python3 convert-hf-to-gguf.py models/microsoft/Phi-3-mini-4k-instruct --outfile models/microsoft/Phi-3-mini-4k-instruct/ggml-vocab-phi-3-mini-4k-instruct.gguf --vocab-only
python3 convert-hf-to-gguf.py models/deepseek-ai/deepseek-llm-7b-base --outfile models/deepseek-ai/deepseek-llm-7b-base/ggml-vocab-deepseek-llm-7b-base.gguf --vocab-only
python3 convert-hf-to-gguf.py models/deepseek-ai/deepseek-coder-6.7b-base --outfile models/deepseek-ai/deepseek-coder-6.7b-base/ggml-vocab-deepseek-coder-6.gguf --vocab-only
python3 convert-hf-to-gguf.py models/tiiuae/falcon-7b --outfile models/tiiuae/falcon-7b/ggml-vocab-falcon-7b.gguf --vocab-only
python3 convert-hf-to-gguf.py models/BAAI/bge-small-en-v1.5 --outfile models/BAAI/bge-small-en-v1.5/ggml-vocab-bge-small-en-v1.gguf --vocab-only
python3 convert-hf-to-gguf.py models/mosaicml/mpt-7b --outfile models/mosaicml/mpt-7b/ggml-vocab-mpt-7b.gguf --vocab-only
python3 convert-hf-to-gguf.py models/bigcode/starcoder2-3b --outfile models/bigcode/starcoder2-3b/ggml-vocab-starcoder2-3b.gguf --vocab-only
python3 convert-hf-to-gguf.py models/openai-community/gpt2 --outfile models/openai-community/gpt2/ggml-vocab-gpt2.gguf --vocab-only
python3 convert-hf-to-gguf.py models/smallcloudai/Refact-1_6-base --outfile models/smallcloudai/Refact-1_6-base/ggml-vocab-refact-1_6-base.gguf --vocab-only
python3 convert-hf-to-gguf.py models/CohereForAI/c4ai-command-r-v01 --outfile models/CohereForAI/c4ai-command-r-v01/ggml-vocab-c4ai-command-r-v01.gguf --vocab-only
python3 convert-hf-to-gguf.py models/Qwen/Qwen1.5-7B --outfile models/Qwen/Qwen1.5-7B/ggml-vocab-qwen1.gguf --vocab-only
python3 convert-hf-to-gguf.py models/allenai/OLMo-1.7-7B-hf --outfile models/allenai/OLMo-1.7-7B-hf/ggml-vocab-olmo-1.gguf --vocab-only
# python3 convert-hf-to-gguf.py models/databricks/dbrx-base --outfile models/databricks/dbrx-base/ggml-vocab-dbrx-base.gguf --vocab-only
python3 convert-hf-to-gguf.py models/jinaai/jina-embeddings-v2-base-en --outfile models/jinaai/jina-embeddings-v2-base-en/ggml-vocab-jina-embeddings-v2-base-en.gguf --vocab-only
python3 convert-hf-to-gguf.py models/jinaai/jina-embeddings-v2-base-es --outfile models/jinaai/jina-embeddings-v2-base-es/ggml-vocab-jina-embeddings-v2-base-es.gguf --vocab-only
python3 convert-hf-to-gguf.py models/jinaai/jina-embeddings-v2-base-de --outfile models/jinaai/jina-embeddings-v2-base-de/ggml-vocab-jina-embeddings-v2-base-de.gguf --vocab-only
python3 convert-hf-to-gguf.py models/microsoft/phi-1 --outfile models/microsoft/phi-1/ggml-vocab-phi-1.gguf --vocab-only
python3 convert-hf-to-gguf.py models/stabilityai/stablelm-2-zephyr-1_6b --outfile models/stabilityai/stablelm-2-zephyr-1_6b/ggml-vocab-stablelm-2-zephyr-1_6b.gguf --vocab-only
python3 convert-hf-to-gguf.py models/mistralai/Mistral-7B-Instruct-v0.2 --outfile models/mistralai/Mistral-7B-Instruct-v0.2/ggml-vocab-mistral-7b-instruct-v0.gguf --vocab-only
python3 convert-hf-to-gguf.py models/mistralai/Mixtral-8x7B-Instruct-v0.1 --outfile models/mistralai/Mixtral-8x7B-Instruct-v0.1/ggml-vocab-mixtral-8x7b-instruct-v0.gguf --vocab-only
