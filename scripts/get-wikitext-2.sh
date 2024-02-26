#!/bin/bash

wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip

echo "Usage:"
echo ""
echo "  ./perplexity -m model.gguf -f wiki.test.raw [other params]"
echo ""

exit 0
