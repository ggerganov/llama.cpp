#!/bin/bash

wget https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-jarvis.cpp/raw/main/winogrande-debiased-eval.csv

echo "Usage:"
echo ""
echo "  ./jarvis-perplexity -m model.gguf -f winogrande-debiased-eval.csv --winogrande [--winogrande-tasks N] [other params]"
echo ""

exit 0
