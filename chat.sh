#!/bin/bash
#
# Temporary script - will be removed in the future
#

./main -m ./models/7B/ggml-model-q4_0.bin -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
