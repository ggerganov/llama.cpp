#!/bin/bash
#
# Temporary script - will be removed in the future
#

./main -m ./models/7B/ggml-model-q4_0.bin -b 128 -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
