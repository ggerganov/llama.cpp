#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

./main --color --instruct --threads 4 \
       --model ./models/gpt4all-7B/gpt4all-lora-quantized.bin \
       --file ./prompts/alpaca.txt \
       --batch_size 8 --ctx_size 2048 \
       --repeat_last_n 64 --repeat_penalty 1.3 \
       --n_predict 128 --temp 0.1 --top_k 40 --top_p 0.95
