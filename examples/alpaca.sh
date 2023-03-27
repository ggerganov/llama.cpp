#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

./instruct -m ./models/alpaca-7B-ggml/ggml-model-q4_0.bin --color -f ./prompts/alpaca.txt -c 2024 -n -1
