#!/bin/bash

#
# Temporary script - will be removed in the future
#


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

${SCRIPT_DIR}/build/bin/main -m /home/g1-s23/dev/Models/vicuna-ggml-vic13b-q4_0.bin -p "Hello! Can you tell me what is the capital of Egypt?" -n 128