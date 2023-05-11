#!/bin/bash
set -e

# Read the first argument into a variable
arg1="$1"

# Shift the arguments to remove the first one
shift

# Join the remaining arguments into a single string
arg2="$@"

if [[ $arg1 == '--convert' || $arg1 == '-c' ]]; then
    python3 ./convert-pth-to-ggml.py $arg2
elif [[ $arg1 == '--quantize' || $arg1 == '-q' ]]; then
    ./quantize $arg2
elif [[ $arg1 == '--run' || $arg1 == '-r' ]]; then
    ./main $arg2
elif [[ $arg1 == '--all-in-one' || $arg1 == '-a' ]]; then
    echo "Converting PTH to GGML..."
    for i in `ls $1/$2/ggml-model-f16.bin*`; do
        if [ -f "${i/f16/q4_0}" ]; then
            echo "Skip model quantization, it already exists: ${i/f16/q4_0}"
        else
            echo "Converting PTH to GGML: $i into ${i/f16/q4_0}..."
            ./quantize "$i" "${i/f16/q4_0}" q4_0
        fi
    done
else
    echo "Unknown command: $arg1"
    echo "Available commands: "
    echo "  --run (-r): Run a model previously converted into ggml"
    echo "              ex: -m /models/7B/ggml-model-q4_0.bin -p \"Building a website can be done in 10 simple steps:\" -n 512"
    echo "  --convert (-c): Convert a llama model into ggml"
    echo "              ex: \"/models/7B/\" 1"
    echo "  --quantize (-q): Optimize with quantization process ggml"
    echo "              ex: \"/models/7B/ggml-model-f16.bin\" \"/models/7B/ggml-model-q4_0.bin\" 2"
    echo "  --all-in-one (-a): Execute --convert & --quantize"
    echo "              ex: \"/models/\" 7B"
fi
