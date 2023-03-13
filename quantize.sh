#!/usr/bin/env bash

if ! [[ "$1" =~ ^[0-9]{1,2}B$ ]]; then
    echo
    echo "Usage: quantize.sh 7B|13B|30B|65B [--remove-f16]"
    echo
    exit 1
fi

for i in `ls models/$1/ggml-model-f16.bin*`; do
    ./quantize "$i" "${i/f16/q4_0}" 2
    if [[ "$2" == "--remove-f16" ]]; then
        rm "$i"
    fi
done
