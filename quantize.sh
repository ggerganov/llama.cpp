#!/usr/bin/env bash

if ! [[ "$1" =~ ^[0-9]{1,2}B$ ]]; then
    echo
    echo "Usage: quantize.sh 7B|13B|30B|65B [--remove-f16]"
    echo
    exit 1
fi

# Determine the binary file extension based on the platform
case "$(uname -s)" in
    Linux*)
        if grep -qE "(Microsoft|WSL)" /proc/version &> /dev/null ; then
            ext=".exe" # WSL environment
        else
            ext=""
        fi
        ;;
    Darwin*)    ext="";;
    CYGWIN*|MINGW32*|MSYS*|MINGW*) ext=".exe";;
    *)          echo "Unknown platform"; exit 1;;
esac

for i in `ls models/$1/ggml-model-f16.bin*`; do
    ./quantize${ext} "$i" "${i/f16/q4_0}" 2
    if [[ "$2" == "--remove-f16" ]]; then
        rm "$i"
    fi
done
