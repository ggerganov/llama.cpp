#!/bin/bash

# Read the first argument into a variable
arg1="$1"

# Shift the arguments to remove the first one
shift

# Join the remaining arguments into a single string
arg2="$@"

if [[ $arg1 == '--convert' || $arg1 == '-c' ]]; then
  python3 ./convert-pth-to-ggml.py $arg2
elif  [[ $arg1 == '--quantize' || $arg1 == '-q' ]]; then
  /app/quantize $arg2
else
  echo "Unknown command: $arg1"
  echo "Valid commands: --convert (-c) or --quantize (-q)"
fi