#!/bin/bash

if [ $# -lt 1 ]
then
    >&2 echo "Usage: $0 model_path [server_args...]"
    exit 1
fi

# kill the server at the end
cleanup() {
    pkill -P $$
}
trap cleanup EXIT

model_path="$1"
shift 1

set -eu

# Start the server in background
../../../build/bin/server \
            --model "$model_path" \
            --alias tinyllama-2 \
            --ctx-size 64 \
            --parallel 2 \
            --n-predict 32 \
            --batch-size 32 \
            --threads 4 \
            --threads-batch 4 \
            --embedding \
            --cont-batching \
            "$@" &

# Start tests
behave