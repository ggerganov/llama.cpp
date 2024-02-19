#!/bin/bash

if [ $# -lt 1 ]
then
    >&2 echo "Usage: $0 model_path [server_args...]"
    exit 1
fi

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

# Wait for the server to start
max_attempts=30
attempts=${max_attempts}
until curl --silent --fail "http://localhost:8080/health" | jq -r '.status' | grep ok; do
  attempts=$(( attempts - 1));
  [ "${attempts}" -eq 0 ] && { echo "Server did not startup" >&2; exit 1; }
  sleep_time=$(( (max_attempts - attempts) * 2 ))
  echo "waiting for server to be ready ${sleep_time}s..."
  sleep ${sleep_time}
done

# Start tests
behave