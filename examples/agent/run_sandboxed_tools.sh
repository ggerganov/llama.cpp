#!/bin/bash
#
# Runs a Python script in a sandboxed environment and makes its functions available as a web service.
#
# git submodule add https://github.com/NousResearch/Hermes-Function-Calling examples/openai/hermes_function_calling
# python examples/openai/fastify.py examples/openai/hermes_function_calling/functions.py
# REQUIREMENTS_FILE=<( cat examples/openai/hermes_function_calling/requirements.txt | grep -vE "bitsandbytes|flash-attn" ) examples/agents/run_sandboxed_tools.sh examples/agents/hermes_function_calling/functions.py -e LOG_FOLDER=/data/inference_logs
set -euo pipefail

script="$( realpath "$1" )"
script_folder="$(dirname "$script")"
shift 1

function cleanup {
  rm -rf "$BUILD_DIR"
  echo "Deleted $BUILD_DIR"
}
trap cleanup EXIT
BUILD_DIR=$(mktemp -d)
DATA_DIR="${DATA_DIR:-$HOME/.llama.cpp/sandbox}"
SCRIPT_DIR=$( cd "$(dirname "$0")" ; pwd )

mkdir -p "$DATA_DIR"

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-}"
if [[ -z "$REQUIREMENTS_FILE" && -f "$script_folder/requirements.txt" ]]; then
    REQUIREMENTS_FILE="$script_folder/requirements.txt"
fi
if [[ -n "$REQUIREMENTS_FILE" ]]; then
    cp "$REQUIREMENTS_FILE" "$BUILD_DIR/script-requirements.txt"
else
    touch $BUILD_DIR/script-requirements.txt
fi

echo "INFO: using DATA_DIR: $DATA_DIR"

cp \
    "$SCRIPT_DIR/fastify-requirements.txt" \
    "$SCRIPT_DIR/fastify.py" \
    "$SCRIPT_DIR/utils.py" \
    "$BUILD_DIR"

mkdir -p "$DATA_DIR"

readonly PORT=${PORT:-8088}
readonly LLAMA_IMAGE_NAME=llama.cpp/tools-base

echo "
    FROM     ${BASE_IMAGE:-python:3.11-slim}
    RUN      apt-get update
    RUN      apt-get install -y gcc python3-dev git cmake
    RUN      pip install --upgrade pip
    RUN      pip install packaging numpy
    RUN      mkdir /src /data

    # Copy resources in increasing likelihood of change, to keep as much as possible cached
    COPY     fastify-requirements.txt /root/
    RUN      pip install -r /root/fastify-requirements.txt
    COPY     script-requirements.txt  /root/
    RUN      pip install -r /root/script-requirements.txt
    COPY     fastify.py utils.py      /root/examples/agent/

    WORKDIR  /data
    ENTRYPOINT PYTHONPATH=/src:/root python -m examples.agent.fastify --port=$PORT '/src/$( basename "$script" )'
" | docker build "$BUILD_DIR" -f - -t "$LLAMA_IMAGE_NAME"

echo "#"
echo "# Binding $script to http://localhost:$PORT/"
echo "#"
set -x
docker run \
    "$@" \
    --mount "type=bind,source=$( realpath "$script_folder" ),target=/src,readonly" \
    --mount "type=bind,source=$DATA_DIR,target=/data" \
    -p "$PORT:$PORT" \
    -it "$LLAMA_IMAGE_NAME"
