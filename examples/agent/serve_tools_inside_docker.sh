#!/bin/bash
#
# Serves tools inside a docker container
#
# Usage:
#   examples/agent/serve_tools_inside_docker.sh [--verbose] [--include="tool1|tool2|..."] [--exclude="tool1|tool2|..."]
#
set -euo pipefail

PORT=${PORT:-8088}
BRAVE_SEARCH_API_KEY=${BRAVE_SEARCH_API_KEY:-}
DATA_DIR=${DATA_DIR:-$HOME/.llama.cpp/agent/tools/data}
UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.llama.cpp/agent/tools/uv_cache}

mkdir -p "$DATA_DIR"
mkdir -p "$UV_CACHE_DIR"

args=( --port $PORT "$@" )
echo "# Warming up the uv cache"
docker run \
    -w /src \
    -v $PWD/examples/agent:/src \
    -v "$UV_CACHE_DIR":/root/.cache/uv:rw \
    --rm -it ghcr.io/astral-sh/uv:python3.12-alpine \
    uv run serve_tools.py --help

echo "# Running inside docker: serve_tools.py ${args[*]}"
docker run \
    -p $PORT:$PORT \
    -w /src \
    -v $PWD/examples/agent:/src \
    -v "$UV_CACHE_DIR":/root/.cache/uv \
    -v "$DATA_DIR":/data:rw \
    --env "MEMORY_SQLITE_DB=/data/memory.db" \
    --env "BRAVE_SEARCH_API_KEY=$BRAVE_SEARCH_API_KEY" \
    --rm -it ghcr.io/astral-sh/uv:python3.12-alpine \
    uv run serve_tools.py "${args[@]}"
