#!/bin/bash
set -euo pipefail

PORT=${PORT:-8088}

docker run -p $PORT:$PORT \
    -w /src \
    -v $PWD/examples/agent:/src \
    --env BRAVE_SEARCH_API_KEY=$BRAVE_SEARCH_API_KEY \
    --rm -it ghcr.io/astral-sh/uv:python3.12-alpine \
    uv run serve_tools.py --port $PORT
