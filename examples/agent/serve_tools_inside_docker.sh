#!/bin/bash
#
# Serves tools inside a docker container
#
# Usage:
#   examples/agent/serve_tools_inside_docker.sh [--verbose] [--include="tool1|tool2|..."] [--exclude="tool1|tool2|..."]
#
set -euo pipefail

PORT=${PORT:-8088}

docker run -p $PORT:$PORT \
    -w /src \
    -v $PWD/examples/agent:/src \
    --env BRAVE_SEARCH_API_KEY=$BRAVE_SEARCH_API_KEY \
    --rm -it ghcr.io/astral-sh/uv:python3.12-alpine \
    uv run serve_tools.py --port $PORT "$@"
