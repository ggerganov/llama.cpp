#!/bin/bash
#
# Serves tools inside a docker container
#
# Usage:
#   examples/agent/serve_tools_inside_docker.sh [--verbose] [--include="tool1|tool2|..."] [--exclude="tool1|tool2|..."]
#
set -euo pipefail

if [[ -z "${BRAVE_SEARCH_API_KEY:-}" ]]; then
    echo "Please set BRAVE_SEARCH_API_KEY environment variable in order to enable the brave_search tool" >&2
fi

PORT=${PORT:-8088}
BRAVE_SEARCH_API_KEY=${BRAVE_SEARCH_API_KEY:-}

docker run -p $PORT:$PORT \
    -w /src \
    -v $PWD/examples/agent:/src \
    --env BRAVE_SEARCH_API_KEY=$BRAVE_SEARCH_API_KEY \
    --rm -it ghcr.io/astral-sh/uv:python3.12-alpine \
    uv run serve_tools.py --port $PORT "$@"
