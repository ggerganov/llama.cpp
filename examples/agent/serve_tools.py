'''
    Runs simple tools as a FastAPI server.

    Usage (docker isolation - with network access):

        docker run -p 8088:8088 -w /src -v $PWD/examples/agent:/src \
            --env BRAVE_SEARCH_API_KEY=$BRAVE_SEARCH_API_KEY \
            --rm -it ghcr.io/astral-sh/uv:python3.12-alpine \
            uv run serve_tools.py --port 8088

    Usage (non-siloed, DANGEROUS):

        uv run examples/agent/serve_tools.py --port 8088
'''
import asyncio
import logging
import re
import fastapi
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from tools.fetch import fetch_page
from tools.search import brave_search
from tools.python import python, python_tools

# try:
#     # https://github.com/aio-libs/aiohttp/discussions/6044
#     setattr(asyncio.sslproto._SSLProtocolTransport, "_start_tls_compatible", True) # type: ignore
# except Exception as e:
#     print(f'Failed to patch asyncio: {e}', file=sys.stderr)

verbose = os.environ.get('VERBOSE', '0') == '1'
include = os.environ.get('INCLUDE_TOOLS')
exclude = os.environ.get('EXCLUDE_TOOLS')

logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

ALL_TOOLS = {
    fn.__name__: fn
    for fn in [
        python,
        fetch_page,
        brave_search,
    ]
}

app = fastapi.FastAPI()
for name, fn in ALL_TOOLS.items():
    if include and not re.match(include, fn.__name__):
        continue
    if exclude and re.match(exclude, fn.__name__):
        continue
    app.post(f'/{name}')(fn)
    if name != 'python':
        python_tools[name] = fn
