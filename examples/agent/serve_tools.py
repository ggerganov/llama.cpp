# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiohttp",
#     "beautifulsoup4",
#     "fastapi",
#     "html2text",
#     "ipython",
#     "pyppeteer",
#     "requests",
#     "typer",
#     "uvicorn",
# ]
# ///
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
import logging
import re
from typing import Optional
import fastapi
import os
import sys
import typer
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

from tools.fetch import fetch_page
from tools.search import brave_search
from tools.python import python, python_tools


ALL_TOOLS = {
    fn.__name__: fn
    for fn in [
        python,
        fetch_page,
        brave_search,
    ]
}


def main(host: str = '0.0.0.0', port: int = 8000, verbose: bool = False, include: Optional[str] = None, exclude: Optional[str] = None):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    def accept_tool(name):
        if include and not re.match(include, name):
            return False
        if exclude and re.match(exclude, name):
            return False
        return True

    app = fastapi.FastAPI()
    for name, fn in python_tools.items():
        if accept_tool(name):
            app.post(f'/{name}')(fn)
            if name != 'python':
                python_tools[name] = fn

    for name, fn in ALL_TOOLS.items():
        app.post(f'/{name}')(fn)

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    typer.run(main)
