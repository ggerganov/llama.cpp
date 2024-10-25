'''
    Runs simple tools as a FastAPI server.

    Usage (docker isolation - with network access):

        export BRAVE_SEARCH_API_KEY=...
        ./examples/agent/serve_tools_inside_docker.sh

    Usage (non-siloed, DANGEROUS):

        pip install -r examples/agent/requirements.txt
        fastapi dev examples/agent/tools/__init__.py --port 8088
'''
import logging
import re
import fastapi
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from .fetch import fetch_page
from .search import brave_search
from .python import python, python_tools_registry

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
        python_tools_registry[name] = fn
