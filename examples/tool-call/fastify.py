# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "typer",
# ]
# ///
'''
    Binds the functions of a python script as a FastAPI server.
'''
import os
import sys
import fastapi, uvicorn
from pathlib import Path
import typer
from typing import List

import importlib.util


def _load_source_as_module(source):
    i = 0
    while (module_name := f'mod_{i}') in sys.modules:
        i += 1

    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec, f'Failed to load {source} as module'
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader, f'{source} spec has no loader'
    spec.loader.exec_module(module)
    return module


def _load_module(f: str):
    if f.endswith('.py'):
        sys.path.insert(0, str(Path(f).parent))
        return _load_source_as_module(f)
    else:
        return importlib.import_module(f)


def main(files: List[str], host: str = '0.0.0.0', port: int = 8000):
    app = fastapi.FastAPI()

    for f in files:
        print(f'Binding functions from {f}')
        module = _load_module(f)
        for k in dir(module):
            if k.startswith('_'):
                continue
            if k == k.capitalize():
                continue
            v = getattr(module, k)
            if not callable(v) or isinstance(v, type):
                continue
            if not hasattr(v, '__annotations__'):
                continue

            vt = type(v)
            if vt.__module__ == 'langchain_core.tools' and vt.__name__.endswith('Tool') and hasattr(v, 'func') and callable(func := getattr(v, 'func')):
                v = func

            print(f'INFO:     Binding /{k}')
            try:
                app.post('/' + k)(v)
            except Exception as e:
                print(f'WARNING:    Failed to bind /{k}\n\t{e}')

    print(f'INFO:     CWD = {os.getcwd()}')
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    typer.run(main)
