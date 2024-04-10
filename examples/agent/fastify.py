'''
    Binds the functions of a python script as a FastAPI server.

    This is useful in combination w/ the examples/agent/run_sandboxed_tools.sh
'''
import os
import fastapi, uvicorn
import typer
from typing import Type, List

from examples.agent.utils import load_module

def bind_functions(app, module):
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
        if vt.__module__ == 'langchain_core.tools' and vt.__name__.endswith('Tool') and hasattr(v, 'func') and callable(v.func):
            v = v.func

        print(f'INFO:     Binding /{k}')
        try:
            app.post('/' + k)(v)
        except Exception as e:
            print(f'WARNING:    Failed to bind /{k}\n\t{e}')

def main(files: List[str], host: str = '0.0.0.0', port: int = 8000):
    app = fastapi.FastAPI()

    for f in files:
        bind_functions(app, load_module(f))

    print(f'INFO:     CWD = {os.getcwd()}')
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    typer.run(main)

