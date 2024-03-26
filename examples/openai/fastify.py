'''
    Binds the functions of a python script as a FastAPI server.

    This is useful in combination w/ the examples/agent/run_sandboxed_tools.sh
'''
import os, sys, typing, importlib.util
from anyio import Path
import fastapi, uvicorn
import typer

def load_source_as_module(source):
    i = 0
    while (module_name := f'mod_{i}') in sys.modules:
        i += 1

    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def bind_functions(app, module):
    for k in dir(module):
        if k.startswith('_'):
            continue
        if k == k.capitalize():
            continue
        v = getattr(module, k)
        if not callable(v) or isinstance(v, typing.Type):
            continue
        if not hasattr(v, '__annotations__'):
            continue

        vt = type(v)
        if vt.__module__ == 'langchain_core.tools' and vt.__name__.endswith('Tool') and hasattr(v, 'func') and callable(v.func):
            v = v.func

        print(f'INFO:     Binding /{k}')
        try:
            app.post(k)(v)
        except Exception as e:
            print(f'WARNING:    Failed to bind /{k}\n\t{e}')

def main(files: typing.List[str], host: str = '0.0.0.0', port: int = 8000):
    app = fastapi.FastAPI()

    for f in files:
        if f.endswith('.py'):
            sys.path.insert(0, str(Path(f).parent))

            module = load_source_as_module(f)
        else:
            module = importlib.import_module(f)

        bind_functions(app, module)

    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    typer.run(main)
    
