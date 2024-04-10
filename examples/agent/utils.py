from pathlib import Path
import sys
import importlib.util
from typing import Type

def load_source_as_module(source):
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

def load_module(f: str):
    if f.endswith('.py'):
        sys.path.insert(0, str(Path(f).parent))

        return load_source_as_module(f)
    else:
        return importlib.import_module(f)

def collect_functions(module):
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

        yield v
