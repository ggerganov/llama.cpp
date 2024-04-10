import json
import sys
import types
from typing import Dict, Union

def _is_serializable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except Exception as e:
        return False

def execute_python(source: str) -> Union[Dict, str]:
    """
        Evaluate a Python program and return the globals it declared.
        Can be used to compute mathematical expressions (e.g. after importing math module).

        Args:
            source: contain valid, executable and pure Python code. Should also import any required Python packages.
                For example: "import math\nresult = math.cos(2) * 10"

        Returns:
            dict | str: A dictionary containing variables declared, or an error message if an exception occurred.
    """
    try:
        namespace = {}
        sys.stderr.write(f"Executing Python program:\n{source}\n")
        exec(source, namespace)
        results = {
            k: v
            for k, v in namespace.items()
            if not k.startswith('_') \
                and not isinstance(v, type) \
                and not isinstance(v, types.ModuleType) \
                and not callable(v) \
                and _is_serializable(v)
        }
        sys.stderr.write(f"Results: {json.dumps(results, indent=2)}\n")
        return results
    except Exception as e:
        msg = f"Error: {sys.exc_info()[1]}"
        sys.stderr.write(f"{msg}\n")
        return msg

