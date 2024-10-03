import re
from IPython.core.interactiveshell import InteractiveShell
from io import StringIO
import logging
import sys


python_tools = {}


def _strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def python(code: str) -> str:
    '''
    Execute Python code in a siloed environment using IPython and returns the output.

    Parameters:
        code (str): The Python code to execute.

    Returns:
        str: The output of the executed code.
    '''
    logging.debug('[python] Executing %s', code)
    shell = InteractiveShell(
        colors='neutral',
    )
    shell.user_global_ns.update(python_tools)

    old_stdout = sys.stdout
    sys.stdout = out = StringIO()

    try:
        shell.run_cell(code)
    except Exception as e:
        # logging.debug('[python] Execution failed: %s\nCode: %s', e, code)
        return f'An error occurred:\n{_strip_ansi_codes(str(e))}'
    finally:
        sys.stdout = old_stdout

    return _strip_ansi_codes(out.getvalue())
