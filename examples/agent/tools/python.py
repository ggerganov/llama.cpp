from IPython.core.interactiveshell import InteractiveShell
from io import StringIO
import sys


def python(code: str) -> str:
    """
    Execute Python code in a siloed environment using IPython and returns the output.

    Parameters:
        code (str): The Python code to execute.

    Returns:
        str: The output of the executed code.
    """
    shell = InteractiveShell()

    old_stdout = sys.stdout
    sys.stdout = out = StringIO()

    try:
        shell.run_cell(code)
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        sys.stdout = old_stdout

    return out.getvalue()
