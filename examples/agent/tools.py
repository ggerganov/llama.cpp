# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ipython",
# ]
# ///
import datetime
import json
from pydantic import BaseModel
import sys
import time
import types
from typing import Union, Optional, Dict


class Duration(BaseModel):
    seconds: Optional[int] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    days: Optional[int] = None
    months: Optional[int] = None
    years: Optional[int] = None

    def __str__(self) -> str:
        return ', '.join([
            x
            for x in [
                f"{self.years} years" if self.years else None,
                f"{self.months} months" if self.months else None,
                f"{self.days} days" if self.days else None,
                f"{self.hours} hours" if self.hours else None,
                f"{self.minutes} minutes" if self.minutes else None,
                f"{self.seconds} seconds" if self.seconds else None,
            ]
            if x is not None
        ])

    @property
    def get_total_seconds(self) -> int:
        return sum([
            self.seconds or 0,
            (self.minutes or 0)*60,
            (self.hours or 0)*3600,
            (self.days or 0)*86400,
            (self.months or 0)*2592000,
            (self.years or 0)*31536000,
        ])

class WaitForDuration(BaseModel):
    duration: Duration

    def __call__(self):
        sys.stderr.write(f"Waiting for {self.duration}...\n")
        time.sleep(self.duration.get_total_seconds)

@staticmethod
def wait_for_duration(duration: Duration) -> None:
    'Wait for a certain amount of time before continuing.'

    # sys.stderr.write(f"Waiting for {duration}...\n")
    time.sleep(duration.get_total_seconds)

@staticmethod
def wait_for_date(target_date: datetime.date) -> None:
    f'''
        Wait until a specific date is reached before continuing.
        Today's date is {datetime.date.today()}
    '''

    # Get the current date
    current_date = datetime.date.today()

    if target_date < current_date:
        raise ValueError("Target date cannot be in the past.")

    time_diff = datetime.datetime.combine(target_date, datetime.time.min) - datetime.datetime.combine(current_date, datetime.time.min)

    days, seconds = time_diff.days, time_diff.seconds

    # sys.stderr.write(f"Waiting for {days} days and {seconds} seconds until {target_date}...\n")
    time.sleep(days * 86400 + seconds)
    # sys.stderr.write(f"Reached the target date: {target_date}\n")

def _is_serializable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except Exception as e:
        return False

def python(code: str) -> str:
    """
    Executes Python code in a siloed environment using IPython and returns the output.

    Parameters:
        code (str): The Python code to execute.

    Returns:
        str: The output of the executed code.
    """
    from IPython import InteractiveShell
    from io import StringIO
    import sys

    # Create an isolated IPython shell instance
    shell = InteractiveShell()

    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    try:
        # Execute the code
        shell.run_cell(code)
    except Exception as e:
        # Restore stdout before returning
        sys.stdout = old_stdout
        return f"An error occurred: {e}"
    finally:
        # Always restore stdout
        sys.stdout = old_stdout

    # Retrieve the output
    output = mystdout.getvalue()
    return output


# def python(source: str) -> Union[Dict, str]:
#     """
#         Evaluate a Python program and return the globals it declared.
#         Can be used to compute mathematical expressions (e.g. after importing math module).
#         Args:
#             source: contain valid, executable and pure Python code. Should also import any required Python packages.
#                 For example: "import math\nresult = math.cos(2) * 10"
#         Returns:
#             dict | str: A dictionary containing variables declared, or an error message if an exception occurred.
#     """
#     try:
#         namespace = {}
#         sys.stderr.write(f"Executing Python program:\n{source}\n")
#         exec(source, namespace)
#         results = {
#             k: v
#             for k, v in namespace.items()
#             if not k.startswith('_') \
#                 and not isinstance(v, type) \
#                 and not isinstance(v, types.ModuleType) \
#                 and not callable(v) \
#                 and _is_serializable(v)
#         }
#         sys.stderr.write(f"Results: {json.dumps(results, indent=2)}\n")
#         return results
#     except Exception as e:
#         msg = f"Error: {sys.exc_info()[1]}"
#         sys.stderr.write(f"{msg}\n")
#         return msg
