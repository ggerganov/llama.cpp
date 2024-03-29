import atexit
from datetime import date
import datetime
import subprocess
import sys
from time import sleep
import time
import typer
from pydantic import BaseModel, Json, TypeAdapter
from annotated_types import MinLen
from typing import Annotated, Callable, List, Union, Literal, Optional, Type, get_args, get_origin
import json, requests

class Duration(BaseModel):
    seconds: Optional[int] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    days: Optional[int] = None
    months: Optional[int] = None
    years: Optional[int] = None

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

class WaitForDate(BaseModel):
    until: date

    def __call__(self):
        # Get the current date
        current_date = datetime.date.today()

        if self.until < current_date:
            raise ValueError("Target date cannot be in the past.")

        time_diff = datetime.datetime.combine(self.until, datetime.time.min) - datetime.datetime.combine(current_date, datetime.time.min)

        days, seconds = time_diff.days, time_diff.seconds

        sys.stderr.write(f"Waiting for {days} days and {seconds} seconds until {d}...\n")
        time.sleep(days * 86400 + seconds)
        sys.stderr.write(f"Reached the target date: {self.until}\n")
        

class StandardTools:

    @staticmethod
    def ask_user(question: str) -> str:
        '''
            Ask the user a question and return the answer.
            This allows getting additional information, requesting disambiguation, etc.
        '''
        return typer.prompt(question)
    
    @staticmethod
    def wait(_for: Union[WaitForDuration, WaitForDate]) -> None:
        '''
            Wait for a certain amount of time before continuing.
            This can be used to wait for a specific duration or until a specific date.
        '''
        return _for()
    
    @staticmethod
    def say_out_loud(something: str) -> str:
        """
            Just says something. Used to say each thought out loud
        """
        return subprocess.check_call(["say", something])
