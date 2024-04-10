from datetime import date
import datetime
from pydantic import BaseModel
import subprocess
import sys
import time
import typer
from typing import Union, Optional

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

class StandardTools:

    @staticmethod
    def ask_user(question: str) -> str:
        '''
            Ask the user a question and return the answer.
            This allows getting additional information, requesting disambiguation, etc.
        '''
        return typer.prompt(question)

    @staticmethod
    def wait_for_duration(duration: Duration) -> None:
        'Wait for a certain amount of time before continuing.'

        # sys.stderr.write(f"Waiting for {duration}...\n")
        time.sleep(duration.get_total_seconds)

    @staticmethod
    def wait_for_date(target_date: date) -> None:
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

    @staticmethod
    def say_out_loud(something: str) -> None:
        """
            Just says something. Used to say each thought out loud
        """
        subprocess.check_call(["say", something])
