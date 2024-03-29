
import random
from typing import Literal


def _weather(w: str, temp, format):
    return f'{w}, {temp}C' if format == 'celsius' \
        else f'{w}, {(temp * 9/5) + 32}F'

def get_current_weather(location: str, format: Literal["celsius", "fahrenheit"]) -> str:
      '''
        Get the current weather

        Args:
            location: The city and state, e.g. San Francisco, CA
            format: The temperature unit to use. Infer this from the users location.
      '''
      return _weather('Sunny', 31, format)

def get_n_day_weather_forecast(location: str, format: Literal["celsius", "fahrenheit"], num_days: int) -> str:
    '''
        Get an N-day weather forecast

        Args:
            location: The city and state, e.g. San Francisco, CA
            format: The temperature unit to use. Infer this from the users location.
            num_days: The number of days to forecast
    '''
    random.seed(123)
    return '\n'.join([
        f'{num_days} forecast for {location}:',
        *(
            f'- in {i} day{"s" if i > 1 else ""}: {_weather("Sunny" if i % 2 == 0 else "Cloudy", random.randrange(15, 35), format)}'
            for i in range(1, num_days)
        )
    ])
