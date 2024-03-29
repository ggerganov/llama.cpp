import math

def add(a: float, b: float) -> float:
    """
        Add a and b reliably.
        Don't use this tool to compute the square of a number (use multiply or pow instead)
    """
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply a with b reliably"""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b reliably"""
    return a / b

def pow(value: float, power: float) -> float:
    """
        Raise a value to a power (exponent) reliably.
        The square of x is pow(x, 2), its cube is pow(x, 3), etc.
    """
    return math.pow(value, power)
