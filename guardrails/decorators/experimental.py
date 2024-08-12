import functools
from guardrails.logger import logger


def experimental(func):
    """Decorator to mark a function as experimental."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.warn(
            f"The function '{func.__name__}' is experimental and subject to change."
        )
        return func(*args, **kwargs)

    return wrapper
