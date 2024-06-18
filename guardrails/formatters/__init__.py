from guardrails.formatters.base_formatter import BaseFormatter, PassthroughFormatter
from guardrails.formatters.json_formatter import JsonFormatter


def get_formatter(name: str, *args, **kwargs) -> BaseFormatter:
    """Returns a class."""
    name = name.lower()
    if name == "jsonformer":
        return JsonFormatter(*args, **kwargs)
    elif name == "none":
        return PassthroughFormatter(*args, **kwargs)
    raise ValueError(f"Unrecognized formatter '{name}'")


__all__ = [
    "get_formatter",
    "BaseFormatter",
    "PassthroughFormatter",
    "JsonFormatter",
]
