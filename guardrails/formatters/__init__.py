from guardrails.formatters.base_formatter import BaseFormatter, PassthroughFormatter
from guardrails.formatters.json_formatter import JsonFormatter


def get_formatter(name: str, *args, **kwargs) -> BaseFormatter:
    """Returns a class"""
    match name.lower():
        case "jsonformer":
            return JsonFormatter(*args, **kwargs)
        case "none":
            return PassthroughFormatter(*args, **kwargs)
    raise ValueError(f"Unrecognized formatter '{name}'")


__all__ = [
    "get_formatter",
    "BaseFormatter",
    "PassthroughFormatter",
    "JsonFormatter",
]
