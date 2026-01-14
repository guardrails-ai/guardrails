from guardrails.formatters.base_formatter import BaseFormatter, PassthroughFormatter

try:
    from guardrails.formatters.json_formatter import JsonFormatter
except ImportError:
    JsonFormatter = None


def get_formatter(name: str, *args, **kwargs) -> BaseFormatter:
    """Returns a class."""
    name = name.lower()
    if name == "jsonformer":
        if JsonFormatter is None:
            raise ValueError("jsonformatter requires transformers to be installed.")
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
