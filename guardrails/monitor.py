import json
from typing import Any, Dict

class Singleton(type):
    """A metaclass that ensures only one instance of a class is created.

    Usage:

        >>> class Example(metaclass=Singleton):
        ...     def __init__(self, x):
        ...         self.x = x
        >>> a = Example(1)
        >>> b = Example(2)
        >>> print(a, id(a))
        <__main__.Example object at 0x7f8b8c0b7a90> 140071000000000
        >>> print(b, id(b))
        <__main__.Example object at 0x7f8b8c0b7a90> 140071000000000
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        # Look up if this cls pair has been created before
        if cls not in cls._instances:
            # If not, we let a new instance be created
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class JsonLinesFile:

    def __init__(self, filepath: str):
        if not filepath.endswith(".jsonl"):
            filepath += ".jsonl"

        self.filepath = filepath
        self.buffer = {}

    def flush(self):
        try:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(self.buffer) + "\n")
        except TypeError:
            # Check which key is not serializable
            for key, value in self.buffer.items():
                try:
                    json.dumps({key: value})
                except TypeError:
                    raise TypeError(
                        f"Data {self.buffer} must be serializable to JSON."
                        "{key}: {value} is not serializable."
                        )
        self.buffer = {}

    def write(self, data: Dict[str, Any]):
        # If any key conflicts, flush the buffer.
        if set(self.buffer.keys()).intersection(set(data.keys())):
            self.flush()
        self.buffer.update(data)


class Sink(metaclass=Singleton):

    def __init__(self, filepath: str):
        
        # TODO: add support for writing to a database
        self.client = JsonLinesFile(filepath)

    def flush(self):
        self.client.flush()

    def write(self, data: Dict[str, Any]):
        self.client.write(data)


def log(data: Dict[str, Any]) -> None:
    try:
        sink = Sink()
    except TypeError:
        # No sink has been initialized, so we don't log anything.
        return
    sink.write(data)
