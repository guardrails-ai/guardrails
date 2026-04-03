from enum import Enum


class SimpleTypes(str, Enum):
    """SimpleTypes."""

    ARRAY = "array"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NULL = "null"
    NUMBER = "number"
    OBJECT = "object"
    STRING = "string"
