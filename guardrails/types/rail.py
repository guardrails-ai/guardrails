from enum import Enum
from typing import Optional


class RailTypes(str, Enum):
    """RailTypes is an Enum that represents the builtin tags for RAIL xml.

    Attributes:
        STRING (Literal["string"]): A string value.
        INTEGER (Literal["integer"]): An integer value.
        FLOAT (Literal["float"]): A float value.
        BOOL (Literal["bool"]): A boolean value.
        DATE (Literal["date"]): A date value.
        TIME (Literal["time"]): A time value.
        DATETIME (Literal["date-time: - A datetime value.
        PERCENTAGE (Literal["percentage"]): A percentage value represented as a string.
            Example "20.5%".
        ENUM (Literal["enum"]): An enum value.
        LIST (Literal["list"]): A list/array value.
        OBJECT (Literal["object"]): An object/dictionary value.
        CHOICE (Literal["choice"]): The options for a discrimated union.
        CASE (Literal["case"]): A dictionary that contains a discrimated union.
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    TIME = "time"
    DATETIME = "date-time"
    PERCENTAGE = "percentage"
    ENUM = "enum"
    LIST = "list"
    OBJECT = "object"
    CHOICE = "choice"
    CASE = "case"

    @classmethod
    def get(cls, key: str) -> Optional["RailTypes"]:
        try:
            return cls(key)
        except Exception:
            return None
