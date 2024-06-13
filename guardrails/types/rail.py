from enum import Enum
from typing import Optional


class RailTypes(str, Enum):
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
