from enum import Enum


class RailTypes(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    TIME = "time"
    DATETIME = "date-time"
    ENUM = "enum"
    LIST = "list"
    OBJECT = "object"
    CHOICE = "choice"
    CASE = "case"

    @classmethod
    def get(cls, key: str) -> "RailTypes":
        try:
            return cls(key)
        except Exception:
            return None
