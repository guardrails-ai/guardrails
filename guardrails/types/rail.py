from enum import Enum


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
