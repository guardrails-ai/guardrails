# TODO: Move this file to guardrails.types
from enum import Enum
from typing import Dict, List, TypeVar

OT = TypeVar("OT", str, List, Dict)


class OutputTypes(str, Enum):
    STRING = "str"
    LIST = "list"
    DICT = "dict"
