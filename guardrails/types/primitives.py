from enum import Enum
from guardrails_api_client import SimpleTypes


class PrimitiveTypes(str, Enum):
    BOOLEAN = SimpleTypes.BOOLEAN
    INTEGER = SimpleTypes.INTEGER
    NUMBER = SimpleTypes.NUMBER
    STRING = SimpleTypes.STRING
