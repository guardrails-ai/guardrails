from enum import Enum
from guardrails_api_client import SimpleTypes


class PrimitiveTypes(str, Enum):
    BOOLEAN = SimpleTypes.BOOLEAN.value
    INTEGER = SimpleTypes.INTEGER.value
    NUMBER = SimpleTypes.NUMBER.value
    STRING = SimpleTypes.STRING.value

    @staticmethod
    def is_primitive(value: str) -> bool:
        try:
            return value in [member.value for member in PrimitiveTypes]
        except Exception as e:
            print(e)
            return False
