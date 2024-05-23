# TODO: Move this file to guardrails.types
from enum import Enum
from typing import Any, Dict, List, TypeVar
from guardrails_api_client import SimpleTypes

OT = TypeVar("OT", str, List, Dict)


class OutputTypes(str, Enum):
    STRING = "str"
    LIST = "list"
    DICT = "dict"

    def get(self, key, default=None):
        try:
            return self[key]
        except Exception:
            return default

    @classmethod
    def __from_json_schema__(cls, json_schema: Dict[str, Any]) -> "OutputTypes":
        if not json_schema:
            return cls("str")

        schema_type = json_schema.get("type")
        if schema_type == SimpleTypes.STRING:
            return cls("str")
        elif schema_type == SimpleTypes.OBJECT:
            return cls("dict")
        elif schema_type == SimpleTypes.ARRAY:
            return cls("list")

        all_of = json_schema.get("allOf")
        if all_of:
            return cls("dict")

        one_of: List[Dict[str, Any]] = [
            s
            for s in json_schema.get("oneOf", [])
            if isinstance(s, dict) and "type" in s
        ]
        if one_of:
            first_sub_schema = one_of[0]
            return cls.__from_json_schema__(first_sub_schema)

        any_of: List[Dict[str, Any]] = [
            s
            for s in json_schema.get("anyOf", [])
            if isinstance(s, dict) and "type" in s
        ]
        if any_of:
            first_sub_schema = any_of[0]
            return cls.__from_json_schema__(first_sub_schema)

        # Fallback to string
        return cls("str")
