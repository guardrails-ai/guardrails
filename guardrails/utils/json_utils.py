from dataclasses import dataclass
from typing import Any, Dict

import lxml.etree as ET


@dataclass
class Placeholder:
    type_map = {
        "string": str,
        "integer": int,
        "float": float,
        "bool": bool,
        "object": dict,
        "list": list,
        "date": str,
        "time": str,
    }
    ignore_types = [
        "pydantic",
        "email",  # email and url should become string validators
        "url",
    ]

    type_string: str

    @property
    def type_object(self):
        if self.type_string in self.ignore_types:
            return Any
        return self.type_map[self.type_string]


@dataclass
class Choice:
    name: str
    cases: Dict[str, Any]


def generate_type_skeleton_from_schema(schema: ET._Element) -> Dict[str, Any]:
    """Generate a JSON skeleton from an XML schema."""

    def _recurse_schema(schema):
        if schema.tag == "object":
            return {child.attrib["name"]: _recurse_schema(child) for child in schema}
        elif schema.tag == "list":
            if len(schema) == 0:
                return []
            return [_recurse_schema(schema[0])]
        elif schema.tag == "choice":
            return Choice(
                schema.attrib["name"],
                {
                    child.attrib["name"]: _recurse_schema(child[0])
                    for child in schema
                    if child.tag == "case"
                },
            )
        else:
            return Placeholder(schema.tag)

    return {child.attrib["name"]: _recurse_schema(child) for child in schema}


def verify_schema_against_json(
    xml_schema: ET._Element,
    generated_json: Dict[str, Any],
    prune_extra_keys: bool = False,
    coerce_types: bool = False,
):
    """Verify that a JSON schema is valid for a given XML."""

    type_skeleton = generate_type_skeleton_from_schema(xml_schema)

    def verification_failed():
        # Sentinel value
        pass

    def _verify_and_cast_value(placeholder, value):
        expected_type = placeholder.type_object
        if expected_type == Any:
            return value
        if not isinstance(value, expected_type):
            if not coerce_types:
                return verification_failed
            try:
                return expected_type(value)
            except ValueError:
                return verification_failed
        return value

    def _verify_dict(schema, json):
        if not isinstance(json, dict):
            return False
        if not schema.keys():
            return True
        json_keys = set(json.keys())

        schema_keys = set(schema.keys())
        choice_keys = set()
        for key in schema_keys:
            if isinstance(schema[key], Choice):
                choice_keys.update(schema[key].cases.keys())

        extra_keys = json_keys - schema_keys - choice_keys
        if prune_extra_keys and extra_keys:
            for key in extra_keys:
                del json[key]
        if any(key not in json_keys for key in schema_keys):
            return False

        for key in schema.keys():
            if isinstance(schema[key], Placeholder):
                value = _verify_and_cast_value(schema[key], json[key])
                if value is verification_failed:
                    return False
                json[key] = value
            elif isinstance(schema[key], dict):
                if not _verify_dict(schema[key], json[key]):
                    return False
            elif isinstance(schema[key], list):
                if not _verify_list(schema[key], json[key]):
                    return False
            elif isinstance(schema[key], Choice):
                if not _verify_choice(schema[key], json):
                    return False
            else:
                raise ValueError(f"Unknown type {type(schema[key])}")

        return True

    def _verify_list(schema, json):
        if not isinstance(json, list):
            return False
        if len(schema) == 0:
            return True
        if len(schema) > 1:
            raise ValueError("List schema should only have one child")
        if not isinstance(json, list):
            return False

        child_schema = schema[0]

        if isinstance(child_schema, Placeholder):
            for i, item in enumerate(json):
                value = _verify_and_cast_value(child_schema, item)
                if value is verification_failed:
                    return False
                json[i] = value
            return True
        elif isinstance(child_schema, dict):
            verification_func = _verify_dict
        elif isinstance(child_schema, list):
            verification_func = _verify_list
        elif isinstance(child_schema, Choice):
            verification_func = _verify_choice
        else:
            raise ValueError(f"Unknown type {type(child_schema)}")

        for item in json:
            if not verification_func(child_schema, item):
                return False

        return True

    def _verify_choice(schema, json):
        if not isinstance(json, dict):
            return False
        if schema.name not in json:
            return False
        value_name = json[schema.name]
        if value_name not in schema.cases:
            return False
        if value_name not in json:
            return False
        if any(key in json for key in schema.cases.keys() if key != value_name):
            return False
        value_schema = schema.cases[value_name]
        value = json[value_name]
        if isinstance(value_schema, Placeholder):
            value = _verify_and_cast_value(value_schema, value)
            if value is verification_failed:
                return False
            json[value_name] = value
        elif isinstance(value_schema, dict):
            if not _verify_dict(value_schema, value):
                return False
        elif isinstance(value_schema, list):
            if not _verify_list(value_schema, value):
                return False
        elif isinstance(value_schema, Choice):
            if not _verify_choice(value_schema, value):
                return False
        else:
            raise ValueError(f"Unknown type {type(value_schema)}")

        return True

    return _verify_dict(type_skeleton, generated_json)
