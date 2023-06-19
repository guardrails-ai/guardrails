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


def generate_type_skeleton_from_schema(schema: ET._Element) -> Dict[str, Any]:
    """Generate a JSON skeleton from an XML schema."""

    def _recurse_schema(schema):
        if schema.tag == "object":
            return {child.attrib["name"]: _recurse_schema(child) for child in schema}
        elif schema.tag == "list":
            if len(schema) == 0:
                return []
            return [_recurse_schema(schema[0])]
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

    def _verify_dict(schema, json):
        if not schema.keys():
            return True
        json_keys = set(json.keys())
        schema_keys = set(schema.keys())
        extra_keys = json_keys - schema_keys
        if prune_extra_keys and extra_keys:
            for key in extra_keys:
                del json[key]
        if any(key not in json_keys for key in schema_keys):
            return False

        for key in schema.keys():
            if isinstance(schema[key], Placeholder):
                expected_type = schema[key].type_object
                if expected_type == Any:
                    continue
                if not isinstance(json[key], expected_type):
                    if not coerce_types:
                        return False
                    try:
                        json[key] = expected_type(json[key])
                    except ValueError:
                        return False
            elif isinstance(schema[key], dict):
                if not isinstance(json[key], dict):
                    return False
                if not _verify_dict(schema[key], json[key]):
                    return False
            elif isinstance(schema[key], list):
                if not isinstance(json[key], list):
                    return False
                if not _verify_list(schema[key], json[key]):
                    return False
            else:
                raise ValueError(f"Unknown type {type(schema[key])}")

        return True

    def _verify_list(schema, json):
        if len(schema) == 0:
            return True
        if len(schema) > 1:
            raise ValueError("List schema should only have one child")
        if not isinstance(json, list):
            return False

        child_schema = schema[0]

        if isinstance(child_schema, Placeholder):
            expected_type = child_schema.type_object
            if expected_type == Any:
                return True

            for i, item in enumerate(json):
                if not isinstance(item, expected_type):
                    if not coerce_types:
                        return False
                    try:
                        json[i] = expected_type(item)
                    except ValueError:
                        return False
        elif isinstance(child_schema, dict):
            for item in json:
                if not isinstance(item, dict):
                    return False
                if not _verify_dict(child_schema, item):
                    return False
        elif isinstance(child_schema, list):
            for item in json:
                if not isinstance(item, list):
                    return False
                if not _verify_list(child_schema, item):
                    return False
        else:
            raise ValueError(f"Unknown type {type(child_schema)}")

        return True

    return _verify_dict(type_skeleton, generated_json)
