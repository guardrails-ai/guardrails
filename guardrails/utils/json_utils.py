from dataclasses import dataclass
from lxml import etree as ET
from typing import Any, Dict


@dataclass
class Placeholder:
    expected_type: str

    @classmethod
    def type_dict(cls):
        return {
            "string": str,
            "integer": int,
            "float": float,
            "bool": bool,
            "object": dict,
            "list": list,
        }


def generate_json_skeleton_from_schema(schema: ET._Element) -> Dict[str, Any]:
    """Generate a JSON skeleton from an XML schema."""

    def _recurse_schema(schema):
        if schema.tag == "object":
            return {
                child.attrib["name"]: _recurse_schema(child)
                for child in schema
            }
        elif schema.tag == "list":
            return [
                _recurse_schema(schema[0])
            ]
        else:
            return Placeholder(schema.tag)

    return {child.attrib["name"]: _recurse_schema(child) for child in schema}


def verify_schema_against_json(xml_schema: ET._Element, generated_json: Dict[str, Any]):
    """Verify that a JSON schema is valid for a given XML."""

    json_schema = generate_json_skeleton_from_schema(xml_schema)

    def _verify_dict(schema, json):
        if set(schema.keys()) != set(json.keys()):
            return False

        for key in schema.keys():
            if isinstance(schema[key], Placeholder):
                expected_type = Placeholder.type_dict[schema[key].expected_type]
                if not isinstance(json[key], expected_type):
                    return False
            else:
                if isinstance(schema[key], dict):
                    if not isinstance(json[key], dict):
                        return False
                    if not _verify_dict(schema[key], json[key]):
                        return False
                elif isinstance(schema[key], list):
                    if not isinstance(json[key], list):
                        return False
                    if not _verify_list(schema[key][0], json[key][0]):
                        return False
                else:
                    raise ValueError(f"Unknown type {type(schema[key])}")

        return True
    
    def _verify_list(schema, json):
        assert len(schema) == 1  # Schema for a list should only have one child
        if not isinstance(json, list):
            return False
        
        if isinstance(schema[0], Placeholder):
            expected_type = Placeholder.type_dict[schema[0].expected_type]

            for item in json:
                if not isinstance(item, expected_type):
                    return False
        else:
            expected_type = type(schema[0])
            for item in json:
                if not isinstance(item, expected_type):
                    return False
                if isinstance(item, dict):
                    if not _verify_dict(schema[0], item):
                        return False
                elif isinstance(item, list):
                    if not _verify_list(schema[0], item):
                        return False
                else:
                    raise ValueError(f"Unknown type {type(item)}")
        
        return True

    return _verify_dict(json_schema, generated_json)
