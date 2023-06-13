from dataclasses import dataclass
import lxml.etree as ET
from typing import Any, Dict


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

    type_string: str

    @property
    def type_object(self):
        return self.type_map[self.type_string]


def generate_type_skeleton_from_schema(schema: ET._Element) -> Dict[str, Any]:
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

    type_skeleton = generate_type_skeleton_from_schema(xml_schema)

    def _verify_dict(schema, json):
        if set(schema.keys()) != set(json.keys()):
            return False

        for key in schema.keys():
            if isinstance(schema[key], Placeholder):
                expected_type = schema[key].type_object
                if not isinstance(json[key], expected_type):
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
        assert len(schema) == 1  # Schema for a list should only have one child
        if not isinstance(json, list):
            return False

        child_schema = schema[0]
        
        if isinstance(child_schema, Placeholder):
            expected_type = child_schema.type_object

            for item in json:
                if not isinstance(item, expected_type):
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
