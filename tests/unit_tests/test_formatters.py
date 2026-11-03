from typing import Any, Dict, List

from pydantic import BaseModel

from guardrails.formatters.json_formatter import _jsonschema_to_jsonformer


def test_basic_schema_conversion():
    class Simple(BaseModel):
        my_age: int
        my_height_in_nanometers: float
        my_name: str
        my_friends: List[str]

    out_schema = _jsonschema_to_jsonformer(Simple.model_json_schema())
    assert out_schema["type"] == "object"
    assert out_schema["properties"]["my_age"]["type"] == "number"
    assert out_schema["properties"]["my_height_in_nanometers"]["type"] == "number"
    assert out_schema["properties"]["my_name"]["type"] == "string"
    assert out_schema["properties"]["my_friends"]["type"] == "array"
    assert out_schema["properties"]["my_friends"]["items"]["type"] == "string"


def test_nested_schema_conversion():
    class Simple(BaseModel):
        name: str

    class Nested(BaseModel):
        best_dog: Simple
        good_dogs: List[Simple]  # May cause OoM if enumerated.  Consider generator.

    out = _jsonschema_to_jsonformer(Nested.model_json_schema())
    assert out["type"] == "object"
    assert out["properties"]["best_dog"]["type"] == "object"
    assert out["properties"]["best_dog"]["properties"]["name"]["type"] == "string"
    assert out["properties"]["good_dogs"]["type"] == "array"
    assert out["properties"]["good_dogs"]["items"]["type"] == "object"


def test_object_schema_without_properties():
    class UserProfile(BaseModel):
        username: str
        metadata: Dict[str, Any]

    out = _jsonschema_to_jsonformer(UserProfile.model_json_schema())
    assert out["properties"]["username"]["type"] == "string"
    metadata = out["properties"]["metadata"]
    assert metadata["type"] == "object"
    assert metadata["properties"] == {}
    assert metadata["additionalProperties"] is True


def test_nested_dict_schema_without_properties():
    class Nested(BaseModel):
        config: Dict[str, Dict[str, Any]]

    out = _jsonschema_to_jsonformer(Nested.model_json_schema())
    config = out["properties"]["config"]
    assert config["type"] == "object"
    assert config["properties"] == {}
    assert config["additionalProperties"]["type"] == "object"


def test_direct_object_schema_without_properties():
    out = _jsonschema_to_jsonformer(
        {"type": "object", "additionalProperties": {"type": "string"}}
    )
    assert out["type"] == "object"
    assert out["properties"] == {}
    assert out["additionalProperties"] == {"type": "string"}
