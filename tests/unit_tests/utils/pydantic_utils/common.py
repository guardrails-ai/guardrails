from copy import deepcopy

import pydantic
from pydantic import BaseModel, Field

from guardrails.utils.pydantic_utils.common import schema_to_bare_model

PYDANTIC_VERSION = pydantic.version.VERSION


class Foo(BaseModel):
    bar: str = Field(description="some string value")


# fmt: off
foo_schema = {
    "title": "Foo",
    "type": "object",
    "properties": {
        "bar": {
            "title": "Bar",
            "description": "some string value",
            "type": "string"
        }
    },
    "required": [
        "bar"
    ]
}
# fmt: on


# These tests demonstrate the issue fixed in PR #616
class TestSchemaToBareModel:
    def test_with_model_type(self):
        expected_schema = deepcopy(foo_schema)
        # When pushed through BareModel it loses the description on any properties.
        del expected_schema["properties"]["bar"]["description"]

        # Current logic
        bare_model = schema_to_bare_model(Foo)

        # Convert Pydantic model to JSON schema
        if PYDANTIC_VERSION.startswith("1"):
            json_schema = bare_model.schema()
        else:
            json_schema = bare_model.model_json_schema()
        json_schema["title"] = Foo.__name__

        assert json_schema == expected_schema

    def test_with_type_of_model_type(self):
        expected_schema = deepcopy(foo_schema)
        # When pushed through BareModel it loses the description on any properties.
        del expected_schema["properties"]["bar"]["description"]

        empty_schema = {"properties": {}, "title": "BareModel", "type": "object"}

        # Previous logic
        bare_model = schema_to_bare_model(type(Foo))

        # Convert Pydantic model to JSON schema
        if PYDANTIC_VERSION.startswith("1"):
            json_schema = bare_model.schema()
        else:
            json_schema = bare_model.model_json_schema()

        assert json_schema != expected_schema
        assert json_schema == empty_schema
