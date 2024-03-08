from copy import deepcopy

import pydantic.version
import pytest
from pydantic import BaseModel, Field

from guardrails.utils.pydantic_utils.v2 import (
    _create_bare_model,
    convert_pydantic_model_to_openai_fn,
)

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


# This test is descriptive, not prescriptive.
@pytest.mark.skipif(
    not PYDANTIC_VERSION.startswith("2"),
    reason="Tests function calling syntax for Pydantic v2",
)
def test_convert_pydantic_model_to_openai_fn():
    expected_schema = deepcopy(foo_schema)
    # When pushed through BareModel it loses the description on any properties.
    del expected_schema["properties"]["bar"]["description"]

    # fmt: off
    expected_fn_params = {
        "name": "Foo",
        "parameters": expected_schema
    }
    # fmt: on

    actual_fn_params = convert_pydantic_model_to_openai_fn(Foo)

    assert actual_fn_params == expected_fn_params


# These tests demonstrate the issue fixed in PR #616
@pytest.mark.skipif(
    not PYDANTIC_VERSION.startswith("2"),
    reason="Tests function calling syntax for Pydantic v2",
)
class TestCreateBareModel:
    def test_with_model_type(self):
        expected_schema = deepcopy(foo_schema)
        # When pushed through BareModel it loses the description on any properties.
        del expected_schema["properties"]["bar"]["description"]

        # Current logic
        bare_model = _create_bare_model(Foo)

        # Convert Pydantic model to JSON schema
        json_schema = bare_model.model_json_schema()
        json_schema["title"] = Foo.__name__

        assert json_schema == expected_schema

    def test_with_type_of_model_type(self):
        expected_schema = deepcopy(foo_schema)
        # When pushed through BareModel it loses the description on any properties.
        del expected_schema["properties"]["bar"]["description"]

        empty_schema = {"properties": {}, "title": "BareModel", "type": "object"}

        # Previous logic
        bare_model = _create_bare_model(type(Foo))

        # Convert Pydantic model to JSON schema
        json_schema = bare_model.model_json_schema()

        assert json_schema != expected_schema
        assert json_schema == empty_schema
