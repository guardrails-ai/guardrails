import pydantic.version
import pytest
from pydantic import BaseModel, Field

from guardrails.utils.pydantic_utils.v1 import convert_pydantic_model_to_openai_fn

PYDANTIC_VERSION = pydantic.version.VERSION


# This test is descriptive, not prescriptive.
@pytest.mark.skipif(
    not PYDANTIC_VERSION.startswith("1"),
    reason="Tests function calling syntax for Pydantic v1",
)
def test_convert_pydantic_model_to_openai_fn():
    class Foo(BaseModel):
        bar: str = Field(description="some string value")

    # fmt: off
    expected_schema = {
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
