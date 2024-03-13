from copy import deepcopy
from typing import List
from warnings import warn

import pydantic.version
import pytest
from pydantic import BaseModel, Field

from guardrails.utils.pydantic_utils.v1 import convert_pydantic_model_to_openai_fn

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
    not PYDANTIC_VERSION.startswith("1"),
    reason="Tests function calling syntax for Pydantic v1",
)
class TestConvertPydanticModelToOpenaiFn:
    def test_object_schema(self):
        expected_schema = deepcopy(foo_schema)
        # When pushed through BareModel it loses the description on any properties.
        del expected_schema["properties"]["bar"]["description"]

        # fmt: off
        expected_fn_params = {  # noqa
            "name": "Foo",
            "parameters": expected_schema
        }
        # fmt: on

        actual_fn_params = convert_pydantic_model_to_openai_fn(Foo)

        # assert actual_fn_params == expected_fn_params
        warn("Function calling is disabled for pydantic 1.x")
        assert actual_fn_params == {}

    def test_list_schema(self):
        expected_schema = deepcopy(foo_schema)
        # When pushed through BareModel it loses the description on any properties.
        del expected_schema["properties"]["bar"]["description"]

        # fmt: off
        expected_schema = {
            "title": f"List<{expected_schema.get('title')}>",
            "type": "array",
            "items": expected_schema
        }
        # fmt: on

        # fmt: off
        expected_fn_params = {  # noqa
            "name": "List<Foo>",
            "parameters": expected_schema
        }
        # fmt: on

        actual_fn_params = convert_pydantic_model_to_openai_fn(List[Foo])

        # assert actual_fn_params == expected_fn_params
        warn("Function calling is disabled for pydantic 1.x")
        assert actual_fn_params == {}
