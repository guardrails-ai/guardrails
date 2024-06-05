from copy import deepcopy
from typing import List

import pydantic.version
import pytest
from pydantic import BaseModel, Field

from guardrails.utils.pydantic_utils import convert_pydantic_model_to_openai_fn

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
class TestConvertPydanticModelToOpenaiFn:
    def test_object_schema(self):
        expected_schema = deepcopy(foo_schema)

        # fmt: off
        expected_fn_params = {
            "name": "Foo",
            "parameters": expected_schema
        }
        # fmt: on

        actual_fn_params = convert_pydantic_model_to_openai_fn(Foo)

        assert actual_fn_params == expected_fn_params

    def test_list_schema(self):
        expected_schema = deepcopy(foo_schema)

        # fmt: off
        expected_schema = {
            "title": f"Array<{expected_schema.get('title')}>",
            "type": "array",
            "items": expected_schema
        }
        # fmt: on

        # fmt: off
        expected_fn_params = {
            "name": "Array<Foo>",
            "parameters": expected_schema
        }
        # fmt: on

        actual_fn_params = convert_pydantic_model_to_openai_fn(List[Foo])

        assert actual_fn_params == expected_fn_params
