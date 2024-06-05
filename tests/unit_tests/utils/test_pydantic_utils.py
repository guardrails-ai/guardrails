import pytest
from pydantic import BaseModel, Field


import pydantic.version as PYDANTIC_VERSION

from guardrails.utils.pydantic_utils import (
    add_pydantic_validators_as_guardrails_validators,
)
from guardrails.validator_base import OnFailAction
from guardrails.validators import FailResult, PassResult, ValidChoices, ValidLength

@pytest.mark.skipif(
    not PYDANTIC_VERSION.startswith("2"),
    reason="Tests validators syntax for Pydantic v2",
)
def test_add_pydantic_validators_as_guardrails_validators_v2():
    class DummyModel(BaseModel):
        name: str = Field(..., validators=[ValidLength(min=1, max=10)])

    model_fields = add_pydantic_validators_as_guardrails_validators(DummyModel)
    name_field = model_fields["name"]

    # Should have 1 field
    assert len(model_fields) == 1, "Should only have one field"

    # Should have 4 validators: 1 from the field, 2 from the add_validator method,
    # and 1 from the validator decorator.
    validators = name_field.json_schema_extra["validators"]
    assert len(validators) == 1, "Should have 1 validator"

    # The BaseModel field should not be modified
    assert len(DummyModel.model_fields["name"].json_schema_extra["validators"]) == 1

    # The first validator should be the ValidLength validator
    assert isinstance(
        validators[0], ValidLength
    ), "First validator should be ValidLength"
    assert isinstance(validators[0].validate("Beatrice", {}), PassResult)
    assert isinstance(validators[0].validate("MrAlexander", {}), FailResult)
