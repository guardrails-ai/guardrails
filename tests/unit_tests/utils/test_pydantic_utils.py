import pytest
from pydantic import BaseModel, Field

from guardrails.utils.pydantic_utils import (
    PYDANTIC_VERSION,
    add_pydantic_validators_as_guardrails_validators,
    add_validator,
)
from guardrails.validators import FailResult, PassResult, ValidChoices, ValidLength


@pytest.mark.skipif(
    not PYDANTIC_VERSION.startswith("1"),
    reason="Tests validators syntax for Pydantic v1",
)
def test_add_pydantic_validators_as_guardrails_validators():
    # TODO(shreya): Uncomment when custom validators are supported
    # def dummy_validator(name: str):
    #     if name != "Alex":
    #         raise ValueError("Name must be Alex")
    #     return name

    class DummyModel(BaseModel):
        name: str = Field(..., validators=[ValidLength(min=1, max=10)])

        _name_choices_validator = add_validator(
            "name", fn=ValidChoices(choices=["Alex", "Bob"])
        )
        # TODO(shreya): Uncomment when custom validators are supported
        # _name_alex_validator = add_validator("name", fn=dummy_validator)

        # @validator("name")
        # def validate_name(cls, name):
        #     if name != "Bob":
        #         raise ValueError("Name must be Alex")
        #     return name

    model_fields = add_pydantic_validators_as_guardrails_validators(DummyModel)
    name_field = model_fields["name"]

    # Should have 1 field
    assert len(model_fields) == 1, "Should only have one field"

    # Should have 4 validators: 1 from the field, 2 from the add_validator method,
    # and 1 from the validator decorator.
    validators = name_field.field_info.extra["validators"]
    assert len(validators) == 2, "Should have 4 validators"

    # The BaseModel field should not be modified
    assert len(DummyModel.__fields__["name"].field_info.extra["validators"]) == 1

    # The first validator should be the ValidLength validator
    assert isinstance(
        validators[0], ValidLength
    ), "First validator should be ValidLength"
    assert isinstance(validators[0].validate("Beatrice", {}), PassResult)
    assert isinstance(validators[0].validate("MrAlexander", {}), FailResult)

    # The second validator should be the ValidChoices validator
    assert isinstance(
        validators[1][0], ValidChoices
    ), "Second validator should be ValidChoices"
    assert validators[1][1] == "reask"
    assert isinstance(validators[1][0].validate("Alex", {}), PassResult)
    assert isinstance(validators[1][0].validate("Bob", {}), PassResult)
    assert isinstance(validators[1][0].validate("Candace", {}), FailResult)

    # TODO(shreya): Uncomment when custom validators are supported
    # # The third validator should be the dummy validator
    # assert (
    #     validators[2].field_validator == dummy_validator
    # ), "Third validator should be dummy_validator"
    # validators[2].validate(None, "Alex", None)
    # with pytest.raises(EventDetail):
    #     validators[2].validate(None, "Bob", None)

    # # The fourth validator should be the validator decorator
    # assert (
    #     validators[3].field_validator.func == DummyModel.validate_name.__func__
    # ), "Fourth validator should be DummyModel.validate_name"
    # validators[3].validate(None, "Bob", None)
    # with pytest.raises(EventDetail):
    #     validators[3].validate(None, "Alex", None)


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
