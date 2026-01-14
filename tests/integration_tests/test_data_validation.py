# ruff: noqa: E501
from typing import Literal, Union

import pytest
from pydantic import BaseModel, Field

from guardrails import Guard
from guardrails.errors import ValidationError
from guardrails.actions.reask import ReAsk
from guardrails.validator_base import OnFailAction
from tests.integration_tests.test_assets.validators import ValidChoices

###   llm_output,                                             raises, fails, has_error ###
test_cases = [
    ('{"choice": {"action": "fight", "fight_move": "kick"}}', False, False, False),
    (
        '{"choice": {"action": "flight", "flight_direction": "north", "flight_speed": 1}}',
        False,
        False,
        False,
    ),
    ('{"choice": {"action": "flight", "fight_move": "punch"}}', False, True, False),
    (
        '{"choice": {"action": "fight", "flight_direction": "north", "flight_speed": 1}}',
        False,
        True,
        False,
    ),
    ('{"choice": {"action": "random_action"}}', False, True, False),
    ('{"choice": {"action": "fight", "fight_move": "random_move"}}', True, True, True),
    (
        '{"choice": {"action": "flight", "random_key": "random_value"}}',
        False,
        True,
        False,
    ),
    (
        '{"choice": {"action": "flight", "random_key": "random_value"}',
        False,
        True,
        True,
    ),
]


@pytest.mark.parametrize("llm_output, raises, fails, has_error", test_cases)
def test_choice_validation(llm_output, raises, fails, has_error):
    rail_spec = """
<rail version="0.1">

<output>
    <choice name="choice" on-fail-choice="exception" discriminator="action">
        <case name="fight">
            <string name="fight_move" validators="valid-choices: {['punch','kick','headbutt']}" on-fail-valid-choices="exception" />
        </case>
        <case name="flight">
            <string name="flight_direction" validators="valid-choices: {['north','south','east','west']}" on-fail-valid-choices="exception" />
            <integer name="flight_speed" validators="valid-choices: {[1,2,3,4]}" on-fail-valid-choices="exception" />
        </case>
    </choice>
</output>

<messages>
    <message role="user">
Dummy prompt.
    </message>
</messages>
</rail>
"""
    guard = Guard.for_rail_string(rail_spec)

    # If raises is True, then the test should raise an exception.
    # For our existing test cases this will always be a ValidationError
    if raises:
        with pytest.raises(ValidationError):
            guard.parse(llm_output, num_reasks=0)
    else:
        result = guard.parse(llm_output, num_reasks=0)
        if fails and not has_error:
            assert result.validation_passed is False
            assert result.reask is not None
            assert result.error is None
        elif fails and has_error:
            assert result.validation_passed is False
            assert result.reask is not None
            assert result.error is not None
        else:
            assert result.validation_passed is True
            assert result.validated_output is not None
            assert not isinstance(result.validated_output, ReAsk)


@pytest.mark.parametrize("llm_output, raises, fails, has_error", test_cases)
def test_choice_validation_pydantic(llm_output, raises, has_error, fails):
    class Fight(BaseModel):
        action: Literal["fight"]
        fight_move: str = Field(
            validators=ValidChoices(
                choices=["punch", "kick", "headbutt"], on_fail=OnFailAction.EXCEPTION
            )
        )

    class Flight(BaseModel):
        action: Literal["flight"]
        flight_direction: str = Field(
            validators=ValidChoices(
                choices=["north", "south", "east", "west"],
                on_fail=OnFailAction.EXCEPTION,
            )
        )
        flight_speed: int = Field(
            validators=ValidChoices(choices=[1, 2, 3, 4], on_fail=OnFailAction.EXCEPTION)
        )

    class Choice(BaseModel):
        choice: Union[Fight, Flight] = Field(..., discriminator="action")

    guard = Guard.for_pydantic(
        output_class=Choice, messages=[{"role": "user", "content": "Dummy prompt."}]
    )

    # If raises is True, then the test should raise an exception.
    # For our existing test cases this will always be a ValidationError
    if raises:
        with pytest.raises(ValidationError):
            guard.parse(llm_output, num_reasks=0)
    else:
        result = guard.parse(llm_output, num_reasks=0)
        if fails and not has_error:
            assert result.validation_passed is False
            assert result.reask is not None
            assert result.error is None
        elif fails and has_error:
            assert result.validation_passed is False
            assert result.reask is not None
            assert result.error is not None
        else:
            assert result.validation_passed is True
            assert result.validated_output is not None
            assert not isinstance(result.validated_output, ReAsk)
