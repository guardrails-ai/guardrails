# flake8: noqa: E501
from typing import Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from guardrails import Guard
from guardrails.utils.reask_utils import ReAsk
from guardrails.validators import ValidChoices

test_cases = [
    ('{"choice": {"action": "fight", "fight_move": "kick"}}', False),
    (
        '{"choice": {"action": "flight", "flight_direction": "north", "flight_speed": 1}}',
        False,
    ),
    ('{"choice": {"action": "flight", "fight_move": "punch"}}', True),
    (
        '{"choice": {"action": "fight", "flight_direction": "north", "flight_speed": 1}}',
        True,
    ),
    ('{"choice": {"action": "random_action"}}', True),
    ('{"choice": {"action": "fight", "fight": "random_move"}}', True),
    ('{"choice": {"action": "flight", "random_key": "random_value"}', True),
]


@pytest.mark.parametrize("llm_output, raises", test_cases)
def test_choice_validation(llm_output, raises):
    rail_spec = """
<rail version="0.1">

<output>
    <choice name="choice" on-fail-choice="exception" discriminator="action">
        <case name="fight">
            <string name="fight_move" format="valid-choices: {['punch','kick','headbutt']}" on-fail-valid-choices="exception" />
        </case>
        <case name="flight">
            <string name="flight_direction" format="valid-choices: {['north','south','east','west']}" on-fail-valid-choices="exception" />
            <integer name="flight_speed" format="valid-choices: {[1,2,3,4]}" on-fail-valid-choices="exception" />
        </case>
    </choice>
</output>


<prompt>
Dummy prompt.
</prompt>

</rail>
"""
    guard = Guard.from_rail_string(rail_spec)

    # If raises is True, then the test should raise an exception.
    if raises:
        with pytest.raises(ValueError):
            result = guard.parse(llm_output, num_reasks=0)
            validated_output = result.validated_output
            if validated_output is None or isinstance(validated_output, ReAsk):
                raise ValueError("Expected a result, but got None or ReAsk.")
    else:
        result = guard.parse(llm_output, num_reasks=0)
        validated_output = result.validated_output
        assert not isinstance(validated_output, ReAsk)


@pytest.mark.parametrize("llm_output, raises", test_cases)
def test_choice_validation_pydantic(llm_output, raises):
    class Fight(BaseModel):
        action: Literal["fight"]
        fight_move: str = Field(
            validators=ValidChoices(
                choices=["punch", "kick", "headbutt"], on_fail="exception"
            )
        )

    class Flight(BaseModel):
        action: Literal["flight"]
        flight_direction: str = Field(
            validators=ValidChoices(
                choices=["north", "south", "east", "west"], on_fail="exception"
            )
        )
        flight_speed: int = Field(
            validators=ValidChoices(choices=[1, 2, 3, 4], on_fail="exception")
        )

    class Choice(BaseModel):
        choice: Union[Fight, Flight] = Field(..., discriminator="action")

    guard = Guard.from_pydantic(output_class=Choice, prompt="Dummy prompt.")

    # If raises is True, then the test should raise an exception.
    if raises:
        with pytest.raises(ValueError):
            result = guard.parse(llm_output, num_reasks=0)
            validated_output = result.validated_output
            if validated_output is None or isinstance(validated_output, ReAsk):
                raise ValueError("Expected a result, but got None or ReAsk.")
    else:
        result = guard.parse(llm_output, num_reasks=0)
        validated_output = result.validated_output
        assert not isinstance(validated_output, ReAsk)
