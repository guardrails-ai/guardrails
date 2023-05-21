# flake8: noqa: E501
from typing import Optional
import pytest

from guardrails import Guard

from guardrails.datatypes import Field, GuardModel
from guardrails.validators import ValidChoices


@pytest.mark.parametrize(
    "llm_output, raises",
    [
        ('{"action": "fight", "fight": "kick"}', False),
        (
            '{"action": "flight", "flight": {"flight_direction": "north", "flight_speed": 1}}',
            False,
        ),
        ('{"action": "flight", "fight": "punch"}', True),
        (
            '{"action": "fight", "flight": {"flight_direction": "north", "flight_speed": 1}}',
            True,
        ),
        ('{"action": "random_action"}', True),
        ('{"action": "fight", "fight": "random_move"}', True),
        ('{"action": "flight", "flight": {"random_key": "random_value"}}', True),
        (
            '{"action": "flight", "flight": {"flight_direction": "north", "flight_speed": 1}, "fight": "punch"}',
            True,
        ),
    ],
)
def test_choice_validation(llm_output, raises):
    rail_spec = """
<rail version="0.1">

<output>
    <choice name="action" on-fail-choice="exception">
        <case name="fight">
            <string name="fight_move" format="valid-choices: {['punch','kick','headbutt']}" on-fail-valid-choices="exception" />
        </case>
        <case name="flight">
            <object name="flight">
                <string name="flight_direction" format="valid-choices: {['north','south','east','west']}" on-fail-valid-choices="exception" />
                <integer name="flight_speed" format="valid-choices: {[1,2,3,4]}" on-fail-valid-choices="exception" />
            </object>
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
        with pytest.raises(Exception):
            guard.parse(llm_output)
    else:
        guard.parse(llm_output)


@pytest.mark.parametrize(
    "llm_output, raises",
    [
        ('{"action": "fight", "fight": "kick"}', False),
        (
            '{"action": "flight", "flight": {"flight_direction": "north", "flight_speed": 1}}',
            False,
        ),
        ('{"action": "flight", "fight": "punch"}', True),
        (
            '{"action": "fight", "flight": {"flight_direction": "north", "flight_speed": 1}}',
            True,
        ),
        ('{"action": "random_action"}', True),
        ('{"action": "fight", "fight": "random_move"}', True),
        ('{"action": "flight", "flight": {"random_key": "random_value"}}', True),
        (
            '{"action": "flight", "flight": {"flight_direction": "north", "flight_speed": 1}, "fight": "punch"}',
            True,
        ),
    ],
)
def test_choice_validation_pydantic(llm_output, raises):
    class FlightDetails(GuardModel):
        flight_direction: str = Field(
            gd_validators=ValidChoices(
                choices=["north", "south", "east", "west"], on_fail="exception"
            )
        )
        flight_speed: int = Field(
            gd_validators=ValidChoices(choices=[1, 2, 3, 4], on_fail="exception")
        )

    class Action(GuardModel):
        action: str = Field(gd_validators=ValidChoices(choices=["fight", "flight"]))
        fight: Optional[str] = Field(
            gd_validators=ValidChoices(choices=["punch", "kick"], on_fail="exception"),
            when="action==fight",
        )
        flight: Optional[FlightDetails] = Field(when="action==flight")
        # flight: Optional[FlightDetails] = Field(when="action")
        # flight: Optional[FlightDetails] = Field(given="action")

    guard = Guard.from_pydantic(output_class=Action, prompt="Dummy prompt.")

    # If raises is True, then the test should raise an exception.
    if raises:
        with pytest.raises(Exception):
            guard.parse(llm_output)
    else:
        guard.parse(llm_output)
