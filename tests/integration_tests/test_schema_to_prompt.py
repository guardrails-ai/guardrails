# flake8: noqa: E501
import pytest

from guardrails import Guard


def test_choice_schema():
    rail_spec = """
<rail version="0.1">

<output>
    <choice name="action" on-fail-choice="exception">
        <case name="fight">
            <string
                name="fight_move"
                format="valid-choices: {['punch','kick','headbutt']}"
                on-fail-valid-choices="exception"
            />
        </case>
        <case name="flight">
            <object name="flight">
                <string
                    name="flight_direction"
                    format="valid-choices: {['north','south','east','west']}"
                    on-fail-valid-choices="exception"
                />
                <integer
                    name="flight_speed"
                    format="valid-choices: {[1,2,3,4]}"
                    on-fail-valid-choices="exception"
                />
            </object>
        </case>
    </choice>
</output>

<prompt>
Dummy prompt
</prompt>

</rail>
"""

    guard = Guard.from_rail_string(rail_spec)
    schema_2_prompt = guard.output_schema.transpile()
    expected_schema_2_prompt = '<output>\n    <string name="action" choices="fight,flight"/>\n    <string name="fight" format="valid-choices: choices=[\'punch\', \'kick\', \'headbutt\']" description="fight_move" if="action==fight"/>\n    <object name="flight" description="flight" if="action==flight">\n        <string name="flight_direction" format="valid-choices: choices=[\'north\', \'south\', \'east\', \'west\']"/>\n        <integer name="flight_speed" format="valid-choices: choices=[1, 2, 3, 4]"/>\n    </object>\n</output>\n'
    assert schema_2_prompt == expected_schema_2_prompt
