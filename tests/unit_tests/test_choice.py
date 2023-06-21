import json

import pytest

from guardrails import Guard


@pytest.mark.parametrize(
    "xml, generated_json, result",
    [
        (
            """
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
""",
            '{"action": "fight", "fight": "kick"}',
            True,
        ),
        (
            """
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
""",
            '{"action": "flight", "flight": {"flight_direction": "north", "flight_speed": 1}}',
            True,
        ),
        (
            """
<rail version="0.1">

<output>
    <choice name="action" on-fail-choice="noop">
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
""",
            '{"action": null}',
            True,
        ),
    ]
)
def test_choice_validation(xml, generated_json, result):
    guard = Guard.from_rail_string(xml)

    if result:
        assert guard.parse(generated_json) == json.loads(generated_json)
    else:
        with pytest.raises(Exception):
            guard.parse(generated_json)
