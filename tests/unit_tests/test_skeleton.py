import pytest

from guardrails.schema.rail_schema import rail_string_to_schema
from guardrails.schema.validator import SchemaValidationError, validate_payload
from guardrails.utils.parsing_utils import coerce_types


# TODO: Make this an integration test instead.
@pytest.mark.parametrize(
    "rail, payload, should_pass, should_coerce_types",
    [
        (
            """
<rail version="0.1"><output>
<list name="my_list">
    <object>
        <string name="my_string" />
    </object>
</list>
<integer name="my_integer" />
<string name="my_string" />
<object name="my_dict">
    <string name="my_string" />
</object>
<object name="my_dict2">
    <list name="my_list">
        <float />
    </list>
</object>
<list name="my_list2">
    <string />
</list>
</output></rail>
            """,
            {
                "my_list": [{"my_string": "string"}],
                "my_integer": 1,
                "my_string": "string",
                "my_dict": {"my_string": "string"},
                "my_dict2": {
                    "my_list": [
                        1.0,
                        2.0,
                    ]
                },
                "my_list2": [],
            },
            True,
            False,
        ),
        (
            """
<rail version="0.1"><output>
<list name="my_list">
    <object>
        <string name="my_string" />
    </object>
</list>
<integer name="my_integer" />
<string name="my_string" />
<object name="my_dict">
    <string name="my_string" />
</object>
<object name="my_dict2">
    <list name="my_list">
        <float />
    </list>
</object>
<list name="my_list2">
    <string />
</list>
</output></rail>
            """,
            {
                "my_list": [{"my_string": "string"}],
                "my_integer": 1,
                "my_string": "string",
                "my_dict": {"my_string": "string"},
                "my_list2": [],
            },
            False,
            False,
        ),
        (
            """
<rail version="0.1"><output>
<choice name="action" discriminator="action_type" on-fail-choice="exception">
    <case name="fight">
        <string
            name="fight_move"
            validators="valid-choices: {['punch','kick','headbutt']}"
            on-fail-valid-choices="exception"
        />
    </case>
    <case name="flight">
        <string
            name="flight_direction"
            validators="valid-choices: {['north','south','east','west']}"
            on-fail-valid-choices="exception"
        />
        <integer
            name="flight_speed"
            validators="valid-choices: {[1,2,3,4]}"
            on-fail-valid-choices="exception"
        />
    </case>
</choice>
</output></rail>
            """,
            {
                "action": {
                    "action_type": "fight",
                    "fight_move": "punch",
                }
            },
            True,
            False,
        ),
        (
            """
<rail version="0.1"><output>
<list name="my_list3">
    <choice discriminator="action_type" on-fail-choice="exception">
        <case name="fight">
            <list name="fight">
                <string
                    validators="valid-choices: {['punch','kick','headbutt']}"
                    on-fail-valid-choices="exception"
                />
            </list>
        </case>
        <case name="flight">
            <string
                name="flight_direction"
                validators="valid-choices: {['north','south','east','west']}"
                on-fail-valid-choices="exception"
            />
            <integer
                name="flight_speed"
                validators="valid-choices: {[1,2,3,4]}"
                on-fail-valid-choices="exception"
            />
        </case>
    </choice>
</list>
</output></rail>
""",
            {
                "my_list3": [
                    {
                        "action_type": "fight",
                        "fight": ["punch", "kick"],
                    },
                    {
                        "action_type": "flight",
                        "flight_direction": "north",
                        "flight_speed": 1,
                    },
                ],
            },
            True,
            False,
        ),
        (
            """
<rail version="0.1"><output>
<object name="mychoices">
    <string name="some random thing"/>
    <choice name="action" discriminator="action_type" on-fail-choice="exception">
        <case name="fight">
            <string
                name="fight_move"
                validators="valid-choices: {['punch','kick','headbutt']}"
                on-fail-valid-choices="exception"
            />
        </case>
        <case name="flight">
            <object name="flight">
                <string
                    name="flight_direction"
                    validators="valid-choices: {['north','south','east','west']}"
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
</object>
</output></rail>
""",
            {
                "mychoices": {
                    "some random thing": "string",
                    "action": {
                        "action_type": "fight",
                        "fight_move": "punch",
                    },
                },
            },
            True,
            False,
        ),
        (
            """
<rail version="0.1"><output>
<string
    name="my_string"
    required="false"
/>
</output></rail>
            """,
            {},
            True,
            False,
        ),
        (
            """
<rail version="0.1"><output>
<string
    name="my_string"
    required="false"
/>
</output></rail>
            """,
            {},
            True,
            False,
        ),
        (
            """
<rail version="0.1"><output>
<string
    name="my_string"
>
</string>
</output></rail>
            """,
            {
                "my_string": "e",
            },
            True,
            True,
        ),
        (
            """
<rail version="0.1"><output>
<string
    name="my_string"
>
</string>
</output></rail>
            """,
            {
                "my_string": ["a"],
            },
            False,
            True,
        ),
        (
            """
<rail version="0.1"><output>
<string
    name="my_string"
>
</string>
</output></rail>
            """,
            {
                "my_string": {"a": "a"},
            },
            False,
            True,
        ),
    ],
)
def test_skeleton(rail, payload, should_pass, should_coerce_types):
    payload = payload
    processed_schema = rail_string_to_schema(rail)
    json_schema = processed_schema.json_schema
    if should_coerce_types:
        payload = coerce_types(payload, json_schema)
    if not should_pass:
        with pytest.raises(SchemaValidationError):
            validate_payload(payload, processed_schema.json_schema)
    else:
        validate_payload(payload, json_schema)
