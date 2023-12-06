import lxml.etree as ET
import pytest

from guardrails.datatypes import Object
from guardrails.utils.json_utils import verify_schema_against_json


@pytest.mark.parametrize(
    "xml, generated_json, result, coerce_types",
    [
        (
            """
<root>
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
</root>
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
<root>
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
</root>
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
<root>
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
</root>
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
<root>
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
</root>
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
<root>
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
</root>
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
<root>
<string
    name="my_string"
    required="false"
/>
</root>
            """,
            {
                "my_string": None,
            },
            True,
            False,
        ),
        (
            """
<root>
<string
    name="my_string"
    required="false"
/>
</root>
            """,
            {
                # "my_string": None,
            },
            True,
            False,
        ),
        (
            """
<root>
<list
    name="my_list"
>
</list>
</root>
            """,
            {
                "my_list": ["e"],
            },
            True,
            False,
        ),
        (
            """
<root>
<string
    name="my_string"
>
</string>
</root>
            """,
            {
                "my_string": "e",
            },
            True,
            True,
        ),
        (
            """
<root>
<string
    name="my_string"
>
</string>
</root>
            """,
            {
                "my_string": ["a"],
            },
            False,
            True,
        ),
        (
            """
<root>
<string
    name="my_string"
>
</string>
</root>
            """,
            {
                "my_string": {"a": "a"},
            },
            False,
            True,
        ),
    ],
)
def test_skeleton(xml, generated_json, result, coerce_types):
    xml_schema = ET.fromstring(xml)
    datatype = Object.from_xml(xml_schema)
    assert verify_schema_against_json(datatype, generated_json) is result
