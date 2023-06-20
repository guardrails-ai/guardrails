import lxml.etree as ET

from guardrails.utils.json_utils import verify_schema_against_json


def test_skeleton():
    xml = """
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
<list name="my_list3">
    <choice name="action" on-fail-choice="exception">
    <case name="fight">
        <list name="fight">
            <string
                format="valid-choices: {['punch','kick','headbutt']}"
                on-fail-valid-choices="exception"
            />
        </list>
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
</list>
<object name="mychoices">
    <string name="some random thing"/>
    <choice name="action" on-fail-choice="exception">
        <case name="fight">
            <string
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
</object>

</root>
"""
    xml_schema = ET.fromstring(xml)
    generated_json = {
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
        "action": "fight",
        "fight": "punch",
        "my_list3": [
            {
                "action": "fight",
                "fight": [
                    "punch",
                    "kick"
                ],
            },
            {
                "action": "flight",
                "flight": {
                    "flight_direction": "north",
                    "flight_speed": 1
                }
            }
        ],
        "mychoices": {
            "some random thing": "string",
            "action": "fight",
            "fight": "punch",
        },
    }
    assert verify_schema_against_json(xml_schema, generated_json)
    del generated_json["my_dict2"]
    assert not verify_schema_against_json(xml_schema, generated_json)
