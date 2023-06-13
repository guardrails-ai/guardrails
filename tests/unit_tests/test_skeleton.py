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
    }
    assert verify_schema_against_json(xml_schema, generated_json)
    del generated_json["my_dict2"]
    assert not verify_schema_against_json(xml_schema, generated_json)
