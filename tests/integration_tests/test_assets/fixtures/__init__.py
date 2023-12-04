import pytest


@pytest.fixture(name="rail_spec")
def fixture_rail_spec():
    return """
<rail version="0.1">

<output>
    <string name="dummy_string" description="Any dummy string" />
    <integer name="dummy_integer" description="Any dummy integer" />
    <float name="dummy_float" description="Any dummy float" />
    <bool name="dummy_boolean" description="Any dummy boolean" />
    <email name="dummy_email" description="Any dummy email" />
    <url name="dummy_url" description="Any dummy url" />
    <date name="dummy_date" description="Any dummy date" />
    <time name="dummy_time" description="Any dummy time" />
    <list name="dummy_list" description="Any dummy list" />
    <object name="dummy_object" description="Any dummy object" />
</output>


<prompt>

Generate a JSON of dummy data, where the data types are specified by the user.

${gr.complete_json_suffix}

</prompt>

</rail>
"""


@pytest.fixture(name="llm_output")
def fixture_llm_output():
    return """
{
    "dummy_string": "Some string",
    "dummy_integer": 42,
    "dummy_float": 3.14,
    "dummy_boolean": true,
    "dummy_email": "example@example.com",
    "dummy_url": "https://www.example.com",
    "dummy_date": "2020-01-01",
    "dummy_time": "12:00:00",
    "dummy_list": ["item1", "item2", "item3"],
    "dummy_object": {
        "key1": "value1",
        "key2": "value2"
    }
}
"""


@pytest.fixture(name="validated_output")
def fixture_validated_output():
    return {
        "dummy_string": "Some string",
        "dummy_integer": 42,
        "dummy_float": 3.14,
        "dummy_boolean": True,
        "dummy_email": "example@example.com",
        "dummy_url": "https://www.example.com",
        "dummy_date": "2020-01-01",
        "dummy_time": "12:00:00",
        "dummy_list": ["item1", "item2", "item3"],
        "dummy_object": {"key1": "value1", "key2": "value2"},
    }
