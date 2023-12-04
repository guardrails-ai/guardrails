import pytest

from guardrails.guard import Guard


def test_passed_date_format():
    rail_spec = """
<rail version="0.1">

<output>
    <string name="name"/>
    <date name="dob" date-format="%Y-%m-%d"/>
</output>


<prompt>
Dummy prompt.
</prompt>

</rail>
"""

    guard = Guard.from_rail_string(rail_spec)
    guard.parse(llm_output='{"name": "John Doe", "dob": "2021-01-01"}', num_reasks=0)


@pytest.mark.parametrize(
    "date_string",
    [
        ("2021-01-01"),  # standard date
        ("2021-01-01T11:10:00+01:00"),  # Cohere-style
        ("2023-10-03T14:18:38.476Z"),  # ISO
    ],
)
def test_defaulted_date_parser(date_string: str):
    rail_spec = """
<rail version="0.1">

<output>
    <string name="name"/>
    <date name="dob"/>
</output>


<prompt>
Dummy prompt.
</prompt>

</rail>
"""

    guard = Guard.from_rail_string(rail_spec)
    # This should not raise an exception
    guard.parse(
        llm_output='{"name": "John Doe", "dob": "' + date_string + '"}', num_reasks=0
    )


@pytest.mark.parametrize(
    "date_string",
    [
        ("1696343743"),  # Unix timestamp/seconds
        ("1697579939213"),  # Unix timestamp/milliseconds
    ],
)
def test_defaulted_date_parser_unsupported_values(date_string: str):
    rail_spec = """
<rail version="0.1">

<output>
    <string name="name"/>
    <date name="dob"/>
</output>


<prompt>
Dummy prompt.
</prompt>

</rail>
"""
    guard = Guard.from_rail_string(rail_spec)
    response = guard.parse(
        llm_output='{"name": "John Doe", "dob": "' + date_string + '"}',
        num_reasks=0,
    )
    assert response.error is not None
    # this should always raise either a ValueError or an OverflowError
