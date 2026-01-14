import pytest
from dateutil.parser import ParserError

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

    guard = Guard.for_rail_string(rail_spec)
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

    guard = Guard.for_rail_string(rail_spec)
    # This should not raise an exception
    guard.parse(llm_output='{"name": "John Doe", "dob": "' + date_string + '"}', num_reasks=0)


@pytest.mark.skip("Must add custom format validators to guardrails/schema/validator.py!")
@pytest.mark.parametrize(
    "date_string,error_type",
    [
        ("1696343743", ParserError),  # Unix timestamp/seconds
        ("1697579939213", OverflowError),  # Unix timestamp/milliseconds
    ],
)
def test_defaulted_date_parser_unsupported_values(date_string: str, error_type: Exception):
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

    guard = Guard.for_rail_string(rail_spec)
    with pytest.raises(Exception) as excinfo:
        guard.parse(
            llm_output='{"name": "John Doe", "dob": "' + date_string + '"}',
            num_reasks=0,
        )
    assert isinstance(excinfo.value, error_type) is True
