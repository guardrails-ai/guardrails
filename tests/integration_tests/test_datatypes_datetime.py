import pytest

from guardrails.guard import Guard


def test_passed_datetime_format():
    rail_spec = """
<rail version="0.1">
<output>
    <string name="name"/>
    <datetime name="dob" datetime-format="%Y-%m-%d %H:%M:%S.%f"/>
</output>
<prompt>
Dummy prompt.
</prompt>
</rail>
"""

    guard = Guard.from_rail_string(rail_spec)
    guard.parse(
        llm_output='{"name": "John Doe", "dob": "2023-10-23 11:06:32.498099"}',
        num_reasks=0,
    )


@pytest.mark.parametrize(
    "datetime_string",
    [
        # ISO 8601 and similar formats
        "2023-10-20T15:30:00+05:00",  # ISO format with timezone
        "2023-10-20T15:30:00.123456",  # ISO format with milliseconds
        "2023-10-20T15:30:59.123",  # ISO format with seconds and milliseconds
        "2023-10-20T15:30:59.999999",  # ISO format with seconds and microseconds
        "2023-03-01T13:45:30",  # ISO format without timezone
        "2023-03-01 13:45:30+02:00",  # Datetime with UTC offset
        "2023-03-01 13:45:30 UTC+2",  # Datetime with UTC offset and UTC string
        "2023-03-01 13:45:30 EST",  # Datetime with timezone abbreviation
        "2023-03-12T01:45:00+05:00",  # ISO 8601 combined date and time with separator
        "20230312T014500+0500",  # Compact ISO 8601 format without colons
        # Formats with various separators
        "2023-10-20 15:30:00.123456",  # Datetime with milliseconds
        "2023-10-20 15:30:59.123",  # Datetime with seconds and milliseconds
        "2023 March 12 01:45:00 +05:00",  # Full month name with time and offset
        "2023-03-12 01:45:00 +05:00",  # Standard datetime with offset
        # Various separators and comma as decimal point
        "2023-10-20T15:30:59,123",  # ISO format with seconds and milliseconds
        "2023-10-20 15:30:59,123",  # Datetime with seconds and milliseconds
        "2023-10-20T15:30:00,999",  # ISO format with milliseconds
        "2023-10-20 15:30:00,123",  # Datetime with milliseconds
        "2023-10-20T15:30:59,999999",  # ISO format with seconds and microseconds
        # Date and time formats with day names
        "Sun, 12 Mar 2023 01:45:00 +0000",  # RFC 822/2822 format with day name
        "2023, Oct 20th 15:30",  # Year, abbreviated month name with time
        "2023 March 12 Sunday 01:45:00 +05:00",  # Wordy format with day name
        "2023 AD March 12 Sunday 01:45:00 +05:00",  # With era AD
        "2023 AD Mar 12 Sun 01:45:00 +05:00",  # Short month and day names with era AD
        "2023 AD 03 12 Sun 01:45:00 +05:00",  # Numeric month with day name and era AD
        # Wordy formats and special cases
        "12th of March, 2023 01:45:00",  # 'of' and 'th' suffix, with time
        "12th December 2022 14:15:29.123456",  # ordinal suffix, full month name
        "12-Dec-2022 14:15:29.999999",  # ordinal suffix, month abbreviation, year, time
        "12 December 2022 14:15:29,999",  # Full date with time and comma separator
        "12/Dec/2022 14:15:29.123",  # Day/MonthAbbreviation/Year
        "12-Dec-2023 13:45:00",  # Short year with time
        "20230312",  # Compact date format without separators
        "2023 AD 03 12th Sun 01:45:00 +05:00",  # Numeric month with 'th', day, era AD
        # Unix/Epoch strings
        "1696343743",  # Unix timestamp/seconds
        "1677649200",  # Epoch timestamp for a specific date
        "1672531199.5",  # Epoch timestamp with fractional seconds
        "1609459200123",  # Epoch timestamp with milliseconds seconds
        "1672531199.123456",  # Epoch timestamp with precision time
        "0",  # Epoch timestamp (start of Unix time)
    ],
)
def test_defaulted_datetime_parser(datetime_string: str):

    rail_spec = """
<rail version="0.1">
<output>
    <string name="name"/>
    <datetime name="dob"/>
</output>
<prompt>
Dummy prompt.
</prompt>
</rail>
"""

    guard = Guard.from_rail_string(rail_spec)
    # This should not raise an exception
    guard.parse(
        llm_output='{"name": "John Doe", "dob": "' + datetime_string + '"}',
        num_reasks=0,
    )


@pytest.mark.parametrize(
    "datetime_string",
    [
        "3rd Thursday in November 2023",  # Informal format
        "2023T03T12T01T45T00+05:00",  # Malformed ISO 8601 with extra 'T' separators
        "2023 CE 03 12th Sun 01:45:00 +05:00",  # CE era
        "2023 CE March 12th Sun 01:45:00 +05:00",  # CE era
        "12 März 2023",  # German month name
    ],
)
def test_defaulted_datetime_parser_unsupported_values(datetime_string: str):
    rail_spec = """
<rail version="0.1">
<output>
    <string name="name"/>
    <datetime name="dob"/>
</output>
<prompt>
Dummy prompt.
</prompt>
</rail>
"""
    guard = Guard.from_rail_string(rail_spec)
    # this should always raise either a ValueError or an OverflowError
    with pytest.raises((ValueError, OverflowError)):
        guard.parse(
            llm_output='{"name": "John Doe", "dob": "' + datetime_string + '"}',
            num_reasks=0,
        )
