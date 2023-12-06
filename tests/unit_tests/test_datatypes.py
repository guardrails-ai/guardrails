import pytest
from lxml.builder import E

import guardrails.datatypes as datatypes
from guardrails.validatorsattr import ValidatorsAttr


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("dummy: a", ["a"]),
        ("dummy: a b", ["a", "b"]),
        (
            "dummy: {list(range(5))} a b",
            [[0, 1, 2, 3, 4], "a", "b"],
        ),
        ("dummy: {[1, 2, 3]} a b", [[1, 2, 3], "a", "b"]),
        (
            "dummy: {{'a': 1, 'b': 2}} c d",
            [{"a": 1, "b": 2}, "c", "d"],
        ),
        (
            "dummy: {1 + 2} {{'a': 1, 'b': 2}} c d",
            [3, {"a": 1, "b": 2}, "c", "d"],
        ),
    ],
)
def test_get_args(input_string, expected):
    _, args = ValidatorsAttr.parse_token(input_string)
    assert args == expected


@pytest.mark.parametrize(
    "date_format,date",
    [
        ("%Y-%m-%d", "2023-04-01"),
        ("%a, %d %b %Y", "Mon, 01 Jan 2019"),
    ],
)
def test_date(date_format, date):
    from datetime import datetime

    date_element = E.date(**{"date-format": date_format})
    date_datatype = datatypes.Date.from_xml(date_element)
    assert date_datatype.from_str(date) == datetime.strptime(date, date_format).date()


@pytest.mark.parametrize(
    "time_format,time",
    [
        ("%H:%M:%S", "12:00:00"),
    ],
)
def test_time(time_format, time):
    from datetime import datetime

    time_element = E.time(**{"time-format": time_format})
    time_datatype = datatypes.Time.from_xml(time_element)
    assert time_datatype.from_str(time) == datetime.strptime(time, time_format).time()
