import pytest

from guardrails.utils.casting_utils import to_float, to_int, to_string

# NOTE: These tests are _currently_ (August 31, 2023) descriptive not prescriptive.
#  We can change the functionality of these methods to make them more flexible
# if desired, but we should _NOT_ make them more strict than what
# basic casting allows.


@pytest.mark.parametrize(
    "input,expected_output",
    [
        (1, 1),
        (1.1, 1),
        ("1", 1),
        ("1.1", None),
        ("abc", None),
        ("None", None),
        (None, None),
    ],
)
def test_to_int(input, expected_output):
    actual_output = to_int(input)

    assert actual_output == expected_output


@pytest.mark.parametrize(
    "input,expected_output",
    [
        (1, 1.0),
        (1.1, 1.1),
        ("1", 1.0),
        ("1.1", 1.1),
        ("abc", None),
        ("None", None),
        (None, None),
    ],
)
def test_to_float(input, expected_output):
    actual_output = to_float(input)

    assert actual_output == expected_output


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("abc", "abc"),
        (1, "1"),
        (1.1, "1.1"),
        (True, "True"),
        (["a", "b", "c"], "['a', 'b', 'c']"),
        ({"a": 1}, "{'a': 1}"),
        (None, "None"),
    ],
)
def test_to_string(input, expected_output):
    actual_output = to_string(input)

    assert actual_output == expected_output
