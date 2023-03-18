import pytest

from guardrails.datatypes import get_args


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("a", ["a"]),
        ("a b", ["a", "b"]),
        ("{list(range(5))} a b", [[0, 1, 2, 3, 4], "a", "b"]),
        ("{[1, 2, 3]} a b", [[1, 2, 3], "a", "b"]),
        ("{{'a': 1, 'b': 2}} c d", [{"a": 1, "b": 2}, "c", "d"]),
    ]
)
def test_get_args(input_string, expected):
    assert get_args(input_string) == expected
