import pytest

from guardrails.schema import FormatAttr


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
    _, args = FormatAttr.parse_token(input_string)
    assert args == expected
