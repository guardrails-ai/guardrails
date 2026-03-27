import pytest

from guardrails.utils.safe_get import safe_get


@pytest.mark.parametrize(
    "container,key,default,expected_value",
    [
        ([1, 2, 3], 1, None, 2),
        ([1, 2, 3], 4, None, None),
        ([1, 2, 3], 4, 42, 42),
        ([1, 2, None], 2, 42, 42),
        ({"a": 1, "b": 2, "c": 3}, "b", None, 2),
        ({"a": 1, "b": 2, "c": 3}, "d", None, None),
        ({"a": 1, "b": 2, "c": 3}, "d", 42, 42),
        ("123", 1, None, "2"),
        ("123", 4, None, None),
        ("123", 4, 42, 42),
        # Falsy values must be returned, not replaced by default
        ([0, "a", "b"], 0, None, 0),
        ([False, "a"], 0, None, False),
        (["", "a"], 0, None, ""),
        # Out-of-bounds and non-subscriptable containers use default
        ([], 0, "fallback", "fallback"),
        (None, 0, "fallback", "fallback"),
    ],
)
def test_safe_get(container, key, default, expected_value):
    actual = safe_get(container, key, default)

    assert actual == expected_value
