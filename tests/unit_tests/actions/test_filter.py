import pytest

from guardrails.actions.filter import Filter, apply_filters


@pytest.mark.parametrize(
    "value,expected",
    [
        (["a", Filter(), "b"], ["a", "b"]),
        (["a", ["b", Filter(), "c"], "d"], ["a", ["b", "c"], "d"]),
        (["a", ["b", "c", "d"], "e"], ["a", ["b", "c", "d"], "e"]),
        (["a", {"b": Filter(), "c": "d"}, "e"], ["a", {"c": "d"}, "e"]),
        ({"a": "b"}, {"a": "b"}),
        ({"a": Filter()}, {}),
        ({"a": "b", "c": {"d": Filter()}}, {"a": "b", "c": {}}),
        ({"a": "b", "c": {"d": "e"}}, {"a": "b", "c": {"d": "e"}}),
        ({"a": "b", "c": ["d", Filter()]}, {"a": "b", "c": ["d"]}),
    ],
)
def test_apply_filters(value, expected):
    assert apply_filters(value) == expected
