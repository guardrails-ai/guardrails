import pytest

from guardrails.actions.filter import Filter, apply_filters


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (Filter(), None),
        ([None], [None]),
        ({"a": None}, {"a": None}),
        ([None, Filter(), 0, False, ""], [None, 0, False, ""]),
        (
            {"a": None, "b": Filter(), "c": [None, Filter()]},
            {"a": None, "c": [None]},
        ),
        (
            [Filter(), {"a": None, "b": Filter()}, [None, Filter()]],
            [{"a": None}, [None]],
        ),
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
