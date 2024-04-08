# Write tests for check_refrain and filter_in_schema in guardrails/validator_base.py
import pytest

from guardrails.validator_base import Filter, Refrain, check_refrain, filter_in_schema


@pytest.mark.parametrize(
    "schema,expected",
    [
        (["a", Refrain(), "b"], True),
        (["a", "b"], False),
        (["a", ["b", Refrain(), "c"], "d"], True),
        (["a", ["b", "c", "d"], "e"], False),
        (["a", {"b": Refrain(), "c": "d"}, "e"], True),
        (["a", {"b": "c", "d": "e"}, "f"], False),
        ({"a": "b"}, False),
        ({"a": Refrain()}, True),
        ({"a": "b", "c": {"d": Refrain()}}, True),
        ({"a": "b", "c": {"d": "e"}}, False),
        ({"a": "b", "c": ["d", Refrain()]}, True),
        ({"a": "b", "c": ["d", "e"]}, False),
    ],
)
def test_check_refrain(schema, expected):
    assert check_refrain(schema) == expected


@pytest.mark.parametrize(
    "schema,expected",
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
def test_filter_in_schema(schema, expected):
    assert filter_in_schema(schema) == expected
