import pytest

from guardrails.actions.refrain import Refrain, apply_refrain, check_for_refrain
from guardrails.classes.output_type import OutputTypes


@pytest.mark.parametrize(
    "value,expected",
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
def test_check_for_refrain(value, expected):
    assert check_for_refrain(value) == expected


@pytest.mark.parametrize(
    "value,output_type,expected",
    [
        (["a", Refrain(), "b"], OutputTypes.LIST, []),
        (["a", "b"], OutputTypes.LIST, ["a", "b"]),
        (["a", ["b", Refrain(), "c"], "d"], OutputTypes.LIST, []),
        (["a", ["b", "c", "d"], "e"], OutputTypes.LIST, ["a", ["b", "c", "d"], "e"]),
        (["a", {"b": Refrain(), "c": "d"}, "e"], OutputTypes.LIST, []),
        (
            ["a", {"b": "c", "d": "e"}, "f"],
            OutputTypes.LIST,
            ["a", {"b": "c", "d": "e"}, "f"],
        ),
        ({"a": "b"}, OutputTypes.DICT, {"a": "b"}),
        ({"a": Refrain()}, OutputTypes.DICT, {}),
        ({"a": "b", "c": {"d": Refrain()}}, OutputTypes.DICT, {}),
        ({"a": "b", "c": {"d": "e"}}, OutputTypes.DICT, {"a": "b", "c": {"d": "e"}}),
        ({"a": "b", "c": ["d", Refrain()]}, OutputTypes.DICT, {}),
        ({"a": "b", "c": ["d", "e"]}, OutputTypes.DICT, {"a": "b", "c": ["d", "e"]}),
    ],
)
def test_apply_refrain(value, output_type, expected):
    assert apply_refrain(value, output_type) == expected
