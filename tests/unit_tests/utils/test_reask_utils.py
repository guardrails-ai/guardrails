import pytest

from guardrails.utils.reask_utils import ReAsk, gather_reasks, sub_reasks_with_fixed_values


@pytest.mark.parametrize(
    "input_dict, expected_dict",
    [
        ({"a": 1, "b": ReAsk(-1, "Error Msg", 1)}, {"a": 1, "b": 1}),
        (
            {"a": 1, "b": {"c": 2, "d": ReAsk(-1, "Error Msg", 2)}},
            {"a": 1, "b": {"c": 2, "d": 2}},
        ),
        (
            {"a": [1, 2, ReAsk(-1, "Error Msg", 3)], "b": 4},
            {"a": [1, 2, 3], "b": 4},
        ),
        (
            {"a": [1, 2, {"c": ReAsk(-1, "Error Msg", 3)}]},
            {"a": [1, 2, {"c": 3}]},
        ),
        (
            {"a": [1, 2, [3, 4, ReAsk(-1, "Error Msg", 5)]]},
            {"a": [1, 2, [3, 4, 5]]},
        ),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_sub_reasks_with_fixed_values(input_dict, expected_dict):
    """Test that sub reasks with fixed values are replaced."""
    assert sub_reasks_with_fixed_values(input_dict) == expected_dict


def test_gather_reasks():
    """Test that reasks are gathered."""
    input_dict = {
        "a": 1,
        "b": ReAsk("b0", "Error Msg", "b1", None),
        "c": {"d": ReAsk("c0", "Error Msg", "c1", "None")},
        "e": [1, 2, ReAsk("e0", "Error Msg", "e1", "None")],
        "f": [1, 2, {"g": ReAsk("f0", "Error Msg", "f1", "None")}],
        "h": [1, 2, [3, 4, ReAsk("h0", "Error Msg", "h1", "None")]],
    }
    expected_reasks = [
        ReAsk("b0", "Error Msg", "b1", ["b"]),
        ReAsk("c0", "Error Msg", "c1", ["c", "d"]),
        ReAsk("e0", "Error Msg", "e1", ["e", 2]),
        ReAsk("f0", "Error Msg", "f1", ["f", 2, "g"]),
        ReAsk("h0", "Error Msg", "h1", ["h", 2, 2]),
    ]
    assert gather_reasks(input_dict) == expected_reasks
