from guardrails.utils import reask_utils


def test_sub_reasks_with_fixed_values():
    """Test that sub reasks with fixed values are replaced."""

    # Test Case 1
    input_dict = {"a": 1, "b": reask_utils.ReAsk(-1, "Error Msg", 1)}
    expected_dict = {"a": 1, "b": 1}
    assert reask_utils.sub_reasks_with_fixed_values(input_dict) == expected_dict

    # Test Case 2
    input_dict = {"a": 1, "b": {"c": 2, "d": reask_utils.ReAsk(-1, "Error Msg", 2)}}
    expected_dict = {"a": 1, "b": {"c": 2, "d": 2}}
    assert reask_utils.sub_reasks_with_fixed_values(input_dict) == expected_dict

    # Test Case 3
    input_dict = {"a": [1, 2, reask_utils.ReAsk(-1, "Error Msg", 3)], "b": 4}
    expected_dict = {"a": [1, 2, 3], "b": 4}
    assert reask_utils.sub_reasks_with_fixed_values(input_dict) == expected_dict

    # Create a test case with a dict in a list
    input_dict = {"a": [1, 2, {"c": reask_utils.ReAsk(-1, "Error Msg", 3)}]}
    expected_dict = {"a": [1, 2, {"c": 3}]}
    assert reask_utils.sub_reasks_with_fixed_values(input_dict) == expected_dict
