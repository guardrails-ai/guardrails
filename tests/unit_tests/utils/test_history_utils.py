from guardrails.classes.generic.stack import Stack
from guardrails.classes.history import Call, Iteration, Outputs
from guardrails.utils.history_utils import merge_valid_output


def test_merge_valid_output():
    first_iteration = Iteration(outputs=Outputs(validated_output={"a": 1, "b": "abc"}))
    second_iteration = Iteration(outputs=Outputs(validated_output={"b": "def", "c": 3}))
    current_call = Call(iterations=Stack(first_iteration, second_iteration))

    merged_output = merge_valid_output(current_call)

    assert merged_output == {"a": 1, "b": "def", "c": 3}
