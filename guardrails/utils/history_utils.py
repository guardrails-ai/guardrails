from typing import Dict, Union

from guardrails.classes.history import Call


def merge_valid_output(current_call: Call) -> Union[str, Dict]:
    print("current_call.validated_output: ", current_call.validated_output)
    if isinstance(current_call.validated_output, str):
        return current_call.validated_output

    valid_output = {}
    # Overwrite the oldest valid values with the newest
    print("merging iterations...")
    print("current_call.iterations.length: ", current_call.iterations.length)
    for iteration in current_call.iterations:
        print("iteration.validated_output: ", iteration.validated_output)
        valid_output.update(iteration.validated_output or {})

    return None if not valid_output else valid_output
