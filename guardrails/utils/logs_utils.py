from copy import deepcopy
from typing import Any, Dict, List, Optional

from guardrails.utils.pydantic_utils import ArbitraryModel
from guardrails.utils.reask_utils import FieldReAsk, ReAsk, prune_obj_for_reasking
from guardrails.validators import ValidationResult


class ValidatorLogs(ArbitraryModel):
    """Logs for a single validator."""

    validator_name: str
    value_before_validation: Any
    validation_result: Optional[ValidationResult] = None
    value_after_validation: Optional[Any] = None
    property_path: str


def update_response_by_path(output: dict, path: List[Any], value: Any) -> None:
    """Update the output by path.

    Args:
        output: The output.
        path: The path to the element to be updated.
        value: The value to be updated.
    """
    for key in path[:-1]:
        output = output[key]
    output[path[-1]] = value


def merge_reask_output(previous_response, reask_response) -> Dict:
    """Merge the reask output into the original output.

    Args:
        prev_logs: validation output object from the previous iteration.
        current_logs: validation output object from the current iteration.

    Returns:
        The merged output.
    """

    if isinstance(previous_response, ReAsk):
        return reask_response

    pruned_reask_json = prune_obj_for_reasking(previous_response)

    # Reask output and reask json have the same structure, except that values
    # of the reask json are ReAsk objects. We want to replace the ReAsk objects
    # with the values from the reask output.
    merged_json = deepcopy(previous_response)

    def update_reasked_elements(pruned_reask_json, reask_response_dict):
        if isinstance(pruned_reask_json, dict):
            for key, value in pruned_reask_json.items():
                if isinstance(value, FieldReAsk):
                    if value.path is None:
                        raise RuntimeError(
                            "FieldReAsk object must have a path attribute."
                        )
                    corrected_value = reask_response_dict.get(key)
                    update_response_by_path(merged_json, value.path, corrected_value)
                else:
                    update_reasked_elements(
                        pruned_reask_json[key], reask_response_dict[key]
                    )
        elif isinstance(pruned_reask_json, list):
            for i, item in enumerate(pruned_reask_json):
                if isinstance(item, FieldReAsk):
                    if item.path is None:
                        raise RuntimeError(
                            "FieldReAsk object must have a path attribute."
                        )
                    corrected_value = reask_response_dict[i]
                    update_response_by_path(merged_json, item.path, corrected_value)
                else:
                    update_reasked_elements(
                        pruned_reask_json[i], reask_response_dict[i]
                    )

    update_reasked_elements(pruned_reask_json, reask_response)

    return merged_json
