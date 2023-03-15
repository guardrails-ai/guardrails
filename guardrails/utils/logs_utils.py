from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

from guardrails.utils.reask_utils import ReAsk, gather_reasks, prune_json_for_reasking


@dataclass
class GuardLogs:
    prompt: str
    output: str
    output_as_dict: dict
    validated_response: dict
    reasks: List[ReAsk]

    @property
    def failed_validations(self) -> List[ReAsk]:
        """Returns the failed validations."""
        return gather_reasks(self.validated_response)


@dataclass
class GuardHistory:
    history: List[GuardLogs]

    def push(self, guard_log: GuardLogs) -> "GuardHistory":
        if len(self.history) > 0:
            last_log = self.history[-1]
            guard_log.validated_response = merge_reask_output(last_log, guard_log)

        return GuardHistory(self.history + [guard_log])

    @property
    def validated_response(self) -> dict:
        """Returns the latest validated output."""
        return self.history[-1].validated_response

    @property
    def output(self) -> str:
        """Returns the latest output."""
        return self.history[-1].output

    @property
    def output_as_dict(self) -> dict:
        """Returns the latest output as a dict."""
        return self.history[-1].output_as_dict

    @property
    def failed_validations(self) -> List[ReAsk]:
        """Returns all failed validations."""
        return [log.failed_validations for log in self.history]


@dataclass
class GuardState:
    all_histories: List[GuardHistory]

    def push(self, guard_history: GuardHistory) -> "GuardState":
        return GuardState(self.all_histories + [guard_history])

    @property
    def most_recent_call(self) -> GuardHistory:
        """Returns the most recent call."""
        if not len(self.all_histories):
            return None
        return self.all_histories[-1]


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


def merge_reask_output(prev_logs: GuardLogs, current_logs: GuardLogs) -> Dict:
    """Merge the reask output into the original output.

    Args:
        prev_logs: GuardLogs object from the previous iteration.
        current_logs: GuardLogs object from the current iteration.

    Returns:
        The merged output.
    """
    previous_response = prev_logs.validated_response
    pruned_reask_json = prune_json_for_reasking(previous_response)
    reask_response = current_logs.validated_response

    # Reask output and reask json have the same structure, except that values
    # of the reask json are ReAsk objects. We want to replace the ReAsk objects
    # with the values from the reask output.
    merged_json = deepcopy(previous_response)

    def update_reasked_elements(pruned_reask_json, reask_response_dict):
        if isinstance(pruned_reask_json, dict):
            for key, value in pruned_reask_json.items():
                if isinstance(value, ReAsk):
                    corrected_value = reask_response_dict[key]
                    update_response_by_path(merged_json, value.path, corrected_value)
                else:
                    update_reasked_elements(
                        pruned_reask_json[key], reask_response_dict[key]
                    )
        elif isinstance(pruned_reask_json, list):
            for i, item in enumerate(pruned_reask_json):
                if isinstance(item, ReAsk):
                    corrected_value = reask_response_dict[i]
                    update_response_by_path(merged_json, item.path, corrected_value)
                else:
                    update_reasked_elements(
                        pruned_reask_json[i], reask_response_dict[i]
                    )

    update_reasked_elements(pruned_reask_json, reask_response)

    return merged_json
