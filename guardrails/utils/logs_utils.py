from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr
from rich.console import Group
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.table import Table
from rich.tree import Tree

from guardrails.prompt import Instructions, Prompt
from guardrails.utils.reask_utils import (
    ReAsk,
    SkeletonReAsk,
    gather_reasks,
    prune_obj_for_reasking,
)
from guardrails.validators import ValidationResult


class ArbitraryModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class ValidatorLogs(ArbitraryModel):
    """Logs for a single validator."""

    validator_name: str
    value_before_validation: Any
    validation_result: Optional[ValidationResult] = None
    value_after_validation: Optional[Any] = None


class FieldValidationLogs(ArbitraryModel):
    """Logs for a single field."""

    validator_logs: List[ValidatorLogs] = Field(default_factory=list)
    children: Dict[Union[int, str], "FieldValidationLogs"] = Field(default_factory=dict)


class LLMResponse(ArbitraryModel):
    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    output: Optional[str] = None


class GuardLogs(ArbitraryModel):
    prompt: Optional[Prompt] = None
    instructions: Optional[Instructions] = None
    llm_response: Optional[LLMResponse] = None
    msg_history: Optional[List[Dict[str, Prompt]]] = None
    parsed_output: Optional[dict] = None
    validated_output: Optional[dict] = None
    reasks: Optional[List[ReAsk]] = None

    field_validation_logs: Optional[FieldValidationLogs] = None

    _previous_logs: Optional["GuardLogs"] = PrivateAttr(None)

    def set_validated_output(self, validated_output, is_full_schema_reask: bool):
        if (
            self._previous_logs is not None
            and self._previous_logs.validated_output is not None
            and not is_full_schema_reask
            and not isinstance(validated_output, SkeletonReAsk)
        ):
            validated_output = merge_reask_output(
                self._previous_logs.validated_output, validated_output
            )
        self.validated_output = validated_output

    @property
    def failed_validations(self) -> List[ReAsk]:
        """Returns the failed validations."""
        return gather_reasks(self.validated_output)

    @property
    def output(self) -> Optional[str]:
        if self.llm_response is None:
            return None
        return self.llm_response.output

    @property
    def rich_group(self) -> Group:
        def create_msg_history_table(
            msg_history: Optional[List[Dict[str, Prompt]]]
        ) -> Table:
            if msg_history is None:
                return "No message history."
            table = Table(show_lines=True)
            table.add_column("Role", justify="right", no_wrap=True)
            table.add_column("Content")

            for msg in msg_history:
                table.add_row(msg["role"], msg["content"].source)

            return table

        table = create_msg_history_table(self.msg_history)

        if self.instructions is not None:
            return Group(
                Panel(
                    self.prompt.source if self.prompt else "No prompt",
                    title="Prompt",
                    style="on #F0F8FF",
                ),
                Panel(
                    self.instructions.source, title="Instructions", style="on #FFF0F2"
                ),
                Panel(table, title="Message History", style="on #E7DFEB"),
                Panel(self.output, title="Raw LLM Output", style="on #F5F5DC"),
                Panel(
                    pretty_repr(self.validated_output),
                    title="Validated Output",
                    style="on #F0FFF0",
                ),
            )
        else:
            return Group(
                Panel(
                    self.prompt.source if self.prompt else "No prompt",
                    title="Prompt",
                    style="on #F0F8FF",
                ),
                Panel(table, title="Message History", style="on #E7DFEB"),
                Panel(self.output, title="Raw LLM Output", style="on #F5F5DC"),
                Panel(
                    pretty_repr(self.validated_output),
                    title="Validated Output",
                    style="on #F0FFF0",
                ),
            )


class GuardHistory(ArbitraryModel):
    history: List[GuardLogs]

    def push(self, guard_log: GuardLogs) -> None:
        if len(self.history) > 0:
            last_log = self.history[-1]
            guard_log._previous_logs = last_log

        self.history += [guard_log]

    @property
    def tree(self) -> Tree:
        """Returns the tree."""
        tree = Tree("Logs")
        for i, log in enumerate(self.history):
            tree.add(Panel(log.rich_group, title=f"Step {i}"))
        return tree

    @property
    def validated_output(self) -> dict:
        """Returns the latest validated output."""
        return self.history[-1].validated_output

    @property
    def output(self) -> str:
        """Returns the latest output."""
        return self.history[-1].output

    @property
    def output_as_dict(self) -> dict:
        """Returns the latest output as a dict."""
        return self.history[-1].parsed_output

    @property
    def failed_validations(self) -> List[ReAsk]:
        """Returns all failed validations."""
        return [log.failed_validations for log in self.history]


class GuardState(ArbitraryModel):
    all_histories: List[GuardHistory]

    def push(self, guard_history: GuardHistory) -> None:
        self.all_histories += [guard_history]

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


def merge_reask_output(previous_response, reask_response) -> Dict:
    """Merge the reask output into the original output.

    Args:
        prev_logs: GuardLogs object from the previous iteration.
        current_logs: GuardLogs object from the current iteration.

    Returns:
        The merged output.
    """
    from guardrails.validators import PydanticReAsk

    if isinstance(previous_response, ReAsk):
        return reask_response

    pruned_reask_json = prune_obj_for_reasking(previous_response)

    # Reask output and reask json have the same structure, except that values
    # of the reask json are ReAsk objects. We want to replace the ReAsk objects
    # with the values from the reask output.
    merged_json = deepcopy(previous_response)

    def update_reasked_elements(pruned_reask_json, reask_response_dict):
        if isinstance(pruned_reask_json, PydanticReAsk):
            corrected_value = reask_response_dict
            # Get the path from any of the ReAsk objects in the PydanticReAsk object
            # all of them have the same path.
            path = [v.path for v in pruned_reask_json.values() if isinstance(v, ReAsk)][
                0
            ]
            update_response_by_path(merged_json, path, corrected_value)

        elif isinstance(pruned_reask_json, dict):
            for key, value in pruned_reask_json.items():
                if isinstance(value, ReAsk):
                    corrected_value = reask_response_dict.get(key)
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
