from typing import Any, Dict, List, Optional, Union
from builtins import id as object_id
from pydantic import Field
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.tree import Tree

from guardrails_api_client import Call as ICall
from guardrails.actions.filter import Filter
from guardrails.actions.refrain import Refrain
from guardrails.actions.reask import merge_reask_output
from guardrails.classes.generic.stack import Stack
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.classes.history.iteration import Iteration
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.validation.validation_result import ValidationResult
from guardrails.constants import error_status, fail_status, not_run_status, pass_status
from guardrails.prompt.messages import Messages
from guardrails.prompt import Prompt, Instructions
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import (
    ReAsk,
    gather_reasks,
    sub_reasks_with_fixed_values,
)
from guardrails.schema.parser import get_value_from_path


# We can't inherit from Iteration because python
# won't let you override a class attribute with a managed attribute
class Call(ICall, ArbitraryModel):
    """A Call represents a single execution of a Guard. One Call is created
    each time the user invokes the `Guard.__call__`, `Guard.parse`, or
    `Guard.validate` method.

    Attributes:
        iterations (Stack[Iteration]): A stack of iterations
            for the initial validation round
            and one for each reask that occurs during a Call.
        inputs (CallInputs): The inputs as passed in to
            `Guard.__call__`, `Guard.parse`, or `Guard.validate`
        exception (Optional[Exception]): The exception that interrupted
            the Guard execution.
    """

    iterations: Stack[Iteration] = Field(
        description="A stack of iterations for each"
        "step/reask that occurred during this call."
    )
    inputs: CallInputs = Field(
        description="The inputs as passed in to Guard.__call__ or Guard.parse"
    )
    exception: Optional[Exception] = Field(
        description="The exception that interrupted the run.",
        default=None,
    )

    # Prevent Pydantic from changing our types
    # Without this, Pydantic casts iterations to a list
    def __init__(
        self,
        iterations: Optional[Stack[Iteration]] = None,
        inputs: Optional[CallInputs] = None,
        exception: Optional[Exception] = None,
    ):
        call_id = str(object_id(self))
        iterations = iterations or Stack()
        inputs = inputs or CallInputs()
        super().__init__(id=call_id, iterations=iterations, inputs=inputs)  # type: ignore - pyright doesn't understand pydantic overrides
        self.iterations = iterations
        self.inputs = inputs
        self.exception = exception

    @property
    def prompt_params(self) -> Optional[Dict]:
        """The prompt parameters as provided by the user when initializing or
        calling the Guard."""
        return self.inputs.prompt_params

    @property
    def messages(self) -> Optional[Union[Messages, list[dict[str, str]]]]:
        """The messages as provided by the user when initializing or calling
        the Guard."""
        return self.inputs.messages

    @property
    def compiled_messages(self) -> Optional[list[dict[str, str]]]:
        """The initial compiled messages that were passed to the LLM on the
        first call."""
        if self.iterations.empty():
            return None
        initial_inputs = self.iterations.first.inputs  # type: ignore
        messages = initial_inputs.messages
        prompt_params = initial_inputs.prompt_params or {}
        compiled_messages = []
        if messages is None:
            return None
        for message in messages:
            content = message["content"].format(**prompt_params)
            if isinstance(content, (Prompt, Instructions)):
                content = content._source
            compiled_messages.append(
                {
                    "role": message["role"],
                    "content": content,
                }
            )

        return compiled_messages

    @property
    def reask_messages(self) -> Stack[Messages]:
        """The compiled messages used during reasks.

        Does not include the initial messages.
        """
        if self.iterations.length > 0:
            reasks = self.iterations.copy()
            initial_messages = reasks.first
            reasks.remove(initial_messages)  # type: ignore
            initial_inputs = self.iterations.first.inputs  # type: ignore
            prompt_params = initial_inputs.prompt_params or {}
            compiled_reasks = []
            for reask in reasks:
                messages = reask.inputs.messages

                if messages is None:
                    compiled_reasks.append(None)
                else:
                    compiled_messages = []
                    for message in messages:
                        content = message["content"].format(**prompt_params)
                        if isinstance(content, (Prompt, Instructions)):
                            content = content._source
                        compiled_messages.append(
                            {
                                "role": message["role"],
                                "content": content,
                            }
                        )
                    compiled_reasks.append(compiled_messages)
            return Stack(*compiled_reasks)

        return Stack()

    @property
    def logs(self) -> Stack[str]:
        """Returns all logs from all iterations as a stack."""
        all_logs = []
        for i in self.iterations:
            all_logs.extend(i.logs)
        return Stack(*all_logs)

    @property
    def tokens_consumed(self) -> Optional[int]:
        """Returns the total number of tokens consumed during all iterations
        with this call."""
        iteration_tokens = [
            i.tokens_consumed for i in self.iterations if i.tokens_consumed is not None
        ]
        if len(iteration_tokens) > 0:
            return sum(iteration_tokens)
        return None

    @property
    def prompt_tokens_consumed(self) -> Optional[int]:
        """Returns the total number of prompt tokens consumed during all
        iterations with this call."""
        iteration_tokens = [
            i.prompt_tokens_consumed
            for i in self.iterations
            if i.prompt_tokens_consumed is not None
        ]
        if len(iteration_tokens) > 0:
            return sum(iteration_tokens)
        return None

    @property
    def completion_tokens_consumed(self) -> Optional[int]:
        """Returns the total number of completion tokens consumed during all
        iterations with this call."""
        iteration_tokens = [
            i.completion_tokens_consumed
            for i in self.iterations
            if i.completion_tokens_consumed is not None
        ]
        if len(iteration_tokens) > 0:
            return sum(iteration_tokens)
        return None

    @property
    def raw_outputs(self) -> Stack[str]:
        """The exact outputs from all LLM calls."""
        return Stack(
            *[
                i.outputs.llm_response_info.output
                if i.outputs.llm_response_info is not None
                else None
                for i in self.iterations
            ]
        )

    @property
    def parsed_outputs(self) -> Stack[Union[str, List, Dict]]:
        """The outputs from the LLM after undergoing parsing but before
        validation."""
        return Stack(*[i.outputs.parsed_output for i in self.iterations])

    @property
    def validation_response(self) -> Optional[Union[str, List, Dict, ReAsk]]:
        """The aggregated responses from the validation process across all
        iterations within the current call.

        This value could contain ReAsks.
        """
        number_of_iterations = self.iterations.length

        if number_of_iterations == 0:
            return None

        # Don't try to merge if
        #   1. We plan to perform full schema reasks
        #   2. There's nothing to merge
        #   3. The output is a top level ReAsk (i.e. SkeletonReAsk or NonParseableReask)
        #   4. The output is a string
        if (
            self.inputs.full_schema_reask
            or number_of_iterations < 2
            or isinstance(
                self.iterations.last.validation_response,  # type: ignore
                ReAsk,  # type: ignore
            )
            or isinstance(self.iterations.last.validation_response, str)  # type: ignore
        ):
            return self.iterations.last.validation_response  # type: ignore

        current_index = 1
        # We've already established that there are iterations,
        #  hence the type ignores
        merged_validation_responses = (
            self.iterations.first.validation_response  # type: ignore
        )
        while current_index < number_of_iterations:
            current_validation_output = self.iterations.at(
                current_index
            ).validation_response  # type: ignore
            merged_validation_responses = merge_reask_output(
                merged_validation_responses, current_validation_output
            )
            current_index = current_index + 1

        return merged_validation_responses

    @property
    def fixed_output(self) -> Optional[Union[str, List, Dict]]:
        """The cumulative output from the validation process across all current
        iterations with any automatic fixes applied.

        Could still contain ReAsks if a fix was not available.
        """
        return sub_reasks_with_fixed_values(self.validation_response)

    @property
    def guarded_output(self) -> Optional[Union[str, List, Dict]]:
        """The complete validated output after all stages of validation are
        completed.

        This property contains the aggregate validated output after all
        validation stages have been completed. Some values in the
        validated output may be "fixed" values that were corrected
        during validation.

        This will only have a value if the Guard is in a passing state
        OR if the action is no-op.
        """
        if self.status == pass_status:
            return self.fixed_output
        last_iteration = self.iterations.last
        if (
            not self.status == pass_status
            and last_iteration
            and last_iteration.failed_validations
        ):
            # check that all failed validations are noop or none
            all_noop = True
            for failed_validation in last_iteration.failed_validations:
                if (
                    failed_validation.value_after_validation
                    is not failed_validation.value_before_validation
                ):
                    all_noop = False
                    break
            if all_noop:
                return last_iteration.guarded_output

    @property
    def reasks(self) -> Stack[ReAsk]:
        """Reasks generated during validation that could not be automatically
        fixed.

        These would be incorporated into the prompt for the next LLM
        call if additional reasks were granted.
        """
        reasks, _ = gather_reasks(self.fixed_output)
        return Stack(*reasks)

    @property
    def validator_logs(self) -> Stack[ValidatorLogs]:
        """The results of each individual validation performed on the LLM
        responses during all iterations."""
        all_validator_logs = Stack()
        for i in self.iterations:
            all_validator_logs.extend(i.validator_logs)
        return all_validator_logs

    @property
    def error(self) -> Optional[str]:
        """The error message from any exception that raised and interrupted the
        run."""
        if self.exception:
            return str(self.exception)
        elif self.iterations.empty():
            return None
        return self.iterations.last.error  # type: ignore

    @property
    def failed_validations(self) -> Stack[ValidatorLogs]:
        """The validator logs for any validations that failed during the
        entirety of the run."""
        return Stack(
            *[
                log
                for log in self.validator_logs
                if log.validation_result is not None
                and isinstance(log.validation_result, ValidationResult)
                and log.validation_result.outcome == "fail"
            ]
        )

    def _has_unresolved_failures(self) -> bool:
        # Check for unresolved ReAsks
        if len(self.reasks) > 0:
            return True

        # Check for scenario where no specified on-fail's produced an unfixed ReAsk,
        #   but valdiation still failed (i.e. Refrain or NoOp).
        output = self.fixed_output
        for failure in self.failed_validations:
            value = get_value_from_path(output, failure.property_path)
            if (
                # NOTE: this means on_fail="fix" was applied
                #       to a Validator without a programmatic fix.
                (value is None and failure.value_before_validation is not None)
                or value == failure.value_before_validation
                or isinstance(failure.value_after_validation, Refrain)
                or isinstance(failure.value_after_validation, Filter)
            ):
                return True

        # No ReAsks and no unresolved failed validations
        return False

    @property
    def status(self) -> str:
        """Returns the cumulative status of the run based on the validity of
        the final merged output."""
        if self.iterations.empty():
            return not_run_status
        elif self.error:
            return error_status
        elif self._has_unresolved_failures():
            return fail_status
        return pass_status

    @property
    def tree(self) -> Tree:
        """Returns the tree."""
        tree = Tree("Logs")
        for i, iteration in enumerate(self.iterations):
            tree.add(Panel(iteration.rich_group, title=f"Step {i}"))

        # Replace the last Validated Output panel if we applied fixes
        if self.failed_validations.length > 0 and self.status == pass_status:
            previous_panels = tree.children[  # type: ignore
                -1
            ].label.renderable._renderables[  # type: ignore
                :-1
            ]
            validated_outcome_panel = Panel(
                pretty_repr(self.guarded_output),
                title="Validated Output",
                style="on #F0FFF0",
            )
            tree.children[-1].label.renderable._renderables = previous_panels + (  # type: ignore
                validated_outcome_panel,
            )

        return tree

    def __str__(self) -> str:
        return pretty_repr(self)

    def to_interface(self) -> ICall:
        return ICall(
            id=self.id,
            iterations=[i.to_interface() for i in self.iterations],
            inputs=self.inputs.to_interface(),
            exception=self.error,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_call: ICall) -> "Call":
        iterations = Stack(
            *[Iteration.from_interface(i) for i in (i_call.iterations or [])]
        )
        inputs = (
            CallInputs.from_interface(i_call.inputs) if i_call.inputs else CallInputs()
        )
        exception = Exception(i_call.exception) if i_call.exception else None
        call_inst = cls(iterations=iterations, inputs=inputs, exception=exception)
        call_inst.id = i_call.id
        return call_inst

    # TODO: Necessary to GET /guards/{guard_name}/history/{call_id}
    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Call":
        i_call = ICall.from_dict(obj)

        if i_call:
            return cls.from_interface(i_call)
        return Call()
