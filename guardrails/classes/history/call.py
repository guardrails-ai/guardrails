from typing import Dict, Optional, Union

from pydantic import Field, PrivateAttr
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.tree import Tree

from guardrails.classes.generic.stack import Stack
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.classes.history.iteration import Iteration
from guardrails.constants import error_status, fail_status, not_run_status, pass_status
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt
from guardrails.utils.logs_utils import ValidatorLogs, merge_reask_output
from guardrails.utils.pydantic_utils import ArbitraryModel
from guardrails.utils.reask_utils import (
    ReAsk,
    gather_reasks,
    sub_reasks_with_fixed_values,
)
from guardrails.utils.safe_get import get_value_from_path
from guardrails.validator_base import Refrain


# We can't inherit from Iteration because python
# won't let you override a class attribute with a managed attribute
class Call(ArbitraryModel):
    iterations: Stack[Iteration] = Field(
        description="A stack of iterations for each"
        "step/reask that occurred during this call."
    )
    inputs: CallInputs = Field(
        description="The inputs as passed in to Guard.__call__ or Guard.parse"
    )
    _exception: Optional[Exception] = PrivateAttr()

    # Prevent Pydantic from changing our types
    # Without this, Pydantic casts iterations to a list
    def __init__(
        self,
        iterations: Optional[Stack[Iteration]] = None,
        inputs: Optional[CallInputs] = None,
        exception: Optional[Exception] = None,
    ):
        iterations = iterations or Stack()
        inputs = inputs or CallInputs()
        super().__init__(
            iterations=iterations,  # type: ignore
            inputs=inputs,  # type: ignore
            _exception=exception,  # type: ignore
        )
        self.iterations = iterations
        self.inputs = inputs
        self._exception = exception

    @property
    def prompt(self) -> Optional[str]:
        """The prompt as provided by the user when intializing or calling the
        Guard."""
        return self.inputs.prompt

    @property
    def prompt_params(self) -> Optional[Dict]:
        """The prompt parameters as provided by the user when intializing or
        calling the Guard."""
        return self.inputs.prompt_params

    @property
    def compiled_prompt(self) -> Optional[str]:
        """The initial compiled prompt that was passed to the LLM on the first
        call."""
        if self.iterations.empty():
            return None
        initial_inputs = self.iterations.first.inputs  # type: ignore
        prompt: Prompt = initial_inputs.prompt  # type: ignore
        prompt_params = initial_inputs.prompt_params or {}
        if initial_inputs.prompt is not None:
            return prompt.format(**prompt_params).source

    @property
    def reask_prompts(self) -> Stack[Optional[str]]:
        """The compiled prompts used during reasks.

        Does not include the initial prompt.
        """
        if self.iterations.length > 0:
            reasks = self.iterations.copy()
            initial_prompt = reasks.first
            reasks.remove(initial_prompt)  # type: ignore
            return Stack(
                *[
                    r.inputs.prompt.source if r.inputs.prompt is not None else None
                    for r in reasks
                ]
            )

        return Stack()

    @property
    def instructions(self) -> Optional[str]:
        """The instructions as provided by the user when intializing or calling
        the Guard."""
        return self.inputs.instructions

    @property
    def compiled_instructions(self) -> Optional[str]:
        """The initial compiled instructions that were passed to the LLM on the
        first call."""
        if self.iterations.empty():
            return None
        initial_inputs = self.iterations.first.inputs  # type: ignore
        instructions: Instructions = initial_inputs.instructions  # type: ignore
        prompt_params = initial_inputs.prompt_params or {}
        if instructions is not None:
            return instructions.format(**prompt_params).source

    @property
    def reask_instructions(self) -> Stack[str]:
        """The compiled instructions used during reasks.

        Does not include the initial instructions.
        """
        if self.iterations.length > 0:
            reasks = self.iterations.copy()
            reasks.remove(reasks.first)  # type: ignore
            return Stack(
                *[
                    r.inputs.instructions.source
                    if r.inputs.instructions is not None
                    else None
                    for r in reasks
                ]
            )

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
    def parsed_outputs(self) -> Stack[Union[str, Dict]]:
        """The outputs from the LLM after undergoing parsing but before
        validation."""
        return Stack(*[i.outputs.parsed_output for i in self.iterations])

    @property
    def validation_output(self) -> Optional[Union[str, Dict, ReAsk]]:
        """The cumulative validation output across all current iterations.

        Could contain ReAsks.
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
            or isinstance(self.iterations.last.validation_output, ReAsk)  # type: ignore
            or isinstance(self.iterations.last.validation_output, str)  # type: ignore
        ):
            return self.iterations.last.validation_output  # type: ignore

        current_index = 1
        # We've already established that there are iterations,
        #  hence the type ignores
        merged_validation_output = (
            self.iterations.first.validation_output  # type: ignore
        )
        while current_index < number_of_iterations:
            current_validation_output = self.iterations.at(
                current_index
            ).validation_output  # type: ignore
            merged_validation_output = merge_reask_output(
                merged_validation_output, current_validation_output
            )
            current_index = current_index + 1

        return merged_validation_output

    @property
    def fixed_output(self) -> Optional[Union[str, Dict]]:
        """The cumulative validation output across all current iterations with
        any automatic fixes applied."""
        return sub_reasks_with_fixed_values(self.validation_output)

    @property
    def validated_output(self) -> Optional[Union[str, Dict]]:
        """The output from the LLM after undergoing validation.

        This will only have a value if the Guard is in a passing state.
        """
        if self.status == pass_status:
            return self.fixed_output

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
        if self._exception:
            return str(self._exception)
        elif self.iterations.empty():
            return None
        return self.iterations.last.error  # type: ignore

    @property
    def exception(self) -> Optional[Exception]:
        """The exception that interrupted the run."""
        if self._exception:
            return self._exception
        elif self.iterations.empty():
            return None
        return self.iterations.last.exception  # type: ignore

    @property
    def failed_validations(self) -> Stack[ValidatorLogs]:
        """The validator logs for any validations that failed during the
        entirety of the run."""
        return Stack(
            *[
                log
                for log in self.validator_logs
                if log.validation_result is not None
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
            if value == failure.value_before_validation or isinstance(
                failure.value_after_validation, Refrain
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
                pretty_repr(self.validated_output),
                title="Validated Output",
                style="on #F0FFF0",
            )
            tree.children[
                -1
            ].label.renderable._renderables = previous_panels + (  # type: ignore
                validated_outcome_panel,
            )

        return tree
