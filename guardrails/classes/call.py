from typing import Dict, List, Optional, Sequence, Union

from pydantic import Field

from guardrails.classes.call_inputs import CallInputs
from guardrails.classes.iteration import Iteration
from guardrails.classes.outputs import Outputs
from guardrails.classes.stack import Stack
from guardrails.constants import not_run_status
from guardrails.utils.logs_utils import ValidatorLogs
from guardrails.utils.pydantic_utils import ArbitraryModel
from guardrails.utils.reask_utils import ReAsk


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

    # Prevent Pydantic from changing our types
    # Without this, Pydantic casts iterations to a list
    def __init__(
        self, iterations: Stack[Iteration] = Stack(), inputs: CallInputs = CallInputs()
    ):
        super().__init__(iterations=iterations, inputs=inputs)
        self.iterations = iterations
        self.inputs = inputs
        print("Call.__init__ called")

    # We might just spread these properties instead of containering them
    @property
    def outputs(self) -> Outputs:
        """The outputs from the last iteration."""
        last_iteration = self.iterations.last
        if last_iteration:
            return last_iteration.outputs
        # To allow chaining without getting AttributeErrors
        return Outputs()

    # TODO
    # @property
    # def logs(self) -> Stack[str]:
    #     """Returns all logs from all iterations as a stack"""

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
    def raw_output(self) -> Optional[str]:
        """The exact output from the LLM."""
        response = self.outputs.llm_response_info
        if response is not None:
            return response.output

    @property
    def parsed_output(self) -> Optional[Union[str, Dict]]:
        """The output from the LLM after undergoing parsing but before
        validation."""
        return self.outputs.parsed_output

    @property
    def validated_output(self) -> Optional[Union[str, Dict]]:
        """The output from the LLM after undergoing validation."""
        return self.outputs.validated_output

    @property
    def reasks(self) -> Sequence[ReAsk]:
        """Reasks generated during validation.

        These would be incorporated into the prompt or the next LLM
        call.
        """
        return self.outputs.reasks

    @property
    def validator_logs(self) -> List[ValidatorLogs]:
        """The results of each individual validation performed on the LLM
        response during this iteration."""
        return self.outputs.validator_logs

    @property
    def error(self) -> Optional[str]:
        """The error message from any exception that raised and interrupted
        this iteration."""
        return self.outputs.error

    @property
    def failed_validations(self) -> List[ValidatorLogs]:
        """The validator logs for any validations that failed during this
        iteration."""
        return self.outputs.failed_validations

    @property
    def status(self) -> str:
        """Returns the status of the last iteration."""
        if self.iterations.empty():
            return not_run_status
        return self.iterations.last.status
