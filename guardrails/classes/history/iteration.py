from typing import Dict, List, Optional, Sequence, Union

from pydantic import Field

from guardrails.classes.history.inputs import Inputs
from guardrails.classes.history.outputs import Outputs
from guardrails.utils.logs_utils import ValidatorLogs

# from guardrails.classes.stack import Stack
from guardrails.utils.pydantic_utils import ArbitraryModel
from guardrails.utils.reask_utils import ReAsk


class Iteration(ArbitraryModel):
    # I think these should be containered since their names slightly overlap with
    #  outputs, but could be convinced otherwise
    inputs: Inputs = Field(
        description="The inputs for the iteration/step.", default_factory=Inputs
    )
    # We might just spread these properties instead of containering them
    outputs: Outputs = Field(
        description="The outputs from the iteration/step.", default_factory=Outputs
    )

    # TODO
    # @property
    # def logs() -> Stack[str]:
    #     """Returns the logs from this iteration as a stack"""

    @property
    def tokens_consumed(self) -> Optional[int]:
        """Returns the total number of tokens consumed during this
        iteration."""
        input_tokens = self.prompt_tokens_consumed
        output_tokens = self.completion_tokens_consumed
        if input_tokens is not None or output_tokens is not None:
            return (input_tokens or 0) + (output_tokens or 0)

    @property
    def prompt_tokens_consumed(self) -> Optional[int]:
        """Returns the number of prompt/input tokens consumed during this
        iteration."""
        response = self.outputs.llm_response_info
        if response is not None:
            return response.prompt_token_count

    @property
    def completion_tokens_consumed(self) -> Optional[int]:
        """Returns the number of completion/output tokens consumed during this
        iteration."""
        response = self.outputs.llm_response_info
        if response is not None:
            return response.response_token_count

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
    def validation_output(self) -> Optional[Union[ReAsk, str, Dict]]:
        """The output from the validation process.

        Could be a combination of valid output and ReAsks
        """
        return self.outputs.validation_output

    @property
    def validated_output(self) -> Optional[Union[str, Dict]]:
        """The valid output from the LLM after undergoing validation."""
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
        """Representation of the end state of this iteration.

        OneOf: pass, fail, error, not run
        """
        return self.outputs.status
