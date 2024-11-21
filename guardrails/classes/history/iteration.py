from typing import Any, Dict, List, Optional, Sequence, Union
from builtins import id as object_id
from pydantic import Field
from rich.console import Group
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.table import Table

from guardrails_api_client import Iteration as IIteration
from guardrails.classes.generic.stack import Stack
from guardrails.classes.history.inputs import Inputs
from guardrails.classes.history.outputs import Outputs
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.logger import get_scope_handler
from guardrails.prompt import Prompt, Instructions
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import ReAsk
from guardrails.classes.validation.validation_result import ErrorSpan


class Iteration(IIteration, ArbitraryModel):
    """An Iteration represents a single iteration of the validation loop
    including a single call to the LLM if applicable.

    Attributes:
        id (str): The unique identifier for the iteration.
        call_id (str): The unique identifier for the Call
            that this iteration is a part of.
        index (int): The index of this iteration within the Call.
        inputs (Inputs): The inputs for the validation loop.
        outputs (Outputs): The outputs from the validation loop.
    """

    # I think these should be containered since their names slightly overlap with
    #  outputs, but could be convinced otherwise
    inputs: Inputs = Field(
        description="The inputs for the iteration/step.", default_factory=Inputs
    )
    # We might just spread these properties instead of containering them
    outputs: Outputs = Field(
        description="The outputs from the iteration/step.", default_factory=Outputs
    )

    def __init__(
        self,
        call_id: str,
        index: int,
        inputs: Optional[Inputs] = None,
        outputs: Optional[Outputs] = None,
    ):
        iteration_id = str(object_id(self))
        inputs = inputs or Inputs()
        outputs = outputs or Outputs()
        super().__init__(
            id=iteration_id,
            call_id=call_id,  # type: ignore
            index=index,
            inputs=inputs,
            outputs=outputs,
        )
        self.inputs = inputs
        self.outputs = outputs

    @property
    def logs(self) -> Stack[str]:
        """Returns the logs from this iteration as a stack."""
        scope = str(id(self))
        scope_handler = get_scope_handler()
        scoped_logs = scope_handler.get_logs(scope)
        return Stack(*[log.getMessage() for log in scoped_logs])

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
        if response is not None and response.output:
            return response.output
        elif self.outputs.raw_output is not None:
            return self.outputs.raw_output

    @property
    def parsed_output(self) -> Optional[Union[str, List, Dict]]:
        """The output from the LLM after undergoing parsing but before
        validation."""
        return self.outputs.parsed_output

    @property
    def validation_response(self) -> Optional[Union[ReAsk, str, List, Dict]]:
        """The response from a single stage of validation.

        Validation response is the output of a single stage of validation
        and could be a combination of valid output and reasks.
        Note that a Guard may run validation multiple times if reasks occur.
        To access the final output after all steps of validation are completed,
        check out `Call.guarded_output`."
        """
        return self.outputs.validation_response

    @property
    def guarded_output(self) -> Optional[Union[str, List, Dict]]:
        """Any valid values after undergoing validation.

        Some values in the validated output may be "fixed" values that
        were corrected during validation. This property may be a partial
        structure if field level reasks occur.
        """
        return self.outputs.guarded_output

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
        if self.inputs.stream:
            filtered_logs = [
                log
                for log in self.outputs.validator_logs
                if log.validation_result and log.validation_result.validated_chunk
            ]
            return filtered_logs
        return self.outputs.validator_logs

    @property
    def error(self) -> Optional[str]:
        """The error message from any exception that raised and interrupted
        this iteration."""
        return self.outputs.error

    @property
    def exception(self) -> Optional[Exception]:
        """The exception that interrupted this iteration."""
        return self.outputs.exception

    @property
    def failed_validations(self) -> List[ValidatorLogs]:
        """The validator logs for any validations that failed during this
        iteration."""
        return self.outputs.failed_validations

    @property
    def error_spans_in_output(self) -> List[ErrorSpan]:
        """The error spans from the LLM response.

        These indices are relative to the complete LLM output.
        """
        return self.outputs.error_spans_in_output

    @property
    def status(self) -> str:
        """Representation of the end state of this iteration.

        OneOf: pass, fail, error, not run
        """
        return self.outputs.status

    @property
    def rich_group(self) -> Group:
        def create_messages_table(
            messages: Optional[List[Dict[str, Union[str, Prompt, Instructions]]]],
        ) -> Union[str, Table]:
            if messages is None:
                return "No messages."
            table = Table(show_lines=True)
            table.add_column("Role", justify="right", no_wrap=True)
            table.add_column("Content")

            for msg in messages:
                if hasattr(msg["content"], "source"):
                    table.add_row(str(msg["role"]), msg["content"].source)  # type: ignore
                else:
                    table.add_row(str(msg["role"]), msg["content"])  # type: ignore

            return table

        table = create_messages_table(self.inputs.messages)  # type: ignore

        return Group(
            Panel(table, title="Messages", style="on #E7DFEB"),
            Panel(self.raw_output or "", title="Raw LLM Output", style="on #F5F5DC"),
            Panel(
                self.validation_response
                if isinstance(self.validation_response, str)
                else pretty_repr(self.validation_response),
                title="Validated Output",
                style="on #F0FFF0",
            ),
        )

    def __str__(self) -> str:
        return pretty_repr(self)

    def to_interface(self) -> IIteration:
        return IIteration(
            id=self.id,
            call_id=self.call_id,  # type: ignore
            index=self.index,
            inputs=self.inputs.to_interface(),
            outputs=self.outputs.to_interface(),
        )

    def to_dict(self) -> Dict:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_iteration: IIteration) -> "Iteration":
        inputs = (
            Inputs.from_interface(i_iteration.inputs) if i_iteration.inputs else None
        )
        outputs = (
            Outputs.from_interface(i_iteration.outputs) if i_iteration.outputs else None
        )
        iteration = cls(
            call_id=i_iteration.call_id,
            index=i_iteration.index,
            inputs=inputs,
            outputs=outputs,
        )
        iteration.id = i_iteration.id
        return iteration

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Iteration":
        id = obj.get("id", "0")
        call_id = obj.get("callId", obj.get("call_id", "0"))
        index = obj.get("index", 0)
        i_iteration = IIteration.from_dict(obj) or IIteration(
            id=id,
            call_id=call_id,  # type: ignore
            index=index,  # type: ignore
        )

        return cls.from_interface(i_iteration)
