from typing import Generic, Iterator, List, Optional, Tuple, Union, cast

from pydantic import Field
from rich.pretty import pretty_repr

from guardrails_api_client import (
    ValidationOutcome as IValidationOutcome,
    ValidationOutcomeValidatedOutput,
)
from guardrails.actions.reask import ReAsk
from guardrails.classes.history import Call, Iteration
from guardrails.classes.output_type import OT
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.validation.validation_summary import ValidationSummary
from guardrails.constants import pass_status
from guardrails.utils.safe_get import safe_get


class ValidationOutcome(IValidationOutcome, ArbitraryModel, Generic[OT]):
    """The final output from a Guard execution.

    Attributes:
        call_id: The id of the Call that produced this ValidationOutcome.
        raw_llm_output: The raw, unchanged output from the LLM call.
        validated_output: The validated, and potentially fixed, output from the LLM call
            after passing through validation.
        reask: If validation continuously fails and all allocated reasks are used,
            this field will contain the final reask that would have been sent
                to the LLM if additional reasks were available.
        validation_passed: A boolean to indicate whether or not the LLM output
            passed validation. If this is False, the validated_output may be invalid.
        error: If the validation failed, this field will contain the error message
    """

    validation_summaries: Optional[List["ValidationSummary"]] = Field(
        description="The summaries of the validation results.", default=[]
    )
    """The summaries of the validation results."""

    raw_llm_output: Optional[str] = Field(
        description="The raw, unchanged output from the LLM call.", default=None
    )
    """The raw, unchanged output from the LLM call."""

    validated_output: Optional[OT] = Field(
        description="The validated, and potentially fixed,"
        " output from the LLM call after passing through validation.",
        default=None,
    )
    """The validated, and potentially fixed, output from the LLM call after
    passing through validation."""

    reask: Optional[ReAsk] = Field(
        description="If validation continuously fails and all allocated"
        " reasks are used, this field will contain the final reask that"
        " would have been sent to the LLM if additional reasks were available.",
        default=None,
    )
    """If validation continuously fails and all allocated reasks are used, this
    field will contain the final reask that would have been sent to the LLM if
    additional reasks were available."""

    validation_passed: bool = Field(
        description="A boolean to indicate whether or not"
        " the LLM output passed validation."
        "  If this is False, the validated_output may be invalid."
    )
    """A boolean to indicate whether or not the LLM output passed validation.

    If this is False, the validated_output may be invalid.
    """

    error: Optional[str] = Field(default=None)
    """If the validation failed, this field will contain the error message."""

    @classmethod
    def from_guard_history(cls, call: Call):
        """Create a ValidationOutcome from a history Call object."""
        last_iteration = call.iterations.last or Iteration(call_id=call.id, index=0)
        last_output = last_iteration.validation_response or safe_get(
            list(last_iteration.reasks), 0
        )
        validation_passed = call.status == pass_status
        validator_logs = last_iteration.validator_logs or []
        validation_summaries = ValidationSummary.from_validator_logs_only_fails(
            validator_logs
        )
        reask = last_output if isinstance(last_output, ReAsk) else None
        error = call.error
        output = cast(OT, call.guarded_output)
        return cls(
            call_id=call.id,  # type: ignore
            raw_llm_output=call.raw_outputs.last,
            validated_output=output,
            reask=reask,
            validation_passed=validation_passed,
            validation_summaries=validation_summaries,
            error=error,
        )

    def __iter__(
        self,
    ) -> Iterator[
        Union[Optional[str], Optional[OT], Optional[ReAsk], bool, Optional[str]]
    ]:
        """Iterate over the ValidationOutcome's fields."""
        as_tuple: Tuple[
            Optional[str], Optional[OT], Optional[ReAsk], bool, Optional[str]
        ] = (
            self.raw_llm_output,
            self.validated_output,
            self.reask,
            self.validation_passed,
            self.error,
        )
        return iter(as_tuple)

    def __getitem__(self, keys):
        """Get a subset of the ValidationOutcome's fields."""
        return iter(getattr(self, k) for k in keys)

    def __str__(self) -> str:
        return pretty_repr(self)

    def to_dict(self):
        i_validation_outcome = IValidationOutcome(
            call_id=self.call_id,  # type: ignore
            raw_llm_output=self.raw_llm_output,  # type: ignore
            validated_output=ValidationOutcomeValidatedOutput(self.validated_output),  # type: ignore
            reask=self.reask,
            validation_passed=self.validation_passed,  # type: ignore
            error=self.error,
        )

        return i_validation_outcome.to_dict()
