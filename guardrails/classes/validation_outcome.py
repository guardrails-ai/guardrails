from typing import Generic, Iterator, Optional, Tuple, Union, cast

from pydantic import Field

from guardrails.classes.history import Call, Iteration
from guardrails.classes.output_type import OT
from guardrails.constants import pass_status
from guardrails.utils.logs_utils import ArbitraryModel
from guardrails.utils.reask_utils import ReAsk


class ValidationOutcome(Generic[OT], ArbitraryModel):
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
        last_iteration = call.iterations.last or Iteration()
        last_output = last_iteration.validation_output or last_iteration.parsed_output
        validation_passed = call.status == pass_status
        reask = last_output if isinstance(last_output, ReAsk) else None
        error = call.error
        output = cast(OT, call.validated_output)
        return cls(
            raw_llm_output=call.raw_outputs.last,
            validated_output=output,
            reask=reask,
            validation_passed=validation_passed,
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
