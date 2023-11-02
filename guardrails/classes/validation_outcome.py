from typing import Dict, Generic, Iterator, Optional, Tuple, Union, cast

from pydantic import Field

from guardrails.utils.logs_utils import ArbitraryModel, GuardHistory
from guardrails.utils.reask_utils import ReAsk
from guardrails.classes.output_type import OT


class ValidationOutcome(Generic[OT], ArbitraryModel):
    raw_llm_output: Optional[str] = Field(
        description="The raw, unchanged output from the LLM call.",
        default=None
    )
    validated_output: Optional[OT] = Field(
        description="The validated, and potentially fixed,"
        " output from the LLM call after passing through validation.",
        default=None
    )
    reask: Optional[ReAsk] = Field(
        description="If validation continuously fails and all allocated"
        " reasks are used, this field will contain the final reask that"
        " would have been sent to the LLM if additional reasks were available.",
        default=None
    )
    validation_passed: bool = Field(
        description="A boolean to indicate whether or not"
        " the LLM output passed validation."
        "  If this is False, the validated_output may be invalid."
    )
    error: Optional[str] = Field(default=None)

    @classmethod
    def from_guard_history(
        cls, guard_history: GuardHistory, error_message: Optional[str]
    ):
        raw_output = guard_history.output
        validated_output = guard_history.validated_output
        any_validations_failed = len(guard_history.failed_validations) > 0
        if error_message:
            return cls(
                raw_llm_output=raw_output or "",
                validation_passed=False,
                error=error_message,
            )
        elif isinstance(validated_output, ReAsk):
            reask: ReAsk = validated_output
            return cls(
                raw_llm_output=raw_output,
                reask=reask,
                validation_passed=any_validations_failed,
            )
        else:
            output = cast(OT, validated_output)
            return cls(
                raw_llm_output=raw_output,
                validated_output=output,
                validation_passed=any_validations_failed,
            )

    def __iter__(self) -> Iterator[Union[Optional[str], Optional[OT], Optional[ReAsk], bool, Optional[str]]]:
        as_tuple: Tuple[Optional[str], Optional[OT], Optional[ReAsk], bool, Optional[str]] = (
            self.raw_llm_output,
            self.validated_output,
            self.reask,
            self.validation_passed,
            self.error,
        )
        return iter(as_tuple)

    def __getitem__(self, keys):
        return iter(getattr(self, k) for k in keys)
