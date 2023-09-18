from typing import Dict, Generic, Optional, TypeVar

from pydantic import Field

from guardrails.utils.logs_utils import ArbitraryModel, GuardHistory
from guardrails.utils.reask_utils import ReAsk

T = TypeVar("T", str, Dict, None)


class ValidationOutcome(Generic[T], ArbitraryModel):
    raw_llm_output: str = Field(
        description="The raw, unchanged output from the LLM call."
    )
    validated_output: Optional[T] = Field(
        description="The validated, and potentially fixed,"
        " output from the LLM call after passing through validation."
    )
    reask: Optional[ReAsk] = Field(
        description="If validation continuously fails and all allocated"
        " reasks are used, this field will contain the final reask that"
        " would have been sent to the LLM if additional reasks were available."
    )
    validation_passed: bool = Field(
        description="A boolean to indicate whether or not"
        " the LLM output passed validation."
        "  If this is False, the validated_output may be invalid."
    )

    @classmethod
    def from_guard_history(cls, guard_history: GuardHistory):
        raw_output = guard_history.output
        validated_output = guard_history.validated_output
        any_validations_failed = len(guard_history.failed_validations) > 0
        if isinstance(validated_output, ReAsk):
            return cls[T](
                raw_llm_output=raw_output,
                reask=validated_output,
                validation_passed=any_validations_failed,
            )
        else:
            return cls[T](
                raw_llm_output=raw_output,
                validated_output=validated_output,
                validation_passed=any_validations_failed,
            )

    def __iter__(self):
        as_tuple = (
            self.raw_llm_output,
            self.validated_output,
            self.reask,
            self.validation_passed,
        )
        return iter(as_tuple)

    def __getitem__(self, keys):
        return iter(getattr(self, k) for k in keys)
