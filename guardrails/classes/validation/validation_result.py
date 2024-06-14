from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from guardrails_api_client import (
    ValidationResult as IValidationResult,  # noqa
    PassResult as IPassResult,
    FailResult as IFailResult,
)
from guardrails.classes.generic.arbitrary_model import ArbitraryModel


class ValidationResult(IValidationResult, ArbitraryModel):
    outcome: str
    metadata: Optional[Dict[str, Any]] = None

    # value argument passed to validator.validate
    # or validator.validate_stream
    # FIXME: Add this to json schema
    validated_chunk: Optional[Any] = None


class PassResult(ValidationResult, IPassResult):
    outcome: Literal["pass"] = "pass"

    class ValueOverrideSentinel:
        pass

    # should only be used if Validator.override_value_on_pass is True
    value_override: Optional[Any] = Field(default=ValueOverrideSentinel)

    def to_dict(self) -> Dict[str, Any]:
        i_pass_result = IPassResult(outcome=self.outcome, metadata=self.metadata)

        if self.value_override is not self.ValueOverrideSentinel:
            i_pass_result.value_override = self.value_override

        return i_pass_result.to_dict()


# FIXME: Add this to json schema
class ErrorSpan(ArbitraryModel):
    start: int
    end: int
    # reason validation failed, specific to this chunk
    reason: str


class FailResult(ValidationResult, IFailResult):
    outcome: Literal["fail"] = "fail"

    error_message: str
    fix_value: Optional[Any] = None
    # FIXME: Add this to json schema
    # segments that caused validation to fail
    error_spans: Optional[List[ErrorSpan]] = None
