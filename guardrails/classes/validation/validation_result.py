from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import Field
from guardrails_api_client import (
    ValidationResult as IValidationResult,  # noqa
    PassResult as IPassResult,
    FailResult as IFailResult,
    ErrorSpan as IErrorSpan,
)
from guardrails.classes.generic.arbitrary_model import ArbitraryModel


class ValidationResult(IValidationResult, ArbitraryModel):
    outcome: str
    metadata: Optional[Dict[str, Any]] = None

    # value argument passed to validator.validate
    # or validator.validate_stream
    validated_chunk: Optional[Any] = None

    @classmethod
    def from_interface(
        cls, i_validation_result: Union[IValidationResult, IPassResult, IFailResult]
    ) -> "ValidationResult":
        if i_validation_result.outcome == "pass":
            return PassResult(
                outcome=i_validation_result.outcome,
                metadata=i_validation_result.metadata,
                validated_chunk=i_validation_result.validated_chunk,
            )
        elif i_validation_result.outcome == "fail":
            return FailResult.from_dict(i_validation_result.to_dict())

        return cls(
            outcome=i_validation_result.outcome or "",
            metadata=i_validation_result.metadata,
            validated_chunk=i_validation_result.validated_chunk,
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ValidationResult":
        i_validation_result = IValidationResult.from_dict(obj) or IValidationResult(
            outcome="pail"
        )
        return cls.from_interface(i_validation_result)


class PassResult(ValidationResult, IPassResult):
    outcome: Literal["pass"] = "pass"

    class ValueOverrideSentinel:
        pass

    # should only be used if Validator.override_value_on_pass is True
    value_override: Optional[Any] = Field(default=ValueOverrideSentinel)

    def to_interface(self) -> IPassResult:
        i_pass_result = IPassResult(outcome=self.outcome, metadata=self.metadata)

        if self.value_override is not self.ValueOverrideSentinel:
            i_pass_result.value_override = self.value_override

        return i_pass_result

    def to_dict(self) -> Dict[str, Any]:
        # Pydantic's model_dump method isn't working properly
        _dict = {
            "outcome": self.outcome,
            "metadata": self.metadata,
            "validatedChunk": self.validated_chunk,
            "valueOverride": (
                self.value_override
                if self.value_override is not self.ValueOverrideSentinel
                else None
            ),
        }
        return _dict


# FIXME: Add this to json schema
class ErrorSpan(IErrorSpan, ArbitraryModel):
    start: int
    end: int
    # reason validation failed, specific to this chunk
    reason: str


class FailResult(ValidationResult, IFailResult):
    outcome: Literal["fail"] = "fail"

    error_message: str
    fix_value: Optional[Any] = None
    # segments that caused validation to fail
    error_spans: Optional[List[ErrorSpan]] = None

    @classmethod
    def from_interface(cls, i_fail_result: IFailResult) -> "FailResult":
        error_spans = None
        if i_fail_result.error_spans:
            error_spans = [
                ErrorSpan(
                    start=error_span.start,
                    end=error_span.end,
                    reason=error_span.reason,
                )
                for error_span in i_fail_result.error_spans
            ]

        return cls(
            outcome="fail",
            metadata=i_fail_result.metadata,
            validated_chunk=i_fail_result.validated_chunk,
            error_message=i_fail_result.error_message or "",
            fix_value=i_fail_result.fix_value,
            error_spans=error_spans,
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "FailResult":
        i_fail_result = IFailResult.from_dict(obj) or IFailResult(
            outcome="Fail",
            error_message="",  # type: ignore - pyright doesn't understand aliases
        )
        return cls.from_interface(i_fail_result)

    def to_dict(self) -> Dict[str, Any]:
        # Pydantic's model_dump method isn't working properly
        _dict = {
            "outcome": self.outcome,
            "metadata": self.metadata,
            "validatedChunk": self.validated_chunk,
            "errorMessage": self.error_message,
            "fixValue": self.fix_value,
            "errorSpans": (
                [error_span.to_dict() for error_span in self.error_spans]
                if self.error_spans
                else []
            ),
        }
        return _dict
