from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import Field
from guardrails_api_client import (
    ValidationResult as IValidationResult,  # noqa
    PassResult as IPassResult,
    FailResult as IFailResult,
    ErrorSpan as IErrorSpan,
)
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from pydantic import BaseModel


class ValidationResult(IValidationResult, ArbitraryModel):
    """ValidationResult is the output type of Validator.validate and the
    abstract base class for all validation results.

    Attributes:
        outcome (str): The outcome of the validation. Must be one of "pass" or "fail".
        metadata (Optional[Dict[str, Any]]): The metadata associated with this
            validation result.
        validated_chunk (Optional[Any]): The value argument passed to
            validator.validate or validator.validate_stream.
    """

    outcome: str
    metadata: Optional[Dict[str, Any]] = None
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
    """PassResult is the output type of Validator.validate when validation
    succeeds.

    Attributes:
        outcome (Literal["pass"]): The outcome of the validation. Must be "pass".
        value_override (Optional[Any]): The value to use as an override
            if validation passes.
    """

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


class FailResult(ValidationResult, IFailResult):
    """FailResult is the output type of Validator.validate when validation
    fails.

    Attributes:
        outcome (Literal["fail"]): The outcome of the validation. Must be "fail".
        error_message (str): The error message indicating why validation failed.
        fix_value (Optional[Any]): The auto-fix value that would be applied
            if the Validator's on_fail method is "fix".
        error_spans (Optional[List[ErrorSpan]]): Segments that caused
            validation to fail.
    """

    outcome: Literal["fail"] = "fail"

    error_message: str
    fix_value: Optional[Any] = None
    """Segments that caused validation to fail.

    May not exist for non-streamed output.
    """
    error_spans: Optional[List["ErrorSpan"]] = None

    def __init__(self, error_message: str, **kwargs) -> None:
        # This is a silly thing to force a friendly error message and to give type hints
        # to IDEs who have a hard time figuring out the constructor parameters.
        kwargs["error_message"] = error_message
        super().__init__(**kwargs)

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


class ErrorSpan(IErrorSpan, ArbitraryModel):
    """ErrorSpan provide additional context for why a validation failed. They
    specify the start and end index of the segment that caused the failure,
    which can be useful when validating large chunks of text or validating
    while streaming with different chunking methods.

    Attributes:
        start (int): Starting index relative to the validated chunk.
        end (int): Ending index relative to the validated chunk.
        reason (str): Reason validation failed for this chunk.
    """

    start: int
    end: int
    # reason validation failed, specific to this chunk
    reason: str


class StreamValidationResult(BaseModel):
    chunk: Any
    original_text: str
    metadata: Dict[str, Any]
