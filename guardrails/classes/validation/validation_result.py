from __future__ import annotations
from typing_extensions import deprecated
from typing import Any, Dict
from pydantic import BaseModel
from guardrails_ai.types import (
    ValidationResult as IValidationResult,
    PassResult as IPassResult,
    FailResult as IFailResult,
    ErrorSpan as ErrorSpan,
)


def to_validation_result(obj: Any) -> PassResult | FailResult | ValidationResult:
    if isinstance(obj, dict):
        outcome = obj.get("outcome")
        if outcome == "pass":
            return PassResult.model_validate(obj)
        elif outcome == "fail":
            return FailResult.model_validate(obj)
    return ValidationResult.model_validate(obj)


class ValidationResult(IValidationResult):
    @classmethod
    @deprecated("Use to_validation_result() instead.")
    def from_interface(cls, i_validation_result: Any) -> "ValidationResult":
        return to_validation_result(i_validation_result)

    @classmethod
    @deprecated("Use to_validation_result instead.")
    def from_dict(cls, obj: Any) -> "ValidationResult":
        return to_validation_result(obj)


class PassResult(IPassResult, ValidationResult):
    @deprecated("Use PassResult.model_dump() instead.")
    def to_interface(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @deprecated("Use PassResult.model_dump() instead.")
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)


class FailResult(IFailResult, ValidationResult):
    @classmethod
    @deprecated("Use FailResult.model_validate() instead.")
    def from_interface(cls, i_fail_result: Any) -> "FailResult":
        return cls.model_validate(i_fail_result)

    @classmethod
    @deprecated("Use FailResult.model_validate() instead.")
    def from_dict(cls, obj: Any) -> "FailResult":
        return cls.model_validate(obj)

    @deprecated("Use FailResult.model_dump() instead.")
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)


class StreamValidationResult(BaseModel):
    chunk: Any
    original_text: str
    metadata: Dict[str, Any]
