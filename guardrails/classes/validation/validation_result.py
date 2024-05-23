from typing import Any, Literal, Optional
from pydantic import Field
from guardrails_api_client import (
    ValidationResult,  # noqa
    PassResult as IPassResult,
    FailResult as IFailResult,
)
from guardrails.classes.generic.arbitrary_model import ArbitraryModel


class PassResult(ValidationResult, IPassResult, ArbitraryModel):
    outcome: Literal["pass"] = "pass"

    class ValueOverrideSentinel:
        pass

    # should only be used if Validator.override_value_on_pass is True
    value_override: Optional[Any] = Field(default=ValueOverrideSentinel)


class FailResult(ValidationResult, IFailResult, ArbitraryModel):
    outcome: Literal["fail"] = "fail"

    error_message: str
    fix_value: Optional[Any] = None
