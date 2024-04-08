from typing import Any, Dict

from guardrails.guard import Guard
from guardrails.validator_base import (
    FailResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator("failure", "string")
class FailureValidator(Validator):
    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        return FailResult(
            error_message=("Failed cuz this is the failure validator"),
            fix_value="FIXED",
        )


def test_failure_mode():
    guard = Guard().use(FailureValidator, on_fail="fix")
    res = guard.parse("hi")
    assert res.validated_output == "FIXED"
    assert res.validation_passed  # Should this even be true though?
