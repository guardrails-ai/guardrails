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


# TODO: Add reask tests. Reask is fairly well covered through notebooks
# but it's good to have it here too.
def test_fix():
    guard = Guard().use(FailureValidator, on_fail="fix")
    res = guard.parse("hi")
    assert res.validated_output == "FIXED"
    assert res.validation_passed  # Should this even be true though?


def test_default_noop():
    guard = Guard().use(FailureValidator, on_fail="noop")
    res = guard.parse("hi")
    assert res.validated_output == "hi"
    assert not res.validation_passed


def test_filter():
    guard = Guard().use(FailureValidator, on_fail="filter")
    res = guard.parse("hi")
    assert res.validated_output is None
    assert not res.validation_passed


def test_refrain():
    guard = Guard().use(FailureValidator, on_fail="refrain")
    res = guard.parse("hi")
    assert res.validated_output is None
    assert not res.validation_passed


def test_exception():
    guard = Guard().use(FailureValidator, on_fail="exception")
    try:
        guard.parse("hi")
    except Exception as e:
        assert "Failed cuz this is the failure validator" in str(e)
    else:
        assert False, "Expected an exception"
