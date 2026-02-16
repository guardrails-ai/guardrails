from typing import Any, Callable, Dict, Optional, Union

import pytest

from guardrails.classes.validation.validation_result import PassResult
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


def test_multiple_validators():
    # throws value error for multiple validators
    with pytest.raises(ValueError):
        Guard().use(FailureValidator, FailureValidator)


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


@register_validator("mycustominstancecheckvalidator", data_type="string")
class MyValidator(Validator):
    def __init__(
        self,
        an_instance_attr: str,
        on_fail: Optional[Union[Callable, str]] = None,
        **kwargs,
    ):
        self.an_instance_attr = an_instance_attr
        super().__init__(on_fail=on_fail, an_instance_attr=an_instance_attr, **kwargs)

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        return PassResult()


@pytest.mark.parametrize(
    "instance_attr",
    [
        "a",
        object(),
    ],
)
def test_validator_instance_attr_equality(mocker, instance_attr):
    validator = MyValidator(an_instance_attr=instance_attr)

    assert validator.an_instance_attr is instance_attr

    guard = Guard.for_string(
        validators=[validator],
    )

    assert guard._validators[0].an_instance_attr == instance_attr
