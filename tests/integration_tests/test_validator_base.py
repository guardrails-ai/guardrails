from typing import Any, Callable, Dict, Optional, Union

import pytest

from guardrails_ai.types import PassResult
from guardrails.guard import Guard
from guardrails.validator_base import (
    FailResult,
    ValidationResult,
    PassResult,
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


@register_validator("other", "string")
class OtherValidator(Validator):
    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        return FailResult(
            error_message=("Failed cuz this is the other validator"),
            fix_value="OTHER",
        )


# TODO: Add reask tests. Reask is fairly well covered through notebooks
# but it's good to have it here too.
def test_fix():
    guard = Guard().use(FailureValidator(on_fail="fix"))
    res = guard.parse("hi")
    assert res.validated_output == "FIXED"
    assert res.validation_passed  # Should this even be true though?


def test_default_noop():
    guard = Guard().use(FailureValidator(on_fail="noop"))
    res = guard.parse("hi")
    assert res.validated_output == "hi"
    assert not res.validation_passed


def test_filter():
    guard = Guard().use(FailureValidator(on_fail="filter"))
    res = guard.parse("hi")
    assert res.validated_output is None
    assert not res.validation_passed


def test_refrain():
    guard = Guard().use(FailureValidator(on_fail="refrain"))
    res = guard.parse("hi")
    assert res.validated_output is None
    assert not res.validation_passed


def test_exception():
    guard = Guard().use(FailureValidator(on_fail="exception"))
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
    
    
def test_on_fail_refrain_only_triggers_on_fail_result():
    """
    Ensures that on_fail='refrain' only triggers when the validator
    explicitly returns FailResult.
    """

    @register_validator(name="pass-despite-issue", data_type="string")
    def pass_despite_issue(value, metadata):
        return PassResult()

    guard = Guard().use(pass_despite_issue(on_fail="refrain"))
    result = guard.parse("harmful input")

    assert result.validation_passed is True
    assert result.validated_output == "harmful input"
    assert result.error is None


def test_on_fail_refrain_blocks_output_on_fail_result():
    """
    Ensures that on_fail='refrain' correctly suppresses output
    when the validator returns FailResult.
    """

    @register_validator(name="always-fails-test", data_type="string")
    def always_fails(value, metadata):
        return FailResult(error_message="blocked")

    guard = Guard().use(always_fails(on_fail="refrain"))
    result = guard.parse("harmful input")

    assert result.validation_passed is False
    assert result.validated_output is None
