"""The two validator services must agree on validated output.

`GUARDRAILS_RUN_SYNC` selects between `SequentialValidatorService` and
`AsyncValidatorService`. It is a performance switch, so the same `Guard` with
the same validators should produce the same `validated_output` either way.

Issue #1633 reports that it does not: with several validators using
`on_fail="fix"`, the default (async) path discards all but the first fix.

The existing dispatcher tests cannot catch this. They patch both services out
and assert which one was constructed:

    mocker.patch("guardrails.validator_service.SequentialValidatorService")
    mocker.patch("guardrails.validator_service.AsyncValidatorService")
    vs.SequentialValidatorService.assert_called_once_with(True)

That passes identically whether or not the two services agree, because nothing
in it looks at what either service returns. The test below closes that gap: it
runs the real services and compares their output.
"""

import os

import pytest

from guardrails import Guard
from guardrails.classes.validation.validation_result import FailResult, PassResult
from guardrails.validator_base import Validator, register_validator


@register_validator(name="equivalence/append-a", data_type="string")
class AppendA(Validator):
    """Appends "A" once, then passes. Idempotent so a retry cannot skew the result."""

    def validate(self, value, metadata):
        if str(value).endswith("A"):
            return PassResult()
        return FailResult(error_message="needs A", fix_value=f"{value}A")


@register_validator(name="equivalence/append-b", data_type="string")
class AppendB(Validator):
    def validate(self, value, metadata):
        if str(value).endswith("B"):
            return PassResult()
        return FailResult(error_message="needs B", fix_value=f"{value}B")


def _validate_under(run_sync: bool) -> str:
    previous = os.environ.get("GUARDRAILS_RUN_SYNC")
    os.environ["GUARDRAILS_RUN_SYNC"] = "true" if run_sync else "false"
    try:
        guard = Guard().use(AppendA(on_fail="fix"), AppendB(on_fail="fix"))
        return guard.validate("x").validated_output
    finally:
        if previous is None:
            del os.environ["GUARDRAILS_RUN_SYNC"]
        else:
            os.environ["GUARDRAILS_RUN_SYNC"] = previous


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#1633: the async path applies only the first fix. Remove this marker "
        "with the fix - strict=True makes the suite fail once it starts passing, "
        "so the marker cannot outlive the bug."
    ),
)
def test_both_services_produce_the_same_validated_output():
    """Both services see two failing validators, each supplying a fix.

    Whatever the canonical semantics are - apply every fix in order, or apply
    the first and stop - the two services have to reach the same answer, because
    the choice between them is meant to be a performance decision.
    """
    sequential = _validate_under(run_sync=True)
    concurrent = _validate_under(run_sync=False)

    assert sequential == concurrent, (
        f"GUARDRAILS_RUN_SYNC=true gave {sequential!r}, "
        f"GUARDRAILS_RUN_SYNC=false gave {concurrent!r}"
    )


def test_a_single_fix_is_applied_by_both_services():
    """Control. With one validator there is nothing to drop, so this passes
    today and shows the failure above is about combining fixes rather than
    about fixes being ignored entirely."""
    previous = os.environ.get("GUARDRAILS_RUN_SYNC")
    outputs = {}
    try:
        for run_sync in (True, False):
            os.environ["GUARDRAILS_RUN_SYNC"] = "true" if run_sync else "false"
            guard = Guard().use(AppendA(on_fail="fix"))
            outputs[run_sync] = guard.validate("x").validated_output
    finally:
        if previous is None:
            os.environ.pop("GUARDRAILS_RUN_SYNC", None)
        else:
            os.environ["GUARDRAILS_RUN_SYNC"] = previous

    assert outputs[True] == outputs[False] == "xA"
