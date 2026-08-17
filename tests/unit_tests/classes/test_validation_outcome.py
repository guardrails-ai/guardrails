import os
from unittest.mock import MagicMock

import pytest

from guardrails import Guard
from guardrails.classes.validation.validation_summary import ValidationSummary
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails_ai.types.validation_outcome import (
    ValidationOutcome as IValidationOutcome,
)
from guardrails_ai.types.validation_summary import (
    ValidationSummary as IValidationSummary,
)


def make_interface_outcome(**overrides) -> IValidationOutcome:
    defaults = dict(
        callId="call-1",
        rawLlmOutput="raw text",
        validatedOutput="validated text",
        validationPassed=False,
        validationSummaries=[
            IValidationSummary(
                validator_name="DetectPII",
                validator_status="fail",
                property_path="$",
                failure_reason="found PII",
                error_spans=[],
            )
        ],
        error="validator failed",
    )
    defaults.update(overrides)
    return IValidationOutcome(**defaults)


class TestFromInterface:
    # Regression tests for issue #1588: the server-call paths hand-copied
    # fields from the wire object, silently dropping validation summaries
    # (stream path), reask, and error.

    def test_carries_all_wire_fields(self):
        outcome = ValidationOutcome.from_interface(make_interface_outcome())

        assert outcome.call_id == "call-1"
        assert outcome.raw_llm_output == "raw text"
        assert outcome.validated_output == "validated text"
        assert outcome.validation_passed is False
        assert outcome.error == "validator failed"
        assert len(outcome.validation_summaries) == 1
        summary = outcome.validation_summaries[0]
        assert isinstance(summary, ValidationSummary)
        assert summary.validator_name == "DetectPII"
        assert summary.failure_reason == "found PII"

    def test_falsy_validated_output_collapses_to_none(self):
        # Preserves the pre-existing server-path semantics.
        outcome = ValidationOutcome.from_interface(
            make_interface_outcome(validatedOutput="")
        )
        assert outcome.validated_output is None

    def test_validation_passed_none_coerces_to_false(self):
        outcome = ValidationOutcome.from_interface(
            make_interface_outcome(validationPassed=None)
        )
        assert outcome.validation_passed is False

    def test_missing_summaries_yield_empty_list(self):
        outcome = ValidationOutcome.from_interface(
            make_interface_outcome(validationSummaries=None)
        )
        assert outcome.validation_summaries == []


class TestServerCallsUseSharedAdapter:
    # Regression tests for issue #1588 at the Guard server-path level.

    @pytest.fixture(autouse=True)
    def disable_guard_history(self, monkeypatch):
        monkeypatch.setitem(os.environ, "GUARD_HISTORY_ENABLED", "false")

    def _make_server_guard(self):
        guard = Guard()
        guard._use_server = True
        guard._api_client = MagicMock()
        return guard

    def test_stream_server_call_preserves_summaries_and_error(self):
        # issue #1588: streaming through the server lost the per-validator
        # failure detail that local streaming provides.
        guard = self._make_server_guard()
        guard._api_client.stream_validate.return_value = iter(
            [make_interface_outcome()]
        )

        outcomes = list(guard._stream_server_call(payload={}))

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert len(outcome.validation_summaries) == 1
        assert outcome.validation_summaries[0].validator_name == "DetectPII"
        assert outcome.error == "validator failed"
        assert outcome.validation_passed is False

    def test_single_server_call_preserves_summaries_and_error(self):
        # Adjacent path: _single_server_call previously carried summaries but
        # still hand-copied fields, dropping error (and reask) on the floor.
        guard = self._make_server_guard()
        guard._api_client.validate.return_value = make_interface_outcome()

        outcome = guard._single_server_call(payload={})

        assert len(outcome.validation_summaries) == 1
        assert outcome.validation_summaries[0].failure_reason == "found PII"
        assert outcome.error == "validator failed"
