from unittest.mock import MagicMock

import pytest

from guardrails import AsyncGuard
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


@pytest.mark.asyncio
async def test_async_stream_server_call_preserves_wire_fields():
    """Regression test for issue #1588 on the AsyncGuard stream path.

    Before the fix, _stream_server_call hand-copied fields from the wire
    object and dropped validation_summaries (and error/reask) entirely,
    unlike the local streaming path.
    """
    guard = AsyncGuard()
    guard._api_client = MagicMock()
    guard._api_client.stream_validate.return_value = iter(
        [None, make_interface_outcome()]
    )

    results = [outcome async for outcome in guard._stream_server_call(payload={})]

    assert len(results) == 2
    assert results[0].error == "The response from the server was empty!"

    outcome = results[1]
    assert isinstance(outcome, ValidationOutcome)
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
