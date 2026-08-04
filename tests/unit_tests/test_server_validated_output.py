import os
from unittest.mock import MagicMock

import pytest

from guardrails import AsyncGuard, Guard
from guardrails_ai.types import ValidationOutcome as IValidationOutcome


FALSY_OUTPUTS = ["", [], {}]


def interface_outcome(validated_output):
    return IValidationOutcome(
        callId="call-1",
        rawLlmOutput="raw output",
        validatedOutput=validated_output,
        validationPassed=True,
    )


@pytest.fixture(autouse=True)
def disable_guard_history(monkeypatch):
    monkeypatch.setitem(os.environ, "GUARD_HISTORY_ENABLED", "false")


@pytest.mark.parametrize("validated_output", FALSY_OUTPUTS)
def test_single_server_call_preserves_falsy_validated_output(validated_output):
    guard = Guard()
    guard._use_server = True
    guard._api_client = MagicMock()
    guard._api_client.validate.return_value = interface_outcome(validated_output)

    outcome = guard._single_server_call(payload={})

    assert outcome.validated_output == validated_output
    assert type(outcome.validated_output) is type(validated_output)


@pytest.mark.parametrize("validated_output", FALSY_OUTPUTS)
def test_stream_server_call_preserves_falsy_validated_output(validated_output):
    guard = Guard()
    guard._use_server = True
    guard._api_client = MagicMock()
    guard._api_client.stream_validate.return_value = iter(
        [interface_outcome(validated_output)]
    )

    [outcome] = guard._stream_server_call(payload={})

    assert outcome.validated_output == validated_output
    assert type(outcome.validated_output) is type(validated_output)


@pytest.mark.asyncio
@pytest.mark.parametrize("validated_output", FALSY_OUTPUTS)
async def test_async_stream_server_call_preserves_falsy_validated_output(
    validated_output,
):
    guard = AsyncGuard()
    guard._api_client = MagicMock()
    guard._api_client.stream_validate.return_value = iter(
        [interface_outcome(validated_output)]
    )

    outcomes = [outcome async for outcome in guard._stream_server_call(payload={})]

    assert outcomes[0].validated_output == validated_output
    assert type(outcomes[0].validated_output) is type(validated_output)
