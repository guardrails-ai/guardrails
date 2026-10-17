import json
from typing import List, Optional

import pytest
from pydantic import BaseModel

from guardrails import AsyncGuard, Guard


class NullableOutput(BaseModel):
    note: Optional[str]
    values: List[Optional[int]]


NULLABLE_OUTPUT = {"note": None, "values": [1, None, 2]}


@pytest.mark.parametrize("run_sync", ["true", "false"])
def test_guard_preserves_nullable_fields_and_list_items(monkeypatch, run_sync):
    monkeypatch.setenv("GUARDRAILS_RUN_SYNC", run_sync)
    guard = Guard.for_pydantic(output_class=NullableOutput)

    outcome = guard.parse(json.dumps(NULLABLE_OUTPUT), num_reasks=0)

    assert outcome.validation_passed
    assert outcome.validated_output == NULLABLE_OUTPUT
    NullableOutput.model_validate(outcome.validated_output)


@pytest.mark.asyncio
async def test_async_guard_preserves_nullable_fields_and_list_items():
    guard = AsyncGuard.for_pydantic(output_class=NullableOutput)

    outcome = await guard.parse(json.dumps(NULLABLE_OUTPUT), num_reasks=0)

    assert outcome.validation_passed
    assert outcome.validated_output == NULLABLE_OUTPUT
    NullableOutput.model_validate(outcome.validated_output)
