import asyncio

import pytest

from guardrails import AsyncGuard, Guard
from guardrails.classes.rc import RC
from guardrails.settings import settings
from tests.integration_tests.test_assets.validators import ValidChoices


@pytest.mark.parametrize("guard_class", [Guard, AsyncGuard])
@pytest.mark.parametrize(
    "configured_budgets,call_budget,expected_retries",
    [
        ((0,), None, 0),
        ((2, 0), None, 0),
        ((0, None), None, 0),
        ((1,), None, 1),
        ((1, None), None, 1),
        ((2,), 0, 0),
        ((2,), 1, 1),
        ((0,), 1, 1),
    ],
)
def test_configure_controls_parse_retry_budget(
    guard_class, configured_budgets, call_budget, expected_retries, mocker
):
    rc = RC(enable_metrics=False, use_remote_inferencing=False)
    mocker.patch.object(RC, "load", return_value=rc)
    mocker.patch.object(settings, "_rc", rc)
    guard = guard_class().use(ValidChoices(["hello"], on_fail="reask"))
    for budget in configured_budgets:
        guard.configure(num_reasks=budget, allow_metrics_collection=False)

    response = mocker.Mock(return_value="hello")

    def sync_llm(*args, messages=None, **kwargs):
        return response(*args, messages=messages, **kwargs)

    async def async_llm(*args, messages=None, **kwargs):
        return response(*args, messages=messages, **kwargs)

    kwargs = {"messages": [{"role": "user", "content": "Return hello."}]}
    if call_budget is not None:
        kwargs["num_reasks"] = call_budget
    llm = async_llm if guard_class is AsyncGuard else sync_llm
    outcome = guard.parse("HELLO", llm_api=llm, **kwargs)
    if guard_class is AsyncGuard:
        outcome = asyncio.run(outcome)

    assert response.call_count == expected_retries
    assert outcome.validation_passed is (expected_retries > 0)
