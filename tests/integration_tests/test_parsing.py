import pytest

import guardrails as gd

from .mock_llm_outputs import MockArbitraryCallable, MockAsyncArbitraryCallable
from .test_assets import pydantic


def test_parsing_reask(mocker):
    """Test re-asking when response is not parseable."""
    mocker.patch(
        "guardrails.llm_providers.ArbitraryCallable", new=MockArbitraryCallable
    )

    guard = gd.Guard.from_pydantic(
        output_class=pydantic.PersonalDetails, prompt=pydantic.PARSING_INITIAL_PROMPT
    )

    def mock_callable(prompt: str):
        return

    _, final_output = guard(
        llm_api=mock_callable,
        prompt_params={"document": pydantic.PARSING_DOCUMENT},
        num_reasks=1,
    )

    assert final_output == pydantic.PARSING_EXPECTED_OUTPUT

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(pydantic.PARSING_COMPILED_PROMPT)
    assert guard_history[0].output == pydantic.PARSING_UNPARSEABLE_LLM_OUTPUT
    assert guard_history[0].validated_output is None

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(pydantic.PARSING_COMPILED_REASK)
    assert guard_history[1].output == pydantic.PARSING_EXPECTED_LLM_OUTPUT
    assert guard_history[1].validated_output == pydantic.PARSING_EXPECTED_OUTPUT


@pytest.mark.asyncio
async def test_async_parsing_reask(mocker):
    """Test re-asking when response is not parseable during async flow."""
    mocker.patch(
        "guardrails.llm_providers.AsyncArbitraryCallable",
        new=MockAsyncArbitraryCallable,
    )

    guard = gd.Guard.from_pydantic(
        output_class=pydantic.PersonalDetails, prompt=pydantic.PARSING_INITIAL_PROMPT
    )

    async def mock_async_callable(prompt: str):
        return

    _, final_output = await guard(
        llm_api=mock_async_callable,
        prompt_params={"document": pydantic.PARSING_DOCUMENT},
        num_reasks=1,
    )

    assert final_output == pydantic.PARSING_EXPECTED_OUTPUT

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(pydantic.PARSING_COMPILED_PROMPT)
    assert guard_history[0].output == pydantic.PARSING_UNPARSEABLE_LLM_OUTPUT
    assert guard_history[0].validated_output is None

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(pydantic.PARSING_COMPILED_REASK)
    assert guard_history[1].output == pydantic.PARSING_EXPECTED_LLM_OUTPUT
    assert guard_history[1].validated_output == pydantic.PARSING_EXPECTED_OUTPUT
