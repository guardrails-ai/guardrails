import openai
import pytest

import guardrails as gd

from .mock_llm_outputs import async_openai_completion_create, openai_completion_create
from .test_assets import pydantic


def test_parsing_reask(mocker):
    """Test re-asking when response is not parseable."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    guard = gd.Guard.from_pydantic(
        output_class=pydantic.PersonalDetails, prompt=pydantic.PARSING_INITIAL_PROMPT
    )
    _, final_output = guard(
        llm_api=openai.Completion.create,
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
        "guardrails.llm_providers.async_openai_wrapper",
        new=async_openai_completion_create,
    )

    guard = gd.Guard.from_pydantic(
        output_class=pydantic.PersonalDetails, prompt=pydantic.PARSING_INITIAL_PROMPT
    )
    raw_output, final_output = await guard(
        llm_api=openai.Completion.acreate,
        prompt_params={"document": pydantic.PARSING_DOCUMENT},
        num_reasks=1,
    )

    print("raw_output: ", raw_output)

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
