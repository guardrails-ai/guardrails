from typing import Dict

import pytest

import guardrails as gd
from guardrails import register_validator
from guardrails.utils.openai_utils import get_static_openai_chat_create_func
from guardrails.validators import FailResult, ValidationResult

from .mock_llm_outputs import (
    MockArbitraryCallable,
    MockAsyncArbitraryCallable,
    MockOpenAIChatCallable,
)
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

    final_output = guard(
        llm_api=mock_callable,
        prompt_params={"document": pydantic.PARSING_DOCUMENT},
        num_reasks=1,
    )

    assert final_output.validated_output == pydantic.PARSING_EXPECTED_OUTPUT

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

    final_output = await guard(
        llm_api=mock_async_callable,
        prompt_params={"document": pydantic.PARSING_DOCUMENT},
        num_reasks=1,
    )

    assert final_output.validated_output == pydantic.PARSING_EXPECTED_OUTPUT

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


def test_reask_prompt_instructions(mocker):
    """Test that the re-ask prompt and instructions are correct.

    This is done implicitly, since if the incorrect prompt or
    instructions are used, the mock LLM will raise a KeyError.
    """

    mocker.patch(
        "guardrails.llm_providers.OpenAIChatCallable",
        new=MockOpenAIChatCallable,
    )

    @register_validator(name="always_fail", data_type="string")
    def always_fail(value: str, metadata: Dict) -> ValidationResult:
        return FailResult(error_message=f"Value {value} should fail.")

    guard = gd.Guard.from_string(
        validators=[(always_fail, "reask")],
        description="Some description",
    )

    guard.parse(
        llm_output="Tomato Cheese Pizza",
        llm_api=get_static_openai_chat_create_func(),
        msg_history=[
            {"role": "system", "content": "Some content"},
            {"role": "user", "content": "Some prompt"},
        ],
    )
