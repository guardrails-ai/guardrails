from typing import Dict
import openai
import pytest

import guardrails as gd
from guardrails import register_validator
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.validator_base import OnFailAction
from guardrails.classes.validation.validation_result import FailResult, ValidationResult
from tests.integration_tests.test_assets.custom_llm import mock_async_llm, mock_llm

from .test_assets import pydantic, string


def test_parsing_reask(mocker):
    """Test re-asking when response is not parseable."""
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.ArbitraryCallable._invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=pydantic.PARSING_UNPARSEABLE_LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=pydantic.PARSING_EXPECTED_LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    guard = gd.Guard.from_pydantic(
        output_class=pydantic.PersonalDetails, prompt=pydantic.PARSING_INITIAL_PROMPT
    )

    final_output = guard(
        llm_api=mock_llm,
        prompt_params={"document": pydantic.PARSING_DOCUMENT},
        num_reasks=1,
    )

    assert final_output.validated_output == pydantic.PARSING_EXPECTED_OUTPUT

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    assert call.compiled_prompt == pydantic.PARSING_COMPILED_PROMPT
    assert call.iterations.first.raw_output == pydantic.PARSING_UNPARSEABLE_LLM_OUTPUT
    assert call.iterations.first.guarded_output is None

    # For re-asked prompt and output
    assert call.iterations.last.inputs.prompt == gd.Prompt(
        pydantic.PARSING_COMPILED_REASK
    )
    # Same as above
    assert call.reask_prompts.last == pydantic.PARSING_COMPILED_REASK
    assert call.raw_outputs.last == pydantic.PARSING_EXPECTED_LLM_OUTPUT
    assert call.guarded_output == pydantic.PARSING_EXPECTED_OUTPUT


@pytest.mark.asyncio
async def test_async_parsing_reask(mocker):
    """Test re-asking when response is not parseable during async flow."""
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncArbitraryCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=pydantic.PARSING_UNPARSEABLE_LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=pydantic.PARSING_EXPECTED_LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    guard = gd.AsyncGuard.from_pydantic(
        output_class=pydantic.PersonalDetails, prompt=pydantic.PARSING_INITIAL_PROMPT
    )

    final_output = await guard(
        llm_api=mock_async_llm,
        prompt_params={"document": pydantic.PARSING_DOCUMENT},
        num_reasks=1,
    )

    assert final_output.validated_output == pydantic.PARSING_EXPECTED_OUTPUT

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    assert call.compiled_prompt == pydantic.PARSING_COMPILED_PROMPT
    assert call.iterations.first.raw_output == pydantic.PARSING_UNPARSEABLE_LLM_OUTPUT
    assert call.iterations.first.guarded_output is None

    # For re-asked prompt and output

    assert call.iterations.last.inputs.prompt == gd.Prompt(
        pydantic.PARSING_COMPILED_REASK
    )
    # Same as above
    assert call.reask_prompts.last == pydantic.PARSING_COMPILED_REASK
    assert call.raw_outputs.last == pydantic.PARSING_EXPECTED_LLM_OUTPUT
    assert call.guarded_output == pydantic.PARSING_EXPECTED_OUTPUT


def test_reask_prompt_instructions(mocker):
    """Test that the re-ask prompt and instructions are correct.

    This is done implicitly, since if the incorrect prompt or
    instructions are used, the mock LLM will raise a KeyError.
    """

    mocker.patch(
        "guardrails.llm_providers.OpenAIChatCallable._invoke_llm",
        return_value=LLMResponse(
            output=string.MSG_LLM_OUTPUT_CORRECT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    )

    @register_validator(name="always_fail", data_type="string")
    def always_fail(value: str, metadata: Dict) -> ValidationResult:
        return FailResult(error_message=f"Value {value} should fail.")

    # We don't support tuple syntax for from_string and never have
    # Once the validator function is decorated though, it becomes a Validator class
    guard = gd.Guard.from_string(
        validators=[always_fail(OnFailAction.REASK)],
        description="Some description",
    )

    guard.parse(
        llm_output="Tomato Cheese Pizza",
        llm_api=openai.chat.completions.create,
        msg_history=[
            {"role": "system", "content": "Some content"},
            {"role": "user", "content": "Some prompt"},
        ],
    )
