import pytest

from guardrails.classes.history.call import Call
from guardrails.classes.history.iteration import Iteration
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.output_type import OutputTypes
from guardrails.llm_providers import AsyncLiteLLMCallable, LiteLLMCallable
from guardrails.run import AsyncRunner, Runner
from guardrails.types.on_fail import OnFailAction

from .test_assets import string
from tests.integration_tests.test_assets.validators.two_words import TwoWords

PROMPT = string.COMPILED_PROMPT
INSTRUCTIONS = """You are a helpful assistant, and you are helping me
     come up with a name for a pizza. ${gr.complete_string_suffix}"""


OUTPUT_SCHEMA = {"type": "string", "description": "Name for the pizza"}
two_words = TwoWords(on_fail=OnFailAction.REASK)
validation_map = {"$": [two_words]}


OUTPUT = "Tomato Cheese Pizza"


def runner_instance(is_sync: bool):
    if is_sync:
        return Runner(
            OutputTypes.STRING,
            output_schema=OUTPUT_SCHEMA,
            num_reasks=0,
            validation_map=validation_map,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": PROMPT},
            ],
            api=LiteLLMCallable,
        )
    else:
        return AsyncRunner(
            OutputTypes.STRING,
            output_schema=OUTPUT_SCHEMA,
            num_reasks=0,
            validation_map=validation_map,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": PROMPT},
            ],
            api=AsyncLiteLLMCallable,
        )


@pytest.mark.asyncio
async def test_sync_async_validate_equivalence(mocker):
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncLiteLLMCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=string.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]

    iteration = Iteration(
        call_id="mock-call",
        index=0,
    )

    parsed_output, _ = runner_instance(True).parse(OUTPUT, OUTPUT_SCHEMA)

    # Call the 'validate' method synchronously
    result_sync = runner_instance(True).validate(iteration, 1, parsed_output, OUTPUT_SCHEMA)

    # Call the 'async_validate' method asynchronously
    result_async = await runner_instance(False).async_validate(
        iteration, 1, parsed_output, OUTPUT_SCHEMA
    )
    assert result_sync == result_async


@pytest.mark.asyncio
async def test_sync_async_step_equivalence(mocker):
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncLiteLLMCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=string.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]

    call_log = Call()

    # Call the 'step' method synchronously
    sync_iteration = runner_instance(True).step(
        1,
        OUTPUT_SCHEMA,
        call_log,
        api=LiteLLMCallable(**{"temperature": 0}),
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": PROMPT},
        ],
        prompt_params={},
        output=OUTPUT,
    )

    # Call the 'async_step' method asynchronously
    async_iteration = await runner_instance(False).async_step(
        1,
        OUTPUT_SCHEMA,
        call_log,
        api=AsyncLiteLLMCallable(**{"temperature": 0}),
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": PROMPT},
        ],
        prompt_params={},
        output=OUTPUT,
    )

    assert sync_iteration.guarded_output == async_iteration.guarded_output
    assert sync_iteration.reasks == async_iteration.reasks
