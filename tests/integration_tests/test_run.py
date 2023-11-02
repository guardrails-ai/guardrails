import os

import pytest
from lxml import etree as ET

import guardrails as gd
from guardrails.llm_providers import AsyncOpenAICallable, OpenAICallable
from guardrails.run import AsyncRunner, Runner
from guardrails.schema import StringSchema
from guardrails.utils.logs_utils import GuardLogs

from .mock_llm_outputs import MockAsyncOpenAICallable, MockOpenAICallable
from .test_assets import string

PROMPT = gd.Prompt(source=string.COMPILED_PROMPT)
INSTRUCTIONS = gd.Instructions(
    """ You are a helpful assistant, and you are helping me
     come up with a name for a pizza. ${gr.complete_string_suffix}"""
)


OUTPUT_SCHEMA = StringSchema.from_element(
    ET.fromstring(
        """<output
    type="string"
    description="Name for the pizza"
    format="two-words"
    on-fail-two-words="reask" />"""
    )
)

OUTPUT = "Tomato Cheese Pizza"


def runner_instance(is_sync: bool):
    if is_sync:
        return Runner(
            instructions=INSTRUCTIONS,
            prompt=PROMPT,
            msg_history=None,
            api=OpenAICallable,
            input_schema=None,
            output_schema=OUTPUT_SCHEMA,
            guard_state={},
            num_reasks=0,
        )
    else:
        return AsyncRunner(
            instructions=INSTRUCTIONS,
            prompt=PROMPT,
            msg_history=None,
            api=AsyncOpenAICallable,
            input_schema=None,
            output_schema=OUTPUT_SCHEMA,
            guard_state={},
            num_reasks=0,
        )


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None, reason="openai api key not set"
)
@pytest.mark.asyncio
async def test_sync_async_call_equivalence(mocker):
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    # Call the 'call' method synchronously
    result_sync = runner_instance(True).call(
        1,
        INSTRUCTIONS,
        PROMPT,
        None,
        OpenAICallable(**{"temperature": 0}),
        "Tomato Cheese Pizza",
    )

    # Call the 'async_call' method asynchronously
    result_async = await runner_instance(False).async_call(
        index=1,
        instructions=INSTRUCTIONS,
        prompt=PROMPT,
        msg_history=None,
        api=AsyncOpenAICallable(**{"temperature": 0}),
        output="Tomato Cheese Pizza",
    )

    assert result_sync.output == result_async.output


@pytest.mark.asyncio
async def test_sync_async_validate_equivalence(mocker):
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)
    guard_logs = GuardLogs()

    parsed_output, _ = runner_instance(True).parse(1, OUTPUT, OUTPUT_SCHEMA)

    # Call the 'validate' method synchronously
    result_sync = runner_instance(True).validate(
        guard_logs, 1, parsed_output, OUTPUT_SCHEMA
    )

    # Call the 'async_validate' method asynchronously
    result_async = await runner_instance(False).async_validate(
        guard_logs, 1, parsed_output, OUTPUT_SCHEMA
    )
    assert result_sync == result_async


@pytest.mark.asyncio
async def test_sync_async_step_equivalence(mocker):
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    # Call the 'step' method synchronously
    result_sync, reask_sync = runner_instance(True).step(
        1,
        OpenAICallable(**{"temperature": 0}),
        INSTRUCTIONS,
        PROMPT,
        None,
        {},
        None,
        OUTPUT_SCHEMA,
        OUTPUT,
    )

    # Call the 'async_step' method asynchronously
    result_async, reask_async = await runner_instance(False).async_step(
        1,
        AsyncOpenAICallable(**{"temperature": 0}),
        INSTRUCTIONS,
        PROMPT,
        None,
        {},
        None,
        OUTPUT_SCHEMA,
        OUTPUT,
    )

    assert result_sync == result_async
    assert reask_sync == reask_async
