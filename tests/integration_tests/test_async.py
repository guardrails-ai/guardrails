import pytest

from guardrails import AsyncGuard
from guardrails.utils import docs_utils
from guardrails.classes.llm.llm_response import LLMResponse
from tests.integration_tests.test_assets.custom_llm import mock_async_llm
from tests.integration_tests.test_assets.fixtures import (  # noqa
    fixture_llm_output,
    fixture_rail_spec,
    fixture_validated_output,
)

from .mock_llm_outputs import entity_extraction


@pytest.mark.asyncio
async def test_entity_extraction_with_reask(mocker):
    """Test that the entity extraction works with re-asking."""
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncArbitraryCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            # TODO: Re-enable once field level reasking is supported
            # output=entity_extraction.LLM_OUTPUT_REASK,
            output=entity_extraction.LLM_OUTPUT_FULL_REASK,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    content = docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = AsyncGuard.for_rail_string(entity_extraction.RAIL_SPEC_WITH_REASK)

    final_output = await guard(
        llm_api=mock_async_llm,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validation_passed is True
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2

    guard_history = guard.history
    call = guard_history.first

    # Check that the guard was only called once and
    # has the correct number of re-asks.
    assert guard_history.length == 1
    assert call.iterations.length == 2

    # For orginal prompt and output
    first = call.iterations.first
    assert (
        first.inputs.messages[0]["content"]._source == entity_extraction.NON_OPENAI_COMPILED_PROMPT
    )
    # Same as above
    assert call.compiled_messages[0]["content"] == entity_extraction.NON_OPENAI_COMPILED_PROMPT
    assert first.prompt_tokens_consumed == 123
    assert first.completion_tokens_consumed == 1234
    assert first.raw_output == entity_extraction.LLM_OUTPUT
    assert first.validation_response == entity_extraction.VALIDATED_OUTPUT_REASK_1

    # For re-asked prompt and output
    final = call.iterations.last
    assert (
        final.inputs.messages[1]["content"]._source
        == entity_extraction.NON_OPENAI_COMPILED_PROMPT_REASK
    )
    # Same as above
    assert (
        call.reask_messages[0][1]["content"] == entity_extraction.NON_OPENAI_COMPILED_PROMPT_REASK
    )

    # TODO: Re-enable once field level reasking is supported
    # assert final.raw_output == entity_extraction.LLM_OUTPUT_REASK
    assert final.raw_output == entity_extraction.LLM_OUTPUT_FULL_REASK
    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_REASK_2


@pytest.mark.asyncio
async def test_entity_extraction_with_noop(mocker):
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncArbitraryCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]
    content = docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = AsyncGuard.for_rail_string(entity_extraction.RAIL_SPEC_WITH_NOOP)
    final_output = await guard(
        llm_api=mock_async_llm,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.

    # Old assertion which is wrong
    # This should not pass validation and therefore will not have a validated output
    # assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_NOOP

    assert final_output.validation_passed is False
    assert final_output.validated_output is not None
    assert final_output.validated_output["fees"]
    assert final_output.validated_output["interest_rates"]

    call = guard.history.first

    # Check that the guard was called once
    # and did not have to reask
    assert guard.history.length == 1
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.NON_OPENAI_COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.validation_response == entity_extraction.VALIDATED_OUTPUT_NOOP


@pytest.mark.asyncio
async def test_entity_extraction_with_noop_pydantic(mocker):
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncArbitraryCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]
    content = docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = AsyncGuard.for_pydantic(
        entity_extraction.PYDANTIC_RAIL_WITH_NOOP,
        messages=[
            {
                "role": "user",
                "content": entity_extraction.PYDANTIC_PROMPT,
            }
        ],
    )
    final_output = await guard(
        llm_api=mock_async_llm,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validation_passed is False
    assert final_output.validated_output is not None
    assert final_output.validated_output["fees"]
    assert final_output.validated_output["interest_rates"]

    call = guard.history.first

    # Check that the guard was called once
    # and did not have toreask
    assert guard.history.length == 1
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.NON_OPENAI_COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.validation_response == entity_extraction.VALIDATED_OUTPUT_NOOP


@pytest.mark.asyncio
async def test_entity_extraction_with_filter(mocker):
    """Test that the entity extraction works with re-asking."""
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncArbitraryCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]

    content = docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = AsyncGuard.for_rail_string(entity_extraction.RAIL_SPEC_WITH_FILTER)
    final_output = await guard(
        llm_api=mock_async_llm,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validation_passed is False
    assert final_output.validated_output is None

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert guard.history.length == 1
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.NON_OPENAI_COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.validation_response == entity_extraction.VALIDATED_OUTPUT_FILTER
    assert call.guarded_output is None
    assert call.status == "fail"


@pytest.mark.asyncio
async def test_entity_extraction_with_fix(mocker):
    """Test that the entity extraction works with re-asking."""
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncArbitraryCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]

    content = docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = AsyncGuard.for_rail_string(entity_extraction.RAIL_SPEC_WITH_FIX)
    final_output = await guard(
        llm_api=mock_async_llm,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validation_passed is True
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_FIX

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert guard.history.length == 1

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.NON_OPENAI_COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_FIX


@pytest.mark.asyncio
async def test_entity_extraction_with_refrain(mocker):
    """Test that the entity extraction works with re-asking."""
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.AsyncArbitraryCallable.invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]

    content = docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = AsyncGuard.for_rail_string(entity_extraction.RAIL_SPEC_WITH_REFRAIN)
    final_output = await guard(
        llm_api=mock_async_llm,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )
    # Assertions are made on the guard state object.

    assert final_output.validation_passed is False
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert guard.history.length == 1

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.NON_OPENAI_COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN


@pytest.mark.asyncio
async def test_rail_spec_output_parse(rail_spec, llm_output, validated_output):
    """Test that the rail_spec fixture is working."""
    guard = AsyncGuard.for_rail_string(rail_spec)
    output = await guard.parse(
        llm_output,
        llm_api=mock_async_llm,
    )
    assert output.validated_output == validated_output


@pytest.fixture
def string_rail_spec():
    return """
<rail version="0.1">
<output
  type="string"
  validators="two-words"
  on-fail-two-words="fix"
/>
<prompt>
Hi please make me a string
</prompt>
</rail>
"""


@pytest.fixture
def string_llm_output():
    return "string output yes"


@pytest.fixture
def validated_string_output():
    return "string output"


@pytest.mark.asyncio
async def test_string_rail_spec_output_parse(
    string_rail_spec, string_llm_output, validated_string_output
):
    """Test that the string_rail_spec fixture is working."""
    guard: AsyncGuard = AsyncGuard.for_rail_string(string_rail_spec)
    output = await guard.parse(
        string_llm_output,
        llm_api=mock_async_llm,
        num_reasks=0,
    )
    assert output.validated_output == validated_string_output
