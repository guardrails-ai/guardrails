from unittest.mock import patch

import openai
import pytest

import guardrails as gd
from guardrails.schema import JsonSchema

from .mock_llm_outputs import MockAsyncOpenAICallable, entity_extraction
from .test_guard import *  # noqa: F403, F401


@pytest.mark.asyncio
@pytest.mark.parametrize("multiprocessing_validators", (True, False))
async def test_entity_extraction_with_reask(mocker, multiprocessing_validators: bool):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )
    mocker.patch(
        "guardrails.validators.Validator.run_in_separate_process",
        new=multiprocessing_validators,
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = gd.Guard.from_rail_string(entity_extraction.RAIL_SPEC_WITH_REASK)

    with patch.object(
        JsonSchema, "preprocess_prompt", wraps=guard.output_schema.preprocess_prompt
    ) as mock_preprocess_prompt:
        final_output = await guard(
            llm_api=openai.Completion.acreate,
            prompt_params={"document": content[:6000]},
            num_reasks=1,
        )

        # Check that the preprocess_prompt method was called.
        mock_preprocess_prompt.assert_called()

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].llm_response.prompt_token_count == 123
    assert guard_history[0].llm_response.response_token_count == 1234
    assert guard_history[0].llm_response.output == entity_extraction.LLM_OUTPUT
    assert (
        guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_1
    )

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT_REASK)
    assert guard_history[1].output == entity_extraction.LLM_OUTPUT_REASK
    assert (
        guard_history[1].validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2
    )


@pytest.mark.asyncio
async def test_entity_extraction_with_noop(mocker):
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )
    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = gd.Guard.from_rail_string(entity_extraction.RAIL_SPEC_WITH_NOOP)
    final_output = await guard(
        llm_api=openai.Completion.acreate,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_NOOP

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_NOOP


@pytest.mark.asyncio
async def test_entity_extraction_with_noop_pydantic(mocker):
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )
    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = gd.Guard.from_pydantic(
        entity_extraction.PYDANTIC_RAIL_WITH_NOOP, entity_extraction.PYDANTIC_PROMPT
    )
    final_output = await guard(
        llm_api=openai.Completion.acreate,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_NOOP

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_NOOP


@pytest.mark.asyncio
async def test_entity_extraction_with_filter(mocker):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = gd.Guard.from_rail_string(entity_extraction.RAIL_SPEC_WITH_FILTER)
    final_output = await guard(
        llm_api=openai.Completion.acreate,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_FILTER

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert (
        guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_FILTER
    )


@pytest.mark.asyncio
async def test_entity_extraction_with_fix(mocker):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = gd.Guard.from_rail_string(entity_extraction.RAIL_SPEC_WITH_FIX)
    final_output = await guard(
        llm_api=openai.Completion.acreate,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_FIX

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_FIX


@pytest.mark.asyncio
async def test_entity_extraction_with_refrain(mocker):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.AsyncOpenAICallable",
        new=MockAsyncOpenAICallable,
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = gd.Guard.from_rail_string(entity_extraction.RAIL_SPEC_WITH_REFRAIN)
    final_output = await guard(
        llm_api=openai.Completion.acreate,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )
    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert (
        guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN
    )


@pytest.mark.asyncio
async def test_rail_spec_output_parse(rail_spec, llm_output, validated_output):
    """Test that the rail_spec fixture is working."""
    guard = gd.Guard.from_rail_string(rail_spec)
    output = await guard.parse(
        llm_output,
        llm_api=openai.Completion.acreate,
    )
    assert output.validated_output == validated_output


@pytest.fixture
def string_rail_spec():
    return """
<rail version="0.1">
<output
  type="string"
  format="two-words"
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
    guard = gd.Guard.from_rail_string(string_rail_spec)
    output = await guard.parse(
        string_llm_output,
        llm_api=openai.Completion.acreate,
        num_reasks=0,
    )
    assert output.validated_output == validated_string_output
