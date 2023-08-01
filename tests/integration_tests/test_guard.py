from typing import Optional, Union

import openai
import pytest
from pydantic import BaseModel

import guardrails as gd
from guardrails.guard import Guard
from guardrails.utils.reask_utils import FieldReAsk

from .mock_llm_outputs import (
    entity_extraction,
    openai_chat_completion_create,
    openai_completion_create,
)
from .test_assets import string


@pytest.fixture(scope="module")
def rail_spec():
    return """
<rail version="0.1">

<output>
    <string name="dummy_string" description="Any dummy string" />
    <integer name="dummy_integer" description="Any dummy integer" />
    <float name="dummy_float" description="Any dummy float" />
    <bool name="dummy_boolean" description="Any dummy boolean" />
    <email name="dummy_email" description="Any dummy email" />
    <url name="dummy_url" description="Any dummy url" />
    <date name="dummy_date" description="Any dummy date" />
    <time name="dummy_time" description="Any dummy time" />
    <list name="dummy_list" description="Any dummy list" />
    <object name="dummy_object" description="Any dummy object" />
</output>


<prompt>

Generate a JSON of dummy data, where the data types are specified by the user.

@complete_json_suffix

</prompt>

</rail>
"""


@pytest.fixture(scope="module")
def llm_output():
    return """
{
    "dummy_string": "Some string",
    "dummy_integer": 42,
    "dummy_float": 3.14,
    "dummy_boolean": true,
    "dummy_email": "example@example.com",
    "dummy_url": "https://www.example.com",
    "dummy_date": "2020-01-01",
    "dummy_time": "12:00:00",
    "dummy_list": ["item1", "item2", "item3"],
    "dummy_object": {
        "key1": "value1",
        "key2": "value2"
    }
}
"""


@pytest.fixture(scope="module")
def validated_output():
    return {
        "dummy_string": "Some string",
        "dummy_integer": 42,
        "dummy_float": 3.14,
        "dummy_boolean": True,
        "dummy_email": "example@example.com",
        "dummy_url": "https://www.example.com",
        "dummy_date": "2020-01-01",
        "dummy_time": "12:00:00",
        "dummy_list": ["item1", "item2", "item3"],
        "dummy_object": {"key1": "value1", "key2": "value2"},
    }


def guard_initializer(
    rail: Union[str, BaseModel], prompt: str, instructions: Optional[str] = None
) -> Guard:
    """Helper function to initialize a Guard object using the correct
    method."""

    if isinstance(rail, str):
        return Guard.from_rail_string(rail)
    else:
        return Guard.from_pydantic(rail, prompt=prompt, instructions=instructions)


def test_rail_spec_output_parse(rail_spec, llm_output, validated_output):
    """Test that the rail_spec fixture is working."""
    guard = gd.Guard.from_rail_string(rail_spec)
    assert guard.parse(llm_output) == validated_output


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (entity_extraction.RAIL_SPEC_WITH_REASK, None),
        (entity_extraction.PYDANTIC_RAIL_WITH_REASK, entity_extraction.PYDANTIC_PROMPT),
    ],
)
def test_entity_extraction_with_reask(mocker, rail, prompt):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)

    _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output == entity_extraction.VALIDATED_OUTPUT_REASK_2

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert (
        guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_1
    )

    # For reask validator logs
    nested_validator_log = (
        guard_history[0]
        .field_validation_logs["fees"]
        .children[1]
        .children["name"]
        .validator_logs[1]
    )
    assert nested_validator_log.value_before_validation == "my chase plan"
    assert nested_validator_log.value_after_validation == FieldReAsk(
        incorrect_value="my chase plan",
        fix_value="my chase",
        error_message="must be exactly two words",
        path=["fees", 1, "name"],
    )

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT_REASK)
    assert guard_history[1].output == entity_extraction.LLM_OUTPUT_REASK
    assert (
        guard_history[1].validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2
    )


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (entity_extraction.RAIL_SPEC_WITH_NOOP, None),
        (entity_extraction.PYDANTIC_RAIL_WITH_NOOP, entity_extraction.PYDANTIC_PROMPT),
    ],
)
def test_entity_extraction_with_noop(mocker, rail, prompt):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)
    _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output == entity_extraction.VALIDATED_OUTPUT_NOOP

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_NOOP


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (entity_extraction.RAIL_SPEC_WITH_FILTER, None),
        (
            entity_extraction.PYDANTIC_RAIL_WITH_FILTER,
            entity_extraction.PYDANTIC_PROMPT,
        ),
    ],
)
def test_entity_extraction_with_filter(mocker, rail, prompt):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)
    _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output == entity_extraction.VALIDATED_OUTPUT_FILTER

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert (
        guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_FILTER
    )


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (entity_extraction.RAIL_SPEC_WITH_FIX, None),
        (entity_extraction.PYDANTIC_RAIL_WITH_FIX, entity_extraction.PYDANTIC_PROMPT),
    ],
)
def test_entity_extraction_with_fix(mocker, rail, prompt):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)
    _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output == entity_extraction.VALIDATED_OUTPUT_FIX

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_FIX


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (entity_extraction.RAIL_SPEC_WITH_REFRAIN, None),
        (
            entity_extraction.PYDANTIC_RAIL_WITH_REFRAIN,
            entity_extraction.PYDANTIC_PROMPT,
        ),
    ],
)
def test_entity_extraction_with_refrain(mocker, rail, prompt):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)
    _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(entity_extraction.COMPILED_PROMPT)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert (
        guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN
    )


@pytest.mark.parametrize(
    "rail,prompt,instructions",
    [
        (entity_extraction.RAIL_SPEC_WITH_FIX_CHAT_MODEL, None, None),
        (
            entity_extraction.PYDANTIC_RAIL_WITH_FIX,
            entity_extraction.PYDANTIC_PROMPT_CHAT_MODEL,
            entity_extraction.PYDANTIC_INSTRUCTIONS_CHAT_MODEL,
        ),
    ],
)
def test_entity_extraction_with_fix_chat_models(mocker, rail, prompt, instructions):
    """Test that the entity extraction works with fix for chat models."""

    mocker.patch(
        "guardrails.llm_providers.openai_chat_wrapper",
        new=openai_chat_completion_create,
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt, instructions)
    _, final_output = guard(
        llm_api=openai.ChatCompletion.create,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output == entity_extraction.VALIDATED_OUTPUT_FIX

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(
        entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS
    )
    assert guard_history[0].instructions == gd.Instructions(
        entity_extraction.COMPILED_INSTRUCTIONS
    )
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_FIX


def test_string_output(mocker):
    """Test single string (non-JSON) generation."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    guard = gd.Guard.from_rail_string(string.RAIL_SPEC_FOR_STRING)
    _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"ingredients": "tomato, cheese, sour cream"},
        num_reasks=1,
    )

    assert final_output == string.LLM_OUTPUT

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For original prompt and output
    assert guard_history[0].prompt == gd.Prompt(string.COMPILED_PROMPT)
    assert guard_history[0].output == string.LLM_OUTPUT


def test_string_reask(mocker):
    """Test single string (non-JSON) generation with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    guard = gd.Guard.from_rail_string(string.RAIL_SPEC_FOR_STRING_REASK)
    _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"ingredients": "tomato, cheese, sour cream"},
        num_reasks=1,
        max_tokens=100,
    )

    assert final_output == string.LLM_OUTPUT_REASK

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    assert guard_history[0].instructions == gd.Instructions(
        string.COMPILED_INSTRUCTIONS
    )
    assert guard_history[0].prompt == gd.Prompt(string.COMPILED_PROMPT)
    assert guard_history[0].output == string.LLM_OUTPUT
    assert guard_history[0].validated_output == string.VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(string.COMPILED_PROMPT_REASK)
    assert guard_history[1].output == string.LLM_OUTPUT_REASK
    assert guard_history[1].validated_output == string.LLM_OUTPUT_REASK


def test_skeleton_reask(mocker):
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = gd.Guard.from_rail_string(entity_extraction.RAIL_SPEC_WITH_SKELETON_REASK)
    _, final_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"document": content[:6000]},
        max_tokens=1000,
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_2

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(
        entity_extraction.COMPILED_PROMPT_SKELETON_REASK_1
    )
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT_SKELETON_REASK_1
    assert (
        guard_history[0].validated_output
        == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_1
    )

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(
        entity_extraction.COMPILED_PROMPT_SKELETON_REASK_2
    )
    assert guard_history[1].output == entity_extraction.LLM_OUTPUT_SKELETON_REASK_2
    assert (
        guard_history[1].validated_output
        == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_2
    )


@pytest.mark.parametrize(
    "rail,prompt,instructions,history,llm_api,expected_prompt,"
    "expected_instructions,expected_reask_prompt,expected_reask_instructions",
    [
        (
            entity_extraction.RAIL_SPEC_WITH_REASK_NO_PROMPT,
            entity_extraction.OPTIONAL_PROMPT_COMPLETION_MODEL,
            None,
            None,
            openai.Completion.create,
            entity_extraction.COMPILED_PROMPT,
            None,
            entity_extraction.COMPILED_PROMPT_REASK,
            None,
        ),
        (
            entity_extraction.RAIL_SPEC_WITH_REASK_NO_PROMPT,
            entity_extraction.OPTIONAL_PROMPT_CHAT_MODEL,
            entity_extraction.OPTIONAL_INSTRUCTIONS_CHAT_MODEL,
            None,
            openai.ChatCompletion.create,
            entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS,
            entity_extraction.COMPILED_INSTRUCTIONS,
            entity_extraction.COMPILED_PROMPT_REASK,
            entity_extraction.COMPILED_INSTRUCTIONS_REASK,
        ),
    ],
)
def test_entity_extraction_with_reask_with_optional_prompts(
    mocker,
    rail,
    prompt,
    instructions,
    history,
    llm_api,
    expected_prompt,
    expected_instructions,
    expected_reask_prompt,
    expected_reask_instructions,
):
    """Test that the entity extraction works with re-asking."""
    if llm_api == openai.Completion.create:
        mocker.patch(
            "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
        )
    else:
        mocker.patch(
            "guardrails.llm_providers.openai_chat_wrapper",
            new=openai_chat_completion_create,
        )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    # guard = guard_initializer(rail, prompt)
    guard = Guard.from_rail_string(rail)

    _, final_output = guard(
        llm_api=llm_api,
        prompt=prompt,
        instructions=instructions,
        chat_history=history,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output == entity_extraction.VALIDATED_OUTPUT_REASK_2

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(expected_prompt)
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert (
        guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_1
    )
    if expected_instructions:
        assert guard_history[0].instructions == gd.Instructions(expected_instructions)

    # For reask validator logs
    nested_validator_log = (
        guard_history[0]
        .field_validation_logs["fees"]
        .children[1]
        .children["name"]
        .validator_logs[1]
    )
    assert nested_validator_log.value_before_validation == "my chase plan"
    assert nested_validator_log.value_after_validation == FieldReAsk(
        incorrect_value="my chase plan",
        fix_value="my chase",
        error_message="must be exactly two words",
        path=["fees", 1, "name"],
    )

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(expected_reask_prompt)
    assert guard_history[1].output == entity_extraction.LLM_OUTPUT_REASK
    assert (
        guard_history[1].validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2
    )
    if expected_reask_instructions:
        assert guard_history[1].instructions == gd.Instructions(
            expected_reask_instructions
        )
