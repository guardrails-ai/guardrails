import json
import os
from typing import Optional, Union

import pytest
from pydantic import BaseModel

import guardrails as gd
from guardrails.guard import Guard
from guardrails.utils.openai_utils import (
    get_static_openai_chat_create_func,
    get_static_openai_create_func,
)
from guardrails.utils.reask_utils import FieldReAsk
from guardrails.validators import FailResult, OneLine

from .mock_llm_outputs import (
    MockOpenAICallable,
    MockOpenAIChatCallable,
    entity_extraction,
)
from .test_assets import pydantic, string


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

${gr.complete_json_suffix}

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


'''def test_rail_spec_output_parse(rail_spec, llm_output, validated_output):
    """Test that the rail_spec fixture is working."""
    guard = gd.Guard.from_rail_string(rail_spec)
    assert guard.parse(llm_output) == validated_output'''


@pytest.mark.parametrize(
    "rail,prompt,test_full_schema_reask",
    [
        (entity_extraction.RAIL_SPEC_WITH_REASK, None, False),
        (entity_extraction.RAIL_SPEC_WITH_REASK, None, True),
        (
            entity_extraction.PYDANTIC_RAIL_WITH_REASK,
            entity_extraction.PYDANTIC_PROMPT,
            False,
        ),
        (
            entity_extraction.PYDANTIC_RAIL_WITH_REASK,
            entity_extraction.PYDANTIC_PROMPT,
            True,
        ),
    ],
)
@pytest.mark.parametrize("multiprocessing_validators", (True, False))
def test_entity_extraction_with_reask(
    mocker, rail, prompt, test_full_schema_reask, multiprocessing_validators
):
    """Test that the entity extraction works with re-asking."""
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)
    mocker.patch(
        "guardrails.validators.Validator.run_in_separate_process",
        new=multiprocessing_validators,
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)

    final_output = guard(
        llm_api=get_static_openai_create_func(),
        prompt_params={"document": content[:6000]},
        num_reasks=1,
        max_tokens=2000,
        full_schema_reask=test_full_schema_reask,
    )

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

    # For reask validator logs
    nested_validator_log = (
        guard_history[0]
        .field_validation_logs.children["fees"]
        .children[1]
        .children["name"]
        .validator_logs[1]
    )

    assert nested_validator_log.value_before_validation == "my chase plan"
    assert nested_validator_log.value_after_validation == FieldReAsk(
        incorrect_value="my chase plan",
        fail_results=[
            FailResult(
                fix_value="my chase",
                error_message="must be exactly two words",
            )
        ],
        path=["fees", 1, "name"],
    )

    # For re-asked prompt and output
    if test_full_schema_reask:
        assert (
            guard_history[1].prompt.source
            == entity_extraction.COMPILED_PROMPT_FULL_REASK
        )
        assert (
            guard_history[1].llm_response.output
            == entity_extraction.LLM_OUTPUT_FULL_REASK
        )
    else:
        assert guard_history[1].prompt.source == entity_extraction.COMPILED_PROMPT_REASK
        assert (
            guard_history[1].llm_response.output == entity_extraction.LLM_OUTPUT_REASK
        )
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
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)
    final_output = guard(
        llm_api=get_static_openai_create_func(),
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
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)
    final_output = guard(
        llm_api=get_static_openai_create_func(),
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


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (entity_extraction.RAIL_SPEC_WITH_FIX, None),
        (entity_extraction.PYDANTIC_RAIL_WITH_FIX, entity_extraction.PYDANTIC_PROMPT),
    ],
)
def test_entity_extraction_with_fix(mocker, rail, prompt):
    """Test that the entity extraction works with re-asking."""
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)
    final_output = guard(
        llm_api=get_static_openai_create_func(),
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
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt)
    final_output = guard(
        llm_api=get_static_openai_create_func(),
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
        "guardrails.llm_providers.OpenAIChatCallable",
        new=MockOpenAIChatCallable,
    )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, prompt, instructions)
    final_output = guard(
        llm_api=get_static_openai_chat_create_func(),
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_FIX

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
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    guard = gd.Guard.from_rail_string(string.RAIL_SPEC_FOR_STRING)
    final_output = guard(
        llm_api=get_static_openai_create_func(),
        prompt_params={"ingredients": "tomato, cheese, sour cream"},
        num_reasks=1,
    )

    assert final_output.validated_output == string.LLM_OUTPUT

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For original prompt and output
    assert guard_history[0].prompt == gd.Prompt(string.COMPILED_PROMPT)
    assert guard_history[0].output == string.LLM_OUTPUT


def test_string_reask(mocker):
    """Test single string (non-JSON) generation with re-asking."""
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    guard = gd.Guard.from_rail_string(string.RAIL_SPEC_FOR_STRING_REASK)
    final_output = guard(
        llm_api=get_static_openai_create_func(),
        prompt_params={"ingredients": "tomato, cheese, sour cream"},
        num_reasks=1,
        max_tokens=100,
    )

    assert final_output.validated_output == string.LLM_OUTPUT_REASK

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
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = gd.Guard.from_rail_string(entity_extraction.RAIL_SPEC_WITH_SKELETON_REASK)
    final_output = guard(
        llm_api=get_static_openai_create_func(),
        prompt_params={"document": content[:6000]},
        max_tokens=1000,
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert (
        final_output.validated_output
        == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_2
    )

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


'''def test_json_output(mocker):
    """Test single string (non-JSON) generation."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    guard = gd.Guard.from_rail_string(string.RAIL_SPEC_FOR_LIST)
    _, final_output, *rest = guard(
        llm_api=get_static_openai_create_func(),
        num_reasks=1,
    )
    assert final_output == string.LIST_LLM_OUTPUT

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 1

    # For original prompt and output
    #assert guard_history[0].prompt == gd.Prompt(string.COMPILED_PROMPT)
    assert guard_history[0].output == string.LLM_OUTPUT

'''


@pytest.mark.parametrize(
    "rail,prompt,instructions,history,llm_api,expected_prompt,"
    "expected_instructions,expected_reask_prompt,expected_reask_instructions",
    [
        (
            entity_extraction.RAIL_SPEC_WITH_REASK_NO_PROMPT,
            entity_extraction.OPTIONAL_PROMPT_COMPLETION_MODEL,
            None,
            None,
            get_static_openai_create_func(),
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
            get_static_openai_chat_create_func(),
            entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS,
            entity_extraction.COMPILED_INSTRUCTIONS,
            entity_extraction.COMPILED_PROMPT_REASK_WITHOUT_INSTRUCTIONS,
            entity_extraction.COMPILED_INSTRUCTIONS_REASK,
        ),
        (
            entity_extraction.RAIL_SPEC_WITH_REASK_NO_PROMPT,
            None,
            None,
            entity_extraction.OPTIONAL_MSG_HISTORY,
            get_static_openai_chat_create_func(),
            None,
            None,
            entity_extraction.COMPILED_PROMPT_REASK_WITHOUT_INSTRUCTIONS,
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
    if llm_api == get_static_openai_create_func():
        mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)
    else:
        mocker.patch(
            "guardrails.llm_providers.OpenAIChatCallable",
            new=MockOpenAIChatCallable,
        )

    content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
    guard = Guard.from_rail_string(rail)

    final_output = guard(
        llm_api=llm_api,
        prompt=prompt,
        instructions=instructions,
        msg_history=history,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    # For orginal prompt and output
    expected_prompt = (
        gd.Prompt(expected_prompt) if expected_prompt is not None else None
    )
    assert guard_history[0].prompt == expected_prompt
    assert guard_history[0].output == entity_extraction.LLM_OUTPUT
    assert (
        guard_history[0].validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_1
    )
    expected_instructions = (
        gd.Instructions(expected_instructions)
        if expected_instructions is not None
        else None
    )
    assert guard_history[0].instructions == expected_instructions

    # For reask validator logs
    nested_validator_log = (
        guard_history[0]
        .field_validation_logs.children["fees"]
        .children[1]
        .children["name"]
        .validator_logs[1]
    )
    assert nested_validator_log.value_before_validation == "my chase plan"
    assert nested_validator_log.value_after_validation == FieldReAsk(
        incorrect_value="my chase plan",
        fail_results=[
            FailResult(
                fix_value="my chase",
                error_message="must be exactly two words",
            )
        ],
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


def test_string_with_message_history_reask(mocker):
    """Test single string (non-JSON) generation with message history and
    reask."""
    mocker.patch(
        "guardrails.llm_providers.OpenAIChatCallable",
        new=MockOpenAIChatCallable,
    )

    guard = gd.Guard.from_rail_string(string.RAIL_SPEC_FOR_MSG_HISTORY)
    final_output = guard(
        llm_api=get_static_openai_chat_create_func(),
        msg_history=string.MOVIE_MSG_HISTORY,
        temperature=0.0,
        model="gpt-3.5-turbo",
    )

    assert final_output.validated_output == string.MSG_LLM_OUTPUT_CORRECT

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    assert guard_history[0].instructions is None
    assert guard_history[0].prompt is None
    assert guard_history[0].output == string.MSG_LLM_OUTPUT_INCORRECT
    assert guard_history[0].validated_output == string.MSG_VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(string.MSG_COMPILED_PROMPT_REASK)
    assert guard_history[1].instructions == gd.Instructions(
        string.MSG_COMPILED_INSTRUCTIONS_REASK
    )
    assert guard_history[1].output == string.MSG_LLM_OUTPUT_CORRECT
    assert guard_history[1].validated_output == string.MSG_LLM_OUTPUT_CORRECT


def test_pydantic_with_message_history_reask(mocker):
    """Test JSON generation with message history re-asking."""
    mocker.patch(
        "guardrails.llm_providers.OpenAIChatCallable",
        new=MockOpenAIChatCallable,
    )

    guard = gd.Guard.from_pydantic(output_class=pydantic.WITH_MSG_HISTORY)
    final_output = guard(
        llm_api=get_static_openai_chat_create_func(),
        msg_history=string.MOVIE_MSG_HISTORY,
        temperature=0.0,
        model="gpt-3.5-turbo",
    )

    assert final_output.raw_llm_output == pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT
    assert final_output.validated_output == json.loads(
        pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT
    )

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 2

    assert guard_history[0].instructions is None
    assert guard_history[0].prompt is None
    assert guard_history[0].output == pydantic.MSG_HISTORY_LLM_OUTPUT_INCORRECT
    assert guard_history[0].validated_output == pydantic.MSG_VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(pydantic.MSG_COMPILED_PROMPT_REASK)
    assert guard_history[1].instructions == gd.Instructions(
        pydantic.MSG_COMPILED_INSTRUCTIONS_REASK
    )
    assert guard_history[1].output == pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT
    assert guard_history[1].validated_output == json.loads(
        pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT
    )


def test_sequential_validator_log_is_not_duplicated(mocker):
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    proc_count_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
    os.environ["GUARDRAILS_PROCESS_COUNT"] = "1"
    try:
        content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
        guard = guard_initializer(
            entity_extraction.PYDANTIC_RAIL_WITH_NOOP, entity_extraction.PYDANTIC_PROMPT
        )

        _, final_output, *rest = guard(
            llm_api=get_static_openai_create_func(),
            prompt_params={"document": content[:6000]},
            num_reasks=1,
        )

        logs = (
            guard.guard_state.most_recent_call.history[0]
            .field_validation_logs.children["fees"]
            .children[0]
            .children["explanation"]
            .validator_logs
        )
        assert len(logs) == 1
        assert logs[0].validator_name == "OneLine"

    finally:
        if proc_count_bak is None:
            del os.environ["GUARDRAILS_PROCESS_COUNT"]
        else:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = proc_count_bak


def test_in_memory_validator_log_is_not_duplicated(mocker):
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    separate_proc_bak = OneLine.run_in_separate_process
    OneLine.run_in_separate_process = False
    try:
        content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
        guard = guard_initializer(
            entity_extraction.PYDANTIC_RAIL_WITH_NOOP, entity_extraction.PYDANTIC_PROMPT
        )

        _, final_output, *rest = guard(
            llm_api=get_static_openai_create_func(),
            prompt_params={"document": content[:6000]},
            num_reasks=1,
        )

        logs = (
            guard.guard_state.most_recent_call.history[0]
            .field_validation_logs.children["fees"]
            .children[0]
            .children["explanation"]
            .validator_logs
        )
        assert len(logs) == 1
        assert logs[0].validator_name == "OneLine"

    finally:
        OneLine.run_in_separate_process = separate_proc_bak
