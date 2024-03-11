import enum
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
    lists_object,
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
    """Mock LLM output for the rail_spec."""
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
    """Mock validated output for the rail_spec."""
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
    """Helper function to initialize a Guard using the correct method."""

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
@pytest.mark.parametrize("multiprocessing_validators", (True,))  # False))
def test_entity_extraction_with_reask(
    mocker, rail, prompt, test_full_schema_reask, multiprocessing_validators
):
    """Test that the entity extraction works with re-asking.

    This test creates a Guard for the entity extraction use case. It
    performs a single call to the LLM and then re-asks the LLM for a
    second time.
    """
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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    first = call.iterations.first
    assert call.compiled_prompt == entity_extraction.COMPILED_PROMPT
    assert first.prompt_tokens_consumed == 123
    assert first.completion_tokens_consumed == 1234
    assert first.raw_output == entity_extraction.LLM_OUTPUT
    assert first.validation_output == entity_extraction.VALIDATED_OUTPUT_REASK_1

    # For reask validator logs
    # TODO: Update once we add json_path to the ValidatorLog class
    nested_validator_logs = list(
        x for x in first.validator_logs if x.value_before_validation == "my chase plan"
    )
    nested_validator_log = nested_validator_logs[1]

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
    # second = call.iterations.at(1)
    if test_full_schema_reask:
        assert (
            # second.inputs.prompt.source # Also valid
            call.reask_prompts.first
            == entity_extraction.COMPILED_PROMPT_FULL_REASK
        )
        assert (
            # second.raw_output # Also valid
            call.raw_outputs.at(1)
            == entity_extraction.LLM_OUTPUT_FULL_REASK
        )
    else:
        # Second iteration is the first reask
        assert call.reask_prompts.first == entity_extraction.COMPILED_PROMPT_REASK
        assert call.raw_outputs.at(1) == entity_extraction.LLM_OUTPUT_REASK
    assert call.validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2


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
    assert final_output.validation_passed is False
    assert final_output.validated_output is None

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_prompt == entity_extraction.COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.validated_output is None
    assert call.validation_output == entity_extraction.VALIDATED_OUTPUT_NOOP


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
    assert final_output.validation_passed is False
    assert final_output.validated_output is None

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_prompt == entity_extraction.COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.status == "fail"
    assert call.validated_output is None


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_prompt == entity_extraction.COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.validated_output == entity_extraction.VALIDATED_OUTPUT_FIX


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_prompt == entity_extraction.COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.validated_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert (
        call.compiled_prompt == entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS
    )
    assert call.compiled_instructions == entity_extraction.COMPILED_INSTRUCTIONS
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.validated_output == entity_extraction.VALIDATED_OUTPUT_FIX


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For original prompt and output
    assert call.compiled_prompt == string.COMPILED_PROMPT
    assert call.raw_outputs.last == string.LLM_OUTPUT


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    assert call.compiled_instructions == string.COMPILED_INSTRUCTIONS
    assert call.compiled_prompt == string.COMPILED_PROMPT
    assert call.iterations.first.raw_output == string.LLM_OUTPUT
    assert call.iterations.first.validation_output == string.VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert call.iterations.last.inputs.prompt == gd.Prompt(string.COMPILED_PROMPT_REASK)
    # Same thing as above
    assert call.reask_prompts.last == string.COMPILED_PROMPT_REASK

    assert call.raw_outputs.last == string.LLM_OUTPUT_REASK
    assert call.validated_output == string.LLM_OUTPUT_REASK


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    assert call.compiled_prompt == entity_extraction.COMPILED_PROMPT_SKELETON_REASK_1
    assert (
        call.iterations.first.raw_output
        == entity_extraction.LLM_OUTPUT_SKELETON_REASK_1
    )
    assert (
        call.iterations.first.validation_output
        == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_1
    )

    # For re-asked prompt and output
    assert call.reask_prompts.last == entity_extraction.COMPILED_PROMPT_SKELETON_REASK_2
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT_SKELETON_REASK_2
    assert call.validated_output == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_2


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For original prompt and output
    #assert call.compiled_prompt == string.COMPILED_PROMPT
    assert call.raw_outputs.last == string.LLM_OUTPUT

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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    assert call.compiled_prompt == expected_prompt
    assert call.iterations.first.raw_output == entity_extraction.LLM_OUTPUT
    assert (
        call.iterations.first.validation_output
        == entity_extraction.VALIDATED_OUTPUT_REASK_1
    )
    assert call.compiled_instructions == expected_instructions

    # For reask validator logs
    # TODO: Update once we add json_path to the ValidatorLog class
    nested_validator_logs = list(
        x
        for x in call.iterations.first.validator_logs
        if x.value_before_validation == "my chase plan"
    )
    nested_validator_log = nested_validator_logs[1]

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
    assert call.reask_prompts.last == expected_reask_prompt
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT_REASK

    assert call.validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2
    if expected_reask_instructions:
        assert call.reask_instructions.last == expected_reask_instructions


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    assert call.compiled_instructions is None
    assert call.compiled_prompt is None
    assert call.iterations.first.raw_output == string.MSG_LLM_OUTPUT_INCORRECT
    assert call.iterations.first.validation_output == string.MSG_VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert call.reask_prompts.last == string.MSG_COMPILED_PROMPT_REASK
    assert call.reask_instructions.last == string.MSG_COMPILED_INSTRUCTIONS_REASK
    assert call.raw_outputs.last == string.MSG_LLM_OUTPUT_CORRECT
    assert call.validated_output == string.MSG_LLM_OUTPUT_CORRECT


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

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    assert call.compiled_instructions is None
    assert call.compiled_prompt is None
    assert call.iterations.first.raw_output == pydantic.MSG_HISTORY_LLM_OUTPUT_INCORRECT
    assert (
        call.iterations.first.validation_output == pydantic.MSG_VALIDATED_OUTPUT_REASK
    )

    # For re-asked prompt and output
    assert call.reask_prompts.last == pydantic.MSG_COMPILED_PROMPT_REASK
    assert call.reask_instructions.last == pydantic.MSG_COMPILED_INSTRUCTIONS_REASK
    assert call.raw_outputs.last == pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT
    assert call.validated_output == json.loads(pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT)


def test_sequential_validator_log_is_not_duplicated(mocker):
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    proc_count_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
    os.environ["GUARDRAILS_PROCESS_COUNT"] = "1"
    try:
        content = gd.docs_utils.read_pdf("docs/examples/data/chase_card_agreement.pdf")
        guard = guard_initializer(
            entity_extraction.PYDANTIC_RAIL_WITH_NOOP, entity_extraction.PYDANTIC_PROMPT
        )

        guard(
            llm_api=get_static_openai_create_func(),
            prompt_params={"document": content[:6000]},
            num_reasks=1,
        )

        # Assert one log per field validation
        # In this case, the OneLine validator should be run once per fee entry
        # because of the explanation field
        one_line_logs = list(
            x
            for x in guard.history.first.iterations.first.validator_logs
            if x.validator_name == "OneLine"
        )
        assert len(one_line_logs) == len(
            guard.history.first.validation_output.get("fees")
        )

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

        guard(
            llm_api=get_static_openai_create_func(),
            prompt_params={"document": content[:6000]},
            num_reasks=1,
        )

        one_line_logs = list(
            x
            for x in guard.history.first.iterations.first.validator_logs
            if x.validator_name == "OneLine"
        )

        assert len(one_line_logs) == len(
            guard.history.first.validation_output.get("fees")
        )

    finally:
        OneLine.run_in_separate_process = separate_proc_bak


def test_enum_datatype(mocker):
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    class TaskStatus(enum.Enum):
        not_started = "not started"
        on_hold = "on hold"
        in_progress = "in progress"

    class Task(BaseModel):
        status: TaskStatus

    guard = gd.Guard.from_pydantic(Task)
    _, dict_o, *rest = guard(
        get_static_openai_create_func(),
        prompt="What is the status of this task?",
    )
    assert dict_o == {"status": "not started"}

    guard = gd.Guard.from_pydantic(Task)
    with pytest.raises(ValueError) as excinfo:
        guard(
            get_static_openai_create_func(),
            prompt="What is the status of this task REALLY?",
        )

    assert str(excinfo.value).startswith("Invalid enum value") is True


@pytest.mark.parametrize(
    "output,throws",
    [
        ("Ice cream is frozen.", False),
        ("Ice cream is a frozen dairy product that is consumed in many places.", True),
        ("This response isn't relevant.", True),
    ],
)
def test_guard_as_runnable(output: str, throws: bool):
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableConfig

    from guardrails.errors import ValidationError
    from guardrails.validators import ReadingTime, RegexMatch

    class MockModel(Runnable):
        def invoke(
            self, input: LanguageModelInput, config: Optional[RunnableConfig] = None
        ) -> BaseMessage:
            return AIMessage(content=output)

    prompt = ChatPromptTemplate.from_template("ELIF: {topic}")
    model = MockModel()
    guard = (
        Guard()
        .use(RegexMatch("Ice cream", match_type="search"))
        .use(ReadingTime(0.05))  # 3 seconds
    )
    output_parser = StrOutputParser()

    chain = prompt | model | guard | output_parser

    topic = "ice cream"
    if throws:
        with pytest.raises(ValidationError) as exc_info:
            chain.invoke({"topic": topic})

        assert str(exc_info.value) == (
            "The response from the LLM failed validation!"
            "See `guard.history` for more details."
        )

        assert guard.history.last.status == "fail"
        assert guard.history.last.status == "fail"

    else:
        result = chain.invoke({"topic": topic})

        assert result == output


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (
            lists_object.PYDANTIC_RAIL_WITH_LIST,
            "Create a list of items that may be found in a grocery store.",
        ),
        (lists_object.RAIL_SPEC_WITH_LIST, None),
    ],
)
def test_guard_with_top_level_list_return_type(mocker, rail, prompt):
    # Create a Guard with a top level list return type

    # Mock the LLM
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    guard = guard_initializer(rail, prompt=prompt)

    output = guard(llm_api=get_static_openai_create_func())

    # Validate the output
    assert output.validated_output == [
        {"name": "apple", "price": 1.0},
        {"name": "banana", "price": 0.5},
        {"name": "orange", "price": 1.5},
    ]
