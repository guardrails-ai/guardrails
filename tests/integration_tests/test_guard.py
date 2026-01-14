import enum
import importlib
import json
import os
from typing import Dict, List, Optional, Union

import pytest
from pydantic import BaseModel, Field
from guardrails_api_client import Guard as IGuard

import guardrails as gd
from guardrails.actions.reask import SkeletonReAsk
from guardrails.classes.generic.stack import Stack
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.classes.validation.validation_result import FailResult
from guardrails.classes.validation.validator_reference import ValidatorReference
from guardrails.guard import Guard
from guardrails.actions.reask import FieldReAsk
from tests.integration_tests.test_assets.validators import (
    RegexMatch,
    ValidLength,
    ValidChoices,
    LowerCase,
    OneLine,
)

from .mock_llm_outputs import (
    MockLiteLLMCallableOther,
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


def guard_initializer(rail: Union[str, BaseModel], messages=None) -> Guard:
    """Helper function to initialize a Guard using the correct method."""

    if isinstance(rail, str):
        return Guard.for_rail_string(rail)
    else:
        return Guard.for_pydantic(rail, messages=messages)


'''def test_rail_spec_output_parse(rail_spec, llm_output, validated_output):
    """Test that the rail_spec fixture is working."""
    guard = gd.Guard.for_rail_string(rail_spec)
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
    """Test that the entity extraction works with re-asking.

    This test creates a Guard for the entity extraction use case. It
    performs a single call to the LLM and then re-asks the LLM for a
    second time.
    """
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")
    second_response = (
        entity_extraction.LLM_OUTPUT_FULL_REASK
        if test_full_schema_reask
        else json.dumps(entity_extraction.VALIDATED_OUTPUT_REASK_2)
        # FIXME: Use this once field level reask schemas are implemented
        # else entity_extraction.LLM_OUTPUT_REASK
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=second_response,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]
    mocker.patch(
        "guardrails.validators.Validator.run_in_separate_process",
        new=multiprocessing_validators,
    )

    content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, messages=[{"role": "user", "content": prompt}])

    final_output: ValidationOutcome = guard(
        model="gpt-3.5-turbo",
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
    assert call.compiled_messages[0]["content"] == entity_extraction.COMPILED_PROMPT
    assert first.prompt_tokens_consumed == 123
    assert first.completion_tokens_consumed == 1234
    assert first.raw_output == entity_extraction.LLM_OUTPUT
    assert first.validation_response == entity_extraction.VALIDATED_OUTPUT_REASK_1

    # For reask validator logs
    two_words_validator_logs = list(
        x
        for x in first.validator_logs
        if x.property_path == "$.fees.1.name" and x.registered_name == "two-words"
    )

    two_words_validator_log = two_words_validator_logs[0]

    assert two_words_validator_log.value_before_validation == "my chase plan"

    expected_value_after_validation = FieldReAsk(
        incorrect_value="my chase plan",
        fail_results=[
            FailResult(
                fix_value="my chase",
                error_message="must be exactly two words",
            )
        ],
        path=["fees", 1, "name"],
    )
    assert two_words_validator_log.value_after_validation == expected_value_after_validation

    # For re-asked prompt and output
    # second = call.iterations.at(1)
    if test_full_schema_reask:
        assert (
            # second.inputs.prompt.source # Also valid
            call.reask_messages.first[1]["content"] == entity_extraction.COMPILED_PROMPT_FULL_REASK
        )
        assert (
            # second.raw_output # Also valid
            call.raw_outputs.at(1) == entity_extraction.LLM_OUTPUT_FULL_REASK
        )
    else:
        # Second iteration is the first reask
        assert call.reask_messages.first[1]["content"] == entity_extraction.COMPILED_PROMPT_REASK
        # FIXME: Switch back to this once field level reask schema pruning is implemented  # noqa
        # assert call.raw_outputs.at(1) == entity_extraction.LLM_OUTPUT_REASK
        assert call.raw_outputs.at(1) == json.dumps(entity_extraction.VALIDATED_OUTPUT_REASK_2)
    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_REASK_2


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (entity_extraction.RAIL_SPEC_WITH_NOOP, None),
        (entity_extraction.PYDANTIC_RAIL_WITH_NOOP, entity_extraction.PYDANTIC_PROMPT),
    ],
)
def test_entity_extraction_with_noop(mocker, rail, prompt):
    """Test that the entity extraction works with re-asking."""
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")

    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT_FULL_REASK,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, messages=[{"role": "user", "content": prompt}])
    final_output = guard(
        model="gpt-3.5-turbo",
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validation_passed is False
    assert (
        final_output.validated_output is not None
        and validated_output.__get__("fees")
        and validated_output.__get__("interest_rates")
    )

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert (
        call.guarded_output is not None
        and call.guarded_output["fees"]
        and call.guarded_output["interest_rates"]
    )
    assert call.validation_response == entity_extraction.VALIDATED_OUTPUT_NOOP


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
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")

    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT_FULL_REASK,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, messages=[{"role": "user", "content": prompt}])
    final_output = guard(
        model="gpt-3.5-turbo",
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validation_passed is False
    assert final_output.validated_output is None

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For original prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.status == "fail"
    assert call.guarded_output is None


@pytest.mark.parametrize(
    "rail,prompt",
    [
        (entity_extraction.RAIL_SPEC_WITH_FIX, None),
        (entity_extraction.PYDANTIC_RAIL_WITH_FIX, entity_extraction.PYDANTIC_PROMPT),
    ],
)
def test_entity_extraction_with_fix(mocker, rail, prompt):
    """Test that the entity extraction works with re-asking."""
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")

    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT_FULL_REASK,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, messages=[{"role": "user", "content": prompt}])
    final_output = guard(
        model="gpt-3.5-turbo",
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_FIX

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_FIX


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
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")

    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT_FULL_REASK,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, messages=[{"role": "user", "content": prompt}])
    final_output = guard(
        model="gpt-3.5-turbo",
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == entity_extraction.COMPILED_PROMPT
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_REFRAIN


@pytest.mark.parametrize(
    "rail,messages",
    [
        (entity_extraction.RAIL_SPEC_WITH_FIX_CHAT_MODEL, None),
        (
            entity_extraction.PYDANTIC_RAIL_WITH_FIX,
            [
                {
                    "role": "system",
                    "content": entity_extraction.PYDANTIC_INSTRUCTIONS_CHAT_MODEL,
                },
                {
                    "role": "user",
                    "content": entity_extraction.PYDANTIC_PROMPT_CHAT_MODEL,
                },
            ],
        ),
    ],
)
def test_entity_extraction_with_fix_chat_models(mocker, rail, messages):
    """Test that the entity extraction works with fix for chat models."""
    mock_invoke_llm = mocker.patch(
        "guardrails.llm_providers.LiteLLMCallable._invoke_llm",
    )
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]

    content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = guard_initializer(rail, messages)
    final_output = guard(
        model="gpt-3.5-turbo",
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
        call.compiled_messages[1]["content"]
        == entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS
    )
    assert call.compiled_messages[0]["content"] == entity_extraction.COMPILED_INSTRUCTIONS
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT
    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_FIX


'''def test_json_output(mocker):
    """Test single string (non-JSON) generation."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    guard = gd.Guard.for_rail_string(string.RAIL_SPEC_FOR_LIST)
    _, final_output, *rest = guard(
        llm_api=openai.completions.create,
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
    "rail,messages,expected_prompt,"
    "expected_instructions,expected_reask_prompt,expected_reask_instructions,"
    "llm_outputs",
    [
        (
            entity_extraction.RAIL_SPEC_WITH_REASK_NO_PROMPT,
            [
                {
                    "role": "user",
                    "content": entity_extraction.OPTIONAL_PROMPT_COMPLETION_MODEL,
                }
            ],
            entity_extraction.COMPILED_PROMPT,
            None,
            entity_extraction.COMPILED_PROMPT_REASK,
            None,
            [
                entity_extraction.LLM_OUTPUT,
                json.dumps(entity_extraction.VALIDATED_OUTPUT_REASK_2),
                # FIXME: Use this once field level reask schemas are implemented
                # else entity_extraction.LLM_OUTPUT_REASK
            ],
        ),
        (
            entity_extraction.RAIL_SPEC_WITH_REASK_NO_PROMPT,
            [
                {
                    "role": "system",
                    "content": entity_extraction.OPTIONAL_INSTRUCTIONS_CHAT_MODEL,
                },
                {
                    "role": "user",
                    "content": entity_extraction.OPTIONAL_PROMPT_CHAT_MODEL,
                },
            ],
            entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS,
            entity_extraction.COMPILED_INSTRUCTIONS,
            entity_extraction.COMPILED_PROMPT_REASK_WITHOUT_INSTRUCTIONS,
            entity_extraction.COMPILED_INSTRUCTIONS_REASK,
            [
                entity_extraction.LLM_OUTPUT,
                json.dumps(entity_extraction.VALIDATED_OUTPUT_REASK_2),
                # FIXME: Use this once field level reask schemas are implemented
                # else entity_extraction.LLM_OUTPUT_REASK
            ],
        ),
        (
            entity_extraction.RAIL_SPEC_WITH_REASK_NO_PROMPT,
            entity_extraction.OPTIONAL_MSG_HISTORY,
            None,
            None,
            entity_extraction.COMPILED_PROMPT_REASK_WITHOUT_INSTRUCTIONS,
            entity_extraction.COMPILED_INSTRUCTIONS_REASK,
            [
                entity_extraction.LLM_OUTPUT,
                json.dumps(entity_extraction.VALIDATED_OUTPUT_REASK_2),
                # FIXME: Use this once field level reask schemas are implemented
                # else entity_extraction.LLM_OUTPUT_REASK
            ],
        ),
    ],
)
def test_entity_extraction_with_reask_with_optional_prompts(
    mocker,
    rail,
    messages,
    expected_prompt,
    expected_instructions,
    expected_reask_prompt,
    expected_reask_instructions,
    llm_outputs,
):
    """Test that the entity extraction works with re-asking."""
    llm_return_values = [
        LLMResponse(
            output=o,
            prompt_token_count=123,
            response_token_count=1234,
        )
        for o in llm_outputs
    ]

    mock_openai_invoke_llm = None

    mock_openai_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")
    mock_openai_invoke_llm.side_effect = llm_return_values

    content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
    guard = Guard.for_rail_string(rail)

    final_output = guard(
        model="gpt-3.5-turbo",
        messages=messages,
        prompt_params={"document": content[:6000]},
        num_reasks=1,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_REASK_2

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    if expected_prompt:
        if len(call.compiled_messages) == 1:
            prompt = call.compiled_messages[0]["content"]
        else:
            prompt = call.compiled_messages[1]["content"]
        assert prompt == expected_prompt
    assert call.iterations.first.raw_output == entity_extraction.LLM_OUTPUT
    assert call.iterations.first.validation_response == entity_extraction.VALIDATED_OUTPUT_REASK_1
    if expected_instructions:
        assert call.compiled_messages[0]["content"] == expected_instructions

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
    if expected_reask_prompt:
        assert call.reask_messages.last[1]["content"] == expected_reask_prompt
    # FIXME: Switch back to this once field level reask schema pruning is implemented  # noqa
    # assert call.raw_outputs.at(1) == entity_extraction.LLM_OUTPUT_REASK
    assert call.raw_outputs.at(1) == json.dumps(entity_extraction.VALIDATED_OUTPUT_REASK_2)

    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_REASK_2
    if expected_reask_instructions:
        assert call.reask_messages.last[0]["content"] == expected_reask_instructions


def test_skeleton_reask(mocker):
    from unittest.mock import patch

    with patch(
        "guardrails.llm_providers.LiteLLMCallable._invoke_llm",
        side_effect=[
            LLMResponse(
                output=entity_extraction.LLM_OUTPUT_SKELETON_REASK_1,
                prompt_token_count=123,
                response_token_count=1234,
            ),
            LLMResponse(
                output=entity_extraction.LLM_OUTPUT_SKELETON_REASK_2,
                prompt_token_count=123,
                response_token_count=1234,
            ),
        ],
    ):
        mocker.patch(
            "guardrails.actions.reask.generate_example",
            return_value={
                "fees": [
                    {
                        "index": 1,
                        "name": "annual membership",
                        "explanation": "Annual Membership Fee",
                        "value": 0,
                    }
                ],
                "interest_rates": {},
            },
        )

        content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
        guard = gd.Guard.for_rail_string(entity_extraction.RAIL_SPEC_WITH_SKELETON_REASK)
        final_output = guard(
            model="gpt-3.5-turbo",
            prompt_params={"document": content[:6000]},
            max_tokens=1000,
            num_reasks=1,
        )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_2

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    assert (
        call.compiled_messages[0]["content"] == entity_extraction.COMPILED_PROMPT_SKELETON_REASK_1
    )
    assert call.iterations.first.raw_output == entity_extraction.LLM_OUTPUT_SKELETON_REASK_1
    assert (
        call.iterations.first.validation_response
        == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_1
    )

    # For re-asked prompt and output
    assert (
        call.reask_messages[0][1]["content"] == entity_extraction.COMPILED_PROMPT_SKELETON_REASK_2
    )
    assert call.raw_outputs.last == entity_extraction.LLM_OUTPUT_SKELETON_REASK_2
    assert call.guarded_output == entity_extraction.VALIDATED_OUTPUT_SKELETON_REASK_2


def test_string_with_message_history_reask(mocker):
    """Test single string (non-JSON) generation with message history and
    reask."""
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")

    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=string.MSG_LLM_OUTPUT_INCORRECT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=string.MSG_LLM_OUTPUT_CORRECT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    guard = gd.Guard.for_rail_string(string.RAIL_SPEC_FOR_MSG_HISTORY)
    final_output = guard(
        messages=string.MOVIE_MSG_HISTORY,
        temperature=0.0,
        model="gpt-3.5-turbo",
    )

    assert final_output.validated_output == string.MSG_LLM_OUTPUT_CORRECT

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    assert call.iterations.first.raw_output == string.MSG_LLM_OUTPUT_INCORRECT
    assert call.iterations.first.validation_response == string.MSG_VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert call.reask_messages[0][1]["content"] == string.MSG_COMPILED_PROMPT_REASK
    assert call.reask_messages[0][0]["content"] == string.MSG_COMPILED_INSTRUCTIONS_REASK
    assert call.raw_outputs.last == string.MSG_LLM_OUTPUT_CORRECT
    assert call.guarded_output == string.MSG_LLM_OUTPUT_CORRECT


def test_pydantic_with_message_history_reask(mocker):
    """Test JSON generation with message history re-asking."""
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=pydantic.MSG_HISTORY_LLM_OUTPUT_INCORRECT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]
    # We need to mock the example generation bc it now uses Faker
    mocker.patch(
        "guardrails.actions.reask.generate_example",
        return_value={
            "name": "Star Wars",
            "director": "George Lucas",
            "release_year": 1977,
        },
    )

    guard = gd.Guard.for_pydantic(output_class=pydantic.WITH_MSG_HISTORY)
    final_output = guard(
        messages=string.MOVIE_MSG_HISTORY,
        temperature=0.0,
        model="gpt-3.5-turbo",
    )

    assert final_output.raw_llm_output == pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT
    assert final_output.validated_output == json.loads(pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT)

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    assert call.iterations.first.raw_output == pydantic.MSG_HISTORY_LLM_OUTPUT_INCORRECT
    assert call.iterations.first.validation_response == pydantic.MSG_VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert call.reask_messages[0][1]["content"] == pydantic.MSG_COMPILED_PROMPT_REASK
    assert call.reask_messages[0][0]["content"] == pydantic.MSG_COMPILED_INSTRUCTIONS_REASK
    assert call.raw_outputs.last == pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT
    assert call.guarded_output == json.loads(pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT)


def test_sequential_validator_log_is_not_duplicated(mocker):
    mocker.patch(
        "guardrails.llm_providers.LiteLLMCallable._invoke_llm",
        return_value=LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    )

    proc_count_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
    os.environ["GUARDRAILS_PROCESS_COUNT"] = "1"
    try:
        content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
        guard = guard_initializer(
            entity_extraction.PYDANTIC_RAIL_WITH_NOOP,
            messages=[{"role": "user", "content": entity_extraction.PYDANTIC_PROMPT}],
        )

        guard(
            model="gpt-3.5-turbo",
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
        assert len(one_line_logs) == len(guard.history.first.validation_response.get("fees"))

    finally:
        if proc_count_bak is None:
            del os.environ["GUARDRAILS_PROCESS_COUNT"]
        else:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = proc_count_bak


def test_in_memory_validator_log_is_not_duplicated(mocker):
    mocker.patch(
        "guardrails.llm_providers.LiteLLMCallable._invoke_llm",
        return_value=LLMResponse(
            output=entity_extraction.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    )

    separate_proc_bak = OneLine.run_in_separate_process
    OneLine.run_in_separate_process = False
    try:
        content = gd.docs_utils.read_pdf("docs/src/examples/data/chase_card_agreement.pdf")
        guard = guard_initializer(
            entity_extraction.PYDANTIC_RAIL_WITH_NOOP,
            messages=[{"role": "user", "content": entity_extraction.PYDANTIC_PROMPT}],
        )

        guard(
            model="gpt-3.5-turbo",
            prompt_params={"document": content[:6000]},
            num_reasks=1,
        )

        one_line_logs = list(
            x
            for x in guard.history.first.iterations.first.validator_logs
            if x.validator_name == "OneLine"
        )

        assert len(one_line_logs) == len(guard.history.first.validation_response.get("fees"))

    finally:
        OneLine.run_in_separate_process = separate_proc_bak


def test_enum_datatype(mocker):
    class TaskStatus(enum.Enum):
        not_started = "not started"
        on_hold = "on hold"
        in_progress = "in progress"

    class Task(BaseModel):
        status: TaskStatus

    return_value = pydantic.LLM_OUTPUT_ENUM

    def custom_llm(
        prompt: Optional[str] = None,
        *args,
        instructions: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        nonlocal return_value
        return return_value

    guard = gd.Guard.for_pydantic(Task)
    _, dict_o, *rest = guard(
        custom_llm,
        messages=[{"role": "user", "content": "What is the status of this task?"}],
    )
    assert dict_o == {"status": "not started"}

    return_value = pydantic.LLM_OUTPUT_ENUM_2
    guard = gd.Guard.for_pydantic(Task)
    result = guard(
        custom_llm,
        messages=[{"role": "user", "content": "What is the status of this task REALLY?"}],
        num_reasks=0,
    )

    assert result.validation_passed is False
    assert isinstance(result.reask, SkeletonReAsk)
    assert result.reask.fail_results[0].error_message.startswith("JSON does not match schema")
    assert "$.status" in result.reask.fail_results[0].error_message
    assert (
        "'i dont know?' is not one of ['not started', 'on hold', 'in progress']"
        in result.reask.fail_results[0].error_message
    )


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
    mocker.patch("guardrails.llm_providers.LiteLLMCallable", new=MockLiteLLMCallableOther)

    guard = guard_initializer(rail, messages=[{"role": "user", "content": prompt}])

    output = guard(model="gpt-3.5-turbo")

    # Validate the output
    assert output.validated_output == [
        {"name": "apple", "price": 1.0},
        {"name": "banana", "price": 0.5},
        {"name": "orange", "price": 1.5},
    ]


def test_pydantic_with_lite_llm(mocker):
    """Test lite llm JSON generation with message history re-asking."""
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=pydantic.MSG_HISTORY_LLM_OUTPUT_INCORRECT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]
    guard = gd.Guard.for_pydantic(output_class=pydantic.WITH_MSG_HISTORY)
    final_output = guard(messages=string.MOVIE_MSG_HISTORY, model="gpt-3.5-turbo", max_tokens=10)
    assert guard.history.last.inputs.messages == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you give me your favorite movie?"},
    ]

    call = guard.history.first
    assert call.iterations.length == 2
    assert final_output.raw_llm_output == pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT


def test_string_output(mocker):
    """Test single string (non-JSON) generation."""
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=string.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]

    guard = gd.Guard.for_rail_string(string.RAIL_SPEC_FOR_STRING)
    final_output = guard(
        model="gpt-3.5-turbo",
        prompt_params={"ingredients": "tomato, cheese, sour cream"},
        num_reasks=1,
    )

    assert final_output.validated_output == string.LLM_OUTPUT

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 1

    # For original prompt and output
    assert call.compiled_messages[1]["content"] == string.COMPILED_PROMPT
    assert call.raw_outputs.last == string.LLM_OUTPUT
    assert mock_invoke_llm.call_count == 1
    mock_invoke_llm = None


def test_json_function_calling_tool(mocker):
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")
    task_list = {
        "list": [
            {
                "status": "not started",
                "priority": 1,
                "description": "Do something",
            },
            {
                "status": "in progress",
                "priority": 2,
                "description": "Do something else",
            },
            {
                "status": "on hold",
                "priority": 3,
                "description": "Do something else again",
            },
        ],
    }

    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=json.dumps(task_list),
            prompt_token_count=123,
            response_token_count=1234,
        )
    ]

    class Task(BaseModel):
        status: str
        priority: int
        description: str

    class Tasks(BaseModel):
        list: List[Task]

    guard = Guard.for_pydantic(Tasks)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    final_output = guard(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "You are a helpful assistant"
                "read this email and return the tasks from it."
                " some email blah blah blah.",
            }
        ],
        tools=guard.json_function_calling_tool(tools),
        tool_choice="required",
    )

    gd_response_tool = mock_invoke_llm.call_args.kwargs["tools"][1]["function"]

    assert mock_invoke_llm.call_count == 1
    assert final_output.validated_output == task_list

    # verify that the tools are augmented with schema
    assert len(mock_invoke_llm.call_args.kwargs["tools"]) == 2
    assert mock_invoke_llm.call_args.kwargs["tools"][0] == tools[0]
    assert gd_response_tool["name"] == "gd_response_tool"
    assert gd_response_tool["parameters"]["$defs"]["Task"] == {
        "properties": {
            "status": {"title": "Status", "type": "string"},
            "priority": {"title": "Priority", "type": "integer"},
            "description": {"title": "Description", "type": "string"},
        },
        "required": ["status", "priority", "description"],
        "title": "Task",
        "type": "object",
    }


def test_string_reask(mocker):
    """Test single string (non-JSON) generation with re-asking."""
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=string.LLM_OUTPUT,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=string.LLM_OUTPUT_REASK,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    guard = gd.Guard.for_rail_string(string.RAIL_SPEC_FOR_STRING_REASK)
    final_output = guard(
        model="gpt-3.5-turbo",
        prompt_params={"ingredients": "tomato, cheese, sour cream"},
        num_reasks=1,
        max_tokens=100,
    )

    assert final_output.validated_output == string.LLM_OUTPUT_REASK

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 2

    # For orginal prompt and output
    assert call.compiled_messages[0]["content"] == string.COMPILED_INSTRUCTIONS
    assert call.compiled_messages[1]["content"] == string.COMPILED_PROMPT
    assert call.iterations.first.raw_output == string.LLM_OUTPUT
    assert call.iterations.first.validation_response == string.VALIDATED_OUTPUT_REASK

    # For re-asked prompt and output
    assert call.iterations.last.inputs.messages[1]["content"] == string.COMPILED_PROMPT_REASK
    # Same thing as above
    assert call.reask_messages[0][1]["content"] == string.COMPILED_PROMPT_REASK

    assert call.raw_outputs.last == string.LLM_OUTPUT_REASK
    assert call.guarded_output == string.LLM_OUTPUT_REASK
    assert mock_invoke_llm.call_count == 2
    mock_invoke_llm = None


class TestSerizlizationAndDeserialization:
    def test_guard_i_guard(self):
        guard = Guard(
            name="name-case", description="Checks that a string is in Name Case format."
        ).use_many(
            RegexMatch(regex="^(?:[A-Z][^\s]*\s?)+$", on_fail="noop"),
            ValidLength(1, 100, on_fail="noop"),
            ValidChoices(["Some Name", "Some Other Name"], on_fail="noop"),
        )

        response = guard.parse("Some Name")

        assert response.validation_passed is True

        response = guard.parse("some-name")

        assert response.validation_passed is False

        i_guard = IGuard(
            id=guard.id,
            name=guard.name,
            description=guard.description,
            validators=guard.validators,
            output_schema=guard.output_schema,
            history=guard.history,
        )

        cls_guard = Guard(
            id=i_guard.id,
            name=i_guard.name,
            description=i_guard.description,
            output_schema=i_guard.output_schema.to_dict(),
            validators=i_guard.validators,
        )
        cls_guard.history = Stack(*i_guard.history)

        assert cls_guard == guard

        response = cls_guard.parse("Some Name")

        assert response.validation_passed is True

        response = cls_guard.parse("some-name")

        assert response.validation_passed is False

    def test_ser_deser(self):
        guard = Guard(
            name="name-case", description="Checks that a string is in Name Case format."
        ).use_many(
            RegexMatch(regex="^(?:[A-Z][^\s]*\s?)+$", on_fail="noop"),
            ValidLength(1, 100, on_fail="noop"),
            ValidChoices(["Some Name", "Some Other Name"], on_fail="noop"),
        )

        response = guard.parse("Some Name")

        assert response.validation_passed is True

        response = guard.parse("some-name")

        assert response.validation_passed is False

        ser_guard = guard.to_dict()

        deser_guard = Guard.from_dict(ser_guard)

        assert deser_guard == guard

        response = deser_guard.parse("Some Name")

        assert response.validation_passed is True

        response = deser_guard.parse("some-name")

        assert response.validation_passed is False


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers") and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_guard_for_pydantic_with_mock_hf_pipeline():
    from tests.unit_tests.mocks.mock_hf_models import make_mock_pipeline

    pipe = make_mock_pipeline()
    guard = Guard()
    _ = guard(pipe, messages=[{"role": "user", "content": "Don't care about the output."}])


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers") and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_guard_for_pydantic_with_mock_hf_model():
    from tests.unit_tests.mocks.mock_hf_models import make_mock_model_and_tokenizer

    model, tokenizer = make_mock_model_and_tokenizer()
    guard = Guard()
    _ = guard(
        model.generate,
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": "Don't care about the output."}],
    )


class TestValidatorInitializedOnce:
    def test_guard_init(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        guard = Guard(validators=[ValidatorReference(id="lower-case", on="$", onFail="noop")])

        # Validator is not initialized until the guard is used
        assert init_spy.call_count == 0

        guard.parse("some-name")

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard.parse("some-other-name")

        assert init_spy.call_count == 1

    def test_for_rail(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        guard = Guard.for_rail_string(
            """
            <rail version="0.1">
            <output
                type="string"
                validators="lower-case"
            />
            </rail>
            """
        )

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard.parse("some-name")

        assert init_spy.call_count == 1

    def test_for_pydantic_validator_instance(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        class MyModel(BaseModel):
            name: str = Field(..., validators=[LowerCase()])

        guard = Guard().for_pydantic(MyModel)

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard.parse('{ "name": "some-name" }')

        assert init_spy.call_count == 1

    def test_for_pydantic_str(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        class MyModel(BaseModel):
            name: str = Field(..., validators=[("lower-case", "noop")])

        guard = Guard().for_pydantic(MyModel)

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard.parse('{ "name": "some-name" }')

        assert init_spy.call_count == 1

    def test_for_pydantic_same_instance_on_two_models(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        lower_case = LowerCase()

        class MyModel(BaseModel):
            name: str = Field(..., validators=[lower_case])

        class MyOtherModel(BaseModel):
            name: str = Field(..., validators=[lower_case])

        guard_1 = Guard.for_pydantic(MyModel)
        guard_2 = Guard.for_pydantic(MyOtherModel)

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard_1.parse("some-name")

        assert init_spy.call_count == 1

        guard_2.parse("some-other-name")

        assert init_spy.call_count == 1

    def test_guard_use_instance(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        guard = Guard().use(LowerCase())

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard.parse("some-name")

        assert init_spy.call_count == 1

    def test_guard_use_class(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        guard = Guard().use(LowerCase)

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard.parse("some-name")

        assert init_spy.call_count == 1

    def test_guard_use_same_instance_on_two_guards(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        lower_case = LowerCase()

        guard_1 = Guard().use(lower_case)
        guard_2 = Guard().use(lower_case)

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard_1.parse("some-name")

        assert init_spy.call_count == 1

        guard_2.parse("some-other-name")

        assert init_spy.call_count == 1

    def test_guard_use_many_instance(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        guard = Guard().use_many(LowerCase())

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard.parse("some-name")

        assert init_spy.call_count == 1

    def test_guard_use_many_class(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        guard = Guard().use_many(LowerCase)

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard.parse("some-name")

        assert init_spy.call_count == 1

    def test_guard_use_many_same_instance_on_two_guards(self, mocker):
        init_spy = mocker.spy(LowerCase, "__init__")

        lower_case = LowerCase()

        guard_1 = Guard().use_many(lower_case)
        guard_2 = Guard().use_many(lower_case)

        assert init_spy.call_count == 1

        # Validator is not initialized again
        guard_1.parse("some-name")

        assert init_spy.call_count == 1

        guard_2.parse("some-other-name")

        assert init_spy.call_count == 1


# These tests are descriptive not prescriptive.
# The method signature for custom LLM APIs needs to be updated to make more sense.
# With 0.6.0 we can drop the baggage of
#   the prompt and instructions and just pass in the messages.
class TestCustomLLMApi:
    def test_WITH_MSG_HISTORY(self, mocker):
        mock_llm = mocker.Mock()

        def custom_llm(
            *args,
            messages: Optional[List[Dict[str, str]]] = None,
            **kwargs,
        ) -> str:
            mock_llm(
                *args,
                messages=messages,
                **kwargs,
            )
            return "Not really, no.  I'm just a static function."

        guard = Guard().use(
            ValidLength(1, 100),
        )
        output = guard(
            llm_api=custom_llm,
            messages=[
                {
                    "role": "system",
                    "content": "You are a list generator.  You can generate a list of things that are not food.",  # noqa
                },
                {
                    "role": "user",
                    "content": "Can you generate a list of 10 things that are not food?",  # noqa
                },
            ],
        )

        assert output.validation_passed is True
        assert output.validated_output == "Not really, no.  I'm just a static function."
        mock_llm.assert_called_once_with(
            messages=[
                {
                    "role": "system",
                    "content": "You are a list generator.  You can generate a list of things that are not food.",  # noqa
                },
                {
                    "role": "user",
                    "content": "Can you generate a list of 10 things that are not food?",  # noqa
                },
            ],
            temperature=0,
        )
