import json
from typing import Dict, List

import pytest
from pydantic import BaseModel

import guardrails as gd
from guardrails.classes.generic.stack import Stack
from guardrails.classes.history.call import Call
from guardrails.utils.history_utils import merge_validation_output
from guardrails.utils.openai_utils import (
    get_static_openai_chat_create_func,
    get_static_openai_create_func,
)

from .mock_llm_outputs import MockOpenAICallable, MockOpenAIChatCallable, pydantic
from .test_assets.pydantic import VALIDATED_RESPONSE_REASK_PROMPT, ListOfPeople


def test_pydantic_with_reask(mocker):
    """Test that the entity extraction works with re-asking."""
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    guard = gd.Guard.from_pydantic(ListOfPeople, prompt=VALIDATED_RESPONSE_REASK_PROMPT)
    final_output = guard(
        get_static_openai_create_func(),
        engine="text-davinci-003",
        max_tokens=512,
        temperature=0.5,
        num_reasks=2,
        full_schema_reask=False,
    )

    # Assertions are made on the guard state object.
    assert final_output.validated_output == pydantic.VALIDATED_OUTPUT_REASK_3

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 3

    # For orginal prompt and output
    assert call.iterations.first.inputs.prompt == gd.Prompt(pydantic.COMPILED_PROMPT)
    assert call.iterations.first.raw_output == pydantic.LLM_OUTPUT
    assert call.iterations.first.validation_output == pydantic.VALIDATED_OUTPUT_REASK_1

    # For re-asked prompt and output
    assert call.iterations.at(1).inputs.prompt == gd.Prompt(pydantic.COMPILED_PROMPT_REASK_1)
    assert call.iterations.at(1).raw_output == pydantic.LLM_OUTPUT_REASK_1
    
    # We don't track merged validation output anymore 
    # Each validation_output is instead tracked as it came back from validation
    # assert call.iterations.at(1).validation_output == pydantic.VALIDATED_OUTPUT_REASK_2
    
    # We can merge down though to achieve the same thing
    merged_output = merge_validation_output(
        Call(iterations=Stack(call.iterations.first, call.iterations.at(1))),
        False
    )
    assert merged_output == pydantic.VALIDATED_OUTPUT_REASK_2

    # For re-asked prompt #2 and output #2
    assert call.iterations.last.inputs.prompt == gd.Prompt(pydantic.COMPILED_PROMPT_REASK_2)
    assert call.raw_output == pydantic.LLM_OUTPUT_REASK_2
    assert call.validated_output == pydantic.VALIDATED_OUTPUT_REASK_3


def test_pydantic_with_full_schema_reask(mocker):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.OpenAIChatCallable", new=MockOpenAIChatCallable
    )

    guard = gd.Guard.from_pydantic(ListOfPeople, prompt=VALIDATED_RESPONSE_REASK_PROMPT)
    _, final_output, *rest = guard(
        get_static_openai_chat_create_func(),
        model="gpt-3.5-turbo",
        max_tokens=512,
        temperature=0.5,
        num_reasks=2,
        full_schema_reask=True,
    )

    # Assertions are made on the guard state object.
    assert final_output == pydantic.VALIDATED_OUTPUT_REASK_3

    call = guard.history.first

    # Check that the guard state object has the correct number of re-asks.
    assert call.iterations.length == 3

    # For orginal prompt and output
    assert call.iterations.first.inputs.prompt == gd.Prompt(pydantic.COMPILED_PROMPT_CHAT)
    assert call.iterations.first.inputs.instructions == gd.Instructions(
        pydantic.COMPILED_INSTRUCTIONS_CHAT
    )
    assert call.iterations.first.raw_output == pydantic.LLM_OUTPUT
    assert call.iterations.first.validation_output == pydantic.VALIDATED_OUTPUT_REASK_1

    # For re-asked prompt and output
    assert call.iterations.at(1).inputs.prompt == gd.Prompt(pydantic.COMPILED_PROMPT_FULL_REASK_1)
    assert call.iterations.at(1).inputs.instructions == gd.Instructions(
        pydantic.COMPILED_INSTRUCTIONS_CHAT
    )
    assert call.iterations.at(1).raw_output == pydantic.LLM_OUTPUT_FULL_REASK_1
    assert call.iterations.at(1).validation_output == pydantic.VALIDATED_OUTPUT_REASK_2

    # For re-asked prompt #2 and output #2
    assert call.iterations.last.inputs.prompt == gd.Prompt(pydantic.COMPILED_PROMPT_FULL_REASK_2)
    assert call.iterations.last.inputs.instructions == gd.Instructions(
        pydantic.COMPILED_INSTRUCTIONS_CHAT
    )
    assert call.raw_output == pydantic.LLM_OUTPUT_FULL_REASK_2
    assert call.validated_output == pydantic.VALIDATED_OUTPUT_REASK_3


class ContainerModel(BaseModel):
    annotated_dict: Dict[str, str] = {}
    annotated_dict_in_list: List[Dict[str, str]] = []
    annotated_list: List[str] = []
    annotated_list_in_dict: Dict[str, List[str]] = {}


class ContainerModel2(BaseModel):
    dict_: Dict = {}
    dict_in_list: List[Dict] = []
    list_: List = []
    list_in_dict: Dict[str, List] = {}


@pytest.mark.parametrize(
    "model, output",
    [
        (
            ContainerModel,
            {
                "annotated_dict": {"a": "b"},
                "annotated_dict_in_list": [{"a": "b"}],
                "annotated_list": ["a"],
                "annotated_list_in_dict": {"a": ["b"]},
            },
        ),
        (
            ContainerModel2,
            {
                "dict_": {"a": "b"},
                "dict_in_list": [{"a": "b"}],
                "list_": ["a"],
                "list_in_dict": {"a": ["b"]},
            },
        ),
    ],
)
def test_container_types(model, output):
    output_str = json.dumps(output)

    guard = gd.Guard.from_pydantic(model)
    out = guard.parse(output_str)
    assert out.validated_output == output
