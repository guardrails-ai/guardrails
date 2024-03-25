# 2 tests
# 1. Test streaming with OpenAICallable (mock openai.Completion.create)
# 2. Test streaming with OpenAIChatCallable (mock openai.ChatCompletion.create)
# Using the LowerCase Validator

import json
from typing import Iterable

import openai
import pytest
from pydantic import BaseModel, Field

import guardrails as gd
from guardrails.utils.openai_utils import OPENAI_VERSION
from guardrails.validator_base import OnFailAction
from guardrails.validators import LowerCase

expected_raw_output = '{"statement": "I am DOING well, and I HOPE you aRe too."}'
expected_fix_output = json.dumps(
    {"statement": "i am doing well, and i hope you are too."}, indent=4
)
expected_noop_output = json.dumps(
    {"statement": "I am DOING well, and I HOPE you aRe too."}, indent=4
)
expected_filter_refrain_output = json.dumps({}, indent=4)


class Delta:
    content: str

    def __init__(self, content):
        self.content = content


class Choice:
    text: str
    finish_reason: str
    index: int
    delta: Delta

    def __init__(self, text, delta, finish_reason, index=0):
        self.index = index
        self.delta = delta
        self.text = text
        self.finish_reason = finish_reason


class MockOpenAIV1ChunkResponse:
    choices: list
    model: str

    def __init__(self, choices, model):
        self.choices = choices
        self.model = model


def mock_openai_completion_create():
    # Returns a generator
    chunks = [
        '{"statement":',
        ' "I am DOING',
        " well, and I",
        " HOPE you aRe",
        ' too."}',
    ]

    def gen():
        for chunk in chunks:
            if OPENAI_VERSION.startswith("0"):
                yield {
                    "choices": [{"text": chunk, "finish_reason": None}],
                    "model": "OpenAI model name",
                }
            else:
                yield MockOpenAIV1ChunkResponse(
                    choices=[
                        Choice(
                            text=chunk,
                            delta=Delta(content=""),
                            finish_reason=None,
                        )
                    ],
                    model="OpenAI model name",
                )

    return gen()


def mock_openai_chat_completion_create():
    # Returns a generator
    chunks = [
        '{"statement":',
        ' "I am DOING',
        " well, and I",
        " HOPE you aRe",
        ' too."}',
    ]

    def gen():
        for chunk in chunks:
            if OPENAI_VERSION.startswith("0"):
                yield {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ]
                }
            else:
                yield MockOpenAIV1ChunkResponse(
                    choices=[
                        Choice(
                            text="",
                            delta=Delta(content=chunk),
                            finish_reason=None,
                        )
                    ],
                    model="OpenAI model name",
                )

    return gen()


class LowerCaseFix(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail=OnFailAction.FIX)],
    )


class LowerCaseNoop(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail=OnFailAction.NOOP)],
    )


class LowerCaseFilter(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail=OnFailAction.FILTER)],
    )


class LowerCaseRefrain(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail=OnFailAction.REFRAIN)],
    )


PROMPT = """
Say something nice to me.

${gr.complete_json_suffix}
"""


@pytest.mark.parametrize(
    "op_class, expected_validated_output",
    [
        (LowerCaseNoop, expected_noop_output),
        (LowerCaseFix, expected_fix_output),
        (LowerCaseFilter, expected_filter_refrain_output),
        (LowerCaseRefrain, expected_filter_refrain_output),
    ],
)
def test_streaming_with_openai_callable(
    mocker,
    op_class,
    expected_validated_output,
):
    """Test streaming with OpenAICallable.

    Mocks openai.Completion.create.
    """
    if OPENAI_VERSION.startswith("0"):
        mocker.patch(
            "openai.Completion.create", return_value=mock_openai_completion_create()
        )
    else:
        mocker.patch(
            "openai.resources.Completions.create",
            return_value=mock_openai_completion_create(),
        )

    # Create a guard object
    guard = gd.Guard.from_pydantic(output_class=op_class, prompt=PROMPT)

    method = (
        openai.Completion.create
        if OPENAI_VERSION.startswith("0")
        else openai.completions.create
    )

    method.__name__ = "mock_openai_completion_create"

    generator = guard(
        method,
        engine="text-davinci-003",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    assert isinstance(generator, Iterable)

    actual_output = ""
    for op in generator:
        actual_output = op

    assert (
        actual_output
        == f"Raw LLM response:\n{expected_raw_output}\n"
        + f"\nValidated response:\n{expected_validated_output}\n"
    )


@pytest.mark.parametrize(
    "op_class, expected_validated_output",
    [
        (LowerCaseNoop, expected_noop_output),
        (LowerCaseFix, expected_fix_output),
        (LowerCaseFilter, expected_filter_refrain_output),
        (LowerCaseRefrain, expected_filter_refrain_output),
    ],
)
def test_streaming_with_openai_chat_callable(
    mocker,
    op_class,
    expected_validated_output,
):
    """Test streaming with OpenAIChatCallable.

    Mocks openai.ChatCompletion.create.
    """
    if OPENAI_VERSION.startswith("0"):
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=mock_openai_chat_completion_create(),
        )
    else:
        mocker.patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=mock_openai_chat_completion_create(),
        )

    # Create a guard object
    guard = gd.Guard.from_pydantic(output_class=op_class, prompt=PROMPT)

    method = (
        openai.ChatCompletion.create
        if OPENAI_VERSION.startswith("0")
        else openai.chat.completions.create
    )

    method.__name__ = "mock_openai_chat_completion_create"

    generator = guard(
        method,
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    assert isinstance(generator, Iterable)

    actual_output = ""
    for op in generator:
        actual_output = op

    assert (
        actual_output
        == f"Raw LLM response:\n{expected_raw_output}\n"
        + f"\nValidated response:\n{expected_validated_output}\n"
    )
