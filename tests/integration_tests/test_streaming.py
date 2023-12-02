# 2 tests
# 1. Test streaming with OpenAICallable (mock openai.Completion.create)
# 2. Test streaming with OpenAIChatCallable (mock openai.ChatCompletion.create)
# Using the LowerCase Validator

import json
import os

import openai
import pytest
from pydantic import BaseModel, Field

import guardrails as gd
from guardrails.utils.openai_utils import OPENAI_VERSION
from guardrails.validators import LowerCase

# Set mock OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxx"


@pytest.fixture(scope="module")
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
            yield {
                "choices": [{"text": chunk, "finish_reason": None}],
                "model": "OpenAI model name",
            }

    return gen()


@pytest.fixture(scope="module")
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
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ]
            }

    return gen()


class LowerCaseValue(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail="fix")],
    )


PROMPT = """
Say something nice to me.

${gr.complete_json_suffix}
"""


def test_streaming_with_openai_callable(mocker, mock_openai_completion_create):
    """Test streaming with OpenAICallable.

    Mocks openai.Completion.create.
    """

    if OPENAI_VERSION.startswith("0"):
        mocker.patch(
            "openai.Completion.create", return_value=mock_openai_completion_create
        )
    else:
        mocker.patch(
            "openai.resources.Completions.create",
            return_value=mock_openai_completion_create,
        )

    # Create a guard object
    guard = gd.Guard.from_pydantic(output_class=LowerCaseValue, prompt=PROMPT)

    method = (
        openai.Completion.create
        if OPENAI_VERSION.startswith("0")
        else openai.completions.create
    )
    generator = guard(
        method,
        engine="text-davinci-003",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    actual_output = ""
    for op in generator:
        actual_output = op

    expected_raw_output = '{"statement": "I am DOING well, and I HOPE you aRe too."}'
    expected_validated_output = json.dumps(
        {"statement": "i am doing well, and i hope you are too."}, indent=4
    )

    assert (
        actual_output
        == f"Raw LLM response:\n{expected_raw_output}\n"
        + f"\nValidated response:\n{expected_validated_output}\n"
    )


def test_streaming_with_openai_chat_callable(
    mocker,
    mock_openai_chat_completion_create,
):
    """Test streaming with OpenAIChatCallable.

    Mocks openai.ChatCompletion.create.
    """

    if OPENAI_VERSION.startswith("0"):
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=mock_openai_chat_completion_create,
        )
    else:
        mocker.patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=mock_openai_chat_completion_create,
        )

    # Create a guard object
    guard = gd.Guard.from_pydantic(output_class=LowerCaseValue, prompt=PROMPT)

    method = (
        openai.ChatCompletion.create
        if OPENAI_VERSION.startswith("0")
        else openai.chat.completions.create
    )
    generator = guard(
        method,
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    actual_output = ""
    for op in generator:
        actual_output = op

    expected_raw_output = '{"statement": "I am DOING well, and I HOPE you aRe too."}'
    expected_validated_output = json.dumps(
        {"statement": "i am doing well, and i hope you are too."}, indent=4
    )

    assert (
        actual_output
        == f"Raw LLM response:\n{expected_raw_output}\n"
        + f"\nValidated response:\n{expected_validated_output}\n"
    )
