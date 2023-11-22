# 2 tests
# 1. Test streaming with OpenAICallable (mock openai.Completion.create)
# 2. Test streaming with OpenAIChatCallable (mock openai.ChatCompletion.create)
# Using the LowerCase Validator

import os

import openai
import pytest
from pydantic import BaseModel, Field

import guardrails as gd
from guardrails.utils.openai_utils import OPENAI_VERSION
from guardrails.utils.safe_get import safe_get_with_brackets
from guardrails.validators import LowerCase

# Set mock OpenAI API key
# os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxx"


@pytest.fixture(scope="module")
def non_chat_token_count_mock() -> int:
    return 10


@pytest.fixture(scope="module")
def chat_token_count_mock() -> int:
    return 10


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
                "choices": [{"text": chunk}],
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


def test_streaming_with_openai_callable(
    mocker, mock_openai_completion_create, non_chat_token_count_mock
):
    """Test streaming with OpenAICallable.

    Mocks openai.Completion.create.
    """

    def mock_os_environ_get(key, *args):
        if key == "OPENAI_API_KEY":
            return "sk-xxxxxxxxxxxxxx"
        return safe_get_with_brackets(os.environ, key, *args)

    mocker.patch("os.environ.get", side_effect=mock_os_environ_get)

    if OPENAI_VERSION.startswith("0"):
        mocker.patch(
            "openai.Completion.create", return_value=mock_openai_completion_create
        )
        mocker.patch(
            "guardrails.utils.openai_utils.v0.num_tokens_from_string",
            return_value=non_chat_token_count_mock,
        )
    else:
        mocker.patch(
            "openai.resources.Completions.create",
            return_value=mock_openai_completion_create,
        )
        mocker.patch(
            "guardrails.utils.openai_utils.v1.num_tokens_from_string",
            return_value=non_chat_token_count_mock,
        )

    # Create a guard object
    guard = gd.Guard.from_pydantic(output_class=LowerCaseValue, prompt=PROMPT)

    method = (
        openai.Completion.create
        if OPENAI_VERSION.startswith("0")
        else openai.completions.create
    )
    raw_output, validated_output, *rest = guard(
        method,
        engine="text-davinci-003",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    assert raw_output == '{"statement": "I am DOING well, and I HOPE you aRe too."}'
    assert validated_output == {"statement": "i am doing well, and i hope you are too."}


def test_streaming_with_openai_chat_callable(
    mocker,
    mock_openai_chat_completion_create,
    chat_token_count_mock,
    non_chat_token_count_mock,
):
    """Test streaming with OpenAIChatCallable.

    Mocks openai.ChatCompletion.create.
    """

    def mock_os_environ_get(key, *args):
        if key == "OPENAI_API_KEY":
            return "sk-xxxxxxxxxxxxxx"
        return safe_get_with_brackets(os.environ, key, *args)

    mocker.patch("os.environ.get", side_effect=mock_os_environ_get)

    if OPENAI_VERSION.startswith("0"):
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=mock_openai_chat_completion_create,
        )
        mocker.patch(
            "guardrails.utils.openai_utils.v0.num_tokens_from_messages",
            return_value=chat_token_count_mock,
        )
        mocker.patch(
            "guardrails.utils.openai_utils.v0.num_tokens_from_string",
            return_value=non_chat_token_count_mock,
        )
    else:
        mocker.patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=mock_openai_chat_completion_create,
        )
        mocker.patch(
            "guardrails.utils.openai_utils.v1.num_tokens_from_messages",
            return_value=chat_token_count_mock,
        )
        mocker.patch(
            "guardrails.utils.openai_utils.v1.num_tokens_from_string",
            return_value=non_chat_token_count_mock,
        )

    # Create a guard object
    guard = gd.Guard.from_pydantic(output_class=LowerCaseValue, prompt=PROMPT)

    method = (
        openai.ChatCompletion.create
        if OPENAI_VERSION.startswith("0")
        else openai.chat.completions.create
    )
    raw_output, validated_output, *rest = guard(
        method,
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    assert raw_output == '{"statement": "I am DOING well, and I HOPE you aRe too."}'
    assert (
        str(validated_output)
        == "{'statement': 'i am doing well, and i hope you are too.'}"
    )
