# 2 tests
# 1. Test streaming with OpenAICallable (mock openai.Completion.create)
# 2. Test streaming with OpenAIChatCallable (mock openai.ChatCompletion.create)
# Using the LowerCase Validator

import openai
import pytest
from pydantic import BaseModel, Field

import guardrails as gd
from guardrails.validators import LowerCase


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

    mocker.patch("openai.Completion.create", return_value=mock_openai_completion_create)
    mocker.patch(
        "guardrails.utils.openai_utils.v0.num_tokens_from_string",
        return_value=non_chat_token_count_mock,
    )

    # Create a guard object
    guard = gd.Guard.from_pydantic(output_class=LowerCaseValue, prompt=PROMPT)

    raw_output, validated_output = guard(
        openai.Completion.create,
        engine="text-davinci-003",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    assert raw_output == '{"statement": "I am DOING well, and I HOPE you aRe too."}'
    assert (
        str(validated_output)
        == "{'statement': 'i am doing well, and i hope you are too.'}"
    )


def test_streaming_with_openai_chat_callable(
    mocker,
    mock_openai_chat_completion_create,
    chat_token_count_mock,
    non_chat_token_count_mock,
):
    """Test streaming with OpenAIChatCallable.

    Mocks openai.ChatCompletion.create.
    """

    mocker.patch(
        "openai.ChatCompletion.create", return_value=mock_openai_chat_completion_create
    )
    mocker.patch(
        "guardrails.utils.openai_utils.v0.num_tokens_from_messages",
        return_value=chat_token_count_mock,
    )
    mocker.patch(
        "guardrails.utils.openai_utils.v0.num_tokens_from_string",
        return_value=non_chat_token_count_mock,
    )

    # Create a guard object
    guard = gd.Guard.from_pydantic(output_class=LowerCaseValue, prompt=PROMPT)

    raw_output, validated_output = guard(
        openai.ChatCompletion.create,
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
