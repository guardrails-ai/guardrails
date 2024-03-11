import importlib.util
from dataclasses import dataclass
from typing import List

import pytest

import guardrails as gd
from guardrails.validators import LowerCase


# Mock the litellm.completion function and
# the classes it returns
@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class MockResponse:
    choices: List[Choice]
    usage: Usage


class MockCompletion:
    @staticmethod
    def create() -> MockResponse:
        return MockResponse(
            choices=[Choice(message=Message(content="GUARDRAILS AI"))],
            usage=Usage(prompt_tokens=10, completion_tokens=20),
        )


TEST_PROMPT = "Suggest a name for an AI company. The name should be short and catchy."
guard = gd.Guard.from_string(validators=[LowerCase(on_fail="fix")], prompt=TEST_PROMPT)


@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
def test_litellm_completion(mocker):
    """Test that Guardrails can use litellm for completions."""
    import litellm

    mocker.patch("litellm.completion", return_value=MockCompletion.create())

    raw, validated, *rest = guard(litellm.completion)
    assert raw == "GUARDRAILS AI"
    assert validated == "guardrails ai"
