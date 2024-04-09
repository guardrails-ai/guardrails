import importlib.util
from dataclasses import dataclass
from typing import List

import pytest

import guardrails as gd
from guardrails.validators import LowerCase, OneLine, UpperCase
import os

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
    def create(output) -> MockResponse:
        return MockResponse(
            choices=[Choice(message=Message(content=output))],
            usage=Usage(prompt_tokens=10, completion_tokens=20),
        )


@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
@pytest.mark.parametrize(
    "input_text, expected",
    [
        (
            """
            Suggestions for a name for an AI company.
            The name should be short and catchy.
            """,
            "GUARDRAILS AI",
        ),
        ("What is the capital of France?", "PARIS"),
    ],
)
def test_litellm_completion(mocker, input_text, expected):
    """Test that Guardrails can use litellm for completions."""
    import litellm

    mocker.patch("litellm.completion", return_value=MockCompletion.create(expected))

    guard = gd.Guard.from_string(
        validators=[LowerCase(on_fail="fix")], prompt=input_text
    )

    raw, validated, *rest = guard(litellm.completion)
    assert raw == expected
    assert validated == expected.lower()


# Test Guard().use() with just output validators
@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
@pytest.mark.parametrize(
    "input_text, raw_response, pass_output",
    [
        ("Name one Oscar-nominated film", "may december", True),
        ("Name one Oscar-nominated film", "PAST LIVES", False),
    ],
)
def test_guard_use_output_validators(mocker, input_text, raw_response, pass_output):
    """Test Guard().use() with just output validators."""
    import litellm

    mocker.patch("litellm.completion", return_value=MockCompletion.create(raw_response))

    guard = (
        gd.Guard()
        .use(LowerCase, on="output", on_fail="fix")
        .use(OneLine, on="output", on_fail="noop")
    )
    raw, validated, *rest = guard(litellm.completion, prompt=input_text)

    assert raw == raw_response
    if pass_output:
        assert validated == raw_response
    else:
        assert validated == raw_response.lower()


# Test Guard().use() with a combination of prompt and output validators
@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
@pytest.mark.parametrize(
    "input_text, pass_input, raw_response, pass_output",
    [
        ("name one oscar-nominated film", True, "MAY DECEMBER", True),
        ("Name one Oscar-nominated film", False, "PAST LIVES", True),
        ("Name one Oscar-nominated film", False, "past lives", False),
        ("name one oscar-nominated film", True, "past lives", False),
    ],
)
def test_guard_use_combination_validators(
    mocker, input_text, pass_input, raw_response, pass_output
):
    """Test Guard().use() with a combination of prompt and output
    validators."""
    import litellm

    mocker.patch("litellm.completion", return_value=MockCompletion.create(raw_response))

    guard = (
        gd.Guard()
        .use(LowerCase, on="prompt", on_fail="exception")
        .use(UpperCase, on="output", on_fail="fix")
    )

    if pass_input:
        raw, validated, *rest = guard(litellm.completion, prompt=input_text)

        assert raw == raw_response
        if pass_output:
            assert validated == raw_response
        else:
            assert validated == raw_response.upper()
    else:
        with pytest.raises(Exception):
            raw, validated, *rest = guard(litellm.completion, prompt=input_text)


# Test Guard().use_many() with just output validators
@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
@pytest.mark.parametrize(
    "input_text, raw_response, pass_output",
    [
        ("Name one Oscar-nominated film", "may december", True),
        ("Name one Oscar-nominated film", "PAST LIVES", False),
    ],
)
def test_guard_use_many_output_validators(
    mocker, input_text, raw_response, pass_output
):
    """Test Guard().use_many() with just output validators."""
    import litellm

    mocker.patch("litellm.completion", return_value=MockCompletion.create(raw_response))

    guard = gd.Guard().use_many(
        LowerCase(on_fail="fix"), OneLine(on_fail="noop"), on="output"
    )
    raw, validated, *rest = guard(litellm.completion, prompt=input_text)

    assert raw == raw_response
    if pass_output:
        assert validated == raw_response
    else:
        assert validated == raw_response.lower()


# Test Guard().use_many() with a combination of prompt and output validators
@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
@pytest.mark.parametrize(
    "input_text, pass_input, raw_response, pass_output",
    [
        ("name one oscar-nominated film", True, "MAY DECEMBER", True),
        ("Name one Oscar-nominated film", False, "PAST LIVES", True),
        ("Name one Oscar-nominated film", False, "past lives", False),
        ("name one oscar-nominated film", True, "past lives", False),
    ],
)
def test_guard_use_many_combination_validators(
    mocker, input_text, pass_input, raw_response, pass_output
):
    """Test Guard().use() with a combination of prompt and output
    validators."""
    import litellm

    mocker.patch("litellm.completion", return_value=MockCompletion.create(raw_response))

    guard = (
        gd.Guard()
        .use_many(LowerCase(on_fail="exception"), on="prompt")
        .use_many(UpperCase(on_fail="fix"), on="output")
    )

    if pass_input:
        raw, validated, *rest = guard(litellm.completion, prompt=input_text)

        assert raw == raw_response
        if pass_output:
            assert validated == raw_response
        else:
            assert validated == raw_response.upper()
    else:
        with pytest.raises(Exception):
            raw, validated, *rest = guard(litellm.completion, prompt=input_text)


# WE EXPLICITLY DO NOT WANT TO MOCK LITELLM FOR THE TESTS BELOW.
# THEY ENSURE THAT WE HAVE A STANDARD WAY TO COMMUNICATE WITH THE LIBRARY
#   OVER TIME.

@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") in [None, "mocked"],
    reason="openai api key not set",
)
def test_litellm_openai():
    from litellm import litellm

    guard = gd.Guard()
    res = guard(
        llm_api=litellm.completion,
        model="gpt-3.5-turbo",
        msg_history=[{'role': 'user', 'content': "Name 10 unique fruits, lowercase only"}]
    )
    assert res.validated_output
    res = guard(
        llm_api=litellm.completion,
        model="gpt-3.5-turbo",
        prompt="Name 10 unique fruits, lowercase only, one per line, no numbers",
    )
    assert res.validated_output

@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") in [None, "mocked"],
    reason="openai api key not set",
)
def test_litellm_openai_streaming():
    from litellm import litellm
    guard = gd.Guard()
    res = guard(
        llm_api=litellm.completion,
        model="gpt-3.5-turbo",
        prompt="Name 10 unique fruits, lowercase only, one per line, no numbers",
        stream=True
    )

    for chunk in res:
        assert chunk.validated_output

@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") in [None, "mocked"],
    reason="openai api key not set",
)
def test_litellm_openai_async():
    from litellm import litellm
    import asyncio
    # from litellm import acompletion
    guard = gd.Guard()
    ares = guard(
        llm_api=litellm.acompletion,
        model="gpt-3.5-turbo",
        prompt="Name 10 unique fruits, lowercase only, one per line, no numbers",
    )

    res = asyncio.run(ares)
    assert res.validated_output
    assert res.validated_output == res.raw_llm_output
    assert len(res.validated_output.split('\n')) == 10