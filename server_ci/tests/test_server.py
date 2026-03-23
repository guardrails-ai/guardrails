import openai
import os
import pytest
from guardrails import AsyncGuard, Guard

# OpenAI compatible Guardrails API Guard
openai.base_url = "http://127.0.0.1:8000/guards/test-guard/openai/v1/"

openai.api_key = os.getenv("OPENAI_API_KEY") or "some key"


def test_guard_validation():
    guard = Guard.load(name="test-guard", api_key="auth-stub")
    if not guard:
        raise RuntimeError("Guard did not load properly!")

    else:
        validation_outcome = guard.validate("France is wonderful in the spring")
        assert validation_outcome.validation_passed is True
        assert validation_outcome.validated_output == "France is"


@pytest.mark.asyncio
async def test_async_guard_validation():
    guard = AsyncGuard.load(name="test-guard", api_key="auth-stub")

    if not guard:
        raise RuntimeError("Guard did not load properly!")

    validation_outcome = await guard(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me about Oranges in 5 words"}],
        temperature=0.0,
    )

    assert validation_outcome.validation_passed is True  # type: ignore # noqa: E712
    assert validation_outcome.validated_output == "Citrus fruit,"  # type: ignore


@pytest.mark.asyncio
async def test_async_streaming_guard_validation():
    guard = AsyncGuard.load(name="test-guard", api_key="auth-stub")

    if not guard:
        raise RuntimeError("Guard did not load properly!")

    async_iterator = await guard(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me about Oranges in 5 words"}],
        stream=True,
        temperature=0.0,
    )

    full_output = ""
    async for validation_chunk in async_iterator:  # type: ignore
        full_output += validation_chunk.validated_output

    assert full_output == "Citrus fruit,"


@pytest.mark.asyncio
async def test_sync_streaming_guard_validation():
    # FIXME: The fact that this is necessary is concerning;
    #   This is also necessary on our latest published versions:
    #       guardrails-ai==0.6.6
    #       guardrails-api==0.1.0a2
    os.environ["GUARD_HISTORY_ENABLED"] = "false"
    guard = Guard.load(name="test-guard", api_key="auth-stub")

    if not guard:
        raise RuntimeError("Guard did not load properly!")

    iterator = guard(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me about Oranges in 5 words"}],
        stream=True,
        temperature=0.0,
    )

    full_output = ""
    for validation_chunk in iterator:
        full_output += validation_chunk.validated_output  # type: ignore

    assert full_output == "Citrus fruit,"


def test_server_guard_llm_integration():
    guard = Guard.load(name="test-guard", api_key="auth-stub")
    if not guard:
        raise RuntimeError("Guard did not load properly!")
    messages = [{"role": "user", "content": "Tell me about Oranges in 5 words"}]

    validation_outcome = guard(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )
    # True because of fix behaviour
    assert validation_outcome.validation_passed is True  # type: ignore
    assert "Citrus fruit" in validation_outcome.validated_output  # type: ignore


def test_server_openai_llm_integration():
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write 5 words of prose."}],
        temperature=0.0,
    )
    assert "Whispers of" in completion.choices[0].message.content  # type: ignore
    assert (completion.guardrails["validation_passed"]) is True  # type: ignore
