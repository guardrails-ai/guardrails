"""Integration tests for Novita AI provider support.

These tests require a valid NOVITA_API_KEY environment variable to run.
They verify that Guardrails can call Novita AI's OpenAI-compatible endpoint
through both the LiteLLM path and the direct OpenAI-SDK (NovitaClient) path.
"""

import asyncio
import importlib
import os

import pytest

import guardrails as gd
from guardrails.utils.novita_utils import (
    NovitaClient,
    novita_completion_kwargs,
    NOVITA_DEFAULT_MODEL,
)


def _has_novita_key() -> bool:
    key = os.environ.get("NOVITA_API_KEY")
    return key not in (None, "", "mocked")


def _has_litellm() -> bool:
    return importlib.util.find_spec("litellm") is not None


# ---------------------------------------------------------------------------
# LiteLLM path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_litellm(), reason="`litellm` is not installed")
@pytest.mark.skipif(not _has_novita_key(), reason="NOVITA_API_KEY not set")
def test_novita_litellm_basic():
    """Guard can call Novita via LiteLLM and return validated output."""
    guard = gd.Guard()
    result = guard(
        messages=[{"role": "user", "content": "Name 5 fruits, one per line."}],
        **novita_completion_kwargs(NOVITA_DEFAULT_MODEL),
    )
    assert result.validated_output


@pytest.mark.skipif(not _has_litellm(), reason="`litellm` is not installed")
@pytest.mark.skipif(not _has_novita_key(), reason="NOVITA_API_KEY not set")
def test_novita_litellm_tools():
    """Guard structured-output (function-calling tools) works with Novita."""
    from typing import List
    from pydantic import BaseModel

    class Fruit(BaseModel):
        name: str
        color: str

    class Fruits(BaseModel):
        items: List[Fruit]

    guard = gd.Guard.for_pydantic(Fruits)
    result = guard(
        messages=[{"role": "user", "content": "Name 5 unique fruits."}],
        tools=guard.json_function_calling_tool([]),
        tool_choice="required",
        **novita_completion_kwargs(NOVITA_DEFAULT_MODEL),
    )
    assert result.validated_output


@pytest.mark.skipif(not _has_litellm(), reason="`litellm` is not installed")
@pytest.mark.skipif(not _has_novita_key(), reason="NOVITA_API_KEY not set")
def test_novita_litellm_async():
    """AsyncGuard works with Novita via LiteLLM."""
    guard = gd.AsyncGuard()
    result = asyncio.run(
        guard(
            messages=[{"role": "user", "content": "Name 5 fruits, one per line."}],
            **novita_completion_kwargs(NOVITA_DEFAULT_MODEL),
        )
    )
    assert result.validated_output


# ---------------------------------------------------------------------------
# Direct NovitaClient path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_novita_key(), reason="NOVITA_API_KEY not set")
def test_novita_client_chat_completion():
    """NovitaClient.create_chat_completion returns an LLMResponse."""
    client = NovitaClient()
    response = client.create_chat_completion(
        model=NOVITA_DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Say hello."}],
    )
    assert response.output


# ---------------------------------------------------------------------------
# Unit-level: helper returns well-formed kwargs dict (no API call needed)
# ---------------------------------------------------------------------------


def test_novita_completion_kwargs_structure():
    """novita_completion_kwargs returns required LiteLLM kwargs."""
    kwargs = novita_completion_kwargs("moonshotai/kimi-k2.5", api_key="test-key")
    assert kwargs["model"] == "openai/moonshotai/kimi-k2.5"
    assert "novita.ai" in kwargs["api_base"]
    assert kwargs["api_key"] == "test-key"


def test_novita_client_uses_novita_base():
    """NovitaClient defaults to the Novita API base URL."""
    client = NovitaClient(api_key="test-key")
    assert "novita.ai" in client.api_base
