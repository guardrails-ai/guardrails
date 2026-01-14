"""Integration tests for the Guardrails server Docker image.

These tests run against a containerized Guardrails server using testcontainers.
The container is started once per test session via fixtures in conftest.py.

Requirements:
- Docker image must be pre-built (default: guardrails-server:ci)
- GUARDRAILS_TOKEN env var for hub extension installation
- GUARDRAILS_EXTENSIONS env var (e.g., "guardrails/two_words")

Optional:
- OLLAMA_HOST: Use host Ollama instead of container (faster local testing)
- OLLAMA_MODEL: Override default model (default: smollm:360m)
"""

import openai
import pytest
from guardrails import AsyncGuard, Guard


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def openai_client(server_url, ollama_model):
    """OpenAI client configured to use the Guardrails server."""
    return openai.OpenAI(
        base_url=f"{server_url}/guards/test-guard/openai/v1/",
        api_key="not-needed-for-ollama",
    )


class TestGuardValidation:
    """Tests for guard.validate() without LLM calls."""

    def test_sync_validate(self):
        guard = Guard(name="test-guard", api_key="auth-stub")
        result = guard.validate("France is wonderful in the spring")

        assert result.validation_passed is True
        assert result.validated_output == "France is"

    @pytest.mark.asyncio
    async def test_async_validate(self):
        guard = AsyncGuard(name="test-guard", api_key="auth-stub")
        result = await guard.validate("France is wonderful in the spring")

        assert result.validation_passed is True
        assert result.validated_output == "France is"


class TestLLMIntegration:
    """Tests that make LLM calls through the Guardrails server using Ollama."""

    def test_sync_guard_call(self, ollama_model):
        guard = Guard(name="test-guard", api_key="auth-stub")
        result = guard(
            model=ollama_model,
            messages=[{"role": "user", "content": "Say hello in two words"}],
            temperature=0.0,
        )

        assert result.validation_passed is True
        # two_words validator ensures output is exactly two words
        assert len(result.validated_output.split()) == 2

    @pytest.mark.asyncio
    async def test_async_guard_call(self, ollama_model):
        guard = AsyncGuard(name="test-guard", api_key="auth-stub")
        result = await guard(
            model=ollama_model,
            messages=[{"role": "user", "content": "Say hello in two words"}],
            temperature=0.0,
        )

        assert result.validation_passed is True

    @pytest.mark.asyncio
    async def test_async_streaming(self, ollama_model):
        guard = AsyncGuard(name="test-guard", api_key="auth-stub")
        stream = await guard(
            model=ollama_model,
            messages=[{"role": "user", "content": "Say hello in two words"}],
            stream=True,
            temperature=0.0,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk.validated_output)

        assert len(chunks) > 0
        full_output = "".join(c for c in chunks if c)
        assert len(full_output) > 0

    def test_sync_streaming(self, ollama_model):
        import os

        # FIXME: This env var requirement is concerning
        os.environ["GUARD_HISTORY_ENABLED"] = "false"

        guard = Guard(name="test-guard", api_key="auth-stub")
        stream = guard(
            model=ollama_model,
            messages=[{"role": "user", "content": "Say hello in two words"}],
            stream=True,
            temperature=0.0,
        )

        chunks = []
        for chunk in stream:
            chunks.append(chunk.validated_output)

        assert len(chunks) > 0

    def test_openai_compatible_endpoint(self, openai_client, ollama_model):
        """Test the OpenAI-compatible API endpoint."""
        completion = openai_client.chat.completions.create(
            model=ollama_model,
            messages=[{"role": "user", "content": "Say hello in two words"}],
            temperature=0.0,
        )

        assert completion.choices[0].message.content
        assert completion.guardrails["validation_passed"] is True
