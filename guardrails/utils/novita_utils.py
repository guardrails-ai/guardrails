"""Utilities for using Novita AI as an LLM provider with Guardrails.

Novita AI exposes an OpenAI-compatible endpoint at https://api.novita.ai/openai,
so it integrates naturally with Guardrails via LiteLLM.

Usage (LiteLLM path, recommended):

    import guardrails as gd
    from guardrails.utils.novita_utils import novita_completion_kwargs

    guard = gd.Guard()
    result = guard(
        messages=[{"role": "user", "content": "Hello!"}],
        **novita_completion_kwargs("moonshotai/kimi-k2.5"),
    )

Usage (direct OpenAI-SDK path):

    from guardrails.utils.novita_utils import NovitaClient

    client = NovitaClient()
    response = client.create_chat_completion(
        model="moonshotai/kimi-k2.5",
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

import os
from typing import Optional

from guardrails.utils.openai_utils.v1 import OpenAIClientV1 as OpenAIClient

NOVITA_API_BASE = "https://api.novita.ai/openai"

# Supported chat/completion model IDs
NOVITA_MODELS = [
    "moonshotai/kimi-k2.5",
    "zai-org/glm-5",
    "minimax/minimax-m2.5",
]

# Default model
NOVITA_DEFAULT_MODEL = "moonshotai/kimi-k2.5"


class NovitaClient(OpenAIClient):
    """OpenAI-compatible client pre-configured for Novita AI.

    Reads ``NOVITA_API_KEY`` from the environment when ``api_key`` is not
    supplied explicitly.  Falls back to ``OPENAI_API_KEY`` for compatibility
    with repos that share a single key env var.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if api_key is None:
            api_key = os.environ.get("NOVITA_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
        if api_base is None:
            api_base = os.environ.get("NOVITA_API_BASE", NOVITA_API_BASE)
        super().__init__(api_key=api_key, api_base=api_base, *args, **kwargs)


def novita_completion_kwargs(
    model: str = NOVITA_DEFAULT_MODEL,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> dict:
    """Return kwargs to pass to ``guard()`` for a Novita AI model via LiteLLM.

    LiteLLM routes to an OpenAI-compatible endpoint when ``api_base`` is
    provided alongside an ``openai/`` prefixed model string.

    Args:
        model: Novita model ID (e.g. ``"moonshotai/kimi-k2.5"``).
        api_key: Novita API key.  Defaults to ``NOVITA_API_KEY`` env var.
        api_base: Novita base URL.  Defaults to ``NOVITA_API_BASE``.

    Returns:
        Dict with ``model``, ``api_base``, and ``api_key`` ready to unpack
        into ``guard()``.
    """
    resolved_key = api_key or os.environ.get("NOVITA_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    resolved_base = api_base or os.environ.get("NOVITA_API_BASE", NOVITA_API_BASE)
    # LiteLLM requires the "openai/" prefix to route to a custom OpenAI-compatible base
    litellm_model = f"openai/{model}"
    return {
        "model": litellm_model,
        "api_base": resolved_base,
        "api_key": resolved_key,
    }


__all__ = [
    "NOVITA_API_BASE",
    "NOVITA_DEFAULT_MODEL",
    "NOVITA_MODELS",
    "NovitaClient",
    "novita_completion_kwargs",
]
