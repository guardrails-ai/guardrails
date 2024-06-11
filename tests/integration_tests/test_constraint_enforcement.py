import importlib.util

import pytest

from guardrails.llm_providers import (
    PromptCallableException,
    chat_prompt,
    get_llm_ask,
)


@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
def test_get_llm_ask_litellm():
    from litellm import completion

    from guardrails.llm_providers import LiteLLMCallable

    prompt_callable = get_llm_ask(completion)

    assert isinstance(prompt_callable, LiteLLMCallable)


def test_chat_prompt():
    # raises when neither msg_history or prompt are provided
    with pytest.raises(PromptCallableException):
        chat_prompt(None)
