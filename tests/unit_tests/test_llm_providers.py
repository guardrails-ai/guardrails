
import pytest
import openai
from guardrails.llm_providers import get_llm_ask


@pytest.mark.parametrize("llm_api, args, kwargs, expected_temperature", [
    (openai.Completion.create, [], {'temperature': 0.5}, 0.5),
    (openai.Completion.create, [], {}, 0),
])

def test_get_llm_ask_temperature(llm_api, args, kwargs, expected_temperature):
    result = get_llm_ask(llm_api, *args, **kwargs)
    assert result.fn.keywords.get('temperature') == expected_temperature