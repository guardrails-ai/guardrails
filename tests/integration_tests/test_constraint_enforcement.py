import importlib.util
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from guardrails.constraint_generator import BalancedBracesGenerator
from guardrails.llm_providers import LLMResponse, get_llm_ask


"""
@pytest.mark.parametrize(
    "model_inputs,tokenizer_call_count", [(None, 1), ({"input_ids": ["Hello"]}, 0)]
)
def test_hugging_face_model_callable(mocker, model_inputs, tokenizer_call_count):
"""


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_hugging_face_model_callable(mocker):
    class MockTensor:
        def __init__(self, input_ids):
            self.input_ids = input_ids
            self.do_sample = None

        def to(self, *args, **kwargs):
            return self

        def __setattr__(self, key, value):
            pass

    class MockTokenizer:
        def __call__(self, prompt: str, *args: Any, **kwds: Any) -> Dict[str, Any]:
            self.prompt = prompt
            tensor = MockTensor(input_ids=["{"])
            return tensor

        def to(self, *args, **kwargs):
            return self

        def decode(self, output: str, *args, **kwargs) -> str:
            return output

    tokenizer = MockTokenizer()

    tokenizer_decode_spy = mocker.spy(tokenizer, "decode")

    class MockModel:
        def __call__(self, input_ids, *args, **kwargs):
            return ["{"]

    # model_generate = MagicMock()
    # model_generate.return_value = ["{"]
    model_generate = MockModel()

    from guardrails.llm_providers import HuggingFaceModelCallable

    hf_model_callable = HuggingFaceModelCallable()
    response = hf_model_callable(
        "Balance these parenthesis:",
        model_generate=model_generate,
        tokenizer=tokenizer,
        constraint_generator=BalancedBracesGenerator(max_depth=1),
    )

    assert tokenizer_decode_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello there!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_hugging_face_pipeline_callable():
    pipeline = MagicMock()
    tokenizer_mock = MagicMock()
    pipeline.tokenizer = tokenizer_mock
    pipeline.return_value = [{"generated_text": "Hello there!"}]

    from guardrails.llm_providers import HuggingFacePipelineCallable

    hf_model_callable = HuggingFacePipelineCallable()
    response = hf_model_callable("Hello", pipeline=pipeline)

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello there!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers"),
    reason="transformers is not installed",
)
def test_get_llm_ask_hugging_face_pipeline():
    from transformers import Pipeline

    from guardrails.llm_providers import HuggingFacePipelineCallable

    class MockPipeline(Pipeline):
        task = "text-generation"

        def __init__(self, *args, **kwargs):
            pass

        def _forward():
            pass

        def _sanitize_parameters():
            pass

        def postprocess():
            pass

        def preprocess():
            pass

    mock_pipeline = MockPipeline()

    prompt_callable = get_llm_ask(mock_pipeline)

    assert isinstance(prompt_callable, HuggingFacePipelineCallable)
