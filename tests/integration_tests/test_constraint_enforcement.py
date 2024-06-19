import importlib.util
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from guardrails.constraint_generator import BalancedBracesGenerator
from guardrails.llm_providers import LLMResponse


def _make_mock_tokenizer(output_token_array: list[int]):
    from transformers import BatchEncoding
    import torch

    class MockTokenizer:
        def __call__(self, prompt: str, *args: Any, **kwds: Any) -> Dict[str, Any]:
            self.prompt = prompt
            result = BatchEncoding()
            result["input_ids"] = torch.Tensor(output_token_array)
            return result

        def to(self, *args, **kwargs):
            return self

        def decode(self, output: str, *args, **kwargs) -> str:
            return output

    return MockTokenizer()


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
    constraint = BalancedBracesGenerator(max_depth=1)

    tokenizer = _make_mock_tokenizer([0])
    tokenizer_decode_spy = mocker.spy(tokenizer, "decode")
    model_generate = MagicMock()
    model_generate.return_value = ["{"]
    from guardrails.llm_providers import HuggingFaceModelCallable

    hf_model_callable = HuggingFaceModelCallable()
    response = hf_model_callable(
        "Balance these parenthesis:",
        model_generate=model_generate,
        tokenizer=tokenizer,
        constraint_generator=constraint,
    )

    assert tokenizer_decode_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "{"

    assert constraint.get_valid_tokens() == {"}"}


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_hugging_face_pipeline_callable():
    constraint = BalancedBracesGenerator(max_depth=1)

    pipeline = MagicMock()
    tokenizer_mock = _make_mock_tokenizer([0])
    pipeline.tokenizer = tokenizer_mock
    pipeline.return_value = [{"generated_text": "{"}]

    from guardrails.llm_providers import HuggingFacePipelineCallable

    assert constraint.get_valid_tokens() == {"{"}  # Can't close an unopened...

    hf_model_callable = HuggingFacePipelineCallable()
    response = hf_model_callable(
        "Balance these parenthesis:",
        pipeline=pipeline,
        constraint_generator=constraint,
    )

    assert isinstance(response, LLMResponse) is True
    assert response.output == "{"
    assert constraint.get_valid_tokens() == {"}"}
