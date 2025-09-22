import importlib
import pytest
from typing import List

from pydantic import BaseModel

from guardrails import Guard


if_transformers_installed = pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    or not importlib.util.find_spec("torch"),
    reason="Transformers / Torch not installed.",
)


@if_transformers_installed
def test_hugging_face_model_callable():
    from tests.unit_tests.mocks.mock_hf_models import make_mock_model_and_tokenizer

    model, tokenizer = make_mock_model_and_tokenizer()

    class Foo(BaseModel):
        bar: str
        bez: List[str]

    g = Guard.for_pydantic(Foo, output_formatter="jsonformer")
    response = g(
        model.generate,
        tokenizer=tokenizer,
        messages=[{"content": "test", "role": "user"}],
    )
    validated_output = response.validated_output
    assert isinstance(validated_output, dict)
    assert "bar" in validated_output
    assert isinstance(validated_output["bez"], list)
    if len(validated_output["bez"]) > 0:
        assert isinstance(validated_output["bez"][0], str)


@if_transformers_installed
def test_hugging_face_pipeline_callable():
    from tests.unit_tests.mocks.mock_hf_models import make_mock_pipeline

    model = make_mock_pipeline()

    class Foo(BaseModel):
        bar: str
        bez: List[str]

    g = Guard.for_pydantic(Foo, output_formatter="jsonformer")
    response = g(model, messages=[{"content": "Sample:", "role": "user"}])
    validated_output = response.validated_output
    assert isinstance(validated_output, dict)
    assert "bar" in validated_output
    assert isinstance(validated_output["bez"], list)
    if len(validated_output["bez"]) > 0:
        assert isinstance(validated_output["bez"][0], str)


@if_transformers_installed
def test_hugging_face_pipeline_complex_schema():
    # NOTE: This is the real GPT-2 model.
    from transformers import pipeline

    model = pipeline("text-generation", "distilgpt2")

    class MultiNum(BaseModel):
        whole: int
        frac: float

    class Tricky(BaseModel):
        foo: MultiNum

    g = Guard.for_pydantic(Tricky, output_formatter="jsonformer")
    response = g(model, messages=[{"content": "Sample:", "role": "user"}])
    out = response.validated_output
    assert isinstance(out, dict)
    assert "foo" in out
    assert isinstance(out["foo"], dict)
    assert isinstance(out["foo"]["whole"], int) or isinstance(
        out["foo"]["whole"], float
    )
    assert isinstance(out["foo"]["frac"], float)
