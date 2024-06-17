import importlib
import pytest

from pydantic import BaseModel

from guardrails import Guard

if_transformers_installed = pytest.mark.skipif(
    "not importlib.util.find_spec('transformers')\
    or not importlib.util.find_spec('torch')",
    reason="Transformers / Torch not installed."
)


@if_transformers_installed
def test_hugging_face_model_callable():
    from tests.unit_tests.mocks.mock_hf_models import make_mock_model_and_tokenizer
    model, tokenizer = make_mock_model_and_tokenizer()

    class Foo(BaseModel):
        bar: str
        bez: list[str]

    g = Guard.from_pydantic(Foo, output_formatter="jsonformer")
    response = g(model.generate, tokenizer=tokenizer, prompt="test")
    validated_output = response.validated_output
    assert isinstance(validated_output, dict)
    assert "bar" in validated_output
    assert isinstance(validated_output["bez"], list)
    if len(validated_output["bez"]) > 0:
        assert isinstance(validated_output["bez"][0], str)


@if_transformers_installed
def test_hugging_face_pipeline_callable():
    from tests.unit_tests.mocks.mock_hf_models import make_random_pipeline
    model = make_random_pipeline()

    class Foo(BaseModel):
        bar: str
        bez: list[str]

    g = Guard.from_pydantic(Foo, output_formatter="jsonformer")
    response = g(model, prompt="Sample:")
    validated_output = response.validated_output
    assert isinstance(validated_output, dict)
    assert "bar" in validated_output
    assert isinstance(validated_output["bez"], list)
    if len(validated_output["bez"]) > 0:
        assert isinstance(validated_output["bez"][0], str)


@pytest.mark.skip(reason="Random model infinitely recurses on complex struct. Use GPT2")
def test_hugging_face_pipeline_complex_schema():
    from transformers import pipeline
    model = pipeline("text-generation", "gpt2")

    class MultiNum(BaseModel):
        whole: int
        frac: float

    class Tricky(BaseModel):
        foo: MultiNum

    # Note: If we used a real model we could do foo: list[MultiNum], but the random
    # model tends to get stuck in an infinite loop during list generation.

    g = Guard.from_pydantic(Tricky, output_formatter="jsonformer")
    response = g(model, prompt="Sample:")
    out = response.validated_output
    assert isinstance(out, dict)
    assert "foo" in out
    assert isinstance(out["foo"], dict)
    assert isinstance(out["foo"]["whole"], int | float)
    assert isinstance(out["foo"]["frac"], float)
