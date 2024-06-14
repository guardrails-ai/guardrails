import importlib
import pytest

from pydantic import BaseModel

from guardrails import Guard


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_hugging_face_model_callable():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # TODO: Don't actually pull GPT-2 during the test.
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    class Foo(BaseModel):
        bar: str
        bez: list[str]

    g = Guard.from_pydantic(Foo, output_formatter="jsonformer")
    response = g(model.generate, tokenizer=tokenizer, prompt="Sample:")
    validated_output = response.validated_output
    assert isinstance(validated_output, dict)
    assert "bar" in validated_output
    assert "bez" in validated_output
    assert isinstance(validated_output["bez"], list)
    if len(validated_output["bez"]) > 0:
        assert isinstance(validated_output["bez"][0], str)


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_hugging_face_pipeline_callable():
    from transformers import pipeline

    # TODO: Don't actually pull GPT-2 during the test.
    model = pipeline("text-generation", "openai-community/gpt2")

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


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_hugging_face_pipeline_complex_schema():
    from transformers import pipeline

    # TODO: Don't actually pull GPT-2 during the test.
    model = pipeline("text-generation", "openai-community/gpt2")

    class MultiNum(BaseModel):
        whole: int
        frac: float

    class Tricky(BaseModel):
        foo: list[str]
        bar: list[MultiNum]
        bez: MultiNum

    g = Guard.from_pydantic(Tricky, output_formatter="jsonformer")
    response = g(model, prompt="Sample:")
    out = response.validated_output
    assert isinstance(out, dict)
    assert "foo" in out
    assert isinstance(out["foo"], list)
    if len(out["foo"]) > 0:
        assert isinstance(out["foo"][0], str)
    assert "bar" in out
    if len(out["bar"]) > 0:
        assert isinstance(out["bar"][0], dict)
    assert "bez" in out
    assert isinstance(out["bez"]["whole"], int | float)
    assert isinstance(out["bez"]["whole"], int | float)
