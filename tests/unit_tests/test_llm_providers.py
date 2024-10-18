import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from guardrails.llm_providers import (
    ArbitraryCallable,
    AsyncArbitraryCallable,
    LLMResponse,
    PromptCallableException,
    chat_prompt,
    get_async_llm_ask,
    get_llm_ask,
)
from guardrails.utils.safe_get import safe_get_with_brackets

from .mocks import MockAsyncOpenAILlm, MockOpenAILlm


def test_openai_callable_does_not_retry_on_success(mocker):
    llm = MockOpenAILlm()
    succeed_spy = mocker.spy(llm, "succeed")

    arbitrary_callable = ArbitraryCallable(llm.succeed, prompt="Hello")
    response = arbitrary_callable()

    assert succeed_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.asyncio
async def test_async_openai_callable_does_not_retry_on_success(mocker):
    llm = MockAsyncOpenAILlm()
    succeed_spy = mocker.spy(llm, "succeed")

    arbitrary_callable = AsyncArbitraryCallable(llm.succeed, prompt="Hello")
    response = await arbitrary_callable()

    assert succeed_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.fixture(scope="module")
def openai_chat_mock():
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    return ChatCompletion(
        id="",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Mocked LLM output",
                    role="assistant",
                ),
            ),
        ],
        created=0,
        model="",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=20,
            prompt_tokens=10,
            total_tokens=30,
        ),
    )


@pytest.fixture(scope="module")
def openai_chat_stream_mock():
    def gen():
        # Returns a generator object
        for i in range(4, 8):
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"{i},"},
                        "finish_reason": None,
                    }
                ]
            }

    return gen()


@pytest.fixture(scope="module")
def openai_mock():
    @dataclass
    class MockCompletionUsage:
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int

    @dataclass
    class MockCompletionChoice:
        finish_reason: str
        index: int
        logprobs: Any
        text: str

    @dataclass
    class MockCompletion:
        id: str
        choices: List[MockCompletionChoice]
        created: int
        model: str
        object: str
        usage: MockCompletionUsage

    return MockCompletion(
        id="",
        choices=[
            MockCompletionChoice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                text="Mocked LLM output",
            ),
        ],
        created=0,
        model="",
        object="text_completion",
        usage=MockCompletionUsage(
            completion_tokens=20,
            prompt_tokens=10,
            total_tokens=30,
        ),
    )


@pytest.fixture(scope="module")
def openai_stream_mock():
    def gen():
        # Returns a generator object
        for i in range(4, 8):
            yield {
                "choices": [{"text": f"{i},", "finish_reason": None}],
                "model": "openai-model-name",
            }

    return gen()


def test_openai_callable(mocker, openai_mock):
    mocker.patch("openai.resources.Completions.create", return_value=openai_mock)

    from guardrails.llm_providers import OpenAICallable

    openai_callable = OpenAICallable()

    response = openai_callable(text="Hello")

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Mocked LLM output"
    assert response.prompt_token_count == 10
    assert response.response_token_count == 20


def test_openai_stream_callable(mocker, openai_stream_mock):
    mocker.patch("openai.resources.Completions.create", return_value=openai_stream_mock)

    from guardrails.llm_providers import OpenAICallable

    openai_callable = OpenAICallable()
    response = openai_callable(text="1,2,3,", stream=True)

    assert isinstance(response, LLMResponse) is True
    assert isinstance(response.stream_output, Iterable) is True

    actual_op = None
    i = 4
    for fragment in response.stream_output:
        actual_op = fragment["choices"][0]["text"]
        assert actual_op == f"{i},"
        i += 1


def test_openai_chat_callable(mocker, openai_chat_mock):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=openai_chat_mock,
    )
    from guardrails.llm_providers import OpenAIChatCallable

    openai_chat_callable = OpenAIChatCallable()
    response = openai_chat_callable(text="Hello")

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Mocked LLM output"
    assert response.prompt_token_count == 10
    assert response.response_token_count == 20


def test_openai_chat_stream_callable(mocker, openai_chat_stream_mock):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=openai_chat_stream_mock,
    )
    from guardrails.llm_providers import OpenAIChatCallable

    openai_chat_callable = OpenAIChatCallable()
    response = openai_chat_callable(text="1,2,3,", stream=True)

    assert isinstance(response, LLMResponse) is True
    assert isinstance(response.stream_output, Iterable) is True

    actual_op = None
    i = 4
    for fragment in response.stream_output:
        actual_op = fragment["choices"][0]["delta"]["content"]
        assert actual_op == f"{i},"
        i += 1


def test_openai_chat_model_callable(mocker, openai_chat_mock):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=openai_chat_mock,
    )

    from guardrails.llm_providers import OpenAIChatCallable

    class MyModel(BaseModel):
        a: str

    openai_chat_model_callable = OpenAIChatCallable()
    response = openai_chat_model_callable(
        text="Hello",
        base_model=MyModel,
    )

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Mocked LLM output"
    assert response.prompt_token_count == 10
    assert response.response_token_count == 20


@pytest.mark.skipif(
    not importlib.util.find_spec("manifest"),
    reason="manifest-ml is not installed",
)
def test_manifest_callable():
    client = MagicMock()
    client.run.return_value = "Hello world!"

    from guardrails.llm_providers import ManifestCallable

    manifest_callable = ManifestCallable()
    response = manifest_callable(text="Hello", client=client)

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("manifest"),
    reason="manifest-ml is not installed",
)
@pytest.mark.asyncio
async def test_async_manifest_callable():
    client = MagicMock()

    async def return_async():
        return ["Hello world!"]

    client.arun_batch.return_value = return_async()

    from guardrails.llm_providers import AsyncManifestCallable

    manifest_callable = AsyncManifestCallable()
    response = await manifest_callable(text="Hello", client=client)

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
@pytest.mark.parametrize(
    "model_inputs,tokenizer_call_count", [(None, 1), ({"input_ids": ["Hello"]}, 0)]
)
def test_hugging_face_model_callable(mocker, model_inputs, tokenizer_call_count):
    class MockTokenizer:
        def __call__(self, prompt: str, *args: Any, **kwds: Any) -> Dict[str, Any]:
            self.prompt = prompt
            return self

        def to(self, *args, **kwargs):
            return {"input_ids": [self.prompt]}

        def decode(self, output: str, *args, **kwargs) -> str:
            return output

    tokenizer = MockTokenizer()

    tokenizer_call_spy = mocker.spy(tokenizer, "to")
    tokenizer_decode_spy = mocker.spy(tokenizer, "decode")

    model_generate = MagicMock()
    model_generate.return_value = ["Hello there!"]

    from guardrails.llm_providers import HuggingFaceModelCallable

    hf_model_callable = HuggingFaceModelCallable()
    response = hf_model_callable(
        "Hello", model_generate=model_generate, tokenizer=tokenizer
    )

    assert tokenizer_call_spy.call_count == 1
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
    pipeline.return_value = [{"generated_text": "Hello there!"}]

    from guardrails.llm_providers import HuggingFacePipelineCallable

    hf_model_callable = HuggingFacePipelineCallable()
    response = hf_model_callable("Hello", pipeline=pipeline)

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello there!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
def test_litellm_callable(mocker):
    # Mock the litellm.completion function and
    # the classes it returns
    @dataclass
    class Message:
        content: str

    @dataclass
    class Choice:
        message: Message

    @dataclass
    class Usage:
        prompt_tokens: int
        completion_tokens: int

    @dataclass
    class MockResponse:
        choices: List[Choice]
        usage: Usage

    class MockCompletion:
        @staticmethod
        def create() -> MockResponse:
            return MockResponse(
                choices=[Choice(message=Message(content="Hello there!"))],
                usage=Usage(prompt_tokens=10, completion_tokens=20),
            )

    mocker.patch("litellm.completion", return_value=MockCompletion.create())

    from guardrails.llm_providers import LiteLLMCallable

    litellm_callable = LiteLLMCallable()
    response = litellm_callable("Hello")

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello there!"
    assert response.prompt_token_count == 10
    assert response.response_token_count == 20


class ReturnTempCallable(Callable):
    def __call__(
        self, prompt: str, *args, instructions=None, msg_history=None, **kwargs
    ) -> Any:
        return ""


@pytest.mark.parametrize(
    "llm_api, args, kwargs, expected_temperature",
    [
        (ReturnTempCallable(), [], {"temperature": 0.5}, 0.5),
        (ReturnTempCallable(), [], {}, 0),
    ],
)
def test_get_llm_ask_temperature(llm_api, args, kwargs, expected_temperature):
    result = get_llm_ask(llm_api, *args, **kwargs)
    assert "temperature" in result.init_kwargs
    assert result.init_kwargs["temperature"] == expected_temperature


@pytest.mark.skipif(
    not importlib.util.find_spec("openai"),
    reason="openai is not installed",
)
def test_get_llm_ask_openai_completion():
    import openai

    from guardrails.llm_providers import OpenAICallable

    completion_create = None
    completion_create = openai.completions.create
    prompt_callable = get_llm_ask(completion_create)

    assert isinstance(prompt_callable, OpenAICallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("openai"),
    reason="openai is not installed",
)
def test_get_llm_ask_openai_chat():
    import openai

    from guardrails.llm_providers import OpenAIChatCallable

    chat_completion_create = openai.chat.completions.create

    prompt_callable = get_llm_ask(chat_completion_create)

    assert isinstance(prompt_callable, OpenAIChatCallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("manifest"),
    reason="manifest is not installed",
)
def test_get_llm_ask_manifest(mocker):
    def mock_os_environ_get(key, *args):
        if key == "OPENAI_API_KEY":
            return "sk-xxxxxxxxxxxxxx"
        return safe_get_with_brackets(os.environ, key, *args)

    mocker.patch("os.environ.get", side_effect=mock_os_environ_get)

    from manifest import Manifest

    from guardrails.llm_providers import ManifestCallable

    manifest_client = Manifest("openai")

    prompt_callable = get_llm_ask(manifest_client)

    assert isinstance(prompt_callable, ManifestCallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("cohere"),
    reason="cohere is not installed",
)
def test_get_llm_ask_cohere():
    from cohere import Client

    from guardrails.llm_providers import CohereCallable

    cohere_client = Client(api_key="mock_api_key")

    prompt_callable = get_llm_ask(cohere_client.chat)

    assert isinstance(prompt_callable, CohereCallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("cohere"),
    reason="cohere is not installed",
)
def test_get_llm_ask_cohere_legacy():
    from cohere import Client

    from guardrails.llm_providers import CohereCallable

    cohere_client = Client(api_key="mock_api_key")

    prompt_callable = get_llm_ask(cohere_client.generate)

    assert isinstance(prompt_callable, CohereCallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("anthropic"),
    reason="anthropic is not installed",
)
def test_get_llm_ask_anthropic():
    if importlib.util.find_spec("anthropic"):
        from anthropic import Anthropic

        from guardrails.llm_providers import AnthropicCallable

        anthropic_client = Anthropic(api_key="my_api_key")
        prompt_callable = get_llm_ask(anthropic_client.completions.create)

        assert isinstance(prompt_callable, AnthropicCallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers"),
    reason="transformers is not installed",
)
def test_get_llm_ask_hugging_face_model(mocker):
    from transformers import PreTrainedModel

    from guardrails.llm_providers import HuggingFaceModelCallable

    class MockModel(PreTrainedModel):
        _modules: Any

        def __init__(self, *args, **kwargs):
            self._modules = {}

    mock_model = MockModel()

    prompt_callable = get_llm_ask(mock_model.generate)

    assert isinstance(prompt_callable, HuggingFaceModelCallable)


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


@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
def test_get_llm_ask_litellm():
    from litellm import completion

    from guardrails.llm_providers import LiteLLMCallable

    prompt_callable = get_llm_ask(completion)

    assert isinstance(prompt_callable, LiteLLMCallable)


def test_get_llm_ask_custom_llm():
    from guardrails.llm_providers import ArbitraryCallable

    def my_llm(prompt: str, *, instructions=None, msg_history=None, **kwargs) -> str:
        return f"Hello {prompt}!"

    prompt_callable = get_llm_ask(my_llm)

    assert isinstance(prompt_callable, ArbitraryCallable)


def test_get_llm_ask_custom_llm_warning():
    from guardrails.llm_providers import ArbitraryCallable

    def my_llm(prompt: str, **kwargs) -> str:
        return f"Hello {prompt}!"

    with pytest.warns(
        UserWarning,
        match=(
            "We recommend including 'instructions' and 'msg_history'"
            " as keyword-only arguments for custom LLM callables."
            " Doing so ensures these arguments are not uninentionally"
            " passed through to other calls via \\*\\*kwargs."
        ),
    ):
        prompt_callable = get_llm_ask(my_llm)

        assert isinstance(prompt_callable, ArbitraryCallable)


def test_get_llm_ask_custom_llm_must_accept_prompt():
    def my_llm() -> str:
        return "Hello!"

    with pytest.raises(
        ValueError,
        match="Custom LLM callables must accept at least one positional argument for prompt!",  # noqa
    ):
        get_llm_ask(my_llm)


def test_get_llm_ask_custom_llm_must_accept_kwargs():
    def my_llm(prompt: str) -> str:
        return f"Hello {prompt}!"

    with pytest.raises(
        ValueError, match="Custom LLM callables must accept \\*\\*kwargs!"
    ):
        get_llm_ask(my_llm)


def test_get_async_llm_ask_custom_llm():
    from guardrails.llm_providers import AsyncArbitraryCallable

    async def my_llm(
        prompt: str, *, instructions=None, msg_history=None, **kwargs
    ) -> str:
        return f"Hello {prompt}!"

    prompt_callable = get_async_llm_ask(my_llm)

    assert isinstance(prompt_callable, AsyncArbitraryCallable)


def test_get_async_llm_ask_custom_llm_warning():
    from guardrails.llm_providers import AsyncArbitraryCallable

    async def my_llm(prompt: str, **kwargs) -> str:
        return f"Hello {prompt}!"

    with pytest.warns(
        UserWarning,
        match=(
            "We recommend including 'instructions' and 'msg_history'"
            " as keyword-only arguments for custom LLM callables."
            " Doing so ensures these arguments are not uninentionally"
            " passed through to other calls via \\*\\*kwargs."
        ),
    ):
        prompt_callable = get_async_llm_ask(my_llm)

        assert isinstance(prompt_callable, AsyncArbitraryCallable)


def test_get_async_llm_ask_custom_llm_must_accept_prompt():
    async def my_llm() -> str:
        return "Hello!"

    with pytest.raises(
        ValueError,
        match="Custom LLM callables must accept at least one positional argument for prompt!",  # noqa
    ):
        get_async_llm_ask(my_llm)


def test_get_async_llm_ask_custom_llm_must_accept_kwargs():
    def my_llm(prompt: str) -> str:
        return f"Hello {prompt}!"

    with pytest.raises(
        ValueError, match="Custom LLM callables must accept \\*\\*kwargs!"
    ):
        get_async_llm_ask(my_llm)


def test_chat_prompt():
    # raises when neither msg_history or prompt are provided
    with pytest.raises(PromptCallableException):
        chat_prompt(None)
