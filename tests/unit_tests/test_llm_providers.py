import importlib.util
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import pytest

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

    arbitrary_callable = ArbitraryCallable(
        llm.succeed, messages=[{"role": "user", "content": "Hello"}]
    )
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

    arbitrary_callable = AsyncArbitraryCallable(
        llm.succeed, messages=[{"role": "user", "content": "Hello"}]
    )
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
        model_generate=model_generate,
        messages=[{"role": "user", "content": "Hello"}],
        tokenizer=tokenizer,
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
    response = hf_model_callable(
        pipeline=pipeline, messages=[{"role": "user", "content": "Hello"}]
    )

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
    def __call__(self, *args, messages=None, **kwargs) -> Any:
        return ""


@pytest.mark.parametrize(
    "llm_api, args, kwargs, expected_temperature",
    [
        (ReturnTempCallable(), [], {"temperature": 0.5}, 0.5),
        (ReturnTempCallable(), [], {}, 0),
        (ReturnTempCallable(), [], {"model": "gpt-5-nano"}, None),
    ],
)
def test_get_llm_ask_temperature(llm_api, args, kwargs, expected_temperature):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = get_llm_ask(llm_api, *args, **kwargs)
        if expected_temperature is None:
            assert "temperature" not in result.init_kwargs
            assert len(w) == 0
        else:
            assert "temperature" in result.init_kwargs
            assert result.init_kwargs["temperature"] == expected_temperature
            if "temperature" not in kwargs:
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "default value of 0 for temperature is deprecated" in str(
                    w[0].message
                )


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
    not importlib.util.find_spec("transformers"),
    reason="transformers is not installed",
)
def test_get_llm_ask_hugging_face_model(mocker):
    from transformers import PreTrainedModel, GenerationMixin

    from guardrails.llm_providers import HuggingFaceModelCallable

    class MockModel(PreTrainedModel, GenerationMixin):
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

    def my_llm(prompt: str, *, messages=None, **kwargs) -> str:
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
            "We recommend including 'messages'"
            " as keyword-only arguments for custom LLM callables."
            " Doing so ensures these arguments are not unintentionally"
            " passed through to other calls via \\*\\*kwargs."
        ),
    ):
        prompt_callable = get_llm_ask(my_llm)

        assert isinstance(prompt_callable, ArbitraryCallable)


def test_get_llm_ask_custom_llm_must_accept_kwargs():
    def my_llm(messages: str) -> str:
        return f"Hello {messages}!"

    with pytest.raises(
        ValueError, match="Custom LLM callables must accept \\*\\*kwargs!"
    ):
        get_llm_ask(my_llm)


def test_get_async_llm_ask_custom_llm():
    from guardrails.llm_providers import AsyncArbitraryCallable

    async def my_llm(messages: str, **kwargs) -> str:
        return f"Hello {messages}!"

    prompt_callable = get_async_llm_ask(my_llm)

    assert isinstance(prompt_callable, AsyncArbitraryCallable)


def test_get_async_llm_ask_custom_llm_warning():
    from guardrails.llm_providers import AsyncArbitraryCallable

    async def my_llm(**kwargs) -> str:
        return "Hello world!"

    with pytest.warns(
        UserWarning,
        match=(
            "We recommend including 'messages'"
            " as keyword-only arguments for custom LLM callables."
            " Doing so ensures these arguments are not unintentionally"
            " passed through to other calls via \\*\\*kwargs."
        ),
    ):
        prompt_callable = get_async_llm_ask(my_llm)

        assert isinstance(prompt_callable, AsyncArbitraryCallable)


def test_get_async_llm_ask_custom_llm_must_accept_kwargs():
    def my_llm(prompt: str) -> str:
        return f"Hello {prompt}!"

    with pytest.raises(
        ValueError, match="Custom LLM callables must accept \\*\\*kwargs!"
    ):
        get_async_llm_ask(my_llm)


def test_chat_prompt():
    # raises when messages are not provided
    with pytest.raises(PromptCallableException):
        chat_prompt(None)


# ---------------------------------------------------------------------------
# MiniMax provider tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimax_chat_response():
    """A minimal mock of openai.types.chat.ChatCompletion for MiniMax."""
    from dataclasses import dataclass

    @dataclass
    class MockUsage:
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int

    @dataclass
    class MockMessage:
        content: str
        role: str

    @dataclass
    class MockChoice:
        finish_reason: str
        index: int
        message: MockMessage

    @dataclass
    class MockResponse:
        id: str
        choices: List[MockChoice]
        created: int
        model: str
        object: str
        usage: MockUsage

    return MockResponse(
        id="minimax-test-id",
        choices=[
            MockChoice(
                finish_reason="stop",
                index=0,
                message=MockMessage(content="Hello from MiniMax!", role="assistant"),
            )
        ],
        created=0,
        model="MiniMax-M2.7",
        object="chat.completion",
        usage=MockUsage(completion_tokens=10, prompt_tokens=5, total_tokens=15),
    )


def test_minimax_callable_uses_text(mocker, minimax_chat_response):
    """MiniMaxCallable converts plain text into a user message."""
    from guardrails.llm_providers import MiniMaxCallable

    mock_create = mocker.MagicMock(return_value=minimax_chat_response)
    mocker.patch(
        "openai.OpenAI",
        return_value=mocker.MagicMock(
            chat=mocker.MagicMock(
                completions=mocker.MagicMock(create=mock_create)
            )
        ),
    )

    callable_obj = MiniMaxCallable(
        text="Say hi",
        model="minimax/MiniMax-M2.7",
        api_key="test-key",
        temperature=0.8,
    )
    response = callable_obj()

    assert isinstance(response, LLMResponse)
    assert response.output == "Hello from MiniMax!"
    assert response.prompt_token_count == 5
    assert response.response_token_count == 10


def test_minimax_callable_strips_prefix(mocker, minimax_chat_response):
    """The 'minimax/' prefix is stripped before calling the OpenAI SDK."""
    from guardrails.llm_providers import MiniMaxCallable

    mock_create = mocker.MagicMock(return_value=minimax_chat_response)
    client_mock = mocker.MagicMock(
        chat=mocker.MagicMock(
            completions=mocker.MagicMock(create=mock_create)
        )
    )
    mocker.patch("openai.OpenAI", return_value=client_mock)

    MiniMaxCallable(
        text="Hello",
        model="minimax/MiniMax-M2.7",
        api_key="test-key",
        temperature=0.7,
    )()

    call_kwargs = mock_create.call_args
    # The first positional-or-keyword argument is 'model'
    assert call_kwargs.kwargs["model"] == "MiniMax-M2.7"


def test_minimax_callable_temperature_clamped(mocker, minimax_chat_response):
    """temperature=0 is clamped to 1.0 for MiniMax."""
    from guardrails.llm_providers import MiniMaxCallable

    mock_create = mocker.MagicMock(return_value=minimax_chat_response)
    mocker.patch(
        "openai.OpenAI",
        return_value=mocker.MagicMock(
            chat=mocker.MagicMock(
                completions=mocker.MagicMock(create=mock_create)
            )
        ),
    )

    MiniMaxCallable(
        text="Hello",
        model="minimax/MiniMax-M2.7",
        api_key="test-key",
        temperature=0,
    )()

    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs["temperature"] == 1.0


def test_minimax_callable_default_base_url(mocker, minimax_chat_response):
    """Default base URL points to the international MiniMax endpoint."""
    from guardrails.llm_providers import MINIMAX_DEFAULT_BASE_URL, MiniMaxCallable

    openai_cls = mocker.patch("openai.OpenAI")
    openai_cls.return_value.chat.completions.create.return_value = minimax_chat_response

    MiniMaxCallable(
        text="Hello",
        model="minimax/MiniMax-M2.7",
        api_key="test-key",
        temperature=0.8,
    )()

    openai_cls.assert_called_once_with(
        api_key="test-key", base_url=MINIMAX_DEFAULT_BASE_URL
    )


def test_minimax_callable_missing_api_key(mocker):
    """PromptCallableException is raised when no API key is provided."""
    from guardrails.llm_providers import MiniMaxCallable

    mocker.patch.dict(os.environ, {}, clear=True)
    # Ensure MINIMAX_API_KEY is not present
    os.environ.pop("MINIMAX_API_KEY", None)

    callable_obj = MiniMaxCallable(
        text="Hello",
        model="minimax/MiniMax-M2.7",
        temperature=0.8,
    )

    with pytest.raises(PromptCallableException, match="MINIMAX_API_KEY"):
        callable_obj()


def test_get_llm_ask_returns_minimax_callable_for_minimax_model():
    """get_llm_ask routes 'minimax/' models to MiniMaxCallable."""
    from guardrails.llm_providers import MiniMaxCallable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        callable_obj = get_llm_ask(None, model="minimax/MiniMax-M2.7")

    assert isinstance(callable_obj, MiniMaxCallable)


def test_get_llm_ask_returns_minimax_callable_for_highspeed_model():
    """get_llm_ask routes 'minimax/MiniMax-M2.7-highspeed' to MiniMaxCallable."""
    from guardrails.llm_providers import MiniMaxCallable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        callable_obj = get_llm_ask(None, model="minimax/MiniMax-M2.7-highspeed")

    assert isinstance(callable_obj, MiniMaxCallable)


def test_get_async_llm_ask_returns_async_minimax_callable():
    """get_async_llm_ask routes 'minimax/' models to AsyncMiniMaxCallable."""
    from guardrails.llm_providers import AsyncMiniMaxCallable

    callable_obj = get_async_llm_ask(None, model="minimax/MiniMax-M2.7")

    assert isinstance(callable_obj, AsyncMiniMaxCallable)
