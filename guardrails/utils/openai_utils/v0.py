from typing import Any, List

import openai
import openai.error
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.openai_utils.base import (
    BaseAsyncOpenAIClient,
    BaseSyncOpenAIClient,
)


def get_static_openai_create_func():
    return openai.Completion.create


def get_static_openai_chat_create_func():
    return openai.ChatCompletion.create


def get_static_openai_acreate_func():
    return openai.Completion.acreate


def get_static_openai_chat_acreate_func():
    return openai.ChatCompletion.acreate


OpenAIServiceUnavailableError = openai.error.ServiceUnavailableError

OPENAI_RETRYABLE_ERRORS = [
    openai.error.APIConnectionError,
    openai.error.APIError,
    openai.error.TryAgain,
    openai.error.Timeout,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError,
]
RETRYABLE_ERRORS = tuple(OPENAI_RETRYABLE_ERRORS)


class OpenAIClientV0(BaseSyncOpenAIClient):
    def create_embedding(
        self,
        model: str,
        input: List[str],
    ):
        response = openai.Embedding.create(
            api_key=self.api_key,
            model=model,
            input=input,
            api_base=self.api_base,
        )
        return [r["embedding"] for r in response["data"]]  # type: ignore

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    def create_completion(
        self, engine: str, prompt: str, *args, **kwargs
    ) -> LLMResponse:
        response = openai.Completion.create(
            api_key=self.api_key, engine=engine, prompt=prompt, *args, **kwargs
        )
        return LLMResponse(
            output=response["choices"][0]["text"],  # type: ignore
            prompt_token_count=response["usage"]["prompt_tokens"],  # type: ignore
            response_token_count=response["usage"]["completion_tokens"],  # type: ignore
        )

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    def create_chat_completion(
        self, model: str, messages: List[Any], *args, **kwargs
    ) -> LLMResponse:
        response = openai.ChatCompletion.create(
            api_key=self.api_key, model=model, prompt=messages, *args, **kwargs
        )

        # Extract string from response
        if "function_call" in response["choices"][0]["message"]:  # type: ignore
            output = response["choices"][0]["message"]["function_call"][  # type: ignore
                "arguments"
            ]
        else:
            output = response["choices"][0]["message"]["content"]  # type: ignore

        return LLMResponse(
            output=output,
            prompt_token_count=response["usage"]["prompt_tokens"],  # type: ignore
            response_token_count=response["usage"]["completion_tokens"],  # type: ignore
        )


class AsyncOpenAIClientV0(BaseAsyncOpenAIClient):
    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    async def create_completion(
        self, engine: str, prompt: str, *args, **kwargs
    ) -> LLMResponse:
        response = await openai.Completion.acreate(
            api_key=self.api_key, engine=engine, prompt=prompt, *args, **kwargs
        )
        return LLMResponse(
            output=response["choices"][0]["text"],  # type: ignore
            prompt_token_count=response["usage"]["prompt_tokens"],  # type: ignore
            response_token_count=response["usage"]["completion_tokens"],  # type: ignore
        )

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    async def create_chat_completion(
        self, model: str, messages: List[Any], *args, **kwargs
    ) -> LLMResponse:
        response = await openai.ChatCompletion.acreate(
            api_key=self.api_key, model=model, prompt=messages, *args, **kwargs
        )

        # Extract string from response
        if "function_call" in response["choices"][0]["message"]:  # type: ignore
            output = response["choices"][0]["message"]["function_call"][  # type: ignore
                "arguments"
            ]
        else:
            output = response["choices"][0]["message"]["content"]  # type: ignore

        return LLMResponse(
            output=output,
            prompt_token_count=response["usage"]["prompt_tokens"],  # type: ignore
            response_token_count=response["usage"]["completion_tokens"],  # type: ignore
        )
