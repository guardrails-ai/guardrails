from typing import Any, AsyncIterable, Dict, Iterable, List, cast

import openai
import openai.error
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.openai_utils.base import (
    BaseAsyncOpenAIClient,
    BaseSyncOpenAIClient,
)
from guardrails.utils.openai_utils.streaming_utils import (
    num_tokens_from_messages,
    num_tokens_from_string,
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
        return self.construct_nonchat_response(
            stream=kwargs.get("stream", False),
            openai_response=response,
        )

    def construct_nonchat_response(
        self,
        stream: bool,
        openai_response: Any,
    ) -> LLMResponse:
        """Construct an LLMResponse from an OpenAI response.

        Splits execution based on whether the `stream` parameter is set
        in the kwargs.
        """
        if stream:
            # If stream is defined and set to True,
            # openai returns a generator
            openai_response = cast(Iterable[Dict[str, Any]], openai_response)

            # Simply return the generator wrapped in an LLMResponse
            return LLMResponse(output="", stream_output=openai_response)

        # If stream is not defined or is set to False,
        # return default behavior
        openai_response = cast(Dict[str, Any], openai_response)
        return LLMResponse(
            output=openai_response["choices"][0]["text"],  # type: ignore
            prompt_token_count=openai_response["usage"][  # type: ignore
                "prompt_tokens"
            ],
            response_token_count=openai_response["usage"][  # type: ignore
                "completion_tokens"
            ],
        )

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    def create_chat_completion(
        self, model: str, messages: List[Any], *args, **kwargs
    ) -> LLMResponse:
        response = openai.ChatCompletion.create(
            api_key=self.api_key, model=model, messages=messages, *args, **kwargs
        )

        return self.construct_chat_response(
            stream=kwargs.get("stream", False),
            openai_response=response,
        )

    def construct_chat_response(
        self,
        stream: bool,
        openai_response: Any,
    ) -> LLMResponse:
        """Construct an LLMResponse from an OpenAI response.

        Splits execution based on whether the `stream` parameter is set
        in the kwargs.
        """
        if stream:
            # If stream is defined and set to True,
            # openai returns a generator object
            openai_response = cast(Iterable[Dict[str, Any]], openai_response)

            # Simply return the generator wrapped in an LLMResponse
            return LLMResponse(output="", stream_output=openai_response)

        # If stream is not defined or is set to False,
        # extract string from response
        openai_response = cast(Dict[str, Any], openai_response)
        if "function_call" in openai_response["choices"][0]["message"]:  # type: ignore
            output = openai_response["choices"][0]["message"][  # type: ignore
                "function_call"
            ]["arguments"]
        else:
            output = openai_response["choices"][0]["message"]["content"]  # type: ignore

        return LLMResponse(
            output=output,
            prompt_token_count=openai_response["usage"][  # type: ignore
                "prompt_tokens"
            ],
            response_token_count=openai_response["usage"][  # type: ignore
                "completion_tokens"
            ],
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
        return await self.construct_nonchat_response(
            stream=kwargs.get("stream", False),
            openai_response=response,
            prompt=prompt,
            engine=engine,
        )

    async def construct_nonchat_response(
        self,
        stream: bool,
        openai_response: Any,
        prompt: str,
        engine: str,
    ) -> LLMResponse:
        if stream:
            # If stream is defined and set to True,
            # openai returns a generator object
            complete_output = ""
            openai_response = cast(AsyncIterable[Dict[str, Any]], openai_response)
            async for response in openai_response:
                complete_output += response["choices"][0]["text"]

            # Also, it no longer returns usage information
            # So manually count the tokens using tiktoken
            prompt_token_count = num_tokens_from_string(
                text=prompt,
                model_name=engine,
            )
            response_token_count = num_tokens_from_string(
                text=complete_output, model_name=engine
            )

            # Return the LLMResponse
            return LLMResponse(
                output=complete_output,
                prompt_token_count=prompt_token_count,
                response_token_count=response_token_count,
            )

        # If stream is not defined or is set to False,
        # extract string from response
        openai_response = cast(Dict[str, Any], openai_response)
        return LLMResponse(
            output=openai_response["choices"][0]["text"],  # type: ignore
            prompt_token_count=openai_response["usage"][  # type: ignore
                "prompt_tokens"
            ],
            response_token_count=openai_response["usage"][  # type: ignore
                "completion_tokens"
            ],
        )

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    async def create_chat_completion(
        self, model: str, messages: List[Any], *args, **kwargs
    ) -> LLMResponse:
        response = await openai.ChatCompletion.acreate(
            api_key=self.api_key, model=model, messages=messages, *args, **kwargs
        )

        return await self.construct_chat_response(
            stream=kwargs.get("stream", False),
            openai_response=response,
            prompt=messages,
            model=model,
        )

    async def construct_chat_response(
        self,
        stream: bool,
        openai_response: Any,
        prompt: List[Any],
        model: str,
    ) -> LLMResponse:
        """Construct an LLMResponse from an OpenAI response.

        Splits execution based on whether the `stream` parameter is set
        in the kwargs.
        """
        if stream:
            # If stream is defined and set to True,
            # openai returns a generator object
            collected_messages = []
            openai_response = cast(AsyncIterable[Dict[str, Any]], openai_response)
            async for chunk in openai_response:
                chunk_message = chunk["choices"][0]["delta"]
                collected_messages.append(chunk_message)  # save the message

            complete_output = "".join(
                [msg.get("content", "") for msg in collected_messages]
            )

            # Also, it no longer returns usage information
            # So manually count the tokens using tiktoken
            prompt_token_count = num_tokens_from_messages(
                messages=prompt,
                model=model,
            )
            response_token_count = num_tokens_from_string(
                text=complete_output, model_name=model
            )

            # Return the LLMResponse
            return LLMResponse(
                output=complete_output,
                prompt_token_count=prompt_token_count,
                response_token_count=response_token_count,
            )

        # If stream is not defined or is set to False,
        # Extract string from response
        openai_response = cast(Dict[str, Any], openai_response)
        if "function_call" in openai_response["choices"][0]["message"]:  # type: ignore
            output = openai_response["choices"][0]["message"][  # type: ignore
                "function_call"
            ]["arguments"]
        else:
            output = openai_response["choices"][0]["message"]["content"]  # type: ignore

        return LLMResponse(
            output=output,
            prompt_token_count=openai_response["usage"][  # type: ignore
                "prompt_tokens"
            ],
            response_token_count=openai_response["usage"][  # type: ignore
                "completion_tokens"
            ],
        )
