from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, cast

import openai

import warnings
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.utils.openai_utils.base import BaseOpenAIClient
from guardrails.utils.openai_utils.streaming_utils import (
    num_tokens_from_messages,
    num_tokens_from_string,
)
from guardrails.utils.safe_get import safe_get
from guardrails.telemetry import trace_llm_call, trace_operation


def get_static_openai_create_func():
    warnings.warn(
        "This function is deprecated. " " and will be removed in 0.6.0",
        DeprecationWarning,
    )
    return openai.completions.create


def get_static_openai_chat_create_func():
    warnings.warn(
        "This function is deprecated and will be removed in 0.6.0",
        DeprecationWarning,
    )
    return openai.chat.completions.create


def get_static_openai_acreate_func():
    warnings.warn(
        "This function is deprecated and will be removed in 0.6.0",
        DeprecationWarning,
    )
    return None


def get_static_openai_chat_acreate_func():
    warnings.warn(
        "This function is deprecated and will be removed in 0.6.0",
        DeprecationWarning,
    )
    return None


def is_static_openai_create_func(llm_api: Optional[Callable]) -> bool:
    try:
        return llm_api == openai.completions.create
    except openai.OpenAIError:
        return False


def is_static_openai_chat_create_func(llm_api: Optional[Callable]) -> bool:
    try:
        return llm_api == openai.chat.completions.create
    except openai.OpenAIError:
        return False


def is_static_openai_acreate_func(llm_api: Optional[Callable]) -> bool:
    # Because the static version of this does not exist in OpenAI 1.x
    # Can we just drop these checks?
    return False


def is_static_openai_chat_acreate_func(llm_api: Optional[Callable]) -> bool:
    # Because the static version of this does not exist in OpenAI 1.x
    # Can we just drop these checks?
    return False


OpenAIServiceUnavailableError = openai.APIError


class OpenAIClientV1(BaseOpenAIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = openai.Client(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def create_embedding(
        self,
        model: str,
        input: List[str],
    ) -> List[List[float]]:
        embeddings = self.client.embeddings.create(
            model=model,
            input=input,
        )
        return [r.embedding for r in embeddings.data]

    def create_completion(
        self, engine: str, prompt: str, *args, **kwargs
    ) -> LLMResponse:
        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "model": engine,
                "prompt": prompt,
                "args": args,
            },
        )

        trace_llm_call(
            invocation_parameters={
                **kwargs,
                "model": engine,
                "prompt": prompt,
            }
        )

        response = self.client.completions.create(
            model=engine, prompt=prompt, *args, **kwargs
        )

        trace_operation(output_mime_type="application/json", output_value=response)

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
            openai_response = cast(Iterator[Dict[str, Any]], openai_response)

            # Simply return the generator wrapped in an LLMResponse
            return LLMResponse(output="", stream_output=openai_response)

        # If stream is not defined or is set to False,
        # return default behavior
        openai_response = cast(Dict[str, Any], openai_response)
        if not openai_response.choices:
            raise ValueError("No choices returned from OpenAI")
        if openai_response.usage is None:
            raise ValueError("No token counts returned from OpenAI")
        trace_llm_call(
            output_messages=[
                {"role": "assistant", "content": openai_response.choices[0].text}
            ],
            token_count_completion=openai_response.usage.completion_tokens,
            token_count_prompt=openai_response.usage.prompt_tokens,
            token_count_total=openai_response.usage.total_tokens,
        )
        return LLMResponse(
            output=openai_response.choices[0].text,  # type: ignore
            prompt_token_count=openai_response.usage.prompt_tokens,  # type: ignore
            response_token_count=openai_response.usage.completion_tokens,  # noqa: E501 # type: ignore
        )

    def create_chat_completion(
        self, model: str, messages: List[Any], *args, **kwargs
    ) -> LLMResponse:
        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "model": model,
                "messages": messages,
                "args": args,
            },
        )
        function_calling_tools = [
            tool.get("function")
            for tool in kwargs.get("tools", [])
            if isinstance(tool, Dict) and tool.get("type") == "function"
        ]
        trace_llm_call(
            input_messages=messages,
            model_name=model,
            invocation_parameters={**kwargs, "model": model, "messages": messages},
            function_call=kwargs.get(
                "function_call", safe_get(function_calling_tools, 0)
            ),
        )
        response = self.client.chat.completions.create(
            model=model, messages=messages, *args, **kwargs
        )

        trace_operation(output_mime_type="application/json", output_value=response)

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
            openai_response = cast(Iterator[Dict[str, Any]], openai_response)

            # Simply return the generator wrapped in an LLMResponse
            return LLMResponse(output="", stream_output=openai_response)

        # If stream is not defined or is set to False,
        # extract string from response
        openai_response = cast(Dict[str, Any], openai_response)
        if not openai_response.choices:
            raise ValueError("No choices returned from OpenAI")
        if not openai_response.choices[0].message:
            raise ValueError("No message returned from OpenAI")
        if openai_response.usage is None:
            raise ValueError("No token counts returned from OpenAI")

        if openai_response.choices[0].message.content is not None:
            output = openai_response.choices[0].message.content
        else:
            try:
                output = openai_response.choices[0].message.function_call.arguments
            except AttributeError:
                try:
                    choice = openai_response.choices[0]
                    output = choice.message.tool_calls[-1].function.arguments
                except AttributeError as ae_tools:
                    raise ValueError(
                        "No message content or function"
                        " call arguments returned from OpenAI"
                    ) from ae_tools
        trace_llm_call(
            output_messages=[choice.message for choice in openai_response.choices],  # type: ignore
            token_count_completion=openai_response.usage.completion_tokens,
            token_count_prompt=openai_response.usage.prompt_tokens,
            token_count_total=openai_response.usage.total_tokens,
        )
        return LLMResponse(
            output=output,
            prompt_token_count=openai_response.usage.prompt_tokens,  # type: ignore
            response_token_count=openai_response.usage.completion_tokens,  # noqa: E501 # type: ignore
        )


class AsyncOpenAIClientV1(BaseOpenAIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = openai.AsyncClient(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    async def create_embedding(
        self,
        model: str,
        input: List[str],
    ) -> List[List[float]]:
        embeddings = await self.client.embeddings.create(
            model=model,
            input=input,
        )
        return [r.embedding for r in embeddings.data]

    async def create_completion(
        self, engine: str, prompt: str, *args, **kwargs
    ) -> LLMResponse:
        response = await self.client.completions.create(
            model=engine, prompt=prompt, *args, **kwargs
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
            openai_response = cast(AsyncIterator[Dict[str, Any]], openai_response)
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
        if not openai_response.choices:
            raise ValueError("No choices returned from OpenAI")
        if openai_response.usage is None:
            raise ValueError("No token counts returned from OpenAI")
        return LLMResponse(
            output=openai_response.choices[0].text,  # type: ignore
            prompt_token_count=openai_response.usage.prompt_tokens,  # type: ignore
            response_token_count=openai_response.usage.completion_tokens,  # noqa: E501 # type: ignore
        )

    async def create_chat_completion(
        self, model: str, messages: List[Any], *args, **kwargs
    ) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=model, messages=messages, *args, **kwargs
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
            openai_response = cast(AsyncIterator[Dict[str, Any]], openai_response)
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
        if not openai_response.choices:
            raise ValueError("No choices returned from OpenAI")
        if not openai_response.choices[0].message:
            raise ValueError("No message returned from OpenAI")
        if openai_response.usage is None:
            raise ValueError("No token counts returned from OpenAI")

        if openai_response.choices[0].message.content is not None:
            output = openai_response.choices[0].message.content
        else:
            try:
                output = openai_response.choices[0].message.function_call.arguments
            except AttributeError:
                try:
                    choice = openai_response.choices[0]
                    output = choice.message.tool_calls[-1].function.arguments
                except AttributeError as ae_tools:
                    raise ValueError(
                        "No message content or function"
                        " call arguments returned from OpenAI"
                    ) from ae_tools

        return LLMResponse(
            output=output,
            prompt_token_count=openai_response.usage.prompt_tokens,  # type: ignore
            response_token_count=openai_response.usage.completion_tokens,  # noqa: E501 # type: ignore
        )
