import os
from typing import Any, List

import openai

from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.openai_utils.base import BaseOpenAIClient


def get_static_openai_create_func():
    if "OPENAI_API_KEY" not in os.environ:
        return None
    return openai.completions.create


def get_static_openai_chat_create_func():
    if "OPENAI_API_KEY" not in os.environ:
        return None
    return openai.chat.completions.create


def get_static_openai_acreate_func():
    return None


def get_static_openai_chat_acreate_func():
    return None


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
        response = self.client.completions.create(
            model=engine, prompt=prompt, *args, **kwargs
        )

        if not response.choices:
            raise ValueError("No choices returned from OpenAI")
        if response.usage is None:
            raise ValueError("No token counts returned from OpenAI")
        return LLMResponse(
            output=response.choices[0].text,
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
        )

    def create_chat_completion(
        self, model: str, messages: List[Any], *args, **kwargs
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=model, messages=messages, *args, **kwargs
        )

        if not response.choices:
            raise ValueError("No choices returned from OpenAI")
        if not response.choices[0].message.content:
            raise ValueError("No message returned from OpenAI")
        if response.usage is None:
            raise ValueError("No token counts returned from OpenAI")
        return LLMResponse(
            output=response.choices[0].message.content,
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
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

        if not response.choices:
            raise ValueError("No choices returned from OpenAI")
        if response.usage is None:
            raise ValueError("No token counts returned from OpenAI")
        return LLMResponse(
            output=response.choices[0].text,
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
        )

    async def create_chat_completion(
        self, model: str, messages: List[Any], *args, **kwargs
    ) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=model, messages=messages, *args, **kwargs
        )

        if not response.choices:
            raise ValueError("No choices returned from OpenAI")
        if not response.choices[0].message.content:
            raise ValueError("No message returned from OpenAI")
        if response.usage is None:
            raise ValueError("No token counts returned from OpenAI")
        return LLMResponse(
            output=response.choices[0].message.content,
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
        )
