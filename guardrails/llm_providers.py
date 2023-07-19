import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

import openai
from guard_rails_api_client.models.validate_payload_llm_api import ValidatePayloadLlmApi
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

try:
    MANIFEST = True
    import manifest
except ImportError:
    MANIFEST = False

OPENAI_RETRYABLE_ERRORS = [
    openai.error.APIConnectionError,
    openai.error.APIError,
    openai.error.TryAgain,
    openai.error.Timeout,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError,
]
RETRYABLE_ERRORS = tuple(OPENAI_RETRYABLE_ERRORS)


class PromptCallableException(Exception):
    pass


###
# Synchronous wrappers
###


@dataclass
class PromptCallable:
    """A wrapper around a callable that takes in a prompt.

    Catches exceptions to let the user know clearly if the callable
    failed, and how to fix it.
    """

    fn: Callable

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    def __call__(self, *args, **kwargs):
        try:
            result = self.fn(*args, **kwargs)
        except Exception as e:
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` failed"
                f" with the following error: `{e}`. "
                "Make sure that `fn` can be called as a function that"
                " takes in a single prompt string "
                "and returns a string."
            )
        if not isinstance(result, str):
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` returned"
                f" a non-string value: {result}. "
                "Make sure that `fn` can be called as a function that"
                " takes in a single prompt string "
                "and returns a string."
            )
        return result


def nonchat_prompt(prompt: str, instructions: Optional[str] = None, **kwargs) -> str:
    """Prepare final prompt for nonchat engine."""
    if instructions:
        prompt = "\n\n".join([instructions, prompt])

    return prompt


def chat_prompt(
    prompt: str, instructions: Optional[str] = None, **kwargs
) -> List[Dict[str, str]]:
    """Prepare final prompt for chat engine."""
    if not instructions:
        instructions = "You are a helpful assistant."
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ]


def openai_wrapper(
    text: str,
    engine: str = "text-davinci-003",
    instructions: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    *args,
    **kwargs,
):
    openai_api_key = (
        openai_api_key
        if openai_api_key is not None
        else os.environ.get("OPENAI_API_KEY")
    )
    openai_response = openai.Completion.create(
        api_key=openai_api_key,
        engine=engine,
        prompt=nonchat_prompt(text, instructions, **kwargs),
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["text"]


def openai_chat_wrapper(
    text: str,
    model="gpt-3.5-turbo",
    instructions: Optional[str] = None,
    base_model: Optional[BaseModel] = None,
    openai_api_key: Optional[str] = None,
    *args,
    **kwargs,
):
    openai_api_key = (
        openai_api_key
        if openai_api_key is not None
        else os.environ.get("OPENAI_API_KEY")
    )
    if base_model:
        base_model_schema = base_model.schema()
        function_params = {
            "name": base_model_schema["title"],
            "description": base_model_schema["description"]
            if "description" in base_model_schema
            else None,
            "parameters": base_model_schema,
        }

    # TODO: update this as new models are released
    if base_model:
        openai_response = openai.ChatCompletion.create(
            api_key=openai_api_key,
            model=model,
            messages=chat_prompt(text, instructions, **kwargs),
            functions=[function_params],
            function_call={"name": function_params["name"]},
            *args,
            **kwargs,
        )
        return openai_response["choices"][0]["message"]["function_call"]["arguments"]
    else:
        openai_response = openai.ChatCompletion.create(
            api_key=openai_api_key,
            model=model,
            messages=chat_prompt(text, instructions, **kwargs),
            *args,
            **kwargs,
        )
        return openai_response["choices"][0]["message"]["content"]


def manifest_wrapper(
    text: str, client: Any, instructions: Optional[str] = None, *args, **kwargs
):
    """Wrapper for manifest client.

    To use manifest for guardrailse, do
    ```
    client = Manifest(client_name=..., client_connection=...)
    raw_llm_response, validated_response = guard(
        client,
        prompt_params={...},
        ...
    ```
    """
    if not MANIFEST:
        raise PromptCallableException(
            "The `manifest` package is not installed. "
            "Install with `pip install manifest-ml`"
        )
    client = cast(manifest.Manifest, client)
    manifest_response = client.run(
        nonchat_prompt(text, instructions, **kwargs), *args, **kwargs
    )
    return manifest_response


def get_llm_ask(
    llm_api: Callable, openai_api_key: Optional[str] = None, *args, **kwargs
) -> PromptCallable:
    openai_api_key = (
        openai_api_key
        if openai_api_key is not None
        else os.environ.get("OPENAI_API_KEY")
    )
    if llm_api == openai.Completion.create:
        fn = partial(openai_wrapper, openai_api_key=openai_api_key, *args, **kwargs)
    elif llm_api == openai.ChatCompletion.create:
        fn = partial(
            openai_chat_wrapper, openai_api_key=openai_api_key, *args, **kwargs
        )
    elif llm_api_is_manifest(llm_api):
        fn = partial(manifest_wrapper, client=llm_api, *args, **kwargs)
    else:
        # Let the user pass in an arbitrary callable.
        fn = partial(llm_api, *args, **kwargs)

    return PromptCallable(fn=fn)


###
# Async wrappers
###


@dataclass
class AsyncPromptCallable:
    """A wrapper around a callable that takes in a prompt.

    Catches exceptions to let the user know clearly if the callable
    failed, and how to fix it.
    """

    fn: Callable[[Any], Awaitable[Any]]

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    async def __call__(self, *args, **kwargs):
        try:
            result = await self.fn(*args, **kwargs)
        except Exception as e:
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` failed"
                f" with the following error: `{e}`. "
                "Make sure that `fn` can be called as a function that"
                " takes in a single prompt string "
                "and returns a string."
            )
        if not isinstance(result, str):
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` returned"
                f" a non-string value: {result}. "
                "Make sure that `fn` can be called as a function that"
                " takes in a single prompt string "
                "and returns a string."
            )
        return result


async def async_openai_wrapper(
    text: str,
    engine: str = "text-davinci-003",
    instructions: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    *args,
    **kwargs,
):
    openai_api_key = (
        openai_api_key
        if openai_api_key is not None
        else os.environ.get("OPENAI_API_KEY")
    )
    openai_response = await openai.Completion.acreate(
        api_key=openai_api_key,
        engine=engine,
        prompt=nonchat_prompt(text, instructions, **kwargs),
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["text"]


async def async_openai_chat_wrapper(
    text: str,
    model="gpt-3.5-turbo",
    instructions: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    *args,
    **kwargs,
):
    openai_api_key = (
        openai_api_key
        if openai_api_key is not None
        else os.environ.get("OPENAI_API_KEY")
    )
    openai_response = await openai.ChatCompletion.acreate(
        api_key=openai_api_key,
        model=model,
        messages=chat_prompt(text, instructions, **kwargs),
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["message"]["content"]


async def async_manifest_wrapper(
    text: str, client: Any, instructions: Optional[str] = None, *args, **kwargs
):
    """Async wrapper for manifest client.

    To use manifest for guardrails, do
    ```
    client = Manifest(client_name=..., client_connection=...)
    raw_llm_response, validated_response = guard(
        client,
        prompt_params={...},
        ...
    ```
    """
    if not MANIFEST:
        raise PromptCallableException(
            "The `manifest` package is not installed. "
            "Install with `pip install manifest-ml`"
        )
    client = cast(manifest.Manifest, client)
    manifest_response = await client.run(
        nonchat_prompt(text, instructions, **kwargs), *args, **kwargs
    )
    return manifest_response


def get_async_llm_ask(
    llm_api: Callable[[Any], Awaitable[Any]],
    openai_api_key: Optional[str] = None,
    *args,
    **kwargs,
):
    openai_api_key = (
        openai_api_key
        if openai_api_key is not None
        else os.environ.get("OPENAI_API_KEY")
    )
    if llm_api == openai.Completion.acreate:
        fn = partial(
            async_openai_wrapper, openai_api_key=openai_api_key, *args, **kwargs
        )
    elif llm_api == openai.ChatCompletion.acreate:
        fn = partial(
            async_openai_chat_wrapper, openai_api_key=openai_api_key, *args, **kwargs
        )
    elif llm_api_is_manifest(llm_api):
        fn = partial(async_manifest_wrapper, client=llm_api, *args, **kwargs)
    else:
        # Let the user pass in an arbitrary callable.
        fn = partial(llm_api, *args, **kwargs)

    return AsyncPromptCallable(fn=fn)


def get_llm_api_enum(llm_api: Callable[[Any], Awaitable[Any]]) -> ValidatePayloadLlmApi:
    if llm_api == openai.Completion.create:
        return ValidatePayloadLlmApi.OPENAI_COMPLETION_CREATE
    elif llm_api == openai.ChatCompletion.create:
        return ValidatePayloadLlmApi.OPENAI_CHATCOMPLETION_CREATE
    elif llm_api == openai.Completion.acreate:
        return ValidatePayloadLlmApi.OPENAI_COMPLETION_ACREATE
    elif llm_api == openai.ChatCompletion.acreate:
        return ValidatePayloadLlmApi.OPENAI_CHATCOMPLETION_ACREATE
    else:
        return None


def llm_api_is_manifest(llm_api: Callable[[Any], Awaitable[Any]]) -> bool:
    return MANIFEST and isinstance(llm_api, manifest.Manifest)
