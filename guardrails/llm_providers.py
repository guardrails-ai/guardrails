import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

import openai
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from guardrails.utils.pydantic_utils import convert_pydantic_model_to_openai_fn

try:
    MANIFEST = True
    import manifest
except ImportError:
    MANIFEST = False

try:
    import cohere
except ImportError:
    cohere = None


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


def nonchat_prompt(prompt: str, instructions: Optional[str] = None) -> str:
    """Prepare final prompt for nonchat engine."""
    if instructions:
        prompt = "\n\n".join([instructions, prompt])
    return prompt


def chat_prompt(
    prompt: str,
    instructions: Optional[str] = None,
    msg_history: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Prepare final prompt for chat engine."""
    if msg_history:
        return msg_history

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
    *args,
    **kwargs,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_response = openai.Completion.create(
        api_key=api_key,
        engine=engine,
        prompt=nonchat_prompt(prompt=text, instructions=instructions),
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["text"]


def openai_chat_wrapper(
    text: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    instructions: Optional[str] = None,
    msg_history: Optional[List[Dict]] = None,
    base_model: Optional[BaseModel] = None,
    function_call: Optional[str] = None,
    *args,
    **kwargs,
) -> str:
    """Wrapper for OpenAI chat engines.

    Use Guardrails with OpenAI chat engines by doing
    ```
    raw_llm_response, validated_response = guard(
        openai.ChatCompletion.create,
        prompt_params={...},
        text=...,
        instructions=...,
        msg_history=...,
        temperature=...,
        ...
    )
    ```

    If `base_model` is passed, the chat engine will be used as a function
    on the base model.
    """

    if msg_history is None and text is None:
        raise PromptCallableException(
            "You must pass in either `text` or `msg_history` to `guard.__call__`."
        )

    # Configure function calling if applicable
    if base_model:
        function_params = [convert_pydantic_model_to_openai_fn(base_model)]
        if function_call is None:
            function_call = {"name": function_params[0]["name"]}
        fn_kwargs = {"functions": function_params, "function_call": function_call}
    else:
        fn_kwargs = {}

    # Call OpenAI
    openai_response = openai.ChatCompletion.create(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=model,
        messages=chat_prompt(
            prompt=text, instructions=instructions, msg_history=msg_history
        ),
        *args,
        **fn_kwargs,
        **kwargs,
    )

    # Extract string from response
    if "function_call" in openai_response["choices"][0]["message"]:
        return openai_response["choices"][0]["message"]["function_call"]["arguments"]
    else:
        return openai_response["choices"][0]["message"]["content"]


def manifest_wrapper(
    text: str, client: Any, instructions: Optional[str] = None, *args, **kwargs
) -> str:
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
        nonchat_prompt(prompt=text, instructions=instructions), *args, **kwargs
    )
    return manifest_response


def cohere_wrapper(
    prompt: str, client_callable: Any, model: str, *args, **kwargs
) -> str:
    """Wrapper for cohere client.

    To use cohere for guardrails, do
    ```
    client = cohere.Client(api_key=...)

    raw_llm_response, validated_response = guard(
        client.generate,
        prompt_params={...},
        model="command-nightly",
        ...
    )
    ```
    """
    cohere_response = client_callable(prompt=prompt, model=model, *args, **kwargs)
    return cohere_response[0].text


def get_llm_ask(llm_api: Callable, *args, **kwargs) -> PromptCallable:
    if llm_api == openai.Completion.create:
        fn = partial(openai_wrapper, *args, **kwargs)
    elif llm_api == openai.ChatCompletion.create:
        fn = partial(openai_chat_wrapper, *args, **kwargs)
    elif MANIFEST and isinstance(llm_api, manifest.Manifest):
        fn = partial(manifest_wrapper, client=llm_api, *args, **kwargs)
    elif (
        cohere
        and isinstance(getattr(llm_api, "__self__", None), cohere.Client)
        and getattr(llm_api, "__name__", None) == "generate"
    ):
        fn = partial(cohere_wrapper, client_callable=llm_api, *args, **kwargs)
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
    *args,
    **kwargs,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_response = await openai.Completion.acreate(
        api_key=api_key,
        engine=engine,
        prompt=nonchat_prompt(prompt=text, instructions=instructions),
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["text"]


async def async_openai_chat_wrapper(
    text: str,
    model="gpt-3.5-turbo",
    instructions: Optional[str] = None,
    *args,
    **kwargs,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_response = await openai.ChatCompletion.acreate(
        api_key=api_key,
        model=model,
        messages=chat_prompt(prompt=text, instructions=instructions),
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
        nonchat_prompt(prompt=text, instructions=instructions), *args, **kwargs
    )
    return manifest_response


def get_async_llm_ask(llm_api: Callable[[Any], Awaitable[Any]], *args, **kwargs):
    if llm_api == openai.Completion.acreate:
        fn = partial(async_openai_wrapper, *args, **kwargs)
    elif llm_api == openai.ChatCompletion.acreate:
        fn = partial(async_openai_chat_wrapper, *args, **kwargs)
    elif MANIFEST and isinstance(llm_api, manifest.Manifest):
        fn = partial(async_manifest_wrapper, client=llm_api, *args, **kwargs)
    else:
        # Let the user pass in an arbitrary callable.
        fn = partial(llm_api, *args, **kwargs)

    return AsyncPromptCallable(fn=fn)
