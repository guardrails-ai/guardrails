import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, cast

import openai

try:
    MANIFEST = True
    import manifest
except ImportError:
    MANIFEST = False


class PromptCallableException(Exception):
    pass


@dataclass
class PromptCallable:
    """A wrapper around a callable that takes in a prompt.

    Catches exceptions to let the user know clearly if the callable
    failed, and how to fix it.
    """

    fn: Callable

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


def nonchat_prompt(prompt: str) -> str:
    """Prepare final prompt for nonchat engine."""
    return prompt + "\n\nJson Output:\n\n"


def chat_prompt(prompt: str, **kwargs) -> List[Dict[str, str]]:
    """Prepare final prompt for chat engine."""
    if "system_prompt" in kwargs:
        system_prompt = kwargs.pop("system_prompt")
    else:
        system_prompt = (
            "You are a helpful assistant, "
            "able to express yourself purely through JSON, "
            "strictly and precisely adhering to the provided XML schemas."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def openai_wrapper(text: str, *args, **kwargs):
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_response = openai.Completion.create(
        api_key=api_key,
        prompt=nonchat_prompt(text),
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["text"]


def openai_chat_wrapper(text: str, *args, model="gpt-3.5-turbo", **kwargs):
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_response = openai.ChatCompletion.create(
        api_key=api_key,
        model=model,
        messages=chat_prompt(text, **kwargs),
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["message"]["content"]


def manifest_wrapper(text: str, client: Any, *args, **kwargs):
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
    manifest_response = client.run(nonchat_prompt(text), *args, **kwargs)
    return manifest_response


def get_llm_ask(llm_api: Callable, *args, **kwargs):
    if llm_api == openai.Completion.create:
        fn = partial(openai_wrapper, *args, **kwargs)
    elif llm_api == openai.ChatCompletion.create:
        fn = partial(openai_chat_wrapper, *args, **kwargs)
    elif MANIFEST and isinstance(llm_api, manifest.Manifest):
        fn = partial(manifest_wrapper, client=llm_api, *args, **kwargs)
    else:
        # Let the user pass in an arbitrary callable.
        fn = partial(llm_api, *args, **kwargs)

    return PromptCallable(fn=fn)