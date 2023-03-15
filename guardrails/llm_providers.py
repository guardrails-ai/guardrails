import os
from dataclasses import dataclass
from functools import partial
from typing import Callable

import openai


class PromptCallableException(Exception):
    pass


@dataclass
class PromptCallable:
    """
    A wrapper around a callable that takes in a prompt.
    Catches exceptions to let the user know clearly
    if the callable failed, and how to fix it.
    """

    fn: Callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.fn(*args, **kwargs)
        except Exception as e:
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` failed"
                f" with the following error: {e}. "
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


def openai_wrapper(text: str, *args, **kwargs):
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_response = openai.Completion.create(
        api_key=api_key,
        prompt=text,
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["text"]


def openai_chat_wrapper(text: str, *args, model="gpt-3.5-turbo", **kwargs):
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_response = openai.ChatCompletion.create(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]
        * args,
        **kwargs,
    )
    return openai_response["choices"][0]["message"]["content"]


def get_llm_ask(llm_api: Callable, *args, **kwargs):
    if llm_api == openai.Completion.create:
        return partial(openai_wrapper, *args, **kwargs)
    elif llm_api == openai.ChatCompletion.create:
        return partial(openai_chat_wrapper, *args, **kwargs)
    else:
        # Let the user pass in an arbitrary callable.
        return PromptCallable(fn=partial(llm_api, *args, **kwargs))
