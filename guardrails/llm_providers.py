import os
from functools import partial
from typing import Callable

import openai


def openai_wrapper(text: str, *args, **kwargs):
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_response = openai.Completion.create(
        api_key=api_key,
        prompt=text,
        *args,
        **kwargs,
    )
    return openai_response["choices"][0]["text"]


def get_llm_ask(llm_api: Callable, *args, **kwargs):
    if llm_api == openai.Completion.create:
        return partial(openai_wrapper, *args, **kwargs)
    else:
        raise ValueError("Only openai.Completion.create supported right now.")
