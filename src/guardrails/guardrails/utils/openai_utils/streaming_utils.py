from typing import Dict, List
from guardrails.logger import logger

import tiktoken


def num_tokens_from_string(text: str, model_name: str) -> int:
    """Returns the number of tokens in a text string.

    Supported for OpenAI models only. This is a helper function
    that is required when OpenAI's `stream` parameter is set to `True`,
    because OpenAI does not return the number of tokens in that case.
    Requires the `tiktoken` package to be installed.

    Args:
        text (str): The text string to count the number of tokens in.
        model_name (str): The name of the OpenAI model to use.

    Returns:
        num_tokens (int): The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def num_tokens_from_messages(
    messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613"
) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.warning(
            """gpt-3.5-turbo may update over time.
            Returning num tokens assuming gpt-3.5-turbo-0613."""
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logger.warning(
            """gpt-4 may update over time.
            Returning num tokens assuming gpt-4-0613."""
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md for
            information on how messages are converted to tokens."""
        )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    # every reply is primed with <|start|>assistant<|message|>
    num_tokens += 3
    return num_tokens
