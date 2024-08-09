import copy
from string import Template
from typing import Dict, cast, Optional, Tuple

from guardrails.classes.output_type import OutputTypes
from guardrails.llm_providers import (
    AsyncOpenAICallable,
    AsyncOpenAIChatCallable,
    OpenAICallable,
    OpenAIChatCallable,
    PromptCallableBase,
)
from guardrails.prompt.prompt import Prompt
from guardrails.types.inputs import MessageHistory
from guardrails.prompt.instructions import Instructions


def msg_history_source(msg_history: MessageHistory) -> MessageHistory:
    msg_history_copy = []
    for msg in msg_history:
        msg_copy = copy.deepcopy(msg)
        content = (
            msg["content"].source
            if isinstance(msg["content"], Prompt)
            else msg["content"]
        )
        msg_copy["content"] = content
        msg_history_copy.append(cast(Dict[str, str], msg_copy))
    return msg_history_copy


def msg_history_string(msg_history: MessageHistory) -> str:
    msg_history_copy = ""
    for msg in msg_history:
        content = (
            msg["content"].source
            if isinstance(msg["content"], Prompt)
            else msg["content"]
        )
        msg_history_copy += content
    return msg_history_copy


def preprocess_prompt_for_string_output(
    prompt_callable: PromptCallableBase,
    instructions: Optional[Instructions],
    prompt: Prompt,
) -> Tuple[Optional[Instructions], Prompt]:
    if isinstance(prompt_callable, OpenAICallable) or isinstance(
        prompt_callable, AsyncOpenAICallable
    ):
        prompt.source += "\n\nString Output:\n\n"
    if (
        isinstance(prompt_callable, OpenAIChatCallable)
        or isinstance(prompt_callable, AsyncOpenAIChatCallable)
    ) and not instructions:
        instructions = Instructions(
            "You are a helpful assistant, expressing yourself through a string."
        )

    return instructions, prompt


def preprocess_prompt_for_json_output(
    prompt_callable: PromptCallableBase,
    instructions: Optional[Instructions],
    prompt: Prompt,
    use_xml: bool,
) -> Tuple[Optional[Instructions], Prompt]:
    if isinstance(prompt_callable, OpenAICallable) or isinstance(
        prompt_callable, AsyncOpenAICallable
    ):
        prompt.source += "\n\nJson Output:\n\n"
    if (
        isinstance(prompt_callable, OpenAIChatCallable)
        or isinstance(prompt_callable, AsyncOpenAIChatCallable)
    ) and not instructions:
        schema_type = "XML schemas" if use_xml else "JSON schema"
        instructions = Instructions(
            Template(
                "You are a helpful assistant, "
                "able to express yourself purely through JSON, "
                "strictly and precisely adhering to the provided ${schema_type}."
            ).safe_substitute(schema_type=schema_type)
        )

    return instructions, prompt


def preprocess_prompt(
    prompt_callable: PromptCallableBase,
    instructions: Optional[Instructions],
    prompt: Prompt,
    output_type: OutputTypes,
    use_xml: bool,
) -> Tuple[Optional[Instructions], Prompt]:
    if output_type == OutputTypes.STRING:
        return preprocess_prompt_for_string_output(
            prompt_callable, instructions, prompt
        )
    return preprocess_prompt_for_json_output(
        prompt_callable, instructions, prompt, use_xml
    )
