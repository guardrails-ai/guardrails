import json
from typing import Any, Dict, Optional, Tuple

from guardrails.classes.output_type import OutputTypes
from guardrails.llm_providers import (
    AsyncOpenAICallable,
    AsyncOpenAIChatCallable,
    OpenAICallable,
    OpenAIChatCallable,
    PromptCallableBase,
)
from guardrails.prompt.prompt import Prompt
from guardrails.prompt.instructions import Instructions
from guardrails.types.validator import ValidatorMap


def preprocess_prompt_for_string_output(
    prompt_callable: PromptCallableBase,
    instructions: Optional[Instructions],
    prompt: Prompt,
) -> Tuple[Instructions, Prompt]:
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
) -> Tuple[Instructions, Prompt]:
    if isinstance(prompt_callable, OpenAICallable) or isinstance(
        prompt_callable, AsyncOpenAICallable
    ):
        prompt.source += "\n\nJson Output:\n\n"
    if (
        isinstance(prompt_callable, OpenAIChatCallable)
        or isinstance(prompt_callable, AsyncOpenAIChatCallable)
    ) and not instructions:
        instructions = Instructions(
            "You are a helpful assistant, "
            "able to express yourself purely through JSON, "
            "strictly and precisely adhering to the provided JSON schema."
        )

    return instructions, prompt


def preprocess_prompt(
    prompt_callable: PromptCallableBase,
    instructions: Optional[Instructions],
    prompt: Prompt,
    output_type: OutputTypes,
) -> Tuple[Instructions, Prompt]:
    if output_type == OutputTypes.STRING:
        return preprocess_prompt_for_string_output(
            prompt_callable, instructions, prompt
        )
    return preprocess_prompt_for_json_output(prompt_callable, instructions, prompt)


def prompt_content_for_string_schema(
    output_schema: Dict[str, Any], validator_map: ValidatorMap, json_path: str
) -> str:
    # NOTE: Is this actually necessary?
    # We should check how LLMs perform this this vs just sending the JSON Schema
    prompt_content = ""
    description = output_schema.get("description")
    if description:
        prompt_content += (
            "Here's a description of what I want you to generate: " f"{description}"
        )
    validators = validator_map.get(json_path, [])
    if len(validators):
        prompt_content += (
            "\n\nYour generated response should satisfy the following properties:"
        )
        for validator in validators:
            prompt_content += f"\n- {validator.to_prompt()}"

    prompt_content += "\n\nDon't talk; just go."
    return prompt_content


# Supersedes Schema.transpile
def prompt_content_for_schema(
    output_type: OutputTypes,
    output_schema: Dict[str, Any],
    validator_map: ValidatorMap,
    json_path: str = "$",
) -> str:
    if output_type == OutputTypes.STRING:
        return prompt_content_for_string_schema(output_schema, validator_map, json_path)
    return json.dumps(output_schema)
