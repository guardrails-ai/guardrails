import json
import re
from typing import Any, Dict, Union

from guardrails.classes.output_type import OutputTypes

from guardrails.types.validator import ValidatorMap
from guardrails.prompt.prompt import Prompt
from guardrails.prompt.instructions import Instructions
from guardrails.types.inputs import MessageHistory


def prompt_uses_xml(prompt: str) -> bool:
    xml_const_regx = re.compile(r"gr\..*xml_.*")
    contains_xml_const = xml_const_regx.search(prompt) is not None
    contains_xml_output = "xml_output_schema" in prompt
    return contains_xml_output or contains_xml_const


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


def messages_to_prompt_string(
    messages: Union[list[dict[str, Union[str, Prompt, Instructions]]], MessageHistory],
) -> str:
    messages_copy = ""
    for msg in messages:
        content = (
            msg["content"].source  # type: ignore
            if isinstance(msg["content"], Prompt)
            or isinstance(msg["content"], Instructions)  # type: ignore
            else msg["content"]  # type: ignore
        )
        messages_copy += content
    return messages_copy
