"""XML utilities."""
from typing import Dict, List, Tuple

from lxml import etree as ET

from guardrails.datatypes import registry as types_registry
from guardrails.validators import Validator
from guardrails.utils.constants import constants
from guardrails.prompt import Prompt
from guardrails.response_schema import Response


def read_aiml(aiml_file: str) -> Tuple[Response, Prompt, Dict]:
    """Read an AIML file.

    Args:
        aiml_file: The path to the AIML file.

    Returns:
        The root element of the AIML file.
    """
    with open(aiml_file, "r") as f:
        xml = f.read()
    parser = ET.XMLParser(encoding="utf-8")
    parsed_aiml = ET.fromstring(xml, parser=parser)

    raw_response_schema = parsed_aiml.find("response")
    if raw_response_schema is None:
        raise ValueError("AIML file must contain a response-schema element.")
    response_schema = load_response_schema(raw_response_schema)

    prompt = parsed_aiml.find("prompt")
    if prompt is None:
        raise ValueError("AIML file must contain a prompt element.")
    prompt = load_prompt(prompt)

    script = parsed_aiml.find("script")
    if script is not None:
        script = load_script(script)

    return response_schema, prompt, script


def load_response_schema(root: ET._Element) -> Response:
    """Validate parsed XML, create a prompt and a Schema object."""

    response = Response(parsed_aiml=root)

    for child in root:
        if isinstance(child, ET._Comment):
            continue
        child_name = child.attrib["name"]
        child_data = types_registry[child.tag].from_xml(child)
        response[child_name] = child_data

    return response


def load_prompt(root: ET._Element) -> Prompt:
    text = root.text

    # Substitute constants by reading the constants file.
    for key, value in constants.items():
        text = text.replace(f"@{key}", value)

    prompt = Prompt(text)

    return prompt


def load_script(root: ET._Element) -> Dict[str, List[Validator]]:
    pass
