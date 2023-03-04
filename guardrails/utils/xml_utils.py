"""XML utilities."""
from typing import Dict, List

from lxml import etree as ET

from guardrails.datatypes import registry as types_registry
from guardrails.validators import types_to_validators, validators_registry, Validator



def read_aiml(aiml_file: str) -> ET._Element:
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

    response_schema = parsed_aiml.find("response-schema")
    if response_schema is None:
        raise ValueError("AIML file must contain a response-schema element.")
    response_schema = load_response_schema(response_schema)

    prompt = parsed_aiml.find("prompt")
    if prompt is None:
        raise ValueError("AIML file must contain a prompt element.")
    prompt = load_prompt(prompt)

    script = parsed_aiml.find("script")
    if script is not None:
        script = load_script(script)

    return response_schema, prompt, script


def load_response_schema(root: ET._Element) -> Dict[str, List[Validator]]:
    """Validate parsed XML, create a prompt and a Schema object."""

    schema = {}

    for child in root:
        if isinstance(child, ET._Comment):
            continue
        child_name = child.attrib["name"]
        child_data_type = child.tag
        child_data_type = types_registry[child_data_type]
        child_data = child_data_type.from_xml(child)
        schema[child_name] = child_data

    return schema


def load_prompt(root: ET._Element) -> Dict[str, List[Validator]]:
    text = root.text

    # Substitute constants by reading the constants file.


def load_script(root: ET._Element) -> Dict[str, List[Validator]]:
    pass

