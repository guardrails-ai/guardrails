"""RAIL utilities."""
from copy import deepcopy
from typing import Dict, Optional, Tuple

from lxml import etree as ET

from guardrails.datatypes import registry as types_registry
from guardrails.output_schema import OutputSchema
from guardrails.prompt import Prompt
from guardrails.utils.reask_utils import extract_prompt_from_xml


def read_rail(
    rail_file: Optional[str] = None, rail_string: Optional[str] = None
) -> Tuple[OutputSchema, Prompt, Dict]:
    """Read an RAIL file.

    Args:
        rail_file: The path to the RAIL file.

    Returns:
        The root element of the RAIL file.
    """
    if rail_file is not None:
        with open(rail_file, "r") as f:
            xml = f.read()
    elif rail_string is not None:
        xml = rail_string
    else:
        raise ValueError("Must pass either rail_file or rail_string.")
    parser = ET.XMLParser(encoding="utf-8")
    parsed_rail = ET.fromstring(xml, parser=parser)

    if "version" not in parsed_rail.attrib or parsed_rail.attrib["version"] != "0.1":
        raise ValueError(
            "RAIL file must have a version attribute set to 0.1."
            "Change the opening <rail> element to: <rail version='0.1'>"
        )

    # Execute the script before validating the rest of the RAIL file.
    script = parsed_rail.find("script")
    if script is not None:
        script = load_script(script)

    raw_output_schema = parsed_rail.find("output")
    if raw_output_schema is None:
        raise ValueError("RAIL file must contain a output-schema element.")
    output_schema = load_output_schema(raw_output_schema)

    prompt = parsed_rail.find("prompt")
    if prompt is None:
        raise ValueError("RAIL file must contain a prompt element.")
    prompt = load_prompt(prompt, output_schema)

    return output_schema, prompt, script


def load_output_schema(root: ET._Element) -> OutputSchema:
    """Validate parsed XML, create a prompt and a Schema object."""

    output = OutputSchema(parsed_rail=root)
    strict = False
    if "strict" in root.attrib and root.attrib["strict"] == "true":
        strict = True

    for child in root:
        if isinstance(child, ET._Comment):
            continue
        child_name = child.attrib["name"]
        child_data = types_registry[child.tag].from_xml(child, strict=strict)
        output[child_name] = child_data

    return output


def load_prompt(root: ET._Element, output_schema: OutputSchema) -> Prompt:
    text = root.text
    output_schema_prompt = extract_prompt_from_xml(deepcopy(output_schema.parsed_rail))

    return Prompt(text, output_schema=output_schema_prompt)


def load_script(root: ET._Element) -> None:
    if "language" not in root.attrib:
        raise ValueError("Script element must have a language attribute.")

    language = root.attrib["language"]
    if language != "python":
        raise ValueError("Only python scripts are supported right now.")

    exec(root.text, globals(), locals())
