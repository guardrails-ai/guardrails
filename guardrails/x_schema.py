import json
from copy import deepcopy
from typing import Any, Dict, Union
from xml.etree import ElementTree as ET

from guardrails.x_datatypes import registry as types_registry
from guardrails.prompt_repo import Prompt


class XSchema:
    def __init__(self, schema: Dict[str, Any], prompt: Prompt):
        self.schema = schema
        self.prompt = prompt
        with open('openai_api_key.txt', 'r') as f:
            self.openai_api_key = f.read()

    def __repr__(self):
        def _print_dict(d: Dict[str, Any], indent: int = 0) -> str:
            """Print a dictionary in a nice way."""

            s = ""
            for k, v in d.items():
                if isinstance(v, dict):
                    s += f"{k}:\n{_print_dict(v, indent=indent + 1)}"
                else:
                    s += f"{' ' * (indent * 4)}{k}: {v}\n"

            return s

        schema = _print_dict(self.schema)

        return f"XSchema({schema})"

    def llm_ask(self, prompt):
        from openai import Completion
        llm_output = Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=2048,
            api_key=self.openai_api_key
        )
        return llm_output['choices'][0]['text']

    @classmethod
    def from_xml(cls, xml_file: str, base_prompt: str) -> "XSchema":
        """Create an XSchema from an XML file."""

        with open(xml_file, "r") as f:
            xml = f.read()
        parser = ET.XMLParser(encoding="utf-8")
        parsed_xml = ET.fromstring(xml, parser=parser)

        # schema = validate_xml(parsed_xml, strict=False)
        schema = load_from_xml(parsed_xml, strict=False)

        prompt = extract_prompt_from_xml(parsed_xml)

        return cls(schema, prompt)

    def ask_with_validation(self, text) -> str:
        """Ask a question, and validate the response."""

        prompt = self.prompt.format(text)
        response = self.llm_ask(prompt)

        response_as_dict = json.loads(response)
        validated_response = self.validate_response(response_as_dict)

        return validated_response

    def validate_response(self, response: Dict[str, Any]):
        """Validate a response against the schema."""

        def _validate_response(response: Dict[str, Any], schema: Dict[str, Any]):
            """Validate a response against a schema."""

            for field, value in response.items():
                if field not in schema:
                    continue

                if isinstance(value, dict):
                    _validate_response(value, schema[field].children)
                else:
                    for validator in schema[field].validators:
                        validator.validate(value)

        _validate_response(response, self.schema)


def load_from_xml(tree: Union[ET.ElementTree, ET.Element], strict: bool = False) -> bool:
    """Validate parsed XML, create a prompt and a Schema object."""

    if type(tree) == ET.ElementTree:
        tree = tree.getroot()

    schema = {}
    for child in tree:
        child_name = child.attrib['name']
        child_data_type = child.tag
        child_data_type = types_registry[child_data_type]
        child_data = child_data_type.from_xml(child)
        schema[child_name] = child_data

    return schema


def extract_prompt_from_xml(tree: Union[ET.ElementTree, ET.Element]) -> str:
    """Extract the prompt from an XML tree.

    Args:
        tree: The XML tree.

    Returns:
        The prompt.
    """

    tree_copy = deepcopy(tree)

    if type(tree_copy) == ET.ElementTree:
        tree_copy = tree_copy.getroot()

    # From the element tree, remove any action attributes like 'on-fail-*'.
    for element in tree_copy.iter():
        for attr in list(element.attrib):
            if attr.startswith('on-fail-'):
                del element.attrib[attr]
    
    # Return the XML as a string.
    return ET.tostring(tree_copy, encoding='unicode', method='xml')
