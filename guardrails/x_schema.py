import json
import logging
from copy import deepcopy
from typing import Any, Dict, Union
from xml.etree import ElementTree as ET

from guardrails.x_datatypes import registry as types_registry, XDataType
from guardrails.prompt_repo import Prompt

logger = logging.getLogger(__name__)


class XSchema:
    def __init__(self, schema: Dict[str, XDataType], prompt: Prompt):
        self.schema: Dict[str, XDataType] = schema
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

    @staticmethod
    def prompt_json_suffix():
        return """\n\nReturn a valid JSON object that respects this XML format and extracts only the information requested in this document. Respect the types indicated in the XML -- the information you extract should be converted into the correct 'type'. Try to be as correct and concise as possible. Find all relevant information in the document. If you are unsure of the answer, enter 'None'. If you answer incorrectly, you will be asked again until you get it right which is expensive."""  # noqa: E501

    @staticmethod
    def prompt_xml_prefix():
        return """\n\nGiven below is XML that describes the information to extract from this document and the tags to extract it into.\n\n"""  # noqa: E501

    @classmethod
    def from_xml(cls, xml_file: str, base_prompt: Prompt) -> "XSchema":
        """Create an XSchema from an XML file."""

        with open(xml_file, "r") as f:
            xml = f.read()
        parser = ET.XMLParser(encoding="utf-8")
        parsed_xml = ET.fromstring(xml, parser=parser)

        schema = load_from_xml(parsed_xml, strict=False)

        prompt = extract_prompt_from_xml(parsed_xml)

        base_prompt.append_to_prompt(cls.prompt_xml_prefix())
        base_prompt.append_to_prompt(prompt)
        base_prompt.append_to_prompt(cls.prompt_json_suffix())

        return cls(schema, base_prompt)

    def ask_with_validation(self, text) -> str:
        """Ask a question, and validate the response."""

        prompt = self.prompt.format(document=text)
        response = self.llm_ask(prompt)

        try:
            response_as_dict = json.loads(response)
            validated_response = self.validate_response(response_as_dict)
        except json.decoder.JSONDecodeError:
            validated_response = None
            response_as_dict = None

        return response, response_as_dict, validated_response

    def validate_response(self, response: Dict[str, Any]):
        """Validate a response against the schema."""

        validated_response = deepcopy(response)

        for field, value in response.items():
            if field not in self.schema:
                logger.debug(f"Field {field} not in schema.")
                continue

            validated_response = self.schema[field].validate(value)

        return validated_response


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
