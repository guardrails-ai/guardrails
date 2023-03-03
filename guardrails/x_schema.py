from collections import defaultdict
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Union
from xml.etree import ElementTree as ET

import manifest

from guardrails.prompt_repo import Prompt
from guardrails.x_datatypes import DataType
from guardrails.x_datatypes import registry as types_registry
from guardrails.x_validators import ReAsk

logger = logging.getLogger(__name__)


class XSchema:
    def __init__(
        self,
        schema: Dict[str, DataType],
        prompt: Prompt,
        parsed_xml: ET.Element,
    ):
        self.schema: Dict[str, DataType] = schema
        self.prompt = prompt
        self.parsed_xml = parsed_xml

        self.openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        self.client = manifest.Manifest(
            client_name="openai",
            client_connection=self.openai_api_key,
        )

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
        return self.client.run(
            prompt,
            engine="text-davinci-003",
            temperature=0,
            max_tokens=2048,
        )

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

        return cls(schema, base_prompt, parsed_xml)

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

    def gather_reasks(self, response: Dict) -> List[tuple]:
        """
        Traverse response and gather all ReAsk objects.
        Response is a nested dictionary, where values can also be lists or
        dictionaries.
        Make sure to also grab the corresponding paths (including list index), and return
        a list of tuples.
        """
        reasks = []

        def _gather_reasks(response: Union[list, dict], path: List[str] = []):
            if isinstance(response, dict):
                iterable = response.items()
            elif isinstance(response, list):
                iterable = enumerate(response)
            else:
                raise ValueError(f"Expected dict or list, got {type(response)}")
            for field, value in iterable:
                if isinstance(value, ReAsk):
                    reasks.append((path + [field], value))

                if isinstance(value, dict):
                    _gather_reasks(value, path + [field])

                if isinstance(value, list):
                    for idx, item in enumerate(value):
                        if isinstance(item, ReAsk):
                            reasks.append((path + [field, idx], item))
                        else:
                            _gather_reasks(item, path + [field, idx])

        _gather_reasks(response)

        return reasks
    
    def get_reasks_by_element(self, reasks: List[tuple], **kwargs) -> Prompt:

        reasks_by_element = defaultdict(list)

        for path, reask in reasks:
            print(path, reask)
            # Make a find query for each path
            # - replace int values in path with '*'
            # TODO: does this work for all cases?
            query = "."
            for part in path:
                if isinstance(part, int):
                    query += "/*"
                else:
                    query += f"/*[@name='{part}']"

            # Find the element
            element = self.parsed_xml.find(query)

            reasks_by_element[element].append((path, reask))

        return reasks_by_element
    
    def get_reask_prompt(
            self, 
            reasks_by_element: Prompt,
        ) -> Prompt:
        # for all previous prompts:
        #       prompt
        #       response
        #       XML schema
        #       reasks_by_element (failed elements, path to failures, reask infos)
        pass

    # prompt = self.prompt.format(document=text)
    # response = self.llm_ask(prompt)
    # response_as_dict = json.loads(response)
    # validated_response = self.validate_response(response_as_dict)
    # print(validated_response)
        
    def do_reask(num_retries: int = 1, **kwargs):
        return # Guardrails(
            # ....
        # ).run()
        # for _ in range(num_retries):
        #     # ...
        #     pass

    def validate_response(self, response: Dict[str, Any]):
        """Validate a response against the schema."""

        validated_response = deepcopy(response)

        for field, value in validated_response.items():
            if field not in self.schema:
                logger.debug(f"Field {field} not in schema.")
                continue

            validated_response = self.schema[field].validate(
                field, value, validated_response
            )
        
        reasks = self.gather_reasks(validated_response)
        if reasks:
            pass

        # validated_response
        # prompt
        # xml

        return validated_response


def load_from_xml(
    tree: Union[ET.ElementTree, ET.Element], strict: bool = False
) -> bool:
    """Validate parsed XML, create a prompt and a Schema object."""

    if type(tree) == ET.ElementTree:
        tree = tree.getroot()

    schema = {}
    for child in tree:
        child_name = child.attrib["name"]
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
            if attr.startswith("on-fail-"):
                del element.attrib[attr]

    # Return the XML as a string.
    return ET.tostring(tree_copy, encoding="unicode", method="xml")
