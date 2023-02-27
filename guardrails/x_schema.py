# from guardrails.prompt_repo import PromptRepo, Prompt
# # from guardrails.types import DataType, String, URL, Email, Date, Time, Percentage, CodeSnippet, Float
# from dataclasses import dataclass
# from guardrails.validators import Validator, FormValidator
# from guardrails.exceptions import SchemaMismatchException

# import re
# from copy import deepcopy

# from pyparsing import CaselessKeyword, Regex

from typing import List, Union

from xml.etree import ElementTree

from guardrails.utils.xml_utils import validate_xml
from guardrails.x_datatypes import DataType
from guardrails.x_validators import Validator


class Field:
    def __init__(self,
                 datatype: DataType,
                 validators: Union[Validator, List[Validator]] = []
    ):
        self.type = datatype
        self.validators = validators
        self.children = {}


class XSchema:
    # def __init__(self, schema: Dict[str, Any]):
    def __init__(self, schema):
        self.schema = schema

    @classmethod
    def from_xml(cls, xml_file: str, base_prompt: str) -> "XSchema":
        """Create an XSchema from an XML file."""

        with open(xml_file, "r") as f:
            xml = f.read()

        tree = ElementTree.fromstring(xml)
        parsed_xml = cls._parse_xml(tree, xml_file, base_prompt)

        # TODO(shreya): Make this return a schema object.
        _ = validate_xml(parsed_xml, strict=False)

        return cls(schema)
