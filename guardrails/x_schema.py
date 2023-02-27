# from guardrails.prompt_repo import PromptRepo, Prompt
# # from guardrails.types import DataType, String, URL, Email, Date, Time, Percentage, CodeSnippet, Float
# from dataclasses import dataclass
# from guardrails.validators import Validator, FormValidator
# from guardrails.exceptions import SchemaMismatchException

# import re
# from copy import deepcopy

# from pyparsing import CaselessKeyword, Regex

# from typing import List, Union

# from xml.etree import ElementTree

# from guardrails.utils.xml_utils import validate_xml
# from guardrails.x_datatypes import DataType
# from guardrails.x_validators import Validator

from typing import Dict, List, Union
import warnings

from xml.etree import ElementTree as ET

from guardrails.x_datatypes import registry as types_registry, DataType
from guardrails.x_validators import types_to_validators, validators_registry, Validator



class Field:
    def __init__(self,
                 datatype: DataType,
                 validators: Union[Validator, List[Validator]] = []
    ):
        self.type = datatype
        self.validators = validators
        self.children = {}

    def __repr__(self):
        return f"Field({self.type}, {self.validators}, {self.children})"


class XSchema:
    # def __init__(self, schema: Dict[str, Any]):
    def __init__(self, schema):
        self.schema = schema

    def __repr__(self):
        return f"XSchema({self.schema})"

    @classmethod
    def from_xml(cls, xml_file: str, base_prompt: str) -> "XSchema":
        """Create an XSchema from an XML file."""

        with open(xml_file, "r") as f:
            xml = f.read()

        parser = ET.XMLParser(encoding="utf-8")
        parsed_xml = ET.fromstring(xml, parser=parser)

        # TODO(shreya): Make this return a schema object.
        schema = validate_xml(parsed_xml, strict=False)

        return cls(schema)


def validate_xml(tree: Union[ET.ElementTree, ET.Element], strict: bool = False) -> bool:
    """Validate parsed XML, create a prompt and a Schema object."""

    if type(tree) == ET.ElementTree:
        tree = tree.getroot()

    schema = validate_element(tree, strict=strict)

    return schema


def get_formatters(element: ET.Element, strict: bool = False) -> List[Validator]:
    """Get the formatters for an element.

    Args:
        element: The XML element.
        strict: If True, raise an error if the element is not registered.

    Returns:
        A list of formatters.
    """
    registered_formatters = types_to_validators[element.tag]
    provided_formatters = element.attrib['format'].split(';')

    valid_formatters = []

    for formatter in provided_formatters:
        # Check if the formatter has any arguments.

        formatter = formatter.strip()
        print(f'formatter: {formatter}')

        args = []
        formatter_with_args = formatter.split(':')
        if len(formatter_with_args) > 1:
            assert len(formatter_with_args) == 2, (
                f"Formatter {formatter} has too many arguments.")
            formatter, args = formatter_with_args
            formatter = formatter.strip()
            args = [x.strip() for x in args.strip().split(' ')]

            for i, arg in enumerate(args):
                # If arg is enclosed within curly braces, then it is a python expression.
                if arg[0] == '{' and arg[-1] == '}':
                    args[i] = eval(arg[1:-1])

        if formatter not in registered_formatters:
            if strict:
                raise ValueError(
                    f"Formatter {formatter} is not valid for element {element.tag}.")
            else:
                warnings.warn(
                    f"Formatter {formatter} is not valid for element {element.tag}.")
            continue

        # See if the formatter has an associated on_fail method.
        on_fail = None
        on_fail_attr_name = f'on-fail-{formatter}'
        if on_fail_attr_name in element.attrib:
            on_fail = element.attrib[on_fail_attr_name]
            # TODO(shreya): Load the on_fail method.
            # This method should be loaded from an optional script given at the
            # beginning of a gxml file.

        formatter = validators_registry[formatter]
        valid_formatters.append(formatter(*args, on_fail=on_fail))

    return valid_formatters


# def validate_element(element: ET.Element, strict: bool = False) -> Tuple[Dict, Prompt]:
def validate_element(element: ET.Element, strict: bool = False) -> Dict:
    """Validate an XML element.

    Creates a schema with only the valid elements.

    Args:
        element: The XML element to validate.
        strict: If True, raise an error if the element is not registered.

    Returns:
        A tuple of the prompt and the schema."""

    schema = {}
    shell_type = False

    if 'name' not in element.attrib:

        if element.tag == 'prompt':
            # TODO(shreya): This needs to be more robust -- which tags are ok?
            shell_type = True
            pass
        else:
            raise ValueError(f"Element {element.tag} does not have a name.")

    if not shell_type:
        name = element.attrib['name']
        if element.tag not in types_registry:
            if strict:
                raise ValueError(f"Element {element.tag} is not registered.")
            else:
                # Raise a warning tha the element is not registered.
                warnings.warn(f"Element {element.tag} is not registered.")
        else:
            schema[name] = Field(datatype=types_registry[element.tag])
            if 'format' in element.attrib:
                schema[name].validators = get_formatters(element, strict=strict)

    # If element has children, check that they are valid.

    if shell_type:
        for child in element:
            schema.update(validate_element(child, strict=strict))
    else:
        for child in element:
            schema[name].children = validate_element(child)

    return schema
