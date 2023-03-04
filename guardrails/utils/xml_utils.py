"""XML utilities."""
from typing import Dict, List, Union
import warnings

from xml.etree import ElementTree as ET

from guardrails.x_schema import Field
from guardrails.x_datatypes import registry as types_registry
from guardrails.x_validators import types_to_validators, validators_registry, Validator


def validate_xml(tree: Union[ET.ElementTree, ET.Element], strict: bool = False) -> bool:
    """Validate parsed XML, create a prompt and a Schema object."""

    if type(tree) == ET.ElementTree:
        tree = tree.getroot()

    for element in tree:
        if not validate_element(element):
            return False

    return True


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
        args = []
        formatter_with_args = formatter.split(':')
        if len(formatter_with_args) > 1:
            assert len(formatter_with_args) == 2, (
                f"Formatter {formatter} has too many arguments.")
            formatter, args = formatter_with_args
            formatter = formatter.strip()
            args = [x.strip() for x in args.split(' ')]

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
        on_fail_attr_name = f'on_fail_{formatter.__name__}'
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

    Creates a schema with only 

    Args:
        element: The XML element to validate.
        strict: If True, raise an error if the element is not registered.

    Returns:
        A tuple of the prompt and the schema."""

    schema = {}

    if 'name' not in element.attrib:
        raise ValueError(f"Element {element.tag} does not have a name.")
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
    for child in element:
        schema[name].children = validate_element(child)

    return schema
