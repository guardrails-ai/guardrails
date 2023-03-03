import warnings
from typing import Any, Dict, List

from xml.etree import ElementTree as ET


def get_validators(element: ET.Element, strict: bool = False) -> List["Validator"]:
    """Get the formatters for an element.

    Args:
        element: The XML element.
        strict: If True, raise an error if the element is not registered.

    Returns:
        A list of formatters.
    """

    from guardrails.x_validators import types_to_validators, validators_registry

    if 'format' not in element.attrib:
        return []

    provided_formatters = element.attrib['format'].split(';')
    registered_formatters = types_to_validators[element.tag]

    valid_formatters = []

    for formatter in provided_formatters:
        # Check if the formatter has any arguments.

        formatter = formatter.strip()

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


class DataType:

    def __init__(self, validators: List, children: Dict[str, Any]) -> None:
        self.validators = validators
        self.children = children

    @classmethod
    def from_str(self, s: str) -> "DataType":
        """Create a DataType from a string."""
        raise NotImplementedError("Abstract method.")

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        """Validate a value."""
        raise NotImplementedError("Abstract method.")

    def set_children(self, element: ET.Element):
        raise NotImplementedError("Abstract method.")

    @classmethod
    def from_xml(cls, element: ET.Element, strict: bool = False) -> "DataType":
        data_type = cls([], {})
        data_type.set_children(element)
        data_type.validators = get_validators(element, strict=strict)
        return data_type


registry: Dict[str, DataType] = {}


# Create a decorator to register a type
def register_type(name: str):
    def decorator(cls: type):
        registry[name] = cls
        return cls
    return decorator


class Scalar(DataType):
    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        """Validate a value."""
        for validator in self.validators:
            schema = validator.validate(key, value, schema)

            if schema is None:
                # The outcome of validation was to refrain from answering.
                return None

            if key not in schema:
                # The key may have been filtered out by a previous validator.
                break
        return schema

    def set_children(self, element: ET.Element):
        for _ in element:
            raise ValueError("Scalar data type must not have any children.")


@register_type("string")
class String(Scalar):

    @classmethod
    def from_str(self, s: str) -> "String":
        """Create a String from a string."""
        return s


@register_type("integer")
class Integer(Scalar):
    pass


@register_type("float")
class Float(Scalar):
    pass


@register_type("date")
class Date(Scalar):
    pass


@register_type("time")
class Time(Scalar):
    pass


@register_type("email")
class Email(Scalar):
    pass


@register_type("url")
class URL(Scalar):
    pass


@register_type("percentage")
class Percentage(Scalar):
    pass


@register_type("list")
class List(DataType):

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        # Validators in the main list data type are applied to the list overall.

        for validator in self.validators:
            schema = validator.validate(key, value, schema)

            if schema is None:
                # The outcome of validation was to refrain from answering.
                return None

            if key not in schema:
                # The key may have been filtered out by a previous validator.
                # In this case, we don't need to validate the items in the list,
                # since the list itself is not present.
                return schema

        if len(self.children) == 0:
            return schema

        item_type = list(self.children.values())[0]

        # TODO(shreya): Edge case: List of lists -- does this still work?
        for item in value:
            value = item_type.validate(None, item, value)

        return schema

    def set_children(self, element: ET.Element):
        idx = 0
        for child in element:
            idx += 1
            if idx > 1:
                # Only one child is allowed in a list data type.
                # The child must be the datatype that all items in the list
                # must conform to.
                raise ValueError("List data type must have exactly one child.")
            child_data_type = registry[child.tag]
            self.children["item"] = child_data_type.from_xml(child)


@register_type("object")
class Object(DataType):

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        # Validators in the main object data type are applied to the object overall.

        for validator in self.validators:
            schema = validator.validate(key, value, schema)

            if schema is None:
                # The outcome of validation was to refrain from answering.
                return None

            if key not in schema:
                # The key may have been filtered out by a previous validator.
                # In this case, we don't need to validate the items in the object,
                # since the object itself is not present.
                return schema

        if len(self.children) == 0:
            return schema

        # Types of supported children
        # 1. key_type
        # 2. value_type
        # 3. List of keys that must be present

        # TODO(shreya): Implement key type and value type later

        # Check for required keys
        # for key, field in self.children.items():
        for child_key, child_data_type in self.children.items():
            # Value should be a dictionary
            # child_key is an expected key that the schema defined
            # child_data_type is the data type of the expected key
            value = child_data_type.validate(
                child_key,
                value.get(child_key, None),
                value
            )
        return True

    def set_children(self, element: ET.Element):
        for child in element:
            child_data_type = registry[child.tag]
            self.children[child.attrib["name"]] = child_data_type.from_xml(child)
        # TODO(shreya): Does this need to return anything?


# @register_type("key")
# class Key(DataType):
#     pass


# @register_type("value")
# class Value(DataType):
#     pass


# @register_type("item")
# class Item(DataType):
#     pass
