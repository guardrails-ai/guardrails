import datetime
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Union

from lxml import etree as ET


def get_validators(
    element: ET._Element, strict: bool = False
) -> List["Validator"]:  # noqa: F821
    """Get the formatters for an element.

    Args:
        element: The XML element.
        strict: If True, raise an error if the element is not registered.

    Returns:
        A list of formatters.
    """

    from guardrails.validators import types_to_validators, validators_registry

    if "format" not in element.attrib:
        return []

    provided_formatters = element.attrib["format"].split(";")
    registered_formatters = types_to_validators[element.tag]

    valid_formatters = []

    for formatter in provided_formatters:
        # Check if the formatter has any arguments.

        formatter = formatter.strip()

        args = []
        formatter_with_args = formatter.split(":")
        if len(formatter_with_args) > 1:
            assert (
                len(formatter_with_args) == 2
            ), f"Formatter {formatter} has too many arguments."
            formatter, args = formatter_with_args
            formatter = formatter.strip()
            args = [x.strip() for x in args.strip().split(" ")]

            for i, arg in enumerate(args):
                # Arg enclosed in curly braces is a python expression.
                if arg[0] == "{" and arg[-1] == "}":
                    args[i] = eval(arg[1:-1])

        if formatter not in registered_formatters:
            if strict:
                raise ValueError(
                    f"Formatter {formatter} is not valid for element {element.tag}."
                )
            else:
                warnings.warn(
                    f"Formatter {formatter} is not valid for element {element.tag}."
                )
            continue

        # See if the formatter has an associated on_fail method.
        on_fail = None
        on_fail_attr_name = f"on-fail-{formatter}"
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
        self._children = children

    @classmethod
    def from_str(cls, s: str) -> "DataType":
        """Create a DataType from a string."""
        raise NotImplementedError("Abstract method.")

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        """Validate a value."""
        raise NotImplementedError("Abstract method.")

    def set_children(self, element: ET._Element):
        raise NotImplementedError("Abstract method.")

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False) -> "DataType":
        data_type = cls([], {})
        data_type.set_children(element)
        data_type.validators = get_validators(element, strict=strict)
        return data_type

    @property
    def children(self) -> SimpleNamespace:
        return SimpleNamespace(**self._children)


registry: Dict[str, DataType] = {}


# Create a decorator to register a type
def register_type(name: str):
    def decorator(cls: type):
        registry[name] = cls
        return cls

    return decorator


class ScalarType(DataType):
    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        """Validate a value."""

        value = self.from_str(value)

        for validator in self.validators:
            schema = validator.validate_with_correction(key, value, schema)

        return schema

    def set_children(self, element: ET._Element):
        for _ in element:
            raise ValueError("ScalarType data type must not have any children.")

    @classmethod
    def from_str(cls, s: str) -> "ScalarType":
        """Create a ScalarType from a string.

        Note: ScalarTypes like int, float, bool, etc. will override this method.
        Other ScalarTypes like string, email, url, etc. will not override this
        """
        return s


class NonScalarType(DataType):
    pass


@register_type("string")
class String(ScalarType):
    """Element tag: `<string>`"""

    @classmethod
    def from_str(cls, s: str) -> "String":
        """Create a String from a string."""
        return s


@register_type("integer")
class Integer(ScalarType):
    """Element tag: `<integer>`"""

    @classmethod
    def from_str(cls, s: str) -> "Integer":
        """Create an Integer from a string."""
        return int(s)


@register_type("float")
class Float(ScalarType):
    """Element tag: `<float>`"""

    @classmethod
    def from_str(cls, s: str) -> "Float":
        """Create a Float from a string."""
        return float(s)


@register_type("bool")
class Boolean(ScalarType):
    """Element tag: `<bool>`"""

    @classmethod
    def from_str(cls, s: Union[str, bool]) -> "Boolean":
        """Create a Boolean from a string."""

        if isinstance(s, bool):
            return s

        if s.lower() == "true":
            return True
        elif s.lower() == "false":
            return False
        else:
            raise ValueError(f"Invalid boolean value: {s}")


@register_type("date")
class Date(ScalarType):
    """Element tag: `<date>`"""

    @classmethod
    def from_str(cls, s: str) -> "Date":
        """Create a Date from a string."""
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()


@register_type("time")
class Time(ScalarType):
    """Element tag: `<time>`"""

    @classmethod
    def from_str(cls, s: str) -> "Time":
        """Create a Time from a string."""
        return datetime.datetime.strptime(s, "%H:%M:%S").time()


@register_type("email")
class Email(ScalarType):
    """Element tag: `<email>`"""


@register_type("url")
class URL(ScalarType):
    """Element tag: `<url>`"""


@register_type("pythoncode")
class PythonCode(ScalarType):
    """Element tag: `<pythoncode>`"""


@register_type("sql")
class SQLCode(ScalarType):
    """Element tag: `<sql>`"""


@register_type("percentage")
class Percentage(ScalarType):
    """Element tag: `<percentage>`"""


@register_type("list")
class List(NonScalarType):
    """Element tag: `<list>`"""

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        # Validators in the main list data type are applied to the list overall.

        for validator in self.validators:
            schema = validator.validate_with_correction(key, value, schema)

        if len(self._children) == 0:
            return schema

        item_type = list(self._children.values())[0]

        # TODO(shreya): Edge case: List of lists -- does this still work?
        for i, item in enumerate(value):
            value = item_type.validate(i, item, value)

        return schema

    def set_children(self, element: ET._Element):
        for idx, child in enumerate(element, start=1):
            if idx > 1:
                # Only one child is allowed in a list data type.
                # The child must be the datatype that all items in the list
                # must conform to.
                raise ValueError("List data type must have exactly one child.")
            child_data_type = registry[child.tag]
            self._children["item"] = child_data_type.from_xml(child)


@register_type("object")
class Object(NonScalarType):
    """Element tag: `<object>`"""

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        # Validators in the main object data type are applied to the object overall.

        for validator in self.validators:
            schema = validator.validate_with_correction(key, value, schema)

        if len(self._children) == 0:
            return schema

        # Types of supported children
        # 1. key_type
        # 2. value_type
        # 3. List of keys that must be present

        # TODO(shreya): Implement key type and value type later

        # Check for required keys
        for child_key, child_data_type in self._children.items():
            # Value should be a dictionary
            # child_key is an expected key that the schema defined
            # child_data_type is the data type of the expected key
            value = child_data_type.validate(
                child_key, value.get(child_key, None), value
            )

        schema[key] = value

        return schema

    def set_children(self, element: ET._Element):
        for child in element:
            child_data_type = registry[child.tag]
            self._children[child.attrib["name"]] = child_data_type.from_xml(child)


# @register_type("key")
# class Key(DataType):
# """
# Element tag: `<string>`
# """


# @register_type("value")
# class Value(DataType):
# """
# Element tag: `<string>`
# """


# @register_type("item")
# class Item(DataType):
# """
# Element tag: `<string>`
# """
