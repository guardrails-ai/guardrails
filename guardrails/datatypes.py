import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Tuple, Type, Union

from lxml import etree as ET
from pydantic import BaseModel

if TYPE_CHECKING:
    from guardrails.schema import FormatAttr


class DataType:
    def __init__(
        self,
        children: Dict[str, Any],
        format_attr: "FormatAttr",
        element: ET._Element,
    ) -> None:
        self._children = children
        self.format_attr = format_attr
        self.element = element

    @property
    def validators(self) -> List:
        return self.format_attr.validators

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._children})"

    def __iter__(self) -> Generator[Tuple[str, "DataType", ET._Element], None, None]:
        """Return a tuple of (name, child_data_type, child_element) for each
        child."""
        for el_child in self.element:
            if "name" in el_child.attrib:
                name: str = el_child.attrib["name"]
                child_data_type: DataType = self._children[name]
                yield name, child_data_type, el_child
            else:
                assert len(self._children) == 1, "Must have exactly one child."
                yield None, list(self._children.values())[0], el_child

    def iter(
        self, element: ET._Element
    ) -> Generator[Tuple[str, "DataType", ET._Element], None, None]:
        """
        Iterate over the children of an element.

        Yields tuples of (name, child_data_type, child_element) for each child.
        """
        for el_child in element:
            if element.tag == "list":
                assert len(self._children) == 1, "Must have exactly one child."
                yield None, list(self._children.values())[0], el_child
            else:
                name: str = el_child.attrib["name"]
                child_data_type: DataType = self._children[name]
                yield name, child_data_type, el_child

    def from_str(self, s: str) -> "DataType":
        """Create a DataType from a string.

        Note: ScalarTypes like int, float, bool, etc. will override this method.
        Other ScalarTypes like string, email, url, etc. will not override this
        """
        return s

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        """Validate a value."""
        value = self.from_str(value)

        for validator in self.validators:
            schema = validator.validate_with_correction(key, value, schema)

        return schema

    def set_children(self, element: ET._Element):
        raise NotImplementedError("Abstract method.")

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False) -> "DataType":
        from guardrails.schema import FormatAttr

        # TODO: don't want to pass strict through to DataType,
        # but need to pass it to FormatAttr.from_element
        # how to handle this?
        format_attr = FormatAttr.from_element(element)
        format_attr.get_validators(strict)

        data_type = cls({}, format_attr, element)
        data_type.set_children(element)
        return data_type

    @property
    def children(self) -> SimpleNamespace:
        """Return a SimpleNamespace of the children of this DataType."""
        return SimpleNamespace(**self._children)


registry: Dict[str, DataType] = {}


# Create a decorator to register a type
def register_type(name: str):
    def decorator(cls: type):
        registry[name] = cls
        return cls

    return decorator


class ScalarType(DataType):
    def set_children(self, element: ET._Element):
        for _ in element:
            raise ValueError("ScalarType data type must not have any children.")


class NonScalarType(DataType):
    pass


@register_type("string")
class String(ScalarType):
    """Element tag: `<string>`"""

    def from_str(self, s: str) -> "String":
        """Create a String from a string."""
        return s


@register_type("integer")
class Integer(ScalarType):
    """Element tag: `<integer>`"""

    def from_str(self, s: str) -> "Integer":
        """Create an Integer from a string."""
        if s is None:
            return None

        return int(s)


@register_type("float")
class Float(ScalarType):
    """Element tag: `<float>`"""

    def from_str(self, s: str) -> "Float":
        """Create a Float from a string."""
        if s is None:
            return None

        return float(s)


@register_type("bool")
class Boolean(ScalarType):
    """Element tag: `<bool>`"""

    def from_str(self, s: Union[str, bool]) -> "Boolean":
        """Create a Boolean from a string."""
        if s is None:
            return None

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
    """Element tag: `<date>`

    To configure the date format, create a date-format attribute on the element.
    E.g. `<date name="..." ... date-format="%Y-%m-%d" />`
    """

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        self.date_format = "%Y-%m-%d"
        super().__init__(children, format_attr, element)

    def from_str(self, s: str) -> "Date":
        """Create a Date from a string."""
        if s is None:
            return None

        return datetime.datetime.strptime(s, self.date_format).date()

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False) -> "DataType":
        datatype = super().from_xml(element, strict)

        if "date-format" in element.attrib:
            datatype.date_format = element.attrib["date-format"]

        return datatype


@register_type("time")
class Time(ScalarType):
    """Element tag: `<time>`

    To configure the date format, create a date-format attribute on the element.
    E.g. `<time name="..." ... time-format="%H:%M:%S" />`
    """

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        self.time_format = "%H:%M:%S"
        super().__init__(children, format_attr, element)

    def from_str(self, s: str) -> "Time":
        """Create a Time from a string."""
        if s is None:
            return None

        return datetime.datetime.strptime(s, self.time_format).time()

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False) -> "DataType":
        datatype = super().from_xml(element, strict)

        if "time-format" in element.attrib:
            datatype.date_format = element.attrib["time-format"]

        return datatype


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


@register_type("choice")
class Choice(NonScalarType):
    """Element tag: `<object>`"""

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        super().__init__(children, format_attr, element)

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        # Call the validate method of the parent class
        super().validate(key, value, schema)

        # Validate the selected choice
        selected_key = value
        selected_value = schema[selected_key]

        self._children[selected_key].validate(selected_key, selected_value, schema)

        schema[key] = value
        return schema

    def set_children(self, element: ET._Element):
        for child in element:
            child_data_type = registry[child.tag]
            assert child_data_type == Case
            self._children[child.attrib["name"]] = child_data_type.from_xml(child)

    @property
    def validators(self) -> List:
        from guardrails.validators import Choice as ChoiceValidator

        # Check if the <choice ... /> element has an `on-fail` attribute.
        # If so, use that as the `on_fail` argument for the PydanticValidator.
        on_fail = None
        on_fail_attr_name = "on-fail-choice"
        if on_fail_attr_name in self.element.attrib:
            on_fail = self.element.attrib[on_fail_attr_name]
        return [ChoiceValidator(choices=list(self._children.keys()), on_fail=on_fail)]


@register_type("case")
class Case(NonScalarType):
    """Element tag: `<case>`"""

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        super().__init__(children, format_attr, element)

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        child = list(self._children.values())[0]
        child.validate(key, value, schema)

        schema[key] = value

        return schema

    def set_children(self, element: ET._Element):
        assert len(element) == 1, "Case must have exactly one child."

        for child in element:
            child_data_type = registry[child.tag]
            self._children[child.attrib["name"]] = child_data_type.from_xml(child)


@register_type("pydantic")
class Pydantic(NonScalarType):
    """Element tag: `<pydantic>`"""

    def __init__(
        self,
        model: Type[BaseModel],
        children: Dict[str, Any],
        format_attr: "FormatAttr",
        element: ET._Element,
    ) -> None:
        super().__init__(children, format_attr, element)
        assert (
            format_attr.empty
        ), "The <pydantic /> data type does not support the `format` attribute."
        assert isinstance(model, type) and issubclass(
            model, BaseModel
        ), "The `model` argument must be a Pydantic model."

        self.model = model

    @property
    def validators(self) -> List:
        from guardrails.validators import Pydantic as PydanticValidator

        # Check if the <pydantic /> element has an `on-fail` attribute.
        # If so, use that as the `on_fail` argument for the PydanticValidator.
        on_fail = None
        on_fail_attr_name = "on-fail-pydantic"
        if on_fail_attr_name in self.element.attrib:
            on_fail = self.element.attrib[on_fail_attr_name]
        return [PydanticValidator(self.model, on_fail=on_fail)]

    def set_children(self, element: ET._Element):
        for child in element:
            child_data_type = registry[child.tag]
            self._children[child.attrib["name"]] = child_data_type.from_xml(child)

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False) -> "DataType":
        from guardrails.schema import FormatAttr
        from guardrails.utils.pydantic_utils import pydantic_models

        model_name = element.attrib["model"]
        model = pydantic_models.get(model_name, None)

        if model is None:
            raise ValueError(f"Invalid Pydantic model: {model_name}")

        data_type = cls(model, {}, FormatAttr(), element)
        data_type.set_children(element)
        return data_type

    def to_object_element(self) -> ET._Element:
        """Convert the Pydantic data type to an <object /> element."""
        from guardrails.utils.pydantic_utils import (
            PYDANTIC_SCHEMA_TYPE_MAP,
            get_field_descriptions,
            pydantic_validators,
        )

        # Get the following attributes
        # TODO: add on-fail
        try:
            name = self.element.attrib["name"]
        except KeyError:
            name = None
        try:
            description = self.element.attrib["description"]
        except KeyError:
            description = None

        # Get the Pydantic model schema.
        schema = self.model.schema()
        field_descriptions = get_field_descriptions(self.model)

        # Make the XML as follows using lxml
        # <object name="..." description="..." format="semicolon separated root validators" pydantic="ModelName"> # noqa: E501
        #     <type name="..." description="..." format="semicolon separated validators" /> # noqa: E501
        # </object>

        # Add the object element, opening tag
        xml = ""
        root_validators = "; ".join(
            list(pydantic_validators[self.model]["__root__"].keys())
        )
        xml += "<object "
        if name:
            xml += f' name="{name}"'
        if description:
            xml += f' description="{description}"'
        if root_validators:
            xml += f' format="{root_validators}"'
        xml += f' pydantic="{self.model.__name__}"'
        xml += ">"

        # Add all the nested fields
        for field in schema["properties"]:
            properties = schema["properties"][field]
            field_type = PYDANTIC_SCHEMA_TYPE_MAP[properties["type"]]
            field_validators = "; ".join(
                list(pydantic_validators[self.model][field].keys())
            )
            try:
                field_description = field_descriptions[field]
            except KeyError:
                field_description = ""
            xml += f"<{field_type}"
            xml += f' name="{field}"'
            if field_description:
                xml += f' description="{field_descriptions[field]}"'
            if field_validators:
                xml += f' format="{field_validators}"'
            xml += " />"

        # Close the object element
        xml += "</object>"

        # Convert the string to an XML element, making sure to format it.
        return ET.fromstring(
            xml, parser=ET.XMLParser(encoding="utf-8", remove_blank_text=True)
        )


@register_type("field")
class Field(ScalarType):
    """Element tag: `<field>`"""


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
