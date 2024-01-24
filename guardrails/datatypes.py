import datetime
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable
from typing import List as TypedList
from typing import Optional, Sequence, Type, TypeVar, Union

from dateutil.parser import parse
from lxml import etree as ET
from typing_extensions import Self

from guardrails.utils.casting_utils import to_float, to_int, to_string
from guardrails.utils.xml_utils import cast_xml_to_string
from guardrails.validator_base import Validator, ValidatorSpec
from guardrails.validatorsattr import ValidatorsAttr

# TODO - deprecate these altogether
deprecated_string_types = {"sql", "email", "url", "pythoncode"}


def update_deprecated_type_to_string(type):
    if type in deprecated_string_types:
        return "string"
    return type


@dataclass
class FieldValidation:
    key: Any
    value: Any
    validators: TypedList[Validator]
    children: TypedList["FieldValidation"]


def verify_metadata_requirements(
    metadata: dict, datatypes: Union["DataType", Iterable["DataType"]]
) -> TypedList[str]:
    missing_keys = set()
    if isinstance(datatypes, DataType):
        datatypes = [datatypes]
    for datatype in datatypes:
        for validator in datatype.validators:
            for requirement in validator.required_metadata_keys:
                if requirement not in metadata:
                    missing_keys.add(requirement)
        nested_missing_keys = verify_metadata_requirements(
            metadata, vars(datatype.children).values()
        )
        missing_keys.update(nested_missing_keys)
    missing_keys = list(missing_keys)
    missing_keys.sort()
    return missing_keys


class DataType:
    rail_alias: str
    tag: str

    def __init__(
        self,
        children: Dict[str, Any],
        validators_attr: ValidatorsAttr,
        optional: bool,
        name: Optional[str],
        description: Optional[str],
    ) -> None:
        self._children = children
        self.validators_attr = validators_attr
        self.name = name
        self.description = description
        self.optional = optional

    def get_example(self):
        raise NotImplementedError

    @property
    def validators(self) -> TypedList:
        return self.validators_attr.validators

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._children})"

    def from_str(self, s: str) -> str:
        """Create a DataType from a string.

        Note: ScalarTypes like int, float, bool, etc. will override this method.
        Other ScalarTypes like string, email, url, etc. will not override this
        """
        return s

    def _constructor_validation(
        self,
        key: str,
        value: Any,
    ) -> FieldValidation:
        """Creates a "FieldValidation" object for ValidatorService to run over,
        which specifies the key, value, and validators for a given field.

        Its children should be populated by its nested fields'
        FieldValidations.
        """
        return FieldValidation(
            key=key, value=value, validators=self.validators, children=[]
        )

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        """Gather validators on a value."""
        value = self.from_str(value)
        return self._constructor_validation(key, value)

    def set_children_from_xml(self, element: ET._Element):
        raise NotImplementedError("Abstract method.")

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False, **kwargs) -> Self:
        # TODO: don't want to pass strict through to DataType,
        # but need to pass it to ValidatorsAttr.from_element
        # how to handle this?
        validators_attr = ValidatorsAttr.from_xml(element, cls.tag, strict)

        is_optional = element.attrib.get("required", "true") == "false"

        name = element.attrib.get("name")
        if name is not None:
            name = cast_xml_to_string(name)

        description = element.attrib.get("description")
        if description is not None:
            description = cast_xml_to_string(description)

        data_type = cls({}, validators_attr, is_optional, name, description, **kwargs)
        data_type.set_children_from_xml(element)
        return data_type

    @property
    def children(self) -> SimpleNamespace:
        """Return a SimpleNamespace of the children of this DataType."""
        return SimpleNamespace(**self._children)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.__dict__ == other.__dict__


registry: Dict[str, Type[DataType]] = {}


T = TypeVar("T", bound=Type[DataType])


# Create a decorator to register a type
def register_type(name: str):
    def decorator(cls: T) -> T:
        registry[name] = cls
        cls.rail_alias = name
        return cls

    return decorator


# Decorator for deprecation
def deprecate_type(cls: type):
    warnings.warn(
        f"""The '{cls.__name__}' type  is deprecated and will be removed in \
versions 0.3.0 and beyond. Use the pydantic 'str' primitive instead.""",
        DeprecationWarning,
    )
    return cls


class ScalarType(DataType):
    def set_children_from_xml(self, element: ET._Element):
        for _ in element:
            raise ValueError("ScalarType data type must not have any children.")


class NonScalarType(DataType):
    pass


@register_type("string")
class String(ScalarType):
    """Element tag: `<string>`"""

    tag = "string"

    def get_example(self):
        return "string"

    def from_str(self, s: str) -> Optional[str]:
        """Create a String from a string."""
        return to_string(s)

    @classmethod
    def from_string_rail(
        cls,
        validators: Sequence[ValidatorSpec],
        description: Optional[str] = None,
        strict: bool = False,
    ) -> Self:
        return cls(
            children={},
            validators_attr=ValidatorsAttr.from_validators(validators, cls.tag, strict),
            optional=False,
            name=None,
            description=description,
        )


@register_type("integer")
class Integer(ScalarType):
    """Element tag: `<integer>`"""

    tag = "integer"

    def get_example(self):
        return 1

    def from_str(self, s: str) -> Optional[int]:
        """Create an Integer from a string."""
        return to_int(s)


@register_type("float")
class Float(ScalarType):
    """Element tag: `<float>`"""

    tag = "float"

    def get_example(self):
        return 1.5

    def from_str(self, s: str) -> Optional[float]:
        """Create a Float from a string."""
        return to_float(s)


@register_type("bool")
class Boolean(ScalarType):
    """Element tag: `<bool>`"""

    tag = "bool"

    def get_example(self):
        return True

    def from_str(self, s: Union[str, bool]) -> Optional[bool]:
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

    To configure the date format, create a date-format attribute on the
    element. E.g. `<date name="..." ... date-format="%Y-%m-%d" />`
    """

    tag = "date"

    def __init__(
        self,
        children: Dict[str, Any],
        validators_attr: "ValidatorsAttr",
        optional: bool,
        name: Optional[str],
        description: Optional[str],
    ) -> None:
        super().__init__(children, validators_attr, optional, name, description)
        self.date_format = None

    def get_example(self):
        return datetime.date.today()

    def from_str(self, s: str) -> Optional[datetime.date]:
        """Create a Date from a string."""
        if s is None:
            return None
        if not self.date_format:
            return parse(s).date()
        return datetime.datetime.strptime(s, self.date_format).date()

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False) -> "Date":
        datatype = super().from_xml(element, strict)

        if "date-format" in element.attrib or "date_format" in element.attrib:
            datatype.date_format = element.attrib["date-format"]

        return datatype


@register_type("time")
class Time(ScalarType):
    """Element tag: `<time>`

    To configure the date format, create a date-format attribute on the
    element. E.g. `<time name="..." ... time-format="%H:%M:%S" />`
    """

    tag = "time"

    def __init__(
        self,
        children: Dict[str, Any],
        validators_attr: "ValidatorsAttr",
        optional: bool,
        name: Optional[str],
        description: Optional[str],
    ) -> None:
        self.time_format = "%H:%M:%S"
        super().__init__(children, validators_attr, optional, name, description)

    def get_example(self):
        return datetime.time()

    def from_str(self, s: str) -> Optional[datetime.time]:
        """Create a Time from a string."""
        if s is None:
            return None

        return datetime.datetime.strptime(s, self.time_format).time()

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False) -> "Time":
        datatype = super().from_xml(element, strict)

        if "time-format" in element.attrib or "time_format" in element.attrib:
            datatype.time_format = element.attrib["time-format"]

        return datatype


@deprecate_type
@register_type("email")
class Email(ScalarType):
    """Element tag: `<email>`"""

    tag = "email"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deprecate_type(type(self))

    def get_example(self):
        return "hello@example.com"


@deprecate_type
@register_type("url")
class URL(ScalarType):
    """Element tag: `<url>`"""

    tag = "url"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deprecate_type(type(self))

    def get_example(self):
        return "https://example.com"


@deprecate_type
@register_type("pythoncode")
class PythonCode(ScalarType):
    """Element tag: `<pythoncode>`"""

    tag = "pythoncode"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deprecate_type(type(self))

    def get_example(self):
        return "print('hello world')"


@deprecate_type
@register_type("sql")
class SQLCode(ScalarType):
    """Element tag: `<sql>`"""

    tag = "sql"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deprecate_type(type(self))

    def get_example(self):
        return "SELECT * FROM table"


@register_type("percentage")
class Percentage(ScalarType):
    """Element tag: `<percentage>`"""

    tag = "percentage"

    def get_example(self):
        return "20%"


@register_type("enum")
class Enum(ScalarType):
    """Element tag: `<enum>`"""

    tag = "enum"

    def __init__(
        self,
        children: Dict[str, Any],
        validators_attr: ValidatorsAttr,
        optional: bool,
        name: Optional[str],
        description: Optional[str],
        enum_values: TypedList[str],
    ) -> None:
        super().__init__(children, validators_attr, optional, name, description)
        self.enum_values = enum_values

    def get_example(self):
        return self.enum_values[0]

    def from_str(self, s: str) -> Optional[str]:
        """Create an Enum from a string."""
        if s is None:
            return None
        if s not in self.enum_values:
            raise ValueError(f"Invalid enum value: {s}")
        return s

    @classmethod
    def from_xml(
        cls,
        enum_values: TypedList[str],
        validators: Sequence[ValidatorSpec],
        description: Optional[str] = None,
        strict: bool = False,
    ) -> "Enum":
        return cls(
            children={},
            validators_attr=ValidatorsAttr.from_validators(validators, cls.tag, strict),
            optional=False,
            name=None,
            description=description,
            enum_values=enum_values,
        )


@register_type("list")
class List(NonScalarType):
    """Element tag: `<list>`"""

    tag = "list"

    def get_example(self):
        return [e.get_example() for e in self._children.values()]

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        # Validators in the main list data type are applied to the list overall.
        validation = self._constructor_validation(key, value)

        if value is None and self.optional:
            return validation

        if len(self._children) == 0:
            return validation

        item_type = list(self._children.values())[0]

        # TODO(shreya): Edge case: List of lists -- does this still work?
        for i, item in enumerate(value):
            child_validation = item_type.collect_validation(i, item, value)
            validation.children.append(child_validation)

        return validation

    def set_children_from_xml(self, element: ET._Element):
        for idx, child in enumerate(element, start=1):
            if idx > 1:
                # Only one child is allowed in a list data type.
                # The child must be the datatype that all items in the list
                # must conform to.
                raise ValueError("List data type must have exactly one child.")
            child_data_type_tag = update_deprecated_type_to_string(child.tag)
            child_data_type = registry[child_data_type_tag]
            self._children["item"] = child_data_type.from_xml(child)


@register_type("object")
class Object(NonScalarType):
    """Element tag: `<object>`"""

    tag = "object"

    def get_example(self):
        return {k: v.get_example() for k, v in self._children.items()}

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        # Validators in the main object data type are applied to the object overall.
        validation = self._constructor_validation(key, value)

        if value is None and self.optional:
            return validation

        if len(self._children) == 0:
            return validation

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
            child_value = value.get(child_key, None)

            # Skip validation for instances where child_value is None
            # by adding a check for child_value
            # This will happen during streaming (sub-schema validation)
            if child_value:
                child_validation = child_data_type.collect_validation(
                    child_key,
                    child_value,
                    value,
                )
                validation.children.append(child_validation)

        return validation

    def set_children_from_xml(self, element: ET._Element):
        for child in element:
            child_data_type = registry[child.tag]

            name = child.attrib["name"]
            name = cast_xml_to_string(name)

            self._children[name] = child_data_type.from_xml(child)


@register_type("choice")
class Choice(NonScalarType):
    """Element tag: `<object>`"""

    tag = "choice"

    def __init__(
        self,
        children: Dict[str, Any],
        validators_attr: "ValidatorsAttr",
        optional: bool,
        name: Optional[str],
        description: Optional[str],
        discriminator_key: str,
    ) -> None:
        super().__init__(children, validators_attr, optional, name, description)
        self.discriminator_key = discriminator_key

    def get_example(self):
        first_discriminator = list(self._children.keys())[0]
        first_child = list(self._children.values())[0]
        return {
            self.discriminator_key: first_discriminator,
            **first_child.get_example(),
        }

    @classmethod
    def from_xml(cls, element: ET._Element, strict: bool = False, **kwargs) -> Self:
        # grab `discriminator` attribute
        disc = element.attrib.get("discriminator")
        if disc is not None:
            disc = cast_xml_to_string(disc)
        else:
            disc = "discriminator"

        datatype = super().from_xml(element, strict, discriminator_key=disc, **kwargs)
        return datatype

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        # Validate the selected choice
        discriminator_value = value[self.discriminator_key]

        validation = self._children[discriminator_value].collect_validation(
            key,
            value,
            schema,
        )

        return validation

    def set_children_from_xml(self, element: ET._Element):
        for child in element:
            child_data_type = registry[child.tag]
            assert child_data_type == Case

            name = child.attrib["name"]
            name = cast_xml_to_string(name)

            self._children[name] = child_data_type.from_xml(child)

    @property
    def validators(self) -> TypedList:
        return []


@register_type("case")
class Case(NonScalarType):
    """Element tag: `<case>`"""

    tag = "case"

    def __init__(
        self,
        children: Dict[str, Any],
        validators_attr: "ValidatorsAttr",
        optional: bool,
        name: Optional[str],
        description: Optional[str],
    ) -> None:
        super().__init__(children, validators_attr, optional, name, description)

    def get_example(self):
        return {k: v.get_example() for k, v in self._children.items()}

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        # Validate the selected choice
        validation = self._constructor_validation(key, value)

        # Collect validation for all children
        for child_key, child_data_type in self._children.items():
            # Value should be a dictionary
            # child_key is an expected key that the schema defined
            # child_data_type is the data type of the expected key
            child_value = value.get(child_key, None)
            child_validation = child_data_type.collect_validation(
                child_key,
                child_value,
                value,
            )
            validation.children.append(child_validation)

        return validation

    def set_children_from_xml(self, element: ET._Element):
        for child in element:
            child_data_type = registry[child.tag]

            name = child.attrib["name"]
            name = cast_xml_to_string(name)

            self._children[name] = child_data_type.from_xml(child)


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
