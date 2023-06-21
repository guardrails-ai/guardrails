from dataclasses import dataclass
import datetime
import re
from types import SimpleNamespace
from typing import Any, Dict, Generator, Optional
from typing import List as TypedList
from typing import Tuple, Type, Union
import warnings

from lxml import etree as ET
from lxml.builder import E
from pydantic import BaseModel
from pydantic.fields import ModelField
from guardrails.utils.pydantic_utils import (
    edit_element_to_add_validators,
    prepare_type_annotation,
    is_pydantic_base_model,
    is_list,
    is_dict,
    get_args,
    type_annotation_to_string,
    extract_pydantic_validators_as_guardrails_validators,
    add_validators_to_element,
)

from collections import defaultdict
from guardrails.validators import Validator
from dataclasses import field


@dataclass
class FormatAttr:
    """Class for parsing and manipulating the `format` attribute of an element.

    The format attribute is a string that contains semi-colon separated
    validators e.g. "valid-url; is-reachable". Each validator is itself either:
    - the name of an parameter-less validator, e.g. "valid-url"
    - the name of a validator with parameters, separated by a colon with a
        space-separated list of parameters, e.g. "is-in: 1 2 3"

    Parameters can either be written in plain text, or in python expressions
    enclosed in curly braces. For example, the following are all valid:
    - "is-in: 1 2 3"
    - "is-in: {1} {2} {3}"
    - "is-in: {1 + 2} {2 + 3} {3 + 4}"
    """

    # The format attribute string.
    format: Optional[str] = None

    # The XML element that this format attribute is associated with.
    element: Optional[ET._Element] = None

    @property
    def empty(self) -> bool:
        """Return True if the format attribute is empty, False otherwise."""
        return self.format is None

    @classmethod
    def from_element(cls, element: ET._Element) -> "FormatAttr":
        """Create a FormatAttr object from an XML element.

        Args:
            element (ET._Element): The XML element.

        Returns:
            A FormatAttr object.
        """
        return cls(element.get("format"), element)

    @property
    def tokens(self) -> TypedList[str]:
        """Split the format attribute into tokens.

        For example, the format attribute "valid-url; is-reachable" will
        be split into ["valid-url", "is-reachable"]. The semicolon is
        used as a delimiter, but not if it is inside curly braces,
        because the format string can contain Python expressions that
        contain semicolons.
        """
        if self.format is None:
            return []
        pattern = re.compile(r";(?![^{}]*})")
        tokens = re.split(pattern, self.format)
        tokens = list(filter(None, tokens))
        return tokens

    @classmethod
    def parse_token(cls, token: str) -> Tuple[str, TypedList[Any]]:
        """Parse a single token in the format attribute, and return the
        validator name and the list of arguments.

        Args:
            token (str): The token to parse, one of the tokens returned by
                `self.tokens`.

        Returns:
            A tuple of the validator name and the list of arguments.
        """
        validator_with_args = token.strip().split(":", 1)
        if len(validator_with_args) == 1:
            return validator_with_args[0].strip(), []

        validator, args_token = validator_with_args

        # Split using whitespace as a delimiter, but not if it is inside curly braces or
        # single quotes.
        pattern = re.compile(r"\s(?![^{}]*})|(?<!')\s(?=[^']*'$)")
        tokens = re.split(pattern, args_token)

        # Filter out empty strings if any.
        tokens = list(filter(None, tokens))

        args = []
        for t in tokens:
            # If the token is enclosed in curly braces, it is a Python expression.
            t = t.strip()
            if t[0] == "{" and t[-1] == "}":
                t = t[1:-1]
                try:
                    # Evaluate the Python expression.
                    t = eval(t)
                except (ValueError, SyntaxError, NameError) as e:
                    raise ValueError(
                        f"Python expression `{t}` is not valid, "
                        f"and raised an error: {e}."
                    )
            args.append(t)

        return validator.strip(), args

    def parse(self) -> Dict:
        """Parse the format attribute into a dictionary of validators.

        Returns:
            A dictionary of validators, where the key is the validator name, and
            the value is a list of arguments.
        """
        if self.format is None:
            return {}

        # Split the format attribute into tokens: each is a validator.
        # Then, parse each token into a validator name and a list of parameters.
        validators = {}
        for token in self.tokens:
            # Parse the token into a validator name and a list of parameters.
            validator_name, args = self.parse_token(token)
            validators[validator_name] = args

        return validators

    @property
    def validators(self) -> TypedList["Validator"]:
        """Get the list of validators from the format attribute.

        Only the validators that are registered for this element will be
        returned.
        """
        try:
            return getattr(self, "_validators")
        except AttributeError:
            raise AttributeError("Must call `get_validators` first.")

    @property
    def unregistered_validators(self) -> TypedList[str]:
        """Get the list of validators from the format attribute that are not
        registered for this element."""
        try:
            return getattr(self, "_unregistered_validators")
        except AttributeError:
            raise AttributeError("Must call `get_validators` first.")

    def get_validators(self, strict: bool = False) -> TypedList["Validator"]:
        """Get the list of validators from the format attribute. Only the
        validators that are registered for this element will be returned.

        For example, if the format attribute is "valid-url; is-reachable", and
        "is-reachable" is not registered for this element, then only the ValidUrl
        validator will be returned, after instantiating it with the arguments
        specified in the format attribute (if any).

        Args:
            strict: If True, raise an error if a validator is not registered for
                this element. If False, ignore the validator and print a warning.

        Returns:
            A list of validators.
        """
        from guardrails.validators import types_to_validators, validators_registry

        _validators = []
        _unregistered_validators = []
        parsed = self.parse().items()
        for validator_name, args in parsed:
            # Check if the validator is registered for this element.
            # The validators in `format` that are not registered for this element
            # will be ignored (with an error or warning, depending on the value of
            # `strict`), and the registered validators will be returned.
            if validator_name not in types_to_validators[self.element.tag]:
                if strict:
                    raise ValueError(
                        f"Validator {validator_name} is not valid for"
                        f" element {self.element.tag}."
                    )
                else:
                    warnings.warn(
                        f"Validator {validator_name} is not valid for"
                        f" element {self.element.tag}."
                    )
                    _unregistered_validators.append(validator_name)
                continue

            validator = validators_registry[validator_name]

            # See if the formatter has an associated on_fail method.
            on_fail = None
            on_fail_attr_name = f"on-fail-{validator_name}"
            if on_fail_attr_name in self.element.attrib:
                on_fail = self.element.attrib[on_fail_attr_name]
                # TODO(shreya): Load the on_fail method.
                # This method should be loaded from an optional script given at the
                # beginning of a rail file.

            # Create the validator.
            _validators.append(validator(*args, on_fail=on_fail))

        self._validators = _validators
        self._unregistered_validators = _unregistered_validators
        return _validators

    def to_prompt(self, with_keywords: bool = True) -> str:
        """Convert the format string to another string representation for use
        in prompting. Uses the validators' to_prompt method in order to
        construct the string to use in prompting.

        For example, the format string "valid-url; other-validator: 1.0
        {1 + 2}" will be converted to "valid-url other-validator:
        arg1=1.0 arg2=3".
        """
        if self.format is None:
            return ""
        # Use the validators' to_prompt method to convert the format string to
        # another string representation.
        prompt = "; ".join([v.to_prompt(with_keywords) for v in self.validators])
        unreg_prompt = "; ".join(self.unregistered_validators)
        if prompt and unreg_prompt:
            prompt += f"; {unreg_prompt}"
        elif unreg_prompt:
            prompt += unreg_prompt
        return prompt


@dataclass
class Element:
    name: str = field(default="root")
    description: Optional[str] = field(default=None)
    children: Dict[str, Any] = field(default_factory=dict)
    value: Optional[Any] = field(default=None)
    path: Optional[TypedList["Element"]] = field(default_factory=list)
    validators: Optional[TypedList["Validator"]] = field(default_factory=list)
    unregistered_validators: Optional[TypedList[str]] = field(default_factory=list)
    strict: bool = field(default=True)

    @property
    def element_type(self) -> str:
        return self._element_type if hasattr(self, "_element_type") else "element"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.children})"

    @classmethod
    def from_xml(cls, xml: ET.Element) -> "Element":
        """
        Convert an XML element to a gd.Element object.

        Args:
            xml (ET.Element): The XML element.

        Returns:
            A gd.Element object.
        """

        # Get the element type, e.g. "string", "integer", etc.
        try:
            el_subtype: Element = registry[xml.tag]
        except KeyError:
            raise KeyError(f"Element type {xml.tag} is not valid.")

        # Parse the `format` attribute of the element to build the list of validators.
        format_attr = FormatAttr.from_element(xml)
        strict = bool(xml.attrib["strict"]) if hasattr(xml, "strict") else False
        validators = format_attr.get_validators(strict)
        unregistered_validators = format_attr.unregistered_validators

        # Recursively parse the children of the element to build the tree.
        if xml.tag == "list":
            children = {"item": Element.from_xml(xml[0])}
        else:
            children = {child.attrib["name"]: Element.from_xml(child) for child in xml}

        if "name" in xml.attrib:
            name = xml.attrib["name"]
        else:
            if xml.tag == "output":
                # If the element is an output, then the name of the element
                name = "root"
            else:
                # If the element is not an output, then the name of the element
                # is `item` since the parent element is a list.
                name = "item"

        if "description" in xml.attrib:
            description = xml.attrib["description"]
        else:
            description = None

        return el_subtype(
            name=name,
            description=description,
            children=children,
            value=None,
            path=[],
            validators=validators,
            unregistered_validators=unregistered_validators,
            strict=strict,
        )

    @classmethod
    def from_pydantic(self, model: BaseModel) -> "Element":
        """
        Convert a Pydantic model to a gd.Element object.

        Args:
            model (BaseModel): The Pydantic model.

        Returns:
            A gd.Element object.
        """
        return create_element_for_base_model(model)

    def to_xml(self) -> ET.Element:
        """
        Convert a Foo object to an XML element.

        Returns:
            An XML element.
        """
        element = E(self.element_type, name=self.name)

        if self.description:
            element.attrib["description"] = self.description
        if self.validators or self.unregistered_validators:
            element = edit_element_to_add_validators(
                element,
                self.validators,
                self.unregistered_validators,
                add_on_fail=False,
            )
        if self.children:
            for child in self.children.values():
                element.append(child.to_xml())
        return element

    def cast(self, value: Any) -> Any:
        """
        Cast a value to the type of the Foo subclass.
        This is implemented by each subclass of Foo.

        Note: scalars like int, float, bool, etc. will override this method.
        Others like string, email, url, etc. will not override this.
        """
        return value

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        """Validate a value against the validators of the Foo object."""
        value = self.from_str(value)

        for validator in self.validators:
            schema = validator.validate_with_correction(key, value, schema)

        return schema

    # def set_children(self, element: ET._Element):
    #     # TODO: fix signature
    #     raise NotImplementedError("Abstract method.")

    # @property
    # def children(self) -> SimpleNamespace:
    #     """Return the children of the Foo object as a SimpleNamespace."""
    #     return SimpleNamespace(**self._children)


def create_element_for_field(
    field: Union[ModelField, Type, type],
    field_name: Optional[str] = None,
) -> "Element":
    """Create an XML element"""

    # Create the element based on the field type
    field_type = type_annotation_to_string(field)
    try:
        sub_el_type = registry.get(field_type)
    except KeyError:
        raise ValueError(f"Unsupported type: {field_type}")
    # element = E(field_type)

    # Add name attribute
    name = None
    if field_name:
        name = field_name

    # Add validators
    # element = add_validators_to_xml_element(field, element)
    validators = add_validators_to_element(field)

    # Add description attribute
    description = None
    if isinstance(field, ModelField):
        if field.field_info.description is not None:
            description = field.field_info.description

    # Create `gd.Element`s for the field's children
    children = {}
    if field_type in ["list", "object"]:
        type_annotation = prepare_type_annotation(field)

        if is_list(type_annotation):
            inner_type = get_args(type_annotation)
            if len(inner_type) == 0:
                # If the list is empty, we cannot infer the type of the elements
                pass

            inner_type = inner_type[0]
            if is_pydantic_base_model(inner_type):
                object_el = create_element_for_base_model(inner_type)
                # element.append(object_el)
                children["item"] = object_el
            else:
                inner_el = create_element_for_field(inner_type)
                # element.append(inner_el)
                children["item"] = inner_el

        elif is_dict(type_annotation):
            if is_pydantic_base_model(type_annotation):
                return create_element_for_base_model(
                    type_annotation,
                    name=name,
                    description=description,
                    validators=validators,
                )
            else:
                dict_args = get_args(type_annotation)
                if len(dict_args) == 2:
                    key_type, val_type = dict_args
                    assert key_type == str, "Only string keys are supported for dicts"
                    inner_element = create_element_for_field(val_type)
                    # element.append(inner_element)
                    inner_element_name = inner_element.name
                    children[inner_element_name] = inner_element
        else:
            raise ValueError(f"Unsupported type: {type_annotation}")

    # Create the gd.Element object
    sub_el = sub_el_type(
        name=name, description=description, validators=validators, children=children
    )

    return sub_el


def create_element_for_base_model(
    model: BaseModel,
    name: Optional[str] = None,
    description: Optional[str] = None,
    validators: Optional[TypedList[Validator]] = [],
) -> "Element":
    """Create a gd.Element object for a Pydantic BaseModel.

    This function does the following:
        1. Iterates through fields of the model and creates gd.Element objects
            for each field.
        2. If a field is a Pydantic BaseModel, it creates a gd.Element object
            that internally nests other gd.Element objects.
        3. If the BaseModel contains a field with a `when` attribute, it creates
           special `Choice` and `Case` elements for the field. # TODO

    Args:
        model: The Pydantic BaseModel to create a gd.Element object for

    Returns:
        The gd.Element object
    """
    # Extract pydantic validators from the model and add them as guardrails
    # validators
    model_fields = extract_pydantic_validators_as_guardrails_validators(model)

    # Identify fields with `when` attribute
    choice_elements = defaultdict(list)
    case_elements = set()
    for field_name, field in model_fields.items():
        if "when" in field.field_info.extra:
            choice_elements[field.field_info.extra["when"]].append((field_name, field))
            case_elements.add(field_name)

    children = {}
    # Add fields to the XML element, except for fields with `when` attribute
    for field_name, field in model_fields.items():
        if field_name in choice_elements or field_name in case_elements:
            continue
        field_el = create_element_for_field(field, field_name)
        children[field_name] = field_el

    # Add `Choice` and `Case` elements for fields with `when` attribute
    for when, discriminator_fields in choice_elements.items():
        # choice_element = E("choice", name=when)
        # TODO(shreya): DONT MERGE WTHOUT SOLVING THIS: How do you set this via SDK?
        # choice_element.set("on-fail-choice", "exception")

        choice_el_children = {}
        for field_name, field in discriminator_fields:
            # case_element = E("case", name=field_name)
            # field_el = create_element_for_field(field, field_name)
            # case_element.append(field_el)
            field_el = create_element_for_field(field)
            case_el = Case(name=field_name, children=[field_el])
            choice_el_children[field_name] = case_el
            # choice_element.append(case_element)

        choice_el = Choice(
            name=when, on_fail_choice="exception", children=choice_el_children
        )
        # element.append(choice_element)
        children[when] = choice_el

    # Create the Foo object
    object_el = Object(
        name=name if name is not None else model.__name__,
        description=description,
        children=children,
        validators=validators,
    )

    return object_el


registry: Dict[str, Element] = {}


# Create a decorator to register a type
def register_type(name: str):
    def decorator(cls: type):
        registry[name] = cls
        print(f"Registered {name} as {cls}")
        cls._element_type = name
        return cls

    return decorator


class ScalarType(Element):
    pass


class NonScalarType(Element):
    pass


@register_type("string")
class String(ScalarType):
    """Element tag: `<string>`"""

    def cast(self, s: str) -> str:
        """Create a String from a string."""
        return s


@register_type("integer")
class Integer(ScalarType):
    """Element tag: `<integer>`"""

    def cast(self, s: Optional[str]) -> Optional[int]:
        """Create an Integer from a string."""
        if s is None:
            return None

        return int(s)


@register_type("float")
class Float(ScalarType):
    """Element tag: `<float>`"""

    def cast(self, s: Optional[str]) -> Optional[float]:
        """Create a Float from a string."""
        if s is None:
            return None

        return float(s)


@register_type("bool")
class Boolean(ScalarType):
    """Element tag: `<bool>`"""

    def cast(self, s: Union[str, bool, None]) -> Optional[bool]:
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

    date_format: str = "%Y-%m-%d"

    @classmethod
    def from_xml(cls, xml: ET._Element) -> "Element":
        el = super().from_xml(xml)

        if "date-format" in xml.attrib:
            el.date_format = xml.attrib["date-format"]

        return el

    def cast(self, s: Optional[str]) -> Optional[datetime.date]:
        """Create a Date from a string."""
        if s is None:
            return None

        return datetime.datetime.strptime(s, self.date_format).date()


@register_type("time")
class Time(ScalarType):
    """Element tag: `<time>`

    To configure the date format, create a date-format attribute on the
    element. E.g. `<time name="..." ... time-format="%H:%M:%S" />`
    """

    time_format: str = "%H:%M:%S"

    @classmethod
    def from_xml(cls, xml: ET._Element) -> "Element":
        el = super().from_xml(xml)

        if "time-format" in xml.attrib:
            el.date_format = xml.attrib["time-format"]

        return el

    def cast(self, s: Optional[str]) -> Optional[datetime.time]:
        """Create a Time from a string."""
        if s is None:
            return None

        return datetime.datetime.strptime(s, self.time_format).time()


@register_type("email")
class Email(ScalarType):
    """Element tag: `<email>`"""


@register_type("url")
class URL(ScalarType):
    """Element tag: `<url>`"""


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

        if len(self.children) == 0:
            return schema

        item_type = list(self.children.values())[0]

        # TODO(shreya): Edge case: List of lists -- does this still work?
        for i, item in enumerate(value):
            value = item_type.validate(i, item, value)

        return schema

    def to_xml(self) -> ET._Element:
        xml = super().to_xml()

        # Update the child of the xml element to not contain the name
        child = xml[0]
        child.attrib.pop("name", None)

        return xml


@register_type("object")
class Object(NonScalarType):
    """Element tag: `<object>`"""

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        # Validators in the main object data type are applied to the object overall.

        for validator in self.validators:
            schema = validator.validate_with_correction(key, value, schema)

        if len(self.children) == 0:
            return schema

        # Types of supported children
        # 1. key_type
        # 2. value_type
        # 3. List of keys that must be present

        # TODO(shreya): Implement key type and value type later

        # Check for required keys
        for child_key, child_data_type in self.children.items():
            # Value should be a dictionary
            # child_key is an expected key that the schema defined
            # child_data_type is the data type of the expected key
            value = child_data_type.validate(
                child_key, value.get(child_key, None), value
            )

        schema[key] = value

        return schema


@register_type("output")
class Output(Object):
    """Element tag: `<output>`"""

    def to_xml(self) -> ET._Element:
        """Convert the Output to an XML element."""
        xml = super().to_xml()
        del xml.attrib["name"]
        return xml


@register_type("choice")
class Choice(NonScalarType):
    """Element tag: `<object>`"""

    on_fail_choice: str = "exception"

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        # Call the validate method of the parent class
        super().validate(key, value, schema)

        # Validate the selected choice
        selected_key = value
        selected_value = schema[selected_key]

        self.children[selected_key].validate(selected_key, selected_value, schema)

        schema[key] = value
        return schema

    # TODO: this should be moved to Foo/Rail.from_xml
    # def set_children(self, element: ET._Element):
    #     for child in element:
    #         child_data_type = registry[child.tag]
    #         assert child_data_type == Case
    #         self._children[child.attrib["name"]] = child_data_type.from_xml(child)

    @property
    def validators(self) -> TypedList:
        from guardrails.validators import Choice as ChoiceValidator

        # Check if the <choice ... /> element has an `on-fail` attribute.
        # If so, use that as the `on_fail` argument for the PydanticValidator.
        on_fail = None
        on_fail_attr_name = "on-fail-choice"
        if on_fail_attr_name in self.element.attrib:
            on_fail = self.element.attrib[on_fail_attr_name]
        return [ChoiceValidator(choices=list(self.children.keys()), on_fail=on_fail)]


@register_type("case")
class Case(NonScalarType):
    """Element tag: `<case>`"""

    def validate(self, key: str, value: Any, schema: Dict) -> Dict:
        child = list(self.children.values())[0]
        child.validate(key, value, schema)

        schema[key] = value
        return schema

    def set_children(self, element: ET._Element):
        assert len(element) == 1, "Case must have exactly one child."

        for child in element:
            child_data_type = registry[child.tag]
            self.children[child.attrib["name"]] = child_data_type.from_xml(child)
