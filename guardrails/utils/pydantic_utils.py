"""Utilities for working with Pydantic models.

Guardrails lets users specify

<pydantic
    model="Person"
    name="person"
    description="Information about a person."
    on-fail-pydantic="reask" / "refrain" / "raise"
/>
"""
import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import date, time
from typing import Any, Callable, Dict, Optional, Type, Union, get_args, get_origin

from griffe.dataclasses import Docstring
from griffe.docstrings.parsers import Parser, parse
from lxml.builder import E
from lxml.etree import Element
from pydantic import BaseModel, HttpUrl, validator
from pydantic.fields import ModelField

from guardrails.validators import Validator

griffe_docstrings_google_logger = logging.getLogger("griffe.docstrings.google")
griffe_agents_nodes_logger = logging.getLogger("griffe.agents.nodes")


def get_field_descriptions(model: "BaseModel") -> Dict[str, str]:
    """Get the descriptions of the fields in a Pydantic model using the
    docstring."""
    griffe_docstrings_google_logger.disabled = True
    griffe_agents_nodes_logger.disabled = True
    try:
        docstring = Docstring(model.__doc__, lineno=1)
    except AttributeError:
        return {}
    parsed = parse(docstring, Parser.google)
    griffe_docstrings_google_logger.disabled = False
    griffe_agents_nodes_logger.disabled = False

    # TODO: change parsed[1] to an isinstance check for the args section
    return {
        field.name: field.description.replace("\n", " ")
        for field in parsed[1].as_dict()["value"]
    }


PYDANTIC_SCHEMA_TYPE_MAP = {
    "string": "string",
    "number": "float",
    "integer": "integer",
    "boolean": "bool",
    "object": "object",
    "array": "list",
}

pydantic_validators = {}
pydantic_models = {}


# Create a class decorator to register all the validators in a BaseModel
def register_pydantic(cls: type):
    """
    Register a Pydantic BaseModel. This is a class decorator that can
    be used in the following way:

    ```
    @register_pydantic
    class MyModel(BaseModel):
        ...
    ```

    This decorator does the following:
        1. Add the model to the pydantic_models dictionary.
        2. Register all pre and post validators.
        3. Register all pre and post root validators.
    """
    # Register the model
    pydantic_models[cls.__name__] = cls

    # Create a dictionary to store all the validators
    pydantic_validators[cls] = {}
    # All all pre and post validators, for each field in the model
    for field in cls.__fields__.values():
        pydantic_validators[cls][field.name] = {}
        if field.pre_validators:
            for pre_validator in field.pre_validators:
                pydantic_validators[cls][field.name][
                    pre_validator.func_name.replace("_", "-")
                ] = pre_validator
        if field.post_validators:
            for post_validator in field.post_validators:
                pydantic_validators[cls][field.name][
                    post_validator.func_name.replace("_", "-")
                ] = post_validator

    pydantic_validators[cls]["__root__"] = {}
    # Add all pre and post root validators
    if cls.__pre_root_validators__:
        for _, pre_validator in cls.__pre_root_validators__:
            pydantic_validators[cls]["__root__"][
                pre_validator.__name__.replace("_", "-")
            ] = pre_validator

    if cls.__post_root_validators__:
        for _, post_validator in cls.__post_root_validators__:
            pydantic_validators[cls]["__root__"][
                post_validator.__name__.replace("_", "-")
            ] = post_validator
    return cls


def is_pydantic_base_model(type_annotation: Any) -> bool:
    """Check if a type_annotation is a Pydantic BaseModel."""
    try:
        if issubclass(type_annotation, BaseModel):
            return True
    except TypeError:
        False
    return False


def is_list(type_annotation: Any) -> bool:
    """Check if a type_annotation is a list."""

    type_annotation = prepare_type_annotation(type_annotation)

    if is_pydantic_base_model(type_annotation):
        return False
    if get_origin(type_annotation) == list:
        return True
    elif type_annotation == list:
        return True
    return False


def is_dict(type_annotation: Any) -> bool:
    """Check if a type_annotation is a dict."""

    type_annotation = prepare_type_annotation(type_annotation)

    if is_pydantic_base_model(type_annotation):
        return True
    if get_origin(type_annotation) == dict:
        return True
    elif type_annotation == dict:
        return True
    return False


def prepare_type_annotation(type_annotation: Any) -> Type:
    """Get the raw type annotation that can be used for downstream processing.

    This function does the following:
        1. If the type_annotation is a Pydantic field, get the annotation
        2. If the type_annotation is a Union, get the first non-None type

    Args:
        type_annotation (Any): The type annotation to prepare

    Returns:
        Type: The prepared type annotation
    """

    if isinstance(type_annotation, ModelField):
        type_annotation = type_annotation.annotation

    # Strip a Union type annotation to the first non-None type
    if get_origin(type_annotation) == Union:
        type_annotation = [
            t for t in get_args(type_annotation) if t != type(None)  # noqa E721
        ]
        assert (
            len(type_annotation) == 1
        ), "Union type must have exactly one non-None type"
        type_annotation = type_annotation[0]

    return type_annotation


def type_annotation_to_string(type_annotation: Any) -> str:
    """Map a type_annotation to the name of the corresponding field type.

    This function checks if the type_annotation is a list, dict, or a
    primitive type, and returns the corresponding type name, e.g.
    "list", "object", "bool", "date", etc.
    """

    # Get the type annotation from the type_annotation
    type_annotation = prepare_type_annotation(type_annotation)

    # Map the type annotation to the corresponding field type
    if is_list(type_annotation):
        return "list"
    elif is_dict(type_annotation):
        return "object"
    elif type_annotation == bool:
        return "bool"
    elif type_annotation == date:
        return "date"
    elif type_annotation == float:
        return "float"
    elif type_annotation == int:
        return "integer"
    elif type_annotation == str:
        return "string"
    elif type_annotation == time:
        return "time"
    elif type_annotation == HttpUrl:
        return "url"
    else:
        raise ValueError(f"Unsupported type: {type_annotation}")


def add_validators_to_xml_element(field_info: ModelField, element: Element) -> Element:
    """Extract validators from a pydantic ModelField and add to XML element.

    Args:
        field_info: The field info to extract validators from
        element: The XML element to add the validators to

    Returns:
        The XML element with the validators added
    """

    if (
        isinstance(field_info, ModelField)
        and "validators" in field_info.field_info.extra
    ):
        validators = field_info.field_info.extra["validators"]
        if isinstance(validators, str) or isinstance(validators, Validator):
            validators = [validators]

        format_prompt = []
        on_fails = {}
        for val in validators:
            validator_prompt = val
            if not isinstance(val, str):
                # `validator` is of type gd.Validator, use the to_xml_attrib method
                validator_prompt = val.to_xml_attrib()
                # Set the on-fail attribute based on the on_fail value
                on_fail = val.on_fail_descriptor
                on_fails[val.rail_alias] = on_fail
            format_prompt.append(validator_prompt)

        if len(format_prompt) > 0:
            format_prompt = "; ".join(format_prompt)
            element.set("format", format_prompt)
            for rail_alias, on_fail in on_fails.items():
                element.set("on-fail-" + rail_alias, on_fail)

    return element


def create_xml_element_for_field(
    field: Union[ModelField, Type, type],
    field_name: Optional[str] = None,
) -> Element:
    """Create an XML element corresponding to a field.

    Args:
        field_info: Field's type. This could be a Pydantic ModelField or a type.
        field_name: Field's name. For some fields (e.g. list), this is not required.

    Returns:
        The XML element corresponding to the field.
    """

    # Create the element based on the field type
    field_type = type_annotation_to_string(field)
    element = E(field_type)

    # Add name attribute
    if field_name:
        element.set("name", field_name)

    # Add validators
    element = add_validators_to_xml_element(field, element)

    # Add description attribute
    if isinstance(field, ModelField):
        if field.field_info.description is not None:
            element.set("description", field.field_info.description)

        # Add other attributes from the field_info
        for key, value in field.field_info.extra.items():
            if key not in ["validators", "description", "when"]:
                element.set(key, value)

    # Create XML elements for the field's children
    if field_type in ["list", "object"]:
        type_annotation = prepare_type_annotation(field)

        if is_list(type_annotation):
            inner_type = get_args(type_annotation)
            if len(inner_type) == 0:
                # If the list is empty, we cannot infer the type of the elements
                pass

            inner_type = inner_type[0]
            if is_pydantic_base_model(inner_type):
                object_element = create_xml_element_for_base_model(inner_type)
                element.append(object_element)
            else:
                inner_element = create_xml_element_for_field(inner_type)
                element.append(inner_element)

        elif is_dict(type_annotation):
            if is_pydantic_base_model(type_annotation):
                element = create_xml_element_for_base_model(type_annotation, element)
            else:
                dict_args = get_args(type_annotation)
                if len(dict_args) == 2:
                    key_type, val_type = dict_args
                    assert key_type == str, "Only string keys are supported for dicts"
                    inner_element = create_xml_element_for_field(val_type)
                    element.append(inner_element)
        else:
            raise ValueError(f"Unsupported type: {type_annotation}")

    return element


def create_xml_element_for_base_model(
    model: BaseModel, element: Optional[Element] = None
) -> Element:
    """Create an XML element for a Pydantic BaseModel.

    This function does the following:
        1. Iterates through fields of the model and creates XML elements for each field
        2. If a field is a Pydantic BaseModel, it creates a nested XML element
        3. If the BaseModel contains a field with a `when` attribute, it creates
           `Choice` and `Case` elements for the field.

    Args:
        model: The Pydantic BaseModel to create an XML element for
        element: The XML element to add the fields to. If None, a new XML element

    Returns:
        The XML element with the fields added
    """

    if element is None:
        element = E("object")

    # Extract pydantic validators from the model and add them as guardrails validators
    model_fields = add_pydantic_validators_as_guardrails_validators(model)

    # Identify fields with `when` attribute
    choice_elements = defaultdict(list)
    case_elements = set()
    for field_name, field in model_fields.items():
        if "when" in field.field_info.extra:
            choice_elements[field.field_info.extra["when"]].append((field_name, field))
            case_elements.add(field_name)

    # Add fields to the XML element, except for fields with `when` attribute
    for field_name, field in model_fields.items():
        if field_name in choice_elements or field_name in case_elements:
            continue
        field_element = create_xml_element_for_field(field, field_name)
        element.append(field_element)

    # Add `Choice` and `Case` elements for fields with `when` attribute
    for when, discriminator_fields in choice_elements.items():
        choice_element = E("choice", name=when)
        # TODO(shreya): DONT MERGE WTHOUT SOLVING THIS: How do you set this via SDK?
        choice_element.set("on-fail-choice", "exception")

        for field_name, field in discriminator_fields:
            case_element = E("case", name=field_name)
            field_element = create_xml_element_for_field(field, field_name)
            case_element.append(field_element)
            choice_element.append(case_element)

        element.append(choice_element)

    return element


def add_validator(
    *fields: str,
    pre: bool = False,
    each_item: bool = False,
    always: bool = False,
    check_fields: bool = True,
    whole: Optional[bool] = None,
    allow_reuse: bool = True,
    fn: Optional[Callable] = None,
) -> Callable:
    return validator(
        *fields,
        pre=pre,
        each_item=each_item,
        always=always,
        check_fields=check_fields,
        whole=whole,
        allow_reuse=allow_reuse,
    )(fn)


def convert_pydantic_validator_to_guardrails_validator(
    model: BaseModel, fn: Callable
) -> Validator:
    """Convert a Pydantic validator to a Guardrails validator.

    Pydantic validators can be defined in three ways:
        1. A method defined in the BaseModel using a `validator` decorator.
        2. Using the `add_validator` function with a Guardrails validator class.
        3. Using the `add_validator` function with a custom function.
    This method converts all three types of validators to a Guardrails validator.

    Args:
        model: The Pydantic BaseModel that the validator is defined in.
        fn: The Pydantic validator function. This is the raw cython function generated
            by calling `BaseModelName.__fields__[field_name].post_validators[idx]`.

    Returns:
        A Guardrails validator
    """

    fn_name = fn.__name__
    callable_fn = fn.__wrapped__

    if hasattr(model, fn_name):
        # # Case 1: fn is a method defined in the BaseModel
        # # Wrap the method in a Guardrails PydanticFieldValidator class
        # field_validator = partial(callable_fn, model)
        # return PydanticFieldValidator(field_validator=field_validator)
        warnings.warn(
            f"Validator {fn_name} is defined as a method in the BaseModel. "
            "This is not supported by Guardrails. "
            "Please define the validator using the `add_validator` function."
        )
        return fn_name

    if issubclass(type(callable_fn), Validator):
        # Case 2: fn is a Guardrails validator
        return callable_fn
    else:
        # # Case 3: fn is a custom function
        # return PydanticFieldValidator(field_validator=callable_fn)
        warnings.warn(
            f"Validator {fn_name} is defined as a custom function. "
            "This is not supported by Guardrails. "
            "Please define the validator using the `add_validator` function."
        )
        return fn_name


def add_pydantic_validators_as_guardrails_validators(
    model: BaseModel,
) -> Dict[str, ModelField]:
    """Extract all validators for a pydantic BaseModel.

    This function converts each Pydantic validator to a GuardRails validator and adds
    it to the corresponding field in the model. The resulting dictionary maps field
    names to ModelField objects.

    Args:
        model: A pydantic BaseModel.

    Returns:
        A dictionary mapping field names to ModelField objects.
    """

    def process_validators(vals, fld):
        if not vals:
            return

        for val in vals:
            gd_validator = convert_pydantic_validator_to_guardrails_validator(
                model, val
            )
            if "validators" not in fld.field_info.extra:
                fld.field_info.extra["validators"] = []
            fld.field_info.extra["validators"].append(gd_validator)

    model_fields = {}
    for field_name, field in model.__fields__.items():
        field_copy = deepcopy(field)
        process_validators(field.pre_validators, field_copy)
        process_validators(field.post_validators, field_copy)
        model_fields[field_name] = field_copy

    # TODO(shreya): Before merging handle root validators
    return model_fields


def convert_pydantic_model_to_openai_fn(model: BaseModel) -> Dict:
    """Convert a Pydantic BaseModel to an OpenAI function.

    Args:
        model: The Pydantic BaseModel to convert.

    Returns:
        OpenAI function paramters.
    """

    # Convert Pydantic model to JSON schema
    json_schema = model.schema()

    # Create OpenAI function parameters
    fn_params = {
        "name": json_schema["title"],
        "parameters": json_schema,
    }
    if "description" in json_schema and json_schema["description"] is not None:
        fn_params["description"] = json_schema["description"]

    return fn_params
