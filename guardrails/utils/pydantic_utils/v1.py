"""Utilities for working with Pydantic models."""
import typing
import warnings
from copy import deepcopy
from datetime import date, time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, validator
from pydantic.fields import ModelField

from guardrails.datatypes import Boolean as BooleanDataType
from guardrails.datatypes import Case as CaseDataType
from guardrails.datatypes import Choice
from guardrails.datatypes import Choice as ChoiceDataType
from guardrails.datatypes import DataType
from guardrails.datatypes import Date as DateDataType
from guardrails.datatypes import Enum as EnumDataType
from guardrails.datatypes import Float as FloatDataType
from guardrails.datatypes import Integer as IntegerDataType
from guardrails.datatypes import List as ListDataType
from guardrails.datatypes import Object as ObjectDataType
from guardrails.datatypes import String as StringDataType
from guardrails.datatypes import Time as TimeDataType
from guardrails.validator_base import Validator
from guardrails.validatorsattr import ValidatorsAttr


class ArbitraryModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


def is_pydantic_base_model(type_annotation: Any) -> bool:
    """Check if a type_annotation is a Pydantic BaseModel."""
    try:
        if issubclass(type_annotation, BaseModel):
            return True
    except TypeError:
        pass
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


def is_enum(type_annotation: Any) -> bool:
    """Check if a type_annotation is an enum."""

    type_annotation = prepare_type_annotation(type_annotation)

    try:
        if issubclass(type_annotation, Enum):
            return True
    except TypeError:
        pass
    return False


def prepare_type_annotation(type_annotation: Union[ModelField, Type]) -> Type:
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
        non_none_type_annotation = [
            t for t in get_args(type_annotation) if t != type(None)  # noqa E721
        ]
        if len(non_none_type_annotation) == 1:
            return non_none_type_annotation[0]
        return type_annotation

    return type_annotation


def add_validator(
    *fields: str,
    pre: bool = False,
    each_item: bool = False,
    always: bool = False,
    check_fields: bool = True,
    whole: Optional[bool] = None,
    allow_reuse: bool = True,
    fn: Callable,
) -> classmethod:
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
    model: Type[BaseModel], fn: Callable
) -> Union[str, Validator]:
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
    model: Type[BaseModel],
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
            fld.field_info.extra["validators"].append((gd_validator, "reask"))

    model_fields = {}
    for field_name, field in model.__fields__.items():
        field_copy = deepcopy(field)

        if "validators" in field.field_info.extra and not isinstance(
            field.field_info.extra["validators"], list
        ):
            field_copy.field_info.extra["validators"] = [
                field_copy.field_info.extra["validators"]
            ]

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

    # Create a bare model with no extra fields
    class BareModel(BaseModel):
        __annotations__ = model.__annotations__

    # Convert Pydantic model to JSON schema
    json_schema = BareModel.schema()

    # Create OpenAI function parameters
    fn_params = {
        "name": json_schema["title"],
        "parameters": json_schema,
    }
    if "description" in json_schema and json_schema["description"] is not None:
        fn_params["description"] = json_schema["description"]

    return fn_params


def field_to_datatype(field: Union[ModelField, Type]) -> Type[DataType]:
    """Map a type_annotation to the name of the corresponding field type.

    This function checks if the type_annotation is a list, dict, or a
    primitive type, and returns the corresponding type name, e.g.
    "list", "object", "bool", "date", etc.
    """

    # FIXME: inaccessible datatypes:
    #   - Email
    #   - SQLCode
    #   - Percentage

    # Get the type annotation from the type_annotation
    type_annotation = prepare_type_annotation(field)

    # Map the type annotation to the corresponding field type
    if is_list(type_annotation):
        return ListDataType
    elif is_dict(type_annotation):
        return ObjectDataType
    elif is_enum(type_annotation):
        return EnumDataType
    elif type_annotation == bool:
        return BooleanDataType
    elif type_annotation == date:
        return DateDataType
    elif type_annotation == float:
        return FloatDataType
    elif type_annotation == int:
        return IntegerDataType
    elif type_annotation == str or typing.get_origin(type_annotation) == typing.Literal:
        return StringDataType
    elif type_annotation == time:
        return TimeDataType
    elif typing.get_origin(type_annotation) == Union:
        return ChoiceDataType
    else:
        raise ValueError(f"Unsupported type: {type_annotation}")


T = typing.TypeVar("T", bound=DataType)


def convert_pydantic_model_to_datatype(
    model_field: Union[ModelField, Type[BaseModel]],
    datatype: Type[T] = ObjectDataType,
    excluded_fields: Optional[typing.List[str]] = None,
    name: Optional[str] = None,
    strict: bool = False,
) -> T:
    """Create an Object from a Pydantic model."""
    if excluded_fields is None:
        excluded_fields = []

    if isinstance(model_field, ModelField):
        model = model_field.type_
    else:
        model = model_field

    model_fields = add_pydantic_validators_as_guardrails_validators(model)

    children = {}
    for field_name, field in model_fields.items():
        if field_name in excluded_fields:
            continue
        type_annotation = prepare_type_annotation(field)
        target_datatype = field_to_datatype(field)
        if target_datatype == ListDataType:
            inner_type = get_args(type_annotation)
            if len(inner_type) == 0:
                # If the list is empty, we cannot infer the type of the elements
                children[field_name] = pydantic_field_to_datatype(
                    ListDataType,
                    field,
                    strict=strict,
                )
                continue
            inner_type = inner_type[0]
            if is_pydantic_base_model(inner_type):
                child = convert_pydantic_model_to_datatype(inner_type)
            else:
                inner_target_datatype = field_to_datatype(inner_type)
                child = construct_datatype(
                    inner_target_datatype,
                    strict=strict,
                )
            children[field_name] = pydantic_field_to_datatype(
                ListDataType,
                field,
                children={"item": child},
                strict=strict,
            )
        elif target_datatype == ChoiceDataType:
            discriminator = field.discriminator_key or "discriminator"
            choice_children = {}
            for case in typing.get_args(field.type_):
                case_discriminator_type = case.__fields__[discriminator].type_
                assert typing.get_origin(case_discriminator_type) is typing.Literal
                assert len(typing.get_args(case_discriminator_type)) == 1
                discriminator_value = typing.get_args(case_discriminator_type)[0]
                choice_children[
                    discriminator_value
                ] = convert_pydantic_model_to_datatype(
                    case,
                    datatype=CaseDataType,
                    name=discriminator_value,
                    strict=strict,
                    excluded_fields=[discriminator],
                )
            children[field_name] = pydantic_field_to_datatype(
                Choice,
                field,
                children=choice_children,
                strict=strict,
                discriminator_key=discriminator,
            )
        elif target_datatype == EnumDataType:
            assert issubclass(type_annotation, Enum)
            valid_choices = [choice.value for choice in type_annotation]
            children[field_name] = pydantic_field_to_datatype(
                EnumDataType, field, strict=strict, enum_values=valid_choices
            )
        elif isinstance(field.type_, type) and issubclass(field.type_, BaseModel):
            children[field_name] = convert_pydantic_model_to_datatype(
                field, datatype=target_datatype, strict=strict
            )
        else:
            children[field_name] = pydantic_field_to_datatype(
                target_datatype,
                field,
                strict=strict,
            )

    if isinstance(model_field, ModelField):
        return pydantic_field_to_datatype(
            datatype,
            model_field,
            children=children,
            strict=strict,
        )
    else:
        return construct_datatype(
            datatype,
            children=children,
            name=name,
        )


def pydantic_field_to_datatype(
    datatype: Type[T],
    field: ModelField,
    children: Optional[Dict[str, "DataType"]] = None,
    strict: bool = False,
    **kwargs,
) -> T:
    if children is None:
        children = {}

    validators = field.field_info.extra.get("validators", [])

    is_optional = field.required is False

    name = field.name
    description = field.field_info.description

    return construct_datatype(
        datatype,
        children,
        validators,
        is_optional,
        name,
        description,
        strict=strict,
        **kwargs,
    )


def construct_datatype(
    datatype: Type[T],
    children: Optional[Dict[str, Any]] = None,
    validators: Optional[typing.List[Validator]] = None,
    optional: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> T:
    if children is None:
        children = {}
    if validators is None:
        validators = []

    validators_attr = ValidatorsAttr.from_validators(validators, datatype.tag, strict)
    return datatype(children, validators_attr, optional, name, description, **kwargs)
