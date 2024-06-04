"""Utilities for working with Pydantic models."""

from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from guardrails.utils.safe_get import safe_get


def create_bare_model(model: Type[BaseModel]):
    class BareModel(BaseModel):
        __annotations__ = getattr(model, "__annotations__", {})

    return BareModel


def reduce_to_annotations(type_annotation: Any) -> Type[Any]:
    if (
        type_annotation
        and isinstance(type_annotation, type)
        and issubclass(type_annotation, BaseModel)
    ):
        return create_bare_model(type_annotation)
    return type_annotation


def find_models_in_type(type_annotation: Any) -> Type[Any]:
    type_origin = get_origin(type_annotation)
    inner_types = get_args(type_annotation)
    if type_origin == Union:
        data_types = tuple([find_models_in_type(t) for t in inner_types])
        return Type[Union[data_types]]  # type: ignore
    elif type_origin == list:
        if len(inner_types) > 1:
            raise ValueError("List data type must have exactly one child.")
        # No List[List] support; we've already declared that in our types
        item_type = safe_get(inner_types, 0)
        return Type[List[find_models_in_type(item_type)]]
    elif type_origin == dict:
        # First arg is key which must be primitive
        # Second arg is potentially a model
        key_value_type = safe_get(inner_types, 1)
        value_value_type = safe_get(inner_types, 1)
        return Type[Dict[key_value_type, find_models_in_type(value_value_type)]]
    else:
        return reduce_to_annotations(type_annotation)


def schema_to_bare_model(model: Type[BaseModel]) -> Type[BaseModel]:
    copy = deepcopy(model)
    for field_key in copy.__fields__:
        field = copy.__fields__.get(field_key)
        if field:
            extras = field.field_info.extra
            if "validators" in extras:
                extras["format"] = list(
                    v.to_prompt()
                    for v in extras.pop("validators", [])
                    if hasattr(v, "to_prompt")
                )

            field.field_info.extra = extras

            value_type = find_models_in_type(field.annotation)
            field.annotation = value_type
            copy.__fields__[field_key] = field

    # root_model = reduce_to_annotations(model)

    # for key in root_model.__annotations__:
    #     value = root_model.__annotations__.get(key)
    #     print("value.field_info: ", value.field_info)
    #     value_type = find_models_in_type(value)
    #     root_model.__annotations__[key] = value_type

    return copy


def convert_pydantic_model_to_openai_fn(
    model: Union[Type[BaseModel], Type[List[Type[BaseModel]]]],
) -> Dict:
    """Convert a Pydantic BaseModel to an OpenAI function.

    Args:
        model: The Pydantic BaseModel to convert.

    Returns:
        OpenAI function paramters.
    """
    return {}

    # schema_model = model

    # type_origin = get_origin(model)
    # if type_origin == list:
    #     item_types = get_args(model)
    #     if len(item_types) > 1:
    #         raise ValueError("List data type must have exactly one child.")
    #     # No List[List] support; we've already declared that in our types
    #     schema_model = safe_get(item_types, 0)

    # # Create a bare model with no extra fields
    # bare_model = schema_to_bare_model(schema_model)

    # # Convert Pydantic model to JSON schema
    # json_schema = bare_model.schema()
    # json_schema["title"] = schema_model.__name__

    # if type_origin == list:
    #     json_schema = {
    #         "title": f"Array<{json_schema.get('title')}>",
    #         "type": "array",
    #         "items": json_schema,
    #     }

    # # Create OpenAI function parameters
    # fn_params = {
    #     "name": json_schema["title"],
    #     "parameters": json_schema,
    # }
    # if "description" in json_schema and json_schema["description"] is not None:
    #     fn_params["description"] = json_schema["description"]

    # return fn_params
