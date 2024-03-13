from typing import Any, Dict, List, Type, Union, get_args, get_origin

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
    root_model = reduce_to_annotations(model)

    for key in root_model.__annotations__:
        value = root_model.__annotations__.get(key)
        value_type = find_models_in_type(value)
        root_model.__annotations__[key] = value_type

    return root_model
