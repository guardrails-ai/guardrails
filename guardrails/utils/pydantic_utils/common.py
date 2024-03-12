from typing import List, Type, Union

from pydantic import BaseModel


def _create_bare_model(
    model: Union[Type[BaseModel], Type[List[Type[BaseModel]]]]
) -> Type[BaseModel]:
    class BareModel(BaseModel):
        __annotations__ = getattr(model, "__annotations__", {})

    return BareModel
