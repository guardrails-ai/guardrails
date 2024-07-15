from typing import Any, Dict, List, Type, Union

from pydantic import BaseModel


ModelOrListOfModels = Union[Type[BaseModel], Type[List[Type[BaseModel]]]]

ModelOrListOrDict = Union[
    Type[BaseModel], Type[List[Type[BaseModel]]], Type[Dict[str, Type[BaseModel]]]
]

ModelOrModelUnion = Union[Type[BaseModel], Union[Type[BaseModel], Any]]
