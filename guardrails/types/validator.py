from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from guardrails.validator_base import Validator


PydanticValidatorTuple = Tuple[Union[Validator, str, Callable], str]
PydanticValidatorSpec = Union[Validator, PydanticValidatorTuple]

UseValidatorSpec = Union[Validator, Type[Validator]]

UseManyValidatorTuple = Tuple[
    Type[Validator],
    Optional[Union[List[Any], Dict[str, Any]]],
    Optional[Dict[str, Any]],
]
UseManyValidatorSpec = Union[Validator, UseManyValidatorTuple]

ValidatorMap = Dict[str, List[Validator]]
