# flake8: noqa
"""This module contains the constants and utils used by the validator.py."""


from typing import Any, Dict, List, Optional, Tuple, Type, Union

from guardrails.utils.safe_get import safe_get
from guardrails.validator_base import Validator

PROVENANCE_V1_PROMPT = """Instruction:
As an Attribution Validator, you task is to verify whether the following contexts support the claim:

Claim:
{}

Contexts:
{}

Just respond with a "Yes" or "No" to indicate whether the given contexts support the claim.
Response:"""


def get_validator(
    validator: Union[
        Validator,
        Type[Validator],
        Tuple[
            Type[Validator],
            Optional[Union[List[Any], Dict[str, Any]]],
            Optional[Dict[str, Any]],
        ],
    ],
    *args,
    **kwargs,
) -> Validator:
    invalid_error = ValueError(f"Invalid arguments! {validator}")
    if isinstance(validator, Validator):
        return validator
    elif isinstance(validator, Type):
        return validator(*args, **kwargs)
    elif isinstance(validator, Tuple):
        validator_cls = safe_get(validator, 0)
        args = safe_get(validator, 1, [])
        kwargs = {}
        if isinstance(args, Dict):
            kwargs = args
            args = []
        kwargs = safe_get(validator, 2, kwargs)
        if validator_cls:
            validator_inst = validator_cls(*args, **kwargs)
            return validator_inst
        raise invalid_error
    else:
        raise invalid_error
