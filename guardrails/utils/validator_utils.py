# ruff: noqa
"""This module contains the constants and utils used by the validator.py."""

from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

from guardrails_api_client import ValidatorReference

from guardrails.types.validator import PydanticValidatorSpec
from guardrails.utils.regex_utils import split_on, ESCAPED_OR_QUOTED
from guardrails.utils.safe_get import safe_get
from guardrails.validator_base import Validator, OnFailAction, get_validator_class
from guardrails.types import UseManyValidatorTuple, PydanticValidatorTuple
from guardrails.constants import hub
from guardrails.logger import logger

PROVENANCE_V1_PROMPT = """Instruction:
As an Attribution Validator, you task is to verify whether the following contexts support the claim:

Claim:
{}

Contexts:
{}

Just respond with a "Yes" or "No" to indicate whether the given contexts support the claim.
Response:"""


def parse_rail_arguments(arg_tokens: List[str]) -> List[Any]:
    """Legacy parsing logic for the Validator aruguments specified in a RAIL
    spec.

    Originally from ValidatorsAttr.
    """
    validator_args = []
    for t in arg_tokens:
        # If the token is enclosed in curly braces, it is a Python expression.
        t = t.strip()
        if t[0] == "{" and t[-1] == "}":
            t = t[1:-1]
            try:
                # FIXME: This is incredibly insecure!
                # We need a better way of escaping and parsing arguments from RAIL.
                # Option 1: Each Validator could accept a spread of argument strings
                #   and be responsible for parsing them to the correct types.
                # Option 2: We use something like the Validator Manifest that describes the arguments
                #   to parse the values from the string WITHOUT an eval.
                t = eval(t)
            except (ValueError, SyntaxError, NameError) as e:
                raise ValueError(
                    f"Python expression `{t}` is not valid, "
                    f"and raised an error: {e}."
                )
        validator_args.append(t)
    return validator_args


def parse_rail_validator(
    validator_spec: str, *, on_fail: Optional[str] = None
) -> Optional[Validator]:
    validator_id = None
    validator_args = []
    if ":" in validator_spec:
        is_hub_validator = validator_spec.startswith(hub)
        max_splits = 2 if is_hub_validator else 1
        parts = validator_spec.split(":", max_splits)
        validator_id = (
            ":".join([parts[0], parts[1].strip()])
            if is_hub_validator
            else parts[0].strip()
        )
        arg_tokens = []
        if len(parts) > max_splits:
            arg_tokens = [
                arg.strip()
                for arg in split_on(
                    parts[max_splits], r"\s", exceptions=ESCAPED_OR_QUOTED
                )
            ]
        validator_args = parse_rail_arguments(arg_tokens)
    else:
        validator_id = validator_spec
    validator_cls = get_validator_class(validator_id)
    if validator_cls:
        return validator_cls(*validator_args, on_fail=OnFailAction.get(on_fail))
    else:
        logger.warning(
            f"Validator with id {validator_id} was not found in the registry!  Ignoring..."
        )


def parse_use_many_validator(
    validator_cls: Type[Validator], use_tuple: UseManyValidatorTuple
) -> Optional[Validator]:
    args = safe_get(use_tuple, 1, [])
    kwargs = {}
    if isinstance(args, Dict):
        kwargs = args
        args = []
    elif not isinstance(args, List):
        args = [args]
    kwargs = safe_get(use_tuple, 2, kwargs)
    if validator_cls:
        validator_inst = validator_cls(*args, **kwargs)
        return validator_inst


def parse_pydantic_validator(
    validator_cls: Union[Validator, str], pydantic_tuple: PydanticValidatorTuple
) -> Optional[Validator]:
    if isinstance(validator_cls, Validator):
        validator_instance = validator_cls
        on_fail = safe_get(pydantic_tuple, 1, OnFailAction.NOOP)
        validator_instance.on_fail_descriptor = on_fail
        return validator_instance
    elif isinstance(validator_cls, str):
        validator_string = validator_cls
        on_fail = safe_get(pydantic_tuple, 1, OnFailAction.NOOP)
        return parse_rail_validator(validator_string, on_fail=on_fail)


def get_validator(
    validator: Union[
        Validator,
        Type[Validator],
        UseManyValidatorTuple,
        PydanticValidatorTuple,
        str,  # RAIL
    ],
    *args,
    **kwargs,
) -> Validator:
    invalid_error = ValueError(f"Invalid arguments! {validator}")
    # Guard.use syntax
    if isinstance(validator, Validator):
        return validator
    # Guard.use syntax
    elif isinstance(validator, Type) and issubclass(validator, Validator):
        return validator(*args, **kwargs)
    # Guard.useMany or Guard.from_pydantic syntax
    elif isinstance(validator, Tuple):
        first_arg = safe_get(validator, 0)
        # useMany Tuple Syntax
        if isinstance(first_arg, type) and issubclass(first_arg, Validator):
            v = parse_use_many_validator(first_arg, validator)  # type: ignore
            if v:
                return v
        # Pydantic Tuple Syntax
        else:
            v = parse_pydantic_validator(first_arg, validator)  # type: ignore
            if v:
                return v
        raise invalid_error
    # Guard.from_rail or Guard.from_rail_string syntax
    elif isinstance(validator, str):
        v = parse_rail_validator(validator)
        if v:
            return v
    raise invalid_error


def safe_get_validator(v: Union[str, PydanticValidatorSpec]) -> Union[Validator, None]:
    try:
        validator = get_validator(v)
        return validator
    except ValueError as e:
        logger.warning(e)
        return None


def verify_metadata_requirements(
    metadata: dict, validators: List[Validator]
) -> List[str]:
    missing_keys = set()
    for validator in validators:
        for requirement in validator.required_metadata_keys:
            if requirement not in metadata:
                missing_keys.add(requirement)
    missing_keys = list(missing_keys)
    missing_keys.sort()
    return missing_keys


def parse_validator_reference(ref: ValidatorReference) -> Optional[Validator]:
    validator_cls = get_validator_class(ref.id)
    if validator_cls:
        args = ref.args or []
        kwargs = ref.kwargs or {}
        validator = validator_cls(
            *args, on_fail=OnFailAction.get(ref.on_fail), **kwargs
        )
        return validator
