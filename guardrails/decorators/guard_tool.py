"""Decorator for validating agent tool inputs and outputs.

Addresses OWASP LLM08 (Excessive Agency) and LLM07 (Insecure Plugin Design)
by validating tool parameters before execution and outputs before returning
to the LLM context.
"""

import asyncio
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError as PydanticValidationError

from guardrails.errors import ValidationError
from guardrails.hub_telemetry.hub_tracing import trace, async_trace
from guardrails.types.on_fail import OnFailAction
from guardrails.validator_base import Validator
from guardrails_ai.types import FailResult, PassResult


def _run_validators(
    value: Any,
    validators: List[Validator],
    metadata: Dict[str, Any],
    on_fail: OnFailAction,
    context: str,
) -> Any:
    """Run validators on a value and handle failures."""
    for validator in validators:
        result = validator.validate(value, metadata)

        if isinstance(result, FailResult):
            if on_fail == OnFailAction.EXCEPTION:
                raise ValidationError(
                    f"Tool {context} validation failed: {result.error_message}"
                )
            elif on_fail == OnFailAction.FIX and result.fix_value is not None:
                value = result.fix_value
            elif on_fail == OnFailAction.NOOP:
                pass
            elif on_fail == OnFailAction.REFRAIN:
                return None
            else:
                raise ValidationError(
                    f"Tool {context} validation failed: {result.error_message}"
                )
        elif isinstance(result, PassResult):
            if (
                validator.override_value_on_pass
                and result.value_override is not result.ValueOverrideSentinel
            ):
                value = result.value_override

    return value


async def _run_validators_async(
    value: Any,
    validators: List[Validator],
    metadata: Dict[str, Any],
    on_fail: OnFailAction,
    context: str,
) -> Any:
    """Run validators asynchronously on a value."""
    for validator in validators:
        result = await validator.async_validate(value, metadata)

        if isinstance(result, FailResult):
            if on_fail == OnFailAction.EXCEPTION:
                raise ValidationError(
                    f"Tool {context} validation failed: {result.error_message}"
                )
            elif on_fail == OnFailAction.FIX and result.fix_value is not None:
                value = result.fix_value
            elif on_fail == OnFailAction.NOOP:
                pass
            elif on_fail == OnFailAction.REFRAIN:
                return None
            else:
                raise ValidationError(
                    f"Tool {context} validation failed: {result.error_message}"
                )
        elif isinstance(result, PassResult):
            if (
                validator.override_value_on_pass
                and result.value_override is not result.ValueOverrideSentinel
            ):
                value = result.value_override

    return value


def _validate_input_schema(
    args: tuple,
    kwargs: Dict[str, Any],
    func: Callable,
    input_schema: Type[BaseModel],
) -> Dict[str, Any]:
    """Validate function arguments against a Pydantic schema."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    try:
        validated = input_schema.model_validate(bound.arguments)
        return validated.model_dump()
    except PydanticValidationError as e:
        raise ValidationError(f"Tool input schema validation failed: {e}")


def guard_tool(
    *,
    input_validators: Optional[List[Validator]] = None,
    output_validators: Optional[List[Validator]] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    on_fail: Union[OnFailAction, str] = OnFailAction.EXCEPTION,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator to validate agent tool inputs and outputs.

    Protects the tool execution boundary in autonomous agent loops by
    validating LLM-generated arguments before execution and tool outputs
    before returning to the LLM context.

    Args:
        input_validators: Validators to run on tool inputs.
        output_validators: Validators to run on tool outputs.
        input_schema: Pydantic model for input validation.
        output_schema: Pydantic model for output validation.
        on_fail: Action on validation failure.
            Defaults to OnFailAction.EXCEPTION.
        metadata: Additional metadata for validators.

    Returns:
        Decorated function with input/output validation.

    Example:
        >>> from guardrails.hub import RegexMatch
        >>> @guard_tool(
        ...     input_validators=[RegexMatch(regex=r'^[a-z]+$')],
        ...     output_validators=[RegexMatch(regex=r'^[A-Z]+$')],
        ... )
        ... def my_tool(query: str) -> str:
        ...     return query.upper()
    """
    if isinstance(on_fail, str):
        on_fail = OnFailAction.get(on_fail, OnFailAction.EXCEPTION)

    input_validators = input_validators or []
    output_validators = output_validators or []
    meta = metadata or {}

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            @async_trace(name="/guard_tool", origin="guard_tool")
            async def async_wrapper(*args, **kwargs) -> Any:
                validated_kwargs = kwargs

                if input_schema:
                    validated_kwargs = _validate_input_schema(
                        args, kwargs, func, input_schema
                    )
                    args = ()

                if input_validators:
                    for key, val in list(validated_kwargs.items()):
                        validated_kwargs[key] = await _run_validators_async(
                            val, input_validators, meta, on_fail, f"input.{key}"
                        )
                        if validated_kwargs[key] is None and on_fail == OnFailAction.REFRAIN:
                            return None

                result = await func(*args, **validated_kwargs)

                if output_schema:
                    try:
                        validated = output_schema.model_validate(
                            result if isinstance(result, dict) else {"value": result}
                        )
                        result = validated.model_dump().get("value", validated.model_dump())
                    except PydanticValidationError as e:
                        raise ValidationError(f"Tool output schema validation failed: {e}")

                if output_validators:
                    result = await _run_validators_async(
                        result, output_validators, meta, on_fail, "output"
                    )

                return result

            return async_wrapper

        else:

            @functools.wraps(func)
            @trace(name="/guard_tool", origin="guard_tool")
            def sync_wrapper(*args, **kwargs) -> Any:
                validated_kwargs = kwargs

                if input_schema:
                    validated_kwargs = _validate_input_schema(
                        args, kwargs, func, input_schema
                    )
                    args = ()

                if input_validators:
                    for key, val in list(validated_kwargs.items()):
                        validated_kwargs[key] = _run_validators(
                            val, input_validators, meta, on_fail, f"input.{key}"
                        )
                        if validated_kwargs[key] is None and on_fail == OnFailAction.REFRAIN:
                            return None

                result = func(*args, **validated_kwargs)

                if output_schema:
                    try:
                        validated = output_schema.model_validate(
                            result if isinstance(result, dict) else {"value": result}
                        )
                        result = validated.model_dump().get("value", validated.model_dump())
                    except PydanticValidationError as e:
                        raise ValidationError(f"Tool output schema validation failed: {e}")

                if output_validators:
                    result = _run_validators(
                        result, output_validators, meta, on_fail, "output"
                    )

                return result

            return sync_wrapper

    return decorator
