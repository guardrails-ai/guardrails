import asyncio
import os
from typing import Any, Iterable, Optional, Tuple
import warnings

from guardrails.actions.filter import apply_filters
from guardrails.actions.refrain import apply_refrain
from guardrails.classes.history import Iteration
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.validation.validation_result import (
    StreamValidationResult,
)
from guardrails.types import ValidatorMap
from guardrails.telemetry.legacy_validator_tracing import trace_validation_result
from guardrails.validator_service.async_validator_service import AsyncValidatorService
from guardrails.validator_service.sequential_validator_service import (
    SequentialValidatorService,
)


try:
    import uvloop  # type: ignore
except ImportError:
    uvloop = None


def should_run_sync():
    process_count = os.environ.get("GUARDRAILS_PROCESS_COUNT")
    if process_count is not None:
        warnings.warn(
            "GUARDRAILS_PROCESS_COUNT is deprecated"
            " and will be removed in a future release."
            " To force synchronous validation, please use GUARDRAILS_RUN_SYNC instead.",
            DeprecationWarning,
        )
        process_count = int(process_count)
    run_sync = os.environ.get("GUARDRAILS_RUN_SYNC", "false")
    bool_values = ["true", "false"]
    if run_sync.lower() not in bool_values:
        warnings.warn(
            f"GUARDRAILS_RUN_SYNC must be one of {bool_values}!"
            f" Defaulting to 'false'."
        )
    return process_count == 1 or run_sync.lower() == "true"


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        raise RuntimeError("An event loop is already running.")

    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    return asyncio.get_event_loop()


def validate(
    value: Any,
    metadata: dict,
    validator_map: ValidatorMap,
    iteration: Iteration,
    disable_tracer: Optional[bool] = True,
    path: Optional[str] = None,
    **kwargs,
):
    if path is None:
        path = "$"

    loop = None
    if should_run_sync():
        validator_service = SequentialValidatorService(disable_tracer)
    else:
        try:
            loop = get_loop()
            validator_service = AsyncValidatorService(disable_tracer)
        except RuntimeError:
            warnings.warn(
                "Could not obtain an event loop."
                " Falling back to synchronous validation."
            )
            validator_service = SequentialValidatorService(disable_tracer)

    return validator_service.validate(
        value,
        metadata,
        validator_map,
        iteration,
        path,
        path,
        loop=loop,  # type: ignore It exists when we need it to.
        **kwargs,
    )


def validate_stream(
    value_stream: Iterable[Tuple[Any, bool]],
    metadata: dict,
    validator_map: ValidatorMap,
    iteration: Iteration,
    disable_tracer: Optional[bool] = True,
    path: Optional[str] = None,
    **kwargs,
) -> Iterable[StreamValidationResult]:
    if path is None:
        path = "$"
    sequential_validator_service = SequentialValidatorService(disable_tracer)
    gen = sequential_validator_service.validate_stream(
        value_stream, metadata, validator_map, iteration, path, path, **kwargs
    )
    return gen


async def async_validate(
    value: Any,
    metadata: dict,
    validator_map: ValidatorMap,
    iteration: Iteration,
    disable_tracer: Optional[bool] = True,
    path: Optional[str] = None,
    stream: Optional[bool] = False,
    **kwargs,
) -> Tuple[Any, dict]:
    if path is None:
        path = "$"
    validator_service = AsyncValidatorService(disable_tracer)
    return await validator_service.async_validate(
        value, metadata, validator_map, iteration, path, path, stream, **kwargs
    )


def post_process_validation(
    validation_response: Any,
    attempt_number: int,
    iteration: Iteration,
    output_type: OutputTypes,
) -> Any:
    validated_response = apply_refrain(validation_response, output_type)

    # Remove all keys that have `Filter` values.
    validated_response = apply_filters(validated_response)

    trace_validation_result(
        validation_logs=iteration.validator_logs, attempt_number=attempt_number
    )

    return validated_response
