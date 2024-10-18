from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
)

from opentelemetry import context, trace
from opentelemetry.trace import StatusCode, Tracer, Span


from guardrails.settings import settings
from guardrails.classes.validation.validation_result import ValidationResult
from guardrails.telemetry.common import get_tracer, add_user_attributes, serialize
from guardrails.telemetry.open_inference import trace_operation
from guardrails.utils.casting_utils import to_string
from guardrails.utils.safe_get import safe_get
from guardrails.version import GUARDRAILS_VERSION


def add_validator_attributes(
    *args,
    validator_span: Span,
    validator_name: str,
    obj_id: int,
    on_fail_descriptor: Optional[str] = None,
    result: Optional[ValidationResult] = None,
    init_kwargs: Dict[str, Any] = {},
    validation_session_id: str,
    **kwargs,
):
    value_arg = serialize(safe_get(args, 0)) or ""
    metadata_arg = serialize(safe_get(args, 1, {})) or "{}"

    # Legacy Span Attributes
    validator_span.set_attribute("on_fail_descriptor", on_fail_descriptor or "noop")
    validator_span.set_attribute(
        "args",
        to_string({k: to_string(v) for k, v in init_kwargs.items()}) or "{}",
    )
    validator_span.set_attribute("instance_id", serialize(obj_id) or "")
    validator_span.set_attribute("input", value_arg)

    # New Span Attributes
    validator_span.set_attribute("type", "guardrails/guard/step/validator")
    validator_span.set_attribute("validation_session_id", validation_session_id)

    ### Validator.__init__ ###
    validator_span.set_attribute("validator.name", validator_name or "unknown")
    validator_span.set_attribute("validator.on_fail", on_fail_descriptor or "noop")
    validator_span.set_attribute("validator.instance_id", serialize(obj_id) or "")
    for k, v in init_kwargs.items():
        if v is not None:
            validator_span.set_attribute(f"validator.init.{k}", serialize(v) or "")

    ### Validator.validate ###
    validator_span.set_attribute("validator.validate.input.value", value_arg)
    validator_span.set_attribute("validator.validate.input.metadata", metadata_arg)
    for k, v in kwargs.items():
        if v is not None:
            validator_span.set_attribute(
                f"validator.validate.input.{k}", serialize(v) or ""
            )
    trace_operation(
        input_value={"value": value_arg, "metadata": metadata_arg},
        input_mime_type="application/json",
    )

    if result is not None:
        output = result.to_dict()
        trace_operation(
            output_value=output,
            output_mime_type="application/json",
        )
        for k, v in output.items():
            if v is not None:
                validator_span.set_attribute(
                    f"validator.validate.output.{k}", serialize(v) or ""
                )


def trace_validator(
    validator_name: str,
    obj_id: int,
    on_fail_descriptor: Optional[str] = None,
    tracer: Optional[Tracer] = None,
    *,
    validation_session_id: str,
    **init_kwargs,
):
    def trace_validator_decorator(fn: Callable[..., Optional[ValidationResult]]):
        @wraps(fn)
        def trace_validator_wrapper(*args, **kwargs):
            if not settings.disable_tracing:
                current_otel_context = context.get_current()
                _tracer = get_tracer(tracer) or trace.get_tracer(
                    "guardrails-ai", GUARDRAILS_VERSION
                )
                validator_span_name = f"{validator_name}.validate"
                with _tracer.start_as_current_span(
                    name=validator_span_name,  # type: ignore
                    context=current_otel_context,  # type: ignore
                ) as validator_span:
                    try:
                        resp = fn(*args, **kwargs)
                        add_user_attributes(validator_span)
                        add_validator_attributes(
                            *args,
                            validator_span=validator_span,
                            validator_name=validator_name,
                            obj_id=obj_id,
                            on_fail_descriptor=on_fail_descriptor,
                            result=resp,
                            init_kwargs=init_kwargs,
                            validation_session_id=validation_session_id,
                            **kwargs,
                        )
                        return resp
                    except Exception as e:
                        validator_span.set_status(
                            status=StatusCode.ERROR, description=str(e)
                        )
                        add_user_attributes(validator_span)
                        add_validator_attributes(
                            *args,
                            validator_span=validator_span,
                            validator_name=validator_name,
                            obj_id=obj_id,
                            on_fail_descriptor=on_fail_descriptor,
                            result=None,
                            init_kwargs=init_kwargs,
                            validation_session_id=validation_session_id,
                            **kwargs,
                        )
                        raise e
            else:
                return fn(*args, **kwargs)

        return trace_validator_wrapper

    return trace_validator_decorator


def trace_async_validator(
    validator_name: str,
    obj_id: int,
    on_fail_descriptor: Optional[str] = None,
    tracer: Optional[Tracer] = None,
    *,
    validation_session_id: str,
    **init_kwargs,
):
    def trace_validator_decorator(
        fn: Callable[..., Awaitable[Optional[ValidationResult]]],
    ):
        @wraps(fn)
        async def trace_validator_wrapper(*args, **kwargs):
            if not settings.disable_tracing:
                current_otel_context = context.get_current()
                _tracer = get_tracer(tracer) or trace.get_tracer(
                    "guardrails-ai", GUARDRAILS_VERSION
                )
                validator_span_name = f"{validator_name}.validate"
                with _tracer.start_as_current_span(
                    name=validator_span_name,  # type: ignore
                    context=current_otel_context,  # type: ignore
                ) as validator_span:
                    try:
                        resp = await fn(*args, **kwargs)
                        add_user_attributes(validator_span)
                        add_validator_attributes(
                            *args,
                            validator_span=validator_span,
                            validator_name=validator_name,
                            obj_id=obj_id,
                            on_fail_descriptor=on_fail_descriptor,
                            result=resp,
                            init_kwargs=init_kwargs,
                            validation_session_id=validation_session_id,
                            **kwargs,
                        )
                        return resp
                    except Exception as e:
                        validator_span.set_status(
                            status=StatusCode.ERROR, description=str(e)
                        )
                        add_user_attributes(validator_span)
                        add_validator_attributes(
                            *args,
                            validator_span=validator_span,
                            validator_name=validator_name,
                            obj_id=obj_id,
                            on_fail_descriptor=on_fail_descriptor,
                            result=None,
                            init_kwargs=init_kwargs,
                            validation_session_id=validation_session_id,
                            **kwargs,
                        )
                        raise e
            else:
                return await fn(*args, **kwargs)

        return trace_validator_wrapper

    return trace_validator_decorator
