from functools import wraps
from typing import (
    Any,
    Dict,
    Optional,
    AsyncGenerator,
)

from opentelemetry.trace import Span
from opentelemetry.trace.propagation import set_span_in_context

from guardrails.classes.validation.validation_result import ValidationResult
from guardrails.hub_token.token import VALIDATOR_HUB_SERVICE
from guardrails.types.primitives import PrimitiveTypes
from guardrails.utils.safe_get import safe_get
from guardrails.utils.hub_telemetry_utils import HubTelemetry


def get_guard_call_attributes(
    attrs: Dict[str, Any], origin: str, *args, **kwargs
) -> Dict[str, Any]:
    attrs["stream"] = kwargs.get("stream", False)

    guard_self = safe_get(args, 0)
    if guard_self is not None:
        attrs["guard_id"] = guard_self.id
        attrs["user_id"] = guard_self._user_id
        attrs["custom_reask_prompt"] = guard_self._exec_opts.reask_prompt is not None
        attrs["custom_reask_instructions"] = (
            guard_self._exec_opts.reask_instructions is not None
        )
        attrs["custom_reask_messages"] = (
            guard_self._exec_opts.reask_messages is not None
        )
        attrs["output_type"] = (
            "unstructured"
            if PrimitiveTypes.is_primitive(
                guard_self.output_schema.type.actual_instance
            )
            else "structured"
        )
        return attrs

    llm_api_str = ""  # noqa
    llm_api = kwargs.get("llm_api")
    if origin in ["Guard.__call__", "AsyncGuard.__call__"]:
        llm_api = safe_get(args, 1, llm_api)

    if llm_api:
        llm_api_module_name = (
            llm_api.__module__ if hasattr(llm_api, "__module__") else ""
        )
        llm_api_name = (
            llm_api.__name__ if hasattr(llm_api, "__name__") else type(llm_api).__name__
        )
        llm_api_str = f"{llm_api_module_name}.{llm_api_name}"
    attrs["llm_api"] = llm_api_str if llm_api_str else "None"

    return attrs


def get_validator_inference_attributes(
    attrs: Dict[str, Any], *args, **kwargs
) -> Dict[str, Any]:
    validator_self = safe_get(args, 0)
    if validator_self is not None:
        used_guardrails_endpoint = (
            VALIDATOR_HUB_SERVICE in validator_self.validation_endpoint
            and not validator_self.use_local
        )
        used_custom_endpoint = (
            not validator_self.use_local and not used_guardrails_endpoint
        )
        attrs["validator_name"] = validator_self.rail_alias
        attrs["used_remote_inference"] = not validator_self.use_local
        attrs["used_local_inference"] = validator_self.use_local
        attrs["used_guardrails_endpoint"] = used_guardrails_endpoint
        attrs["used_custom_endpoint"] = used_custom_endpoint
    return attrs


def get_validator_usage_attributes(
    attrs: Dict[str, Any], response, *args, **kwargs
) -> Dict[str, Any]:
    # We're wrapping a wrapped function,
    #   so the first arg is the validator service
    validator_self = safe_get(args, 1)
    if validator_self is not None:
        attrs["validator_name"] = validator_self.rail_alias
        attrs["validator_on_fail"] = validator_self.on_fail_descriptor

    if response is not None:
        attrs["validator_result"] = (
            response.outcome if isinstance(response, ValidationResult) else None
        )

    return attrs


def add_attributes(
    span: Span,
    attrs: Dict[str, Any],
    name: str,
    origin: str,
    *args,
    response=None,
    **kwargs,
):
    attrs["origin"] = origin
    if name == "/guard_call":
        attrs = get_guard_call_attributes(attrs, origin, *args, **kwargs)
    elif name == "/reasks":
        if response is not None and hasattr(response, "iterations"):
            attrs["reask_count"] = len(response.iterations) - 1
        else:
            attrs["reask_count"] = 0
    elif name == "/validator_inference":
        attrs = get_validator_inference_attributes(attrs, *args, **kwargs)
    elif name == "/validator_usage":
        attrs = get_validator_usage_attributes(attrs, response, *args, **kwargs)

    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


def trace(
    *,
    name: str,
    origin: Optional[str] = None,
    **attrs,
):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            hub_telemetry = HubTelemetry()
            if hub_telemetry._enabled and hub_telemetry._tracer is not None:
                with hub_telemetry._tracer.start_span(
                    name,
                    context=hub_telemetry.extract_current_context(),
                    set_status_on_exception=True,
                ) as span:  # noqa
                    context = set_span_in_context(span)
                    hub_telemetry.inject_current_context(context=context)
                    nonlocal origin
                    origin = origin if origin is not None else name

                    resp = fn(*args, **kwargs)
                    add_attributes(
                        span, attrs, name, origin, *args, response=resp, **kwargs
                    )
                    return resp
            else:
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def async_trace(
    *,
    name: str,
    origin: Optional[str] = None,
):
    def decorator(fn):
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            hub_telemetry = HubTelemetry()
            if hub_telemetry._enabled and hub_telemetry._tracer is not None:
                with hub_telemetry._tracer.start_span(
                    name,
                    context=hub_telemetry.extract_current_context(),
                    set_status_on_exception=True,
                ) as span:  # noqa
                    context = set_span_in_context(span)
                    hub_telemetry.inject_current_context(context=context)

                    nonlocal origin
                    origin = origin if origin is not None else name
                    add_attributes(span, {"async": True}, name, origin, *args, **kwargs)
                    return await fn(*args, **kwargs)
            else:
                return await fn(*args, **kwargs)

        return async_wrapper

    return decorator


def _run_gen(fn, *args, **kwargs):
    gen = fn(*args, **kwargs)
    for item in gen:
        yield item


def trace_stream(
    *,
    name: str,
    origin: Optional[str] = None,
    **attrs,
):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            hub_telemetry = HubTelemetry()
            if hub_telemetry._enabled and hub_telemetry._tracer is not None:
                with hub_telemetry._tracer.start_span(
                    name,
                    context=hub_telemetry.extract_current_context(),
                    set_status_on_exception=True,
                ) as span:  # noqa
                    context = set_span_in_context(span)
                    hub_telemetry.inject_current_context(context=context)

                    nonlocal origin
                    origin = origin if origin is not None else name
                    add_attributes(span, attrs, name, origin, *args, **kwargs)
                    return _run_gen(fn, *args, **kwargs)
            else:
                return fn(*args, **kwargs)

        return wrapper

    return decorator


async def _run_async_gen(fn, *args, **kwargs) -> AsyncGenerator[Any, None]:
    gen = fn(*args, **kwargs)
    async for item in gen:
        yield item


def async_trace_stream(
    *,
    name: str,
    origin: Optional[str] = None,
    **attrs,
):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            hub_telemetry = HubTelemetry()
            if hub_telemetry._enabled and hub_telemetry._tracer is not None:
                with hub_telemetry._tracer.start_span(
                    name,
                    context=hub_telemetry.extract_current_context(),
                    set_status_on_exception=True,
                ) as span:  # noqa
                    context = set_span_in_context(span)
                    hub_telemetry.inject_current_context(context=context)

                    nonlocal origin
                    origin = origin if origin is not None else name
                    add_attributes(span, attrs, name, origin, *args, **kwargs)
                    return fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)

        return wrapper

    return decorator
