import sys
from functools import wraps
from operator import attrgetter
from typing import Any, List, Optional, Union

from opentelemetry import context
from opentelemetry.context import Context
from opentelemetry.trace import StatusCode, Tracer

from guardrails.stores.context import get_tracer as get_context_tracer
from guardrails.stores.context import get_tracer_context
from guardrails.utils.casting_utils import to_string
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import ReAsk
from guardrails.actions import Filter, Refrain


def get_result_type(before_value: Any, after_value: Any, outcome: str):
    try:
        if isinstance(after_value, (Filter, Refrain, ReAsk)):
            name = after_value.__class__.__name__.lower()
        elif after_value != before_value:
            name = "fix"
        else:
            name = outcome
        return name
    except Exception:
        return type(after_value)


def get_tracer(tracer: Optional[Tracer] = None) -> Union[Tracer, None]:
    # TODO: Do we ever need to consider supporting non-otel tracers?
    _tracer = tracer if tracer is not None else get_context_tracer()
    return _tracer


def get_current_context() -> Union[Context, None]:
    otel_current_context = (
        context.get_current()
        if context is not None and hasattr(context, "get_current")
        else None
    )
    tracer_context = get_tracer_context()
    return otel_current_context or tracer_context


def get_span(span=None):
    if span is not None and hasattr(span, "add_event"):
        return span
    try:
        from opentelemetry import trace

        current_context = get_current_context()
        current_span = trace.get_current_span(current_context)
        return current_span
    except Exception as e:
        print(e)
        return None


def trace_validator_result(
    current_span, validator_log: ValidatorLogs, attempt_number: int, **kwargs
):
    (
        validator_name,
        value_before_validation,
        validation_result,
        value_after_validation,
        start_time,
        end_time,
        instance_id,
    ) = attrgetter(
        "registered_name",
        "value_before_validation",
        "validation_result",
        "value_after_validation",
        "start_time",
        "end_time",
        "instance_id",
    )(validator_log)
    result = (
        validation_result.outcome
        if hasattr(validation_result, "outcome")
        and validation_result.outcome is not None
        else "unknown"
    )
    result_type = get_result_type(
        value_before_validation, value_after_validation, result
    )

    event = {
        "validator_name": validator_name,
        "attempt_number": attempt_number,
        "result": result,
        "result_type": result_type,
        "input": to_string(value_before_validation),
        "output": to_string(value_after_validation),
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": end_time.isoformat() if end_time else None,
        "instance_id": instance_id,
        **kwargs,
    }
    current_span.add_event(
        f"{validator_name}_result",
        {k: v for k, v in event.items() if v is not None},
    )


def trace_validation_result(
    validation_logs: List[ValidatorLogs],
    attempt_number: int,
    current_span=None,
):
    # Duplicate logs are showing here
    # print("validation_logs.validator_logs: ", validation_logs.validator_logs)
    _current_span = get_span(current_span)
    if _current_span is not None:
        for log in validation_logs:
            # Duplicate logs are showing here
            # print("calling trace_validator_result with: ", log, attempt_number)
            trace_validator_result(_current_span, log, attempt_number)

        # CHECKME: disabled these because I think we flattened this structure?
        # if validation_logs.children:
        #     for child in validation_logs.children:
        #         # print("calling trace_validation_result with child logs")
        #         trace_validation_result(
        #             validation_logs.children.get(child), attempt_number, _current_span
        #         )


def trace_validator(
    validator_name: str,
    obj_id: int,
    # TODO - re-enable once we have namespace support
    # namespace: str = None,
    on_fail_descriptor: Optional[str] = None,
    tracer: Optional[Tracer] = None,
    **init_kwargs,
):
    def trace_validator_wrapper(fn):
        _tracer = get_tracer(tracer)

        @wraps(fn)
        def with_trace(*args, **kwargs):
            span_name = (
                # TODO - re-enable once we have namespace support
                # f"{namespace}.{validator_name}.validate"
                # if namespace is not None
                # else f"{validator_name}.validate"
                f"{validator_name}.validate"
            )
            trace_context = get_current_context()
            if _tracer is None:
                return fn(*args, **kwargs)
            with _tracer.start_as_current_span(
                span_name,  # type: ignore (Fails in Python 3.9 for invalid reason)
                trace_context,
            ) as validator_span:
                try:
                    validator_span.set_attribute(
                        "on_fail_descriptor", on_fail_descriptor or "noop"
                    )
                    validator_span.set_attribute(
                        "args",
                        to_string({k: to_string(v) for k, v in init_kwargs.items()})
                        or "{}",
                    )
                    validator_span.set_attribute("instance_id", to_string(obj_id) or "")

                    # NOTE: Update if Validator.validate method signature ever changes
                    if args is not None and len(args) > 1:
                        validator_span.set_attribute("input", to_string(args[1]) or "")

                    return fn(*args, **kwargs)
                except Exception as e:
                    validator_span.set_status(
                        status=StatusCode.ERROR, description=str(e)
                    )
                    raise e

        @wraps(fn)
        def without_a_trace(*args, **kwargs):
            return fn(*args, **kwargs)

        if _tracer is not None and hasattr(_tracer, "start_as_current_span"):
            return with_trace
        else:
            return without_a_trace

    return trace_validator_wrapper


def trace(name: str, tracer: Optional[Tracer] = None):
    def trace_wrapper(fn):
        @wraps(fn)
        def to_trace_or_not_to_trace(*args, **kwargs):
            _tracer = get_tracer(tracer)

            if _tracer is not None and hasattr(_tracer, "start_as_current_span"):
                trace_context = get_current_context()
                with _tracer.start_as_current_span(name, trace_context) as trace_span:  # type: ignore (Fails in Python 3.9 for invalid reason)
                    try:
                        # TODO: Capture args and kwargs as attributes?
                        response = fn(*args, **kwargs)
                        return response
                    except Exception as e:
                        trace_span.set_status(
                            status=StatusCode.ERROR, description=str(e)
                        )
                        raise e
            else:
                return fn(*args, **kwargs)

        return to_trace_or_not_to_trace

    return trace_wrapper


def async_trace(name: str, tracer: Optional[Tracer] = None):
    def trace_wrapper(fn):
        @wraps(fn)
        async def to_trace_or_not_to_trace(*args, **kwargs):
            _tracer = get_tracer(tracer)

            if _tracer is not None and hasattr(_tracer, "start_as_current_span"):
                trace_context = get_current_context()
                with _tracer.start_as_current_span(name, trace_context) as trace_span:  # type: ignore (Fails in Python 3.9 for invalid reason)
                    try:
                        # TODO: Capture args and kwargs as attributes?
                        response = await fn(*args, **kwargs)
                        return response
                    except Exception as e:
                        trace_span.set_status(
                            status=StatusCode.ERROR, description=str(e)
                        )
                        raise e
            else:
                response = await fn(*args, **kwargs)
                return response

        return to_trace_or_not_to_trace

    return trace_wrapper


def default_otel_collector_tracer(resource_name: str = "guardsrails"):
    """This is the standard otel tracer set to talk to a grpc open telemetry
    collector running on port 4317."""

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource(attributes={SERVICE_NAME: resource_name})

    traceProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    traceProvider.add_span_processor(processor)
    trace.set_tracer_provider(traceProvider)

    return trace.get_tracer("gr")


def default_otlp_tracer(resource_name: str = "guardsrails"):
    """This tracer will emit spans directly to an otlp endpoint, configured by
    the following environment variables:

    OTEL_EXPORTER_OTLP_PROTOCOL
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    OTEL_EXPORTER_OTLP_ENDPOINT
    OTEL_EXPORTER_OTLP_HEADERS

    We recommend using Grafana to collect your metrics. A full example of how to
    do that is in our (docs)[https://docs.guardsrails.com/telemetry]
    """
    import os

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    envvars_exist = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") and (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )

    resource = Resource(attributes={SERVICE_NAME: resource_name})

    traceProvider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter()
    if envvars_exist:
        processor = BatchSpanProcessor(span_exporter=span_exporter)
    else:
        processor = SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr))

    traceProvider.add_span_processor(processor)
    trace.set_tracer_provider(traceProvider)

    return trace.get_tracer("guardrails-ai")
