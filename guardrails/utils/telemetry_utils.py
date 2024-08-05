import json
import sys
from functools import wraps
from operator import attrgetter
from typing import Any, Callable, Dict, List, Optional, Union

from opentelemetry import context
from opentelemetry.context import Context
from opentelemetry.trace import StatusCode, Tracer, Span

from guardrails_api_client.models import Reask

from guardrails.call_tracing import TraceHandler
from guardrails.stores.context import get_tracer as get_context_tracer
from guardrails.stores.context import get_tracer_context
from guardrails.utils.casting_utils import to_string
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.logger import logger
from guardrails.actions.filter import Filter
from guardrails.actions.refrain import Refrain


def get_result_type(before_value: Any, after_value: Any, outcome: str):
    try:
        if isinstance(after_value, (Filter, Refrain, Reask)):
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


def get_span(span: Optional[Span] = None) -> Optional[Span]:
    if span is not None and hasattr(span, "add_event"):
        return span
    try:
        from opentelemetry import trace

        current_context = get_current_context()
        current_span = trace.get_current_span(current_context)
        return current_span
    except Exception as e:
        logger.error(e)
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

    TraceHandler().log_validator(validator_log)

    current_span.add_event(
        f"{validator_name}_result",
        {k: v for k, v in event.items() if v is not None},
    )


def trace_validation_result(
    validation_logs: List[ValidatorLogs],
    attempt_number: int,
    current_span=None,
):
    _current_span = get_span(current_span)
    if _current_span is not None:
        for log in validation_logs:
            trace_validator_result(_current_span, log, attempt_number)


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


def serialize(val: Any) -> Optional[str]:
    try:
        if val is None:
            return None
        if hasattr(val, "to_dict"):
            return json.dumps(val.to_dict())
        elif hasattr(val, "__dict__"):
            return json.dumps(val.__dict__)
        elif isinstance(val, dict) or isinstance(val, list):
            return json.dumps(val)
        return str(val)
    except Exception:
        return None


def to_dict(val: Any) -> Dict:
    try:
        if val is None:
            return {}
        elif isinstance(val, dict):
            return val
        elif hasattr(val, "to_dict"):
            return val.to_dict()
        elif hasattr(val, "__dict__"):
            return val.__dict__
        else:
            return dict(val)
    except Exception:
        return {}


def trace(name: str, tracer: Optional[Tracer] = None):
    def trace_wrapper(fn):
        @wraps(fn)
        def to_trace_or_not_to_trace(*args, **kwargs):
            _tracer = get_tracer(tracer)

            if _tracer is not None and hasattr(_tracer, "start_as_current_span"):
                trace_context = get_current_context()
                with _tracer.start_as_current_span(name, trace_context) as trace_span:  # type: ignore (Fails in Python 3.9 for invalid reason)
                    try:
                        ser_args = [serialize(arg) for arg in args]
                        ser_kwargs = {k: serialize(v) for k, v in kwargs.items()}
                        inputs = {
                            "args": [sarg for sarg in ser_args if sarg is not None],
                            "kwargs": {
                                k: v for k, v in ser_kwargs.items() if v is not None
                            },
                        }
                        trace_span.set_attribute("input.mime_type", "application/json")
                        trace_span.set_attribute("input.value", json.dumps(inputs))
                        # TODO: Capture args and kwargs as attributes?
                        response = fn(*args, **kwargs)

                        ser_output = serialize(response)
                        if ser_output:
                            trace_span.set_attribute(
                                "output.mime_type", "application/json"
                            )
                            trace_span.set_attribute(
                                "output.value",
                                (
                                    json.dumps(ser_output)
                                    if isinstance(ser_output, dict)
                                    else ser_output
                                ),
                            )
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
                        ser_args = [serialize(arg) for arg in args]
                        ser_kwargs = {k: serialize(v) for k, v in kwargs.items()}
                        inputs = {
                            "args": [sarg for sarg in ser_args if sarg is not None],
                            "kwargs": {
                                k: v for k, v in ser_kwargs.items() if v is not None
                            },
                        }
                        trace_span.set_attribute("input.mime_type", "application/json")
                        trace_span.set_attribute("input.value", json.dumps(inputs))
                        # TODO: Capture args and kwargs as attributes?
                        response = await fn(*args, **kwargs)

                        ser_output = serialize(response)
                        if ser_output:
                            trace_span.set_attribute(
                                "output.mime_type", "application/json"
                            )
                            trace_span.set_attribute(
                                "output.value",
                                (
                                    json.dumps(ser_output)
                                    if isinstance(ser_output, dict)
                                    else ser_output
                                ),
                            )

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


def wrap_with_otel_context(
    outer_scope_otel_context: Context, func: Callable[..., Any]
) -> Callable[..., Any]:
    """This function is designed to ensure that a given OpenTelemetry context
    is applied when executing a specified function. It is particularly useful
    for preserving the trace context when a guardrails is executed in a
    different execution flow or when integrating with other frameworks.

    Args:
        outer_scope_otel_context (Context): The OpenTelemetry context to apply
            when executing the function.
        func (Callable[..., Any]): The function to be executed within
            the given OpenTelemetry context.

    Returns:
        Callable[..., Any]: A wrapped version of 'func' that, when called,
            executes with 'outer_scope_otel_context' applied.
    """

    def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        # Attach the specified OpenTelemetry context before executing 'func'
        token = context.attach(outer_scope_otel_context)
        try:
            # Execute 'func' within the attached context
            return func(*args, **kwargs)
        finally:
            # Ensure the context is detached after execution
            #   to maintain correct context management
            context.detach(token)

    return wrapped_func


def default_otel_collector_tracer(resource_name: str = "guardrails"):
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


def default_otlp_tracer(resource_name: str = "guardrails"):
    """This tracer will emit spans directly to an otlp endpoint, configured by
    the following environment variables:

    OTEL_EXPORTER_OTLP_PROTOCOL
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    OTEL_EXPORTER_OTLP_ENDPOINT
    OTEL_EXPORTER_OTLP_HEADERS

    We recommend using Grafana to collect your metrics. A full example of how to
    do that is in our (docs)[https://docs.guardrails.com/telemetry]
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


def trace_operation(
    *,
    input_mime_type: Optional[str] = None,
    input_value: Optional[Any] = None,
    output_mime_type: Optional[str] = None,
    output_value: Optional[Any] = None,
):
    """Traces an operation (any function call) using OpenInference semantic
    conventions."""
    current_span = get_span()

    if current_span is None:
        return

    ser_input_mime_type = serialize(input_mime_type)
    if ser_input_mime_type:
        current_span.set_attribute("input.mime_type", ser_input_mime_type)

    ser_input_value = serialize(input_value)
    if ser_input_value:
        current_span.set_attribute("input.value", ser_input_value)

    ser_output_mime_type = serialize(output_mime_type)
    if ser_output_mime_type:
        current_span.set_attribute("output.mime_type", ser_output_mime_type)

    ser_output_value = serialize(output_value)
    if ser_output_value:
        current_span.set_attribute("output.value", ser_output_value)


def trace_llm_call(
    *,
    function_call: Optional[
        Dict[str, Any]
    ] = None,  # JSON String	"{function_name: 'add', args: [1, 2]}"	Object recording details of a function call in models or APIs  # noqa
    input_messages: Optional[
        List[Dict[str, Any]]
    ] = None,  # List of objects†	[{"message.role": "user", "message.content": "hello"}]	List of messages sent to the LLM in a chat API request  # noqa
    invocation_parameters: Optional[
        Dict[str, Any]
    ] = None,  # JSON string	"{model_name: 'gpt-3', temperature: 0.7}"	Parameters used during the invocation of an LLM or API  # noqa
    model_name: Optional[
        str
    ] = None,  # String	"gpt-3.5-turbo"	The name of the language model being utilized  # noqa
    output_messages: Optional[
        List[Dict[str, Any]]
    ] = None,  # List of objects	[{"message.role": "user", "message.content": "hello"}]	List of messages received from the LLM in a chat API request  # noqa
    prompt_template_template: Optional[
        str
    ] = None,  # String	"Weather forecast for {city} on {date}"	Template used to generate prompts as Python f-strings  # noqa
    prompt_template_variables: Optional[
        Dict[str, Any]
    ] = None,  # JSON String	{ context: "<context from retrieval>", subject: "math" }	JSON of key value pairs applied to the prompt template  # noqa
    prompt_template_version: Optional[
        str
    ] = None,  # String	"v1.0"	The version of the prompt template  # noqa
    token_count_completion: Optional[
        int
    ] = None,  # Integer	15	The number of tokens in the completion  # noqa
    token_count_prompt: Optional[
        int
    ] = None,  # Integer	5	The number of tokens in the prompt  # noqa
    token_count_total: Optional[
        int
    ] = None,  # Integer	20	Total number of tokens, including prompt and completion  # noqa
):
    """Traces an LLM call using OpenInference semantic conventions."""
    current_span = get_span()

    if current_span is None:
        return

    ser_function_call = serialize(function_call)
    if ser_function_call:
        current_span.set_attribute("llm.function_call", ser_function_call)

    if input_messages and isinstance(input_messages, list):
        for i, message in enumerate(input_messages):
            msg_obj = to_dict(message)
            for key, value in msg_obj.items():
                standardized_key = f"message.{key}" if "message" not in key else key
                current_span.set_attribute(
                    f"llm.input_messages.{i}.{standardized_key}",
                    serialize(value),  # type: ignore
                )

    ser_invocation_parameters = serialize(invocation_parameters)
    if ser_invocation_parameters:
        current_span.set_attribute(
            "llm.invocation_parameters", ser_invocation_parameters
        )

    ser_model_name = serialize(model_name)
    if ser_model_name:
        current_span.set_attribute("llm.model_name", ser_model_name)

    if output_messages and isinstance(output_messages, list):
        for i, message in enumerate(output_messages):
            # Most responses are either dictionaries or Pydantic models
            msg_obj = to_dict(message)
            for key, value in msg_obj.items():
                standardized_key = f"message.{key}" if "message" not in key else key
                current_span.set_attribute(
                    f"llm.output_messages.{i}.{standardized_key}",
                    serialize(value),  # type: ignore
                )

    ser_prompt_template_template = serialize(prompt_template_template)
    if ser_prompt_template_template:
        current_span.set_attribute(
            "llm.prompt_template.template", ser_prompt_template_template
        )

    ser_prompt_template_variables = serialize(prompt_template_variables)
    if ser_prompt_template_variables:
        current_span.set_attribute(
            "llm.prompt_template.variables", ser_prompt_template_variables
        )

    ser_prompt_template_version = serialize(prompt_template_version)
    if ser_prompt_template_version:
        current_span.set_attribute(
            "llm.prompt_template.version", ser_prompt_template_version
        )

    if token_count_completion:
        current_span.set_attribute("llm.token_count.completion", token_count_completion)

    if token_count_prompt:
        current_span.set_attribute("llm.token_count.prompt", token_count_prompt)

    if token_count_total:
        current_span.set_attribute("llm.token_count.total", token_count_total)
