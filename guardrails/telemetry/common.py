import json
from typing import Any, Callable, Dict, Optional, Union
from opentelemetry import context
from opentelemetry.context import Context
from opentelemetry.trace import Tracer, Span

from guardrails.logger import logger
from guardrails.stores.context import (
    get_tracer as get_context_tracer,
    get_tracer_context,
)


def get_tracer(tracer: Optional[Tracer] = None) -> Optional[Tracer]:
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
