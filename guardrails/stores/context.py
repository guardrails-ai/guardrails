from contextvars import ContextVar, copy_context
from typing import Any, Dict, Literal, Optional, Union

try:
    from opentelemetry import context
    from opentelemetry.context import Context as TracerContext
    from opentelemetry.trace import Tracer
except Exception:

    class Tracer:
        pass

    class TracerContext:
        pass

    context = None


TRACER_KEY: Literal["tracer"] = "gr.reserved.tracer"
TRACER_CONTEXT_KEY: Literal["tracer"] = "gr.reserved.tracer.context"
DOCUMENT_STORE_KEY: Literal["document_store"] = "gr.reserved.document_store"
CALL_KWARGS_KEY: Literal["call_kwargs"] = "gr.reserved.call_kwargs"


def set_tracer(tracer: Optional[Tracer] = None) -> None:
    set_context_var(TRACER_KEY, tracer)


def get_tracer() -> Union[Tracer, None]:
    return get_context_var(TRACER_KEY)


def set_tracer_context(tracer_context: Optional[TracerContext] = None) -> None:
    tracer_context = (
        tracer_context
        if tracer_context
        else (context.get_current() if context is not None else None)
    )
    set_context_var(TRACER_CONTEXT_KEY, tracer_context)


def get_tracer_context() -> Union[TracerContext, None]:
    return get_context_var(TRACER_CONTEXT_KEY)


def set_call_kwargs(kwargs: Dict[str, Any]) -> None:
    set_context_var(CALL_KWARGS_KEY, kwargs)


def get_call_kwargs() -> Dict[str, Any]:
    return get_context_var(CALL_KWARGS_KEY) or {}


def get_call_kwarg(kwarg_key: str) -> Union[Any, None]:
    kwargs = get_call_kwargs()
    return kwargs.get(kwarg_key)


def _get_contextvar(key):
    context = copy_context()
    context_var = None
    for c_key in context.keys():
        if c_key.name == key:
            context_var = c_key
            break
    return context_var


def set_context_var(key, value):
    context_var = _get_contextvar(key) or ContextVar(key)
    context_var.set(value)


def get_context_var(key):
    context_var = _get_contextvar(key)
    return context_var.get() if context_var else None
