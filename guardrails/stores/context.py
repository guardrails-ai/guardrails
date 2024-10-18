from contextvars import ContextVar, copy_context
from typing import Any, Dict, Literal, Optional, Union, cast

from opentelemetry import context
from opentelemetry.context import Context
from opentelemetry.trace import Tracer

GUARD_NAME_KEY: Literal["gr.reserved.guard.name"] = "gr.reserved.guard.name"
TRACER_KEY: Literal["gr.reserved.tracer"] = "gr.reserved.tracer"
TRACER_CONTEXT_KEY: Literal["gr.reserved.tracer.context"] = "gr.reserved.tracer.context"
DOCUMENT_STORE_KEY: Literal["gr.reserved.document_store"] = "gr.reserved.document_store"
CALL_KWARGS_KEY: Literal["gr.reserved.call_kwargs"] = "gr.reserved.call_kwargs"


def set_guard_name(guard_name: str) -> None:
    set_context_var(GUARD_NAME_KEY, guard_name)


def get_guard_name() -> str:
    return get_context_var(GUARD_NAME_KEY) or ""


def set_tracer(tracer: Optional[Tracer] = None) -> None:
    set_context_var(TRACER_KEY, tracer)


def get_tracer() -> Union[Tracer, None]:
    cvar = get_context_var(TRACER_KEY)
    if cvar is None:
        return None
    return cast(Tracer, cvar)


def set_tracer_context(tracer_context: Optional[Context] = None) -> None:
    tracer_context = (
        tracer_context
        if tracer_context
        else (context.get_current() if context is not None else None)
    )
    set_context_var(TRACER_CONTEXT_KEY, tracer_context)


def get_tracer_context() -> Union[Context, None]:
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
