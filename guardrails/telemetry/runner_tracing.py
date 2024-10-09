import json
from functools import wraps
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Optional,
)

from opentelemetry import context, trace
from opentelemetry.trace import StatusCode, Span

from guardrails.classes.history.iteration import Iteration
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.settings import settings
from guardrails.classes.output_type import OT
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.stores.context import get_guard_name
from guardrails.telemetry.common import get_tracer, add_user_attributes, serialize
from guardrails.utils.safe_get import safe_get
from guardrails.version import GUARDRAILS_VERSION

import sys

if sys.version_info.minor < 10:
    from guardrails.utils.polyfills import anext

#########################################
### START Runner.step Instrumentation ###
#########################################


# TODO: Track input arguments and outputs explicitly as named attributes
def add_step_attributes(
    step_span: Span, response: Optional[Iteration], *args, **kwargs
):
    step_number = safe_get(args, 1, kwargs.get("index", 0))
    guard_name = get_guard_name()

    step_span.set_attribute("guardrails.version", GUARDRAILS_VERSION)
    step_span.set_attribute("type", "guardrails/guard/step")
    step_span.set_attribute("guard.name", guard_name)
    step_span.set_attribute("step.index", step_number)

    ser_args = [serialize(arg) for arg in args]
    ser_kwargs = {k: serialize(v) for k, v in kwargs.items()}
    inputs = {
        "args": [sarg for sarg in ser_args if sarg is not None],
        "kwargs": {k: v for k, v in ser_kwargs.items() if v is not None},
    }
    step_span.set_attribute("input.mime_type", "application/json")
    step_span.set_attribute("input.value", json.dumps(inputs))

    ser_output = serialize(response)
    if ser_output:
        step_span.set_attribute("output.mime_type", "application/json")
        step_span.set_attribute(
            "output.value",
            (json.dumps(ser_output) if isinstance(ser_output, dict) else ser_output),
        )


def trace_step(fn: Callable[..., Iteration]):
    @wraps(fn)
    def trace_step_wrapper(*args, **kwargs) -> Iteration:
        if not settings.disable_tracing:
            current_otel_context = context.get_current()
            tracer = get_tracer()
            tracer = tracer or trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)

            with tracer.start_as_current_span(
                name="step",  # type: ignore
                context=current_otel_context,  # type: ignore
            ) as step_span:
                try:
                    response = fn(*args, **kwargs)
                    add_step_attributes(step_span, response, *args, **kwargs)
                    add_user_attributes(step_span)
                    return response
                except Exception as e:
                    step_span.set_status(status=StatusCode.ERROR, description=str(e))
                    add_step_attributes(step_span, None, *args, **kwargs)
                    add_user_attributes(step_span)
                    raise e
        else:
            return fn(*args, **kwargs)

    return trace_step_wrapper


def trace_stream_step_generator(
    fn: Callable[..., Iterator[ValidationOutcome[OT]]], *args, **kwargs
) -> Iterator[ValidationOutcome[OT]]:
    current_otel_context = context.get_current()
    tracer = get_tracer()
    tracer = tracer or trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)

    exception = None
    with tracer.start_as_current_span(
        name="step",  # type: ignore
        context=current_otel_context,  # type: ignore
    ) as step_span:
        try:
            gen = fn(*args, **kwargs)
            next_exists = True
            while next_exists:
                try:
                    res = next(gen)
                    yield res
                except StopIteration:
                    next_exists = False
        except Exception as e:
            step_span.set_status(status=StatusCode.ERROR, description=str(e))
            exception = e
        finally:
            call = safe_get(args, 8, kwargs.get("call_log", None))
            iteration = call.iterations.last if call else None
            add_step_attributes(step_span, iteration, *args, **kwargs)
            add_user_attributes(step_span)
            if exception:
                raise exception


def trace_stream_step(
    fn: Callable[..., Iterator[ValidationOutcome[OT]]],
) -> Callable[..., Iterator[ValidationOutcome[OT]]]:
    @wraps(fn)
    def trace_stream_step_wrapper(*args, **kwargs) -> Iterator[ValidationOutcome[OT]]:
        if not settings.disable_tracing:
            return trace_stream_step_generator(fn, *args, **kwargs)
        else:
            return fn(*args, **kwargs)

    return trace_stream_step_wrapper


def trace_async_step(fn: Callable[..., Awaitable[Iteration]]):
    @wraps(fn)
    async def trace_async_step_wrapper(*args, **kwargs) -> Iteration:
        if not settings.disable_tracing:
            current_otel_context = context.get_current()
            tracer = get_tracer()
            tracer = tracer or trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)

            with tracer.start_as_current_span(
                name="step",  # type: ignore
                context=current_otel_context,  # type: ignore
            ) as step_span:
                try:
                    response = await fn(*args, **kwargs)
                    add_user_attributes(step_span)
                    add_step_attributes(step_span, response, *args, **kwargs)
                    return response
                except Exception as e:
                    step_span.set_status(status=StatusCode.ERROR, description=str(e))
                    add_user_attributes(step_span)
                    add_step_attributes(step_span, None, *args, **kwargs)
                    raise e

        else:
            return await fn(*args, **kwargs)

    return trace_async_step_wrapper


async def trace_async_stream_step_generator(
    fn: Callable[..., AsyncIterator[ValidationOutcome[OT]]], *args, **kwargs
) -> AsyncIterator[ValidationOutcome[OT]]:
    current_otel_context = context.get_current()
    tracer = get_tracer()
    tracer = tracer or trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)

    exception = None
    with tracer.start_as_current_span(
        name="step",  # type: ignore
        context=current_otel_context,  # type: ignore
    ) as step_span:
        try:
            gen = fn(*args, **kwargs)
            next_exists = True
            while next_exists:
                try:
                    res = await anext(gen)  # type: ignore
                    yield res
                except StopIteration:
                    next_exists = False
                except StopAsyncIteration:
                    next_exists = False
        except Exception as e:
            step_span.set_status(status=StatusCode.ERROR, description=str(e))
            exception = e
        finally:
            call = safe_get(args, 3, kwargs.get("call_log", None))
            iteration = call.iterations.last if call else None
            add_step_attributes(step_span, iteration, *args, **kwargs)
            if exception:
                raise exception


def trace_async_stream_step(
    fn: Callable[..., AsyncIterator[ValidationOutcome[OT]]],
):
    @wraps(fn)
    async def trace_async_stream_step_wrapper(
        *args, **kwargs
    ) -> AsyncIterator[ValidationOutcome[OT]]:
        if not settings.disable_tracing:
            return trace_async_stream_step_generator(fn, *args, **kwargs)
        else:
            return fn(*args, **kwargs)

    return trace_async_stream_step_wrapper


#########################################
#### END Runner.step Instrumentation ####
#########################################


#########################################
### START Runner.call Instrumentation ###
#########################################


# TODO: Track input arguments and outputs explicitly as named attributes
def add_call_attributes(
    call_span: Span, response: Optional[LLMResponse], *args, **kwargs
):
    guard_name = get_guard_name()

    call_span.set_attribute("guardrails.version", GUARDRAILS_VERSION)
    call_span.set_attribute("type", "guardrails/guard/step/call")
    call_span.set_attribute("guard.name", guard_name)

    ser_args = [serialize(arg) for arg in args]
    ser_kwargs = {k: serialize(v) for k, v in kwargs.items()}
    inputs = {
        "args": [sarg for sarg in ser_args if sarg is not None],
        "kwargs": {k: v for k, v in ser_kwargs.items() if v is not None},
    }
    call_span.set_attribute("input.mime_type", "application/json")
    call_span.set_attribute("input.value", json.dumps(inputs))

    ser_output = serialize(response)
    if ser_output:
        call_span.set_attribute("output.mime_type", "application/json")
        call_span.set_attribute(
            "output.value",
            (json.dumps(ser_output) if isinstance(ser_output, dict) else ser_output),
        )


def trace_call(fn: Callable[..., LLMResponse]):
    @wraps(fn)
    def trace_call_wrapper(*args, **kwargs):
        if not settings.disable_tracing:
            current_otel_context = context.get_current()
            tracer = get_tracer()
            tracer = tracer or trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)

            with tracer.start_as_current_span(
                name="call",  # type: ignore
                context=current_otel_context,  # type: ignore
            ) as call_span:
                try:
                    response = fn(*args, **kwargs)
                    if isinstance(response, LLMResponse) and (
                        response.async_stream_output or response.stream_output
                    ):
                        # TODO: Iterate, add a call attr each time
                        return response
                    add_call_attributes(call_span, response, *args, **kwargs)
                    return response
                except Exception as e:
                    call_span.set_status(status=StatusCode.ERROR, description=str(e))
                    add_call_attributes(call_span, None, *args, **kwargs)
                    raise e
        else:
            return fn(*args, **kwargs)

    return trace_call_wrapper


def trace_async_call(fn: Callable[..., Awaitable[LLMResponse]]):
    @wraps(fn)
    async def trace_async_call_wrapper(*args, **kwargs):
        if not settings.disable_tracing:
            current_otel_context = context.get_current()
            tracer = get_tracer()
            tracer = tracer or trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)

            with tracer.start_as_current_span(
                name="call",  # type: ignore
                context=current_otel_context,  # type: ignore
            ) as call_span:
                try:
                    response = await fn(*args, **kwargs)
                    add_call_attributes(call_span, response, *args, **kwargs)
                    return response
                except Exception as e:
                    call_span.set_status(status=StatusCode.ERROR, description=str(e))
                    add_call_attributes(call_span, None, *args, **kwargs)
                    raise e

        else:
            return await fn(*args, **kwargs)

    return trace_async_call_wrapper


#########################################
#### END Runner.call Instrumentation ####
#########################################
