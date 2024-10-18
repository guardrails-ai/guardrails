import inspect
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Optional,
    Union,
)

from opentelemetry import context, trace
from opentelemetry.trace import StatusCode, Tracer, Span, Link, get_tracer

from guardrails.settings import settings
from guardrails.classes.generic.stack import Stack
from guardrails.classes.history.call import Call
from guardrails.classes.output_type import OT
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.telemetry.open_inference import trace_operation
from guardrails.telemetry.common import add_user_attributes
from guardrails.version import GUARDRAILS_VERSION

import sys

if sys.version_info.minor < 10:
    from guardrails.utils.polyfills import anext

# from sentence_transformers import SentenceTransformer
# import numpy as np
# from numpy.linalg import norm

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def add_guard_attributes(
    guard_span: Span,
    history: Stack[Call],
    resp: ValidationOutcome,
):
    instructions = history.last.compiled_instructions if history.last else ""
    prompt = history.last.compiled_prompt if history.last else ""
    messages = []
    if history.last and history.last.iterations.last:
        messages = history.last.iterations.last.inputs.msg_history or []
    if not instructions:
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        system_message = system_messages[-1] if system_messages else {}
        instructions = system_message.get("content", "")
    if not prompt:
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        user_message = user_messages[-1] if user_messages else {}
        prompt = user_message.get("content", "")
    input_value = f"""
        {instructions}
        {prompt}
        """
    trace_operation(
        input_mime_type="text/plain",
        input_value=input_value,
        output_mime_type="text/plain",
        output_value=resp.validated_output,
    )
    guard_span.set_attribute("type", "guardrails/guard")
    guard_span.set_attribute("validation_passed", resp.validation_passed)

    execution_id = history.last.id if history.last else None
    if execution_id is not None:
        guard_span.set_attribute("execution_id", execution_id)

    token_consumption = history.last.tokens_consumed if history.last else None
    if token_consumption is not None:
        guard_span.set_attribute("token_consumption", token_consumption)

    number_of_reasks = (
        history.last.iterations.last.index
        if history.last and history.last.iterations.last
        else None
    )
    if number_of_reasks is not None:
        guard_span.set_attribute("number_of_reasks", number_of_reasks)

    number_of_llm_calls = number_of_reasks + 1 if number_of_reasks is not None else None
    if number_of_llm_calls is not None:
        guard_span.set_attribute("number_of_llm_calls", number_of_llm_calls)

    # # FIXME: Find a lighter weight library to do this.
    # raw_embed = model.encode(resp.raw_llm_output)
    # validated_embed = model.encode(resp.validated_output)
    # input_embed = model.encode(input_value)

    # # define two arrays
    # raw_embed_np = np.array(raw_embed)
    # validated_embed_np = np.array(validated_embed)
    # input_embed_np = np.array(input_embed)

    # # compute cosine similarity
    # raw_output_x_validated_output_cosine = (
    #     np.sum(raw_embed_np*validated_embed_np, axis=0)
    #     /
    #     (
    #         norm(raw_embed_np, axis=0)*norm(validated_embed_np, axis=0)
    #     )
    # )

    # input_x_validated_output_cosine = (
    #     np.sum(input_embed_np*validated_embed_np, axis=0)
    #     /
    #     (
    #         norm(input_embed_np, axis=0)*norm(validated_embed_np, axis=0)
    #     )
    # )

    # input_x_raw_output_cosine = (
    #     np.sum(input_embed_np*raw_embed_np, axis=0)
    #     /
    #     (
    #         norm(input_embed_np, axis=0)*norm(raw_embed_np, axis=0)
    #     )
    # )

    # guard_span.set_attribute(
    #     "raw_output_x_validated_output_cosine",
    #     float(str(raw_output_x_validated_output_cosine))
    # )
    # guard_span.set_attribute(
    #     "input_x_validated_output_cosine",
    #     float(str(input_x_validated_output_cosine))
    # )
    # guard_span.set_attribute(
    #     "input_x_raw_output_cosine",
    #     float(str(input_x_raw_output_cosine))
    # )


def trace_stream_guard(
    guard_span: Span,
    result: Iterator[ValidationOutcome[OT]],
    history: Stack[Call],
) -> Iterator[ValidationOutcome[OT]]:
    next_exists = True
    while next_exists:
        try:
            res = next(result)  # type: ignore
            # FIXME: This should only be called once;
            # Accumulate the validated output and call at the end
            add_guard_attributes(guard_span, history, res)
            add_user_attributes(guard_span)
            yield res
        except StopIteration:
            next_exists = False


def trace_guard_execution(
    guard_name: str,
    history: Stack[Call],
    _execute_fn: Callable[
        ..., Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]
    ],
    tracer: Optional[Tracer] = None,
    *args,
    **kwargs,
) -> Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]:
    if not settings.disable_tracing:
        current_otel_context = context.get_current()
        tracer = tracer or trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)

        with tracer.start_as_current_span(
            name="guard",  # type: ignore
            context=current_otel_context,  # type: ignore
        ) as guard_span:
            guard_span.set_attribute("guardrails.version", GUARDRAILS_VERSION)
            guard_span.set_attribute("type", "guardrails/guard")
            guard_span.set_attribute("guard.name", guard_name)

            try:
                result = _execute_fn(*args, **kwargs)
                if isinstance(result, Iterator) and not isinstance(
                    result, ValidationOutcome
                ):
                    return trace_stream_guard(guard_span, result, history)
                add_guard_attributes(guard_span, history, result)
                add_user_attributes(guard_span)
                return result
            except Exception as e:
                guard_span.set_status(status=StatusCode.ERROR, description=str(e))
                raise e
    else:
        return _execute_fn(*args, **kwargs)


async def trace_async_stream_guard(
    guard_span: Span,
    result: AsyncIterator[ValidationOutcome[OT]],
    history: Stack[Call],
) -> AsyncIterator[ValidationOutcome[OT]]:
    next_exists = True
    while next_exists:
        try:
            res = await anext(result)  # type: ignore
            if not guard_span.is_recording():
                # Assuming you have a tracer instance
                tracer = get_tracer(__name__)
                # Create a new span and link it to the previous span
                with tracer.start_as_current_span(
                    "new_guard_span",  # type: ignore
                    links=[Link(guard_span.get_span_context())],
                ) as new_span:
                    guard_span = new_span

                    add_guard_attributes(guard_span, history, res)
                    add_user_attributes(guard_span)
            yield res
        except StopIteration:
            next_exists = False
        except StopAsyncIteration:
            next_exists = False


async def trace_async_guard_execution(
    guard_name: str,
    history: Stack[Call],
    _execute_fn: Callable[
        ...,
        Coroutine[
            Any,
            Any,
            Union[
                ValidationOutcome[OT],
                Awaitable[ValidationOutcome[OT]],
                AsyncIterator[ValidationOutcome[OT]],
            ],
        ],
    ],
    tracer: Optional[Tracer] = None,
    *args,
    **kwargs,
) -> Union[
    ValidationOutcome[OT],
    Awaitable[ValidationOutcome[OT]],
    AsyncIterator[ValidationOutcome[OT]],
]:
    if not settings.disable_tracing:
        current_otel_context = context.get_current()
        tracer = tracer or trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)

        with tracer.start_as_current_span(
            name="guard",  # type: ignore
            context=current_otel_context,  # type: ignore
        ) as guard_span:
            guard_span.set_attribute("guardrails.version", GUARDRAILS_VERSION)
            guard_span.set_attribute("type", "guardrails/guard")
            guard_span.set_attribute("guard.name", guard_name)

            try:
                result = await _execute_fn(*args, **kwargs)
                if isinstance(result, AsyncIterator):
                    return trace_async_stream_guard(guard_span, result, history)

                res = result
                if inspect.isawaitable(result):
                    res = await result
                add_guard_attributes(guard_span, history, res)  # type: ignore
                add_user_attributes(guard_span)
                return res
            except Exception as e:
                guard_span.set_status(status=StatusCode.ERROR, description=str(e))
                add_user_attributes(guard_span)
                raise e
    else:
        return await _execute_fn(*args, **kwargs)
