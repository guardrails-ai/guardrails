from functools import wraps
import inspect
import sys
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Union,
)

from guardrails import Guard, AsyncGuard, settings
from guardrails.classes.validation.validation_result import ValidationResult
from guardrails.run import Runner, StreamRunner, AsyncRunner, AsyncStreamRunner
from guardrails.validator_base import Validator
from guardrails.version import GUARDRAILS_VERSION
from guardrails.telemetry.guard_tracing import (
    add_guard_attributes,
    trace_stream_guard,
    trace_async_stream_guard,
)
from guardrails.telemetry.runner_tracing import add_step_attributes, add_call_attributes
from guardrails.telemetry.validator_tracing import add_validator_attributes
from guardrails.classes.generic.stack import Stack
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.history.iteration import Iteration
from guardrails.classes.output_type import OT
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.utils.safe_get import safe_get

try:
    import mlflow
    import mlflow.tracing
    import mlflow.tracing.provider
    from mlflow.entities.span_status import SpanStatusCode
except ImportError:
    raise ImportError("Please install mlflow to use this instrumentor")


if sys.version_info.minor < 10:
    from guardrails.utils.polyfills import anext


# TODO: Abstract these methods and common logic into a base class
#   that can be extended by other instrumentors
class MlFlowInstrumentor:
    """Instruments Guardrails to send traces to MLFlow."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        # Disable legacy OTEL tracing to avoid duplicate spans
        settings.disable_tracing = True

    def instrument(self):
        if not mlflow.tracing.provider._is_enabled():
            mlflow.tracing.enable()
        mlflow.set_experiment(self.experiment_name)

        wrapped_guard_execute = self._instrument_guard(Guard._execute)
        setattr(Guard, "_execute", wrapped_guard_execute)

        wrapped_async_guard_execute = self._instrument_async_guard(AsyncGuard._execute)
        setattr(AsyncGuard, "_execute", wrapped_async_guard_execute)

        wrapped_runner_step = self._instrument_runner_step(Runner.step)
        setattr(Runner, "step", wrapped_runner_step)

        wrapped_stream_runner_step = self._instrument_stream_runner_step(
            StreamRunner.step
        )
        setattr(StreamRunner, "step", wrapped_stream_runner_step)

        wrapped_async_runner_step = self._instrument_async_runner_step(
            AsyncRunner.async_step
        )
        setattr(AsyncRunner, "async_step", wrapped_async_runner_step)

        wrapped_async_stream_runner_step = self._instrument_async_stream_runner_step(
            AsyncStreamRunner.async_step  # type: ignore
        )
        setattr(AsyncStreamRunner, "async_step", wrapped_async_stream_runner_step)

        wrapped_runner_call = self._instrument_runner_call(Runner.call)
        setattr(Runner, "call", wrapped_runner_call)

        wrapped_async_runner_call = self._instrument_async_runner_call(
            AsyncRunner.async_call
        )
        setattr(AsyncRunner, "async_call", wrapped_async_runner_call)

        import guardrails

        validators = guardrails.hub.__dir__()  # type: ignore

        for validator_name in validators:
            export = getattr(guardrails.hub, validator_name)  # type: ignore
            if isinstance(export, type) and issubclass(export, Validator):
                wrapped_validator_validate = self._instrument_validator_validate(
                    export.validate
                )
                setattr(export, "validate", wrapped_validator_validate)
                setattr(guardrails.hub, validator_name, export)  # type: ignore

    def _instrument_guard(
        self,
        guard_execute: Callable[
            ..., Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]
        ],
    ):
        @wraps(guard_execute)
        def _guard_execute_wrapper(
            *args, **kwargs
        ) -> Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]:
            with mlflow.start_span(
                name="guardrails/guard",
                span_type="guard",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard",
                },
            ) as guard_span:
                guard_self = args[0]
                history = Stack()

                if guard_self is not None and isinstance(guard_self, Guard):
                    guard_span.set_attribute("guard.name", guard_self.name)
                    history = guard_self.history

                try:
                    result = guard_execute(*args, **kwargs)
                    if isinstance(result, Iterator) and not isinstance(
                        result, ValidationOutcome
                    ):
                        return trace_stream_guard(guard_span, result, history)  # type: ignore
                    add_guard_attributes(guard_span, history, result)  # type: ignore
                    return result
                except Exception as e:
                    guard_span.set_status(status=SpanStatusCode.ERROR)
                    raise e

        return _guard_execute_wrapper

    def _instrument_async_guard(
        self,
        guard_execute: Callable[
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
    ):
        @wraps(guard_execute)
        async def _async_guard_execute_wrapper(
            *args, **kwargs
        ) -> Union[
            ValidationOutcome[OT],
            Awaitable[ValidationOutcome[OT]],
            AsyncIterator[ValidationOutcome[OT]],
        ]:
            with mlflow.start_span(
                name="guardrails/guard",
                span_type="guard",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard",
                    "async": True,
                },
            ) as guard_span:
                guard_self = args[0]
                history = Stack()

                if guard_self is not None and isinstance(guard_self, Guard):
                    guard_span.set_attribute("guard.name", guard_self.name)
                    history = guard_self.history

                try:
                    result = await guard_execute(*args, **kwargs)
                    if isinstance(result, AsyncIterator):
                        return trace_async_stream_guard(guard_span, result, history)  # type: ignore
                    res = result
                    if inspect.isawaitable(result):
                        res = await result
                    add_guard_attributes(guard_span, history, res)  # type: ignore
                    return res
                except Exception as e:
                    guard_span.set_status(status=SpanStatusCode.ERROR)
                    raise e

        return _async_guard_execute_wrapper

    def _instrument_runner_step(self, runner_step: Callable[..., Iteration]):
        @wraps(runner_step)
        def trace_step_wrapper(*args, **kwargs) -> Iteration:
            with mlflow.start_span(
                name="guardrails/guard/step",
                span_type="step",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard/step",
                },
            ) as step_span:
                try:
                    response = runner_step(*args, **kwargs)
                    add_step_attributes(step_span, response, *args, **kwargs)  # type: ignore
                    return response
                except Exception as e:
                    step_span.set_status(status=SpanStatusCode.ERROR)
                    add_step_attributes(step_span, None, *args, **kwargs)  # type: ignore
                    raise e

        return trace_step_wrapper

    def _instrument_stream_runner_step(
        self, runner_step: Callable[..., Iterator[ValidationOutcome[OT]]]
    ):
        @wraps(runner_step)
        def trace_stream_step_wrapper(
            *args, **kwargs
        ) -> Iterator[ValidationOutcome[OT]]:
            with mlflow.start_span(
                name="guardrails/guard/step",
                span_type="step",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard/step",
                    "stream": True,
                },
            ) as step_span:
                exception = None
                try:
                    gen = runner_step(*args, **kwargs)
                    next_exists = True
                    while next_exists:
                        try:
                            res = next(gen)
                            yield res
                        except StopIteration:
                            next_exists = False
                except Exception as e:
                    step_span.set_status(status=SpanStatusCode.ERROR)
                    exception = e
                finally:
                    call = safe_get(args, 8, kwargs.get("call_log", None))
                    iteration = call.iterations.last if call else None
                    add_step_attributes(step_span, iteration, *args, **kwargs)  # type: ignore
                    if exception:
                        raise exception

        return trace_stream_step_wrapper

    def _instrument_async_runner_step(
        self, runner_step: Callable[..., Awaitable[Iteration]]
    ):
        @wraps(runner_step)
        async def trace_async_step_wrapper(*args, **kwargs) -> Iteration:
            with mlflow.start_span(
                name="guardrails/guard/step",
                span_type="step",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard/step",
                    "async": True,
                },
            ) as step_span:
                try:
                    response = await runner_step(*args, **kwargs)
                    add_step_attributes(step_span, response, *args, **kwargs)  # type: ignore
                    return response
                except Exception as e:
                    step_span.set_status(status=SpanStatusCode.ERROR)
                    add_step_attributes(step_span, None, *args, **kwargs)  # type: ignore
                    raise e

        return trace_async_step_wrapper

    def _instrument_async_stream_runner_step(
        self, runner_step: Callable[..., AsyncIterator[ValidationOutcome[OT]]]
    ) -> Callable[..., AsyncIterator[ValidationOutcome[OT]]]:
        @wraps(runner_step)
        async def trace_async_stream_step_wrapper(
            *args, **kwargs
        ) -> AsyncIterator[ValidationOutcome[OT]]:
            with mlflow.start_span(
                name="guardrails/guard/step",
                span_type="step",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard/step",
                    "async": True,
                    "stream": True,
                },
            ) as step_span:
                exception = None
                try:
                    gen = runner_step(*args, **kwargs)
                    next_exists = True
                    while next_exists:
                        try:
                            res = await anext(gen)
                            yield res
                        except StopIteration:
                            next_exists = False
                        except StopAsyncIteration:
                            next_exists = False
                except Exception as e:
                    step_span.set_status(status=SpanStatusCode.ERROR)
                    exception = e
                finally:
                    call = safe_get(args, 3, kwargs.get("call_log", None))
                    iteration = call.iterations.last if call else None
                    add_step_attributes(step_span, iteration, *args, **kwargs)  # type: ignore
                    if exception:
                        raise exception

        return trace_async_stream_step_wrapper

    def _instrument_runner_call(self, runner_call: Callable[..., LLMResponse]):
        @wraps(runner_call)
        def trace_call_wrapper(*args, **kwargs):
            with mlflow.start_span(
                name="guardrails/guard/step/call",
                span_type="LLM",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard/step/call",
                },
            ) as call_span:
                try:
                    response = runner_call(*args, **kwargs)
                    add_call_attributes(call_span, response, *args, **kwargs)  # type: ignore
                    return response
                except Exception as e:
                    call_span.set_status(status=SpanStatusCode.ERROR)
                    add_call_attributes(call_span, None, *args, **kwargs)  # type: ignore
                    raise e

        return trace_call_wrapper

    def _instrument_async_runner_call(
        self, runner_call: Callable[..., Awaitable[LLMResponse]]
    ):
        @wraps(runner_call)
        async def trace_async_call_wrapper(*args, **kwargs):
            with mlflow.start_span(
                name="guardrails/guard/step/call",
                span_type="LLM",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard/step/call",
                    "async": True,
                },
            ) as call_span:
                try:
                    response = await runner_call(*args, **kwargs)
                    add_call_attributes(call_span, response, *args, **kwargs)  # type: ignore
                    return response
                except Exception as e:
                    call_span.set_status(status=SpanStatusCode.ERROR)
                    add_call_attributes(call_span, None, *args, **kwargs)  # type: ignore
                    raise e

        return trace_async_call_wrapper

    def _instrument_validator_validate(
        self, validator_validate: Callable[..., ValidationResult]
    ):
        @wraps(validator_validate)
        def trace_validator_wrapper(*args, **kwargs):
            validator_name = "validator"
            obj_id = id(validator_validate)
            on_fail_descriptor = "unknown"
            init_kwargs = {}
            validation_session_id = "unknown"

            validator_self = args[0]
            if validator_self is not None and isinstance(validator_self, Validator):
                validator_name = validator_self.rail_alias
                obj_id = id(validator_self)
                on_fail_descriptor = validator_self.on_fail_descriptor
                init_kwargs = validator_self._kwargs

            validator_span_name = f"{validator_name}.validate"
            with mlflow.start_span(
                name=validator_span_name,
                span_type="validator",
                attributes={
                    "guardrails.version": GUARDRAILS_VERSION,
                    "type": "guardrails/guard/step/validator",
                },
            ) as validator_span:
                try:
                    resp = validator_validate(*args, **kwargs)
                    add_validator_attributes(
                        *args,
                        validator_span=validator_span,  # type: ignore
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
                    validator_span.set_status(status=SpanStatusCode.ERROR)
                    add_validator_attributes(
                        *args,
                        validator_span=validator_span,  # type: ignore
                        validator_name=validator_name,
                        obj_id=obj_id,
                        on_fail_descriptor=on_fail_descriptor,
                        result=None,
                        init_kwargs=init_kwargs,
                        validation_session_id=validation_session_id,
                        **kwargs,
                    )
                    raise e

        return trace_validator_wrapper
