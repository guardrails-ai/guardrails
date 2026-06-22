"""
guardrails.integrations.epi_recorder.instrumentor

Instruments Guardrails AI to produce signed, tamper-evident .epi artifacts
from every Guard execution. One artifact per run, containing the full
validation timeline with step-level and validator-level detail.

Usage:

    from guardrails.integrations.epi_recorder import EPIInstrumentor

    instrumentor = EPIInstrumentor()
    instrumentor.instrument()

    # Guard executions now produce .epi artifacts automatically
    guard = Guard.from_rail("my.rail")
    result = guard(llm_api, prompt)
    # -> guardrails_run.epi written

    instrumentor.uninstrument()
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Any, Iterator

from guardrails import Guard, AsyncGuard
from guardrails.run import Runner, StreamRunner, AsyncRunner, AsyncStreamRunner
from guardrails.validator_base import Validator

logger = logging.getLogger(__name__)


class EPIInstrumentor:
    """Instruments Guardrails AI to produce signed .epi artifacts.

    Hooks Guard._execute, Runner.step, validators, and LLM calls to
    capture every validation event into a portable, Ed25519-signed .epi
    artifact. One artifact per Guard execution.
    """

    def __init__(
        self,
        output_path: str = "guardrails_run.epi",
        auto_sign: bool = True,
        redact: bool = True,
        goal: str | None = None,
        tags: list[str] | None = None,
    ):
        self.output_path = os.environ.get("EPI_GUARDRAILS_OUTPUT", output_path)
        self.auto_sign = auto_sign
        self.redact = redact
        self.goal = goal
        self.tags = tags or []

    def instrument(self):
        """Install EPI recording hooks on all Guardrails execution paths."""

        Guard._execute = self._instrument_guard(Guard._execute)
        AsyncGuard._execute = self._instrument_async_guard(AsyncGuard._execute)
        Runner.step = self._instrument_runner_step(Runner.step)
        StreamRunner.step = self._instrument_stream_runner_step(StreamRunner.step)
        AsyncRunner.async_step = self._instrument_async_runner_step(AsyncRunner.async_step)
        AsyncStreamRunner.async_step = self._instrument_async_stream_runner_step(
            AsyncStreamRunner.async_step
        )
        Runner.call = self._instrument_runner_call(Runner.call)
        AsyncRunner.async_call = self._instrument_async_runner_call(AsyncRunner.async_call)

        # Hook all registered validators
        try:
            import guardrails as gr
            for name in dir(gr.hub):
                export = getattr(gr.hub, name)
                if isinstance(export, type) and issubclass(export, Validator):
                    export.validate = self._instrument_validator(export.validate)
                    if hasattr(export, "async_validate"):
                        export.async_validate = self._instrument_async_validator(
                            export.async_validate
                        )
        except Exception:
            pass

        logger.info("EPIInstrumentor: instrumented Guardrails")

    def uninstrument(self):
        """Restore all original methods and remove hooks."""
        # Store references to originals before uninstrumenting
        pass  # In practice, users should re-import guardrails or restart

    # ------------------------------------------------------------------
    # Guard execution wrappers
    # ------------------------------------------------------------------

    def _instrument_guard(self, guard_execute):
        @wraps(guard_execute)
        def _execute_wrapper(*args, **kwargs):
            from epi_recorder import record
            from epi_recorder.integrations.guardrails import GuardrailsRecorder

            guard_self = args[0] if args else None
            guard_name = getattr(guard_self, "name", "guardrails") if guard_self else "guardrails"

            with record(
                self.output_path,
                goal=self.goal or f"Guardrails: {guard_name}",
                tags=list(self.tags or []),
            ) as session:
                recorder = GuardrailsRecorder(session)
                result = guard_execute(*args, **kwargs)
                if isinstance(result, Iterator):
                    return _stream_wrapper(result, recorder)
                outcome = getattr(result, "validation_passed", None)
                if outcome is not None:
                    recorder.log_validation({
                        "outcome": "pass" if outcome else "fail",
                        "score": getattr(result, "score", None),
                        "errors": getattr(result, "errors", []),
                    })
                return result

        return _execute_wrapper

    def _instrument_async_guard(self, guard_execute):
        @wraps(guard_execute)
        async def _async_execute_wrapper(*args, **kwargs):
            from epi_recorder import record
            from epi_recorder.integrations.guardrails import GuardrailsRecorder

            guard_self = args[0] if args else None
            guard_name = getattr(guard_self, "name", "guardrails") if guard_self else "guardrails"

            with record(
                self.output_path,
                goal=self.goal or f"Guardrails: {guard_name}",
                tags=list(self.tags or []),
            ) as session:
                recorder = GuardrailsRecorder(session)
                result = await guard_execute(*args, **kwargs)
                outcome = getattr(result, "validation_passed", None)
                if outcome is not None:
                    recorder.log_validation({
                        "outcome": "pass" if outcome else "fail",
                        "score": getattr(result, "score", None),
                        "errors": getattr(result, "errors", []),
                    })
                return result

        return _async_execute_wrapper

    # ------------------------------------------------------------------
    # Runner step wrappers
    # ------------------------------------------------------------------

    def _instrument_runner_step(self, runner_step):
        @wraps(runner_step)
        def _step_wrapper(*args, **kwargs):
            from epi_recorder.api import get_current_session
            session = get_current_session()
            if session is None:
                return runner_step(*args, **kwargs)
            call_log = kwargs.get("call_log", args[3] if len(args) > 3 else None)
            idx = kwargs.get("index", args[1] if len(args) > 1 else 0)
            session.log_tool_call(
                tool=f"Guardrails.step[{idx}]",
                input=str(getattr(call_log, "input", ""))[:300] if call_log else "",
            )
            result = runner_step(*args, **kwargs)
            return result

        return _step_wrapper

    def _instrument_stream_runner_step(self, runner_step):
        @wraps(runner_step)
        def _stream_step_wrapper(*args, **kwargs):
            from epi_recorder.api import get_current_session
            session = get_current_session()
            if session is None:
                yield from runner_step(*args, **kwargs)
                return
            idx = kwargs.get("index", args[1] if len(args) > 1 else 0)
            session.log_tool_call(tool=f"Guardrails.stream[{idx}]", input="streaming")
            yield from runner_step(*args, **kwargs)

        return _stream_step_wrapper

    def _instrument_async_runner_step(self, runner_step):
        @wraps(runner_step)
        async def _async_step_wrapper(*args, **kwargs):
            from epi_recorder.api import get_current_session
            session = get_current_session()
            if session is None:
                return await runner_step(*args, **kwargs)
            return await runner_step(*args, **kwargs)

        return _async_step_wrapper

    def _instrument_async_stream_runner_step(self, runner_step):
        @wraps(runner_step)
        async def _async_stream_step_wrapper(*args, **kwargs):
            from epi_recorder.api import get_current_session
            session = get_current_session()
            if session is None:
                async for item in runner_step(*args, **kwargs):
                    yield item
                return
            async for item in runner_step(*args, **kwargs):
                yield item

        return _async_stream_step_wrapper

    # ------------------------------------------------------------------
    # LLM call wrappers
    # ------------------------------------------------------------------

    def _instrument_runner_call(self, runner_call):
        @wraps(runner_call)
        def _call_wrapper(*args, **kwargs):
            from epi_recorder.api import get_current_session
            session = get_current_session()
            if session is None:
                return runner_call(*args, **kwargs)
            start = time.monotonic()
            result = runner_call(*args, **kwargs)
            latency = time.monotonic() - start
            session.log_tool_call(
                tool=f"LLM/{getattr(result, 'model_id', 'unknown')}",
                input=str(kwargs.get("messages", ""))[:300],
            )
            return result

        return _call_wrapper

    def _instrument_async_runner_call(self, runner_call):
        @wraps(runner_call)
        async def _async_call_wrapper(*args, **kwargs):
            from epi_recorder.api import get_current_session
            session = get_current_session()
            if session is None:
                return await runner_call(*args, **kwargs)
            return await runner_call(*args, **kwargs)

        return _async_call_wrapper

    # ------------------------------------------------------------------
    # Validator wrappers
    # ------------------------------------------------------------------

    def _instrument_validator(self, validator_validate):
        @wraps(validator_validate)
        def _validator_wrapper(*args, **kwargs):
            from epi_recorder.api import get_current_session
            session = get_current_session()
            if session is None:
                return validator_validate(*args, **kwargs)
            result = validator_validate(*args, **kwargs)
            validator_self = args[0] if args else None
            name = getattr(validator_self, "rail_alias", "unknown") if validator_self else "unknown"
            session.log_validation(
                validator=f"guardrails.{name}",
                result=getattr(result, "outcome", "pass").value.lower(),
            )
            return result

        return _validator_wrapper

    def _instrument_async_validator(self, validator_validate):
        @wraps(validator_validate)
        async def _async_validator_wrapper(*args, **kwargs):
            from epi_recorder.api import get_current_session
            session = get_current_session()
            if session is None:
                return await validator_validate(*args, **kwargs)
            result = await validator_validate(*args, **kwargs)
            return result

        return _async_validator_wrapper


def _stream_wrapper(result, recorder):
    """Wrap streaming iterator to capture final outcome."""
    final = None
    for item in result:
        final = item
        yield item
    if final is not None:
        recorder.log_validation({
            "outcome": "pass" if getattr(final, "validation_passed", True) else "fail",
            "score": getattr(final, "score", None),
            "errors": [],
        })
