"""
EPI Recorder instrumentor for Guardrails AI.

Exports Guardrails validation runs as cryptographically signed .epi artifacts.
Follows the OpenInference lifecycle pattern (_instrument / _uninstrument)
and hooks into MLFlow-style integration points.
"""

import logging
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Dict

try:
    from epi_recorder import EpiRecorderSession, get_current_session

    _EPI_AVAILABLE = True
except ImportError:
    _EPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class EPIInstrumentor:
    """Optional instrumentor for exporting Guardrails runs as signed .epi artifacts."""

    def __init__(self, output_dir: str = "./epi-recordings", auto_sign: bool = True):
        self.output_dir = output_dir
        self.auto_sign = auto_sign
        self._originals: Dict[str, Any] = {}
        self._patched = False

    def instrument(self) -> None:
        """Enable EPI instrumentation. Idempotent."""
        self._instrument()

    def uninstrument(self) -> None:
        """Disable EPI instrumentation and restore original functions."""
        self._uninstrument()

    def _instrument(self) -> None:
        if self._patched:
            return
        if not _EPI_AVAILABLE:
            logger.warning("epi-recorder not installed; skipping EPI instrumentation")
            return

        import guardrails as gd

        Guard = gd.Guard
        Runner = gd.run.Runner
        ValidatorServiceBase = gd.validator_service.ValidatorServiceBase

        if not hasattr(Guard, "_execute"):
            logger.warning("Guard._execute not found; skipping EPI instrumentation")
            return

        self._originals["Guard._execute"] = Guard._execute
        Guard._execute = self._wrap_guard_execute(Guard._execute)

        if hasattr(Runner, "step"):
            self._originals["Runner.step"] = Runner.step
            Runner.step = self._wrap_runner_step(Runner.step)

        if hasattr(ValidatorServiceBase, "after_run_validator"):
            self._originals["ValidatorServiceBase.after_run_validator"] = (
                ValidatorServiceBase.after_run_validator
            )
            ValidatorServiceBase.after_run_validator = self._wrap_validator_after_run(
                ValidatorServiceBase.after_run_validator
            )

        self._patched = True
        logger.info("EPI instrumentation enabled")

    def _uninstrument(self) -> None:
        if not self._patched:
            return

        import guardrails as gd

        Guard = gd.Guard
        Runner = gd.run.Runner
        ValidatorServiceBase = gd.validator_service.ValidatorServiceBase

        Guard._execute = self._originals.pop("Guard._execute", Guard._execute)
        if "Runner.step" in self._originals:
            Runner.step = self._originals.pop("Runner.step")
        if "ValidatorServiceBase.after_run_validator" in self._originals:
            ValidatorServiceBase.after_run_validator = self._originals.pop(
                "ValidatorServiceBase.after_run_validator"
            )

        self._patched = False
        logger.info("EPI instrumentation disabled")

    def _guard_name(self, args: tuple) -> str:
        guard = args[0] if args else None
        return getattr(guard, "name", "unknown") if guard else "unknown"

    def _wrap_guard_execute(self, original):
        @wraps(original)
        def wrapper(*args, **kwargs):
            guard_name = self._guard_name(args)
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            output_path = (
                Path(self.output_dir)
                / f"guardrails_{guard_name}_{uuid.uuid4().hex[:8]}.epi"
            )
            session = EpiRecorderSession(
                output_path=output_path,
                workflow_name=guard_name,
                auto_sign=self.auto_sign,
            )
            with session:
                session.log_step(
                    "guardrails.execution.start", {"guard_name": guard_name}
                )
                try:
                    return original(*args, **kwargs)
                finally:
                    session.log_step("guardrails.execution.end", {})

        return wrapper

    def _wrap_runner_step(self, original):
        @wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            session = get_current_session()
            if session and result is not None:
                status = "unknown"
                try:
                    outputs = getattr(result, "outputs", None)
                    if outputs:
                        vr = getattr(outputs, "validation_response", None)
                        if vr and hasattr(vr, "passed"):
                            status = "pass" if vr.passed else "fail"
                        guarded = getattr(outputs, "guarded_output", None)
                        parsed = getattr(outputs, "parsed_output", None)
                        if (
                            guarded is not None
                            and guarded != parsed
                            and status != "fail"
                        ):
                            status = "corrected"
                except Exception:
                    pass
                session.log_step("guardrails.validation", {"status": status})
            return result

        return wrapper

    def _wrap_validator_after_run(self, original):
        @wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            session = get_current_session()
            if session:
                validator = args[0] if args else None
                validator_result = None
                if len(args) >= 3:
                    validator_result = args[2]
                elif "result" in kwargs:
                    validator_result = kwargs["result"]
                if validator_result is not None:
                    try:
                        name = getattr(validator, "rail_alias", "unknown")
                        outcome = getattr(validator_result, "outcome", "unknown")
                    except Exception:
                        name, outcome = "unknown", "unknown"
                    session.log_step(
                        "guardrails.validator.result",
                        {"validator": name, "outcome": outcome},
                    )
            return result

        return wrapper
