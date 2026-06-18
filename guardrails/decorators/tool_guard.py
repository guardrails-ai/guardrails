import functools
import inspect
import uuid
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    cast,
    Tuple,
)

try:
    from openinference.semconv.trace import SpanAttributes  # type: ignore
except ImportError:
    SpanAttributes = None

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from guardrails.validator_base import Validator
from guardrails_ai.types import FailResult
from guardrails.errors import ValidationError
from guardrails.settings import settings
from guardrails.telemetry.open_inference import trace_operation
from guardrails.telemetry.common import add_user_attributes
from guardrails.version import GUARDRAILS_VERSION
from guardrails.types.on_fail import OnFailAction


class ToolValidationError(ValidationError):
    """Raised when validation of tool inputs or outputs fails."""

    def __init__(self, message: str, failures: Dict[str, Any]):
        super().__init__(message)
        self.failures = failures


class ToolValidationResult:
    """Result of tool validation containing input/output validation outcomes."""

    def __init__(
        self,
        tool_id: str,
        tool_name: str,
        input_validation_passed: bool,
        output_validation_passed: bool,
        input_failures: Optional[Dict[str, Any]] = None,
        output_failures: Optional[Dict[str, Any]] = None,
        validated_inputs: Optional[Dict[str, Any]] = None,
        validated_output: Optional[Any] = None,
        execution_time_ms: float = 0.0,
        input_on_fail_action: Optional[OnFailAction] = None,
        output_on_fail_action: Optional[OnFailAction] = None,
    ):
        """Initialize a ToolValidationResult.

        Args:
            tool_id: Unique identifier for the tool execution.
            tool_name: Name of the tool.
            input_validation_passed: Whether input validation succeeded.
            output_validation_passed: Whether output validation succeeded.
            input_failures: Dict of input validation failures by parameter.
            output_failures: Dict of output validation failures.
            validated_inputs: The validated input parameters.
            validated_output: The validated output value.
            execution_time_ms: Time taken to execute the tool.
            input_on_fail_action: OnFailAction taken for input failures.
            output_on_fail_action: OnFailAction taken for output failures.
        """
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.input_validation_passed = input_validation_passed
        self.output_validation_passed = output_validation_passed
        self.input_failures = input_failures or {}
        self.output_failures = output_failures or {}
        self.validated_inputs = validated_inputs or {}
        self.validated_output = validated_output
        self.execution_time_ms = execution_time_ms
        self.input_on_fail_action = input_on_fail_action
        self.output_on_fail_action = output_on_fail_action

    @property
    def validation_passed(self) -> bool:
        """Check if all validations passed."""
        return self.input_validation_passed and self.output_validation_passed

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "validation_passed": self.validation_passed,
            "input_validation_passed": self.input_validation_passed,
            "output_validation_passed": self.output_validation_passed,
            "input_failures": self.input_failures,
            "output_failures": self.output_failures,
            "execution_time_ms": self.execution_time_ms,
            "input_on_fail_action": (
                self.input_on_fail_action.value
                if self.input_on_fail_action
                else None
            ),
            "output_on_fail_action": (
                self.output_on_fail_action.value
                if self.output_on_fail_action
                else None
            ),
        }


class ToolGuard:
    """Manager for tool execution with input/output validation.

    Tracks tool executions, maintains validation history, and integrates with
    OpenTelemetry/OpenInference for observability.
    """

    def __init__(self, name: str = "default"):
        """Initialize ToolGuard manager.

        Args:
            name: Name for this ToolGuard instance.
        """
        self.name = name
        self.execution_history: List[ToolValidationResult] = []
        self._tool_validators: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        tool_name: str,
        input_validators: Optional[Union[List[Validator], Dict[str, List[Validator]]]] = None,
        output_validators: Optional[List[Validator]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a tool with its validators.

        Args:
            tool_name: Name of the tool.
            input_validators: Validators for tool inputs.
            output_validators: Validators for tool outputs.
            metadata: Optional metadata for validators.

        Returns:
            Tool ID for tracking.
        """
        tool_id = str(uuid.uuid4())
        self._tool_validators[tool_id] = {
            "name": tool_name,
            "input_validators": input_validators,
            "output_validators": output_validators,
            "metadata": metadata or {},
        }
        return tool_id

    def _record_execution(self, result: ToolValidationResult) -> None:
        """Record a tool execution result in history.

        Args:
            result: ToolValidationResult to record.
        """
        self.execution_history.append(result)

    def get_execution_history(
        self, tool_name: Optional[str] = None
    ) -> List[ToolValidationResult]:
        """Get execution history, optionally filtered by tool name.

        Args:
            tool_name: Optional tool name to filter by.

        Returns:
            List of ToolValidationResult instances.
        """
        if tool_name:
            return [
                result
                for result in self.execution_history
                if result.tool_name == tool_name
            ]
        return self.execution_history

    def get_statistics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get validation statistics.

        Args:
            tool_name: Optional tool name to filter by.

        Returns:
            Dict containing validation statistics.
        """
        history = self.get_execution_history(tool_name)
        if not history:
            return {
                "total_executions": 0,
                "validation_pass_rate": 0.0,
                "input_pass_rate": 0.0,
                "output_pass_rate": 0.0,
            }

        total = len(history)
        passed = sum(1 for r in history if r.validation_passed)
        input_passed = sum(1 for r in history if r.input_validation_passed)
        output_passed = sum(1 for r in history if r.output_validation_passed)

        return {
            "total_executions": total,
            "validation_pass_rate": passed / total if total > 0 else 0.0,
            "input_pass_rate": input_passed / total if total > 0 else 0.0,
            "output_pass_rate": output_passed / total if total > 0 else 0.0,
        }


def guard_tool(
    input_validators: Optional[
        Union[List[Validator], Dict[str, List[Validator]]]
    ] = None,
    output_validators: Optional[List[Validator]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tool_guard: Optional[ToolGuard] = None,
    on_input_fail: Optional[Union[str, OnFailAction]] = None,
    on_output_fail: Optional[Union[str, OnFailAction]] = None,
):
    """Decorator to apply Guardrails validators to a tool's inputs and outputs.

    This decorator validates tool inputs before execution and outputs after
    execution, integrating with OpenTelemetry for observability. Supports
    multiple failure handling strategies via OnFailAction.

    Args:
        input_validators: A list of validators applied to the arguments dict,
            or a dictionary mapping parameter names to lists of validators.
        output_validators: A list of validators applied to the function's return value.
        metadata: Optional metadata passed to the validators.
        tool_guard: Optional ToolGuard instance for tracking execution.
        on_input_fail: OnFailAction to apply when input validation fails.
            Valid values: "exception", "fix", "filter", "refrain", "noop", etc.
            Defaults to "exception".
        on_output_fail: OnFailAction to apply when output validation fails.
            Valid values: "exception", "fix", "filter", "refrain", "noop", etc.
            Defaults to "exception".

    Returns:
        Decorated function that performs validation.

    Raises:
        ToolValidationError: When validation fails and on_fail action is "exception".

    Example:
        >>> from guardrails.validators import ValidEmail
        >>> from guardrails.types.on_fail import OnFailAction
        >>> 
        >>> @guard_tool(
        ...     input_validators={"email": [ValidEmail()]},
        ...     output_validators=[SomeOutputValidator()],
        ...     on_input_fail=OnFailAction.FIX,
        ...     on_output_fail=OnFailAction.EXCEPTION
        ... )
        ... def send_email(email: str, message: str) -> dict:
        ...     return {"status": "sent"}
    """
    metadata = metadata or {}
    on_input_fail = OnFailAction.get(on_input_fail, OnFailAction.EXCEPTION)
    on_output_fail = OnFailAction.get(on_output_fail, OnFailAction.EXCEPTION)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)
        func_name = func.__name__
        tool_id = str(uuid.uuid4())

        # Register tool with ToolGuard if provided
        if tool_guard:
            tool_guard.register_tool(
                func_name,
                input_validators=input_validators,
                output_validators=output_validators,
                metadata=metadata,
            )

        def apply_on_fail_action(
            action: OnFailAction,
            failures: Dict[str, Any],
            value: Any,
            is_input: bool = True,
        ) -> Tuple[bool, Optional[Any]]:
            """Apply OnFailAction strategy.

            Args:
                action: The OnFailAction to apply.
                failures: The validation failures.
                value: The value that failed validation.
                is_input: Whether this is input or output validation.

            Returns:
                Tuple of (should_continue, processed_value)
            """
            if action == OnFailAction.EXCEPTION:
                return False, None
            elif action == OnFailAction.FIX:
                # FIX action requires the validator to have set fix_value
                return True, value
            elif action == OnFailAction.FILTER:
                # FILTER action removes invalid values
                if isinstance(value, dict):
                    return True, {}
                elif isinstance(value, list):
                    return True, []
                else:
                    return True, None
            elif action == OnFailAction.REFRAIN:
                # REFRAIN returns empty/default value
                return True, None
            elif action == OnFailAction.NOOP:
                # NOOP does nothing but allows continuation
                return True, value
            elif action == OnFailAction.REASK:
                # REASK would require LLM interaction - for tools, treat as exception
                return False, None
            else:
                # Default to exception
                return False, None

        def run_input_validation(
            *args, **kwargs
        ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
            """Run input validation and return (validated_params, passed, failures)."""
            if not input_validators:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return dict(bound.arguments), True, {}

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments

            failures = {}
            validated_params = dict(params)

            if isinstance(input_validators, list):
                for val in input_validators:
                    res = val.validate(validated_params, metadata)
                    if isinstance(res, FailResult):
                        failures["__all__"] = failures.get("__all__", []) + [
                            res.error_message
                        ]

            elif isinstance(input_validators, dict):
                for param_name, validators in input_validators.items():
                    if param_name in validated_params:
                        val_input = validated_params[param_name]
                        for val in validators:
                            res = val.validate(val_input, metadata)
                            if isinstance(res, FailResult):
                                failures[param_name] = failures.get(
                                    param_name, []
                                ) + [res.error_message]

            return validated_params, not bool(failures), failures

        def run_output_validation(
            result: Any,
        ) -> Tuple[Any, bool, Dict[str, Any]]:
            """Run output validation and return (validated_result, passed, failures)."""
            if not output_validators:
                return result, True, {}

            failures = []
            validated_result = result
            for val in output_validators:
                res = val.validate(validated_result, metadata)
                if isinstance(res, FailResult):
                    failures.append(res.error_message)

            return (
                validated_result,
                not bool(failures),
                {"output": failures} if failures else {},
            )

        def trace_tool_execution(
            tool_name: str,
            validated_inputs: Dict[str, Any],
            input_validation_passed: bool,
            output_validation_passed: bool,
            input_failures: Optional[Dict[str, Any]] = None,
            output_failures: Optional[Dict[str, Any]] = None,
            validated_output: Optional[Any] = None,
        ) -> None:
            """Record tool execution metrics and traces."""
            if not settings.disable_tracing:
                tracer = trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)
                with tracer.start_as_current_span(
                    name=f"tool/{tool_name}"
                ) as tool_span:
                    tool_span.set_attribute("type", "guardrails/tool")
                    tool_span.set_attribute("tool.name", tool_name)
                    tool_span.set_attribute("tool.id", tool_id)
                    tool_span.set_attribute(
                        "tool.input_validation_passed", input_validation_passed
                    )
                    tool_span.set_attribute(
                        "tool.output_validation_passed", output_validation_passed
                    )

                    if SpanAttributes is not None:
                        tool_span.set_attribute(
                            SpanAttributes.OPENINFERENCE_SPAN_KIND, "TOOL"
                        )

                    try:
                        trace_operation(
                            input_mime_type="application/json",
                            input_value=validated_inputs,
                            output_mime_type="application/json",
                            output_value=validated_output,
                        )
                        add_user_attributes(tool_span)

                        if input_failures:
                            tool_span.set_attribute(
                                "tool.input_failures",
                                str(input_failures),
                            )
                        if output_failures:
                            tool_span.set_attribute(
                                "tool.output_failures",
                                str(output_failures),
                            )

                        if not (input_validation_passed and output_validation_passed):
                            tool_span.set_status(
                                status=StatusCode.ERROR,
                                description="Tool validation failed",
                            )

                    except Exception as e:
                        tool_span.set_status(
                            status=StatusCode.ERROR, description=str(e)
                        )
                        raise e

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                input_failures = {}
                output_failures = {}
                input_validation_passed = True
                output_validation_passed = True
                validated_inputs = {}
                validated_output = None

                try:
                    validated_inputs, input_validation_passed, input_failures = (
                        run_input_validation(*args, **kwargs)
                    )
                    if not input_validation_passed:
                        should_continue, _ = apply_on_fail_action(
                            on_input_fail,
                            input_failures,
                            validated_inputs,
                            is_input=True,
                        )
                        if not should_continue:
                            trace_tool_execution(
                                func_name,
                                {},
                                input_validation_passed,
                                False,
                                input_failures=input_failures,
                            )
                            if tool_guard:
                                result = ToolValidationResult(
                                    tool_id=tool_id,
                                    tool_name=func_name,
                                    input_validation_passed=False,
                                    output_validation_passed=False,
                                    input_failures=input_failures,
                                    input_on_fail_action=on_input_fail,
                                    execution_time_ms=time.time() - start_time,
                                )
                                tool_guard._record_execution(result)
                            raise ToolValidationError(
                                "Tool input validation failed", input_failures
                            )
                        # If should_continue is True (NOOP), mark as failed but continue
                        input_validation_passed = False
                except ToolValidationError:
                    raise

                bound = sig.bind_partial()
                bound.arguments.update(validated_inputs)

                try:
                    result = await func(*bound.args, **bound.kwargs)
                    validated_output, output_validation_passed, output_failures = (
                        run_output_validation(result)
                    )
                    if not output_validation_passed:
                        should_continue, _ = apply_on_fail_action(
                            on_output_fail,
                            output_failures,
                            validated_output,
                            is_input=False,
                        )
                        if not should_continue:
                            trace_tool_execution(
                                func_name,
                                validated_inputs,
                                input_validation_passed,
                                output_validation_passed,
                                input_failures=input_failures,
                                output_failures=output_failures,
                            )
                            if tool_guard:
                                exec_result = ToolValidationResult(
                                    tool_id=tool_id,
                                    tool_name=func_name,
                                    input_validation_passed=input_validation_passed,
                                    output_validation_passed=False,
                                    input_failures=input_failures,
                                    output_failures=output_failures,
                                    validated_inputs=validated_inputs,
                                    input_on_fail_action=on_input_fail,
                                    output_on_fail_action=on_output_fail,
                                    execution_time_ms=time.time() - start_time,
                                )
                                tool_guard._record_execution(exec_result)
                            raise ToolValidationError(
                                "Tool output validation failed", output_failures
                            )
                        # If should_continue is True (NOOP), mark as failed but continue
                        output_validation_passed = False
                except ToolValidationError:
                    raise

                trace_tool_execution(
                    func_name,
                    validated_inputs,
                    input_validation_passed,
                    output_validation_passed,
                    validated_output=validated_output,
                )

                if tool_guard:
                    exec_result = ToolValidationResult(
                        tool_id=tool_id,
                        tool_name=func_name,
                        input_validation_passed=input_validation_passed,
                        output_validation_passed=output_validation_passed,
                        validated_inputs=validated_inputs,
                        validated_output=validated_output,
                        input_on_fail_action=on_input_fail,
                        output_on_fail_action=on_output_fail,
                        execution_time_ms=time.time() - start_time,
                    )
                    tool_guard._record_execution(exec_result)

                return validated_output

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                input_failures = {}
                output_failures = {}
                input_validation_passed = True
                output_validation_passed = True
                validated_inputs = {}
                validated_output = None

                try:
                    validated_inputs, input_validation_passed, input_failures = (
                        run_input_validation(*args, **kwargs)
                    )
                    if not input_validation_passed:
                        should_continue, _ = apply_on_fail_action(
                            on_input_fail,
                            input_failures,
                            validated_inputs,
                            is_input=True,
                        )
                        if not should_continue:
                            trace_tool_execution(
                                func_name,
                                {},
                                input_validation_passed,
                                False,
                                input_failures=input_failures,
                            )
                            if tool_guard:
                                result = ToolValidationResult(
                                    tool_id=tool_id,
                                    tool_name=func_name,
                                    input_validation_passed=False,
                                    output_validation_passed=False,
                                    input_failures=input_failures,
                                    input_on_fail_action=on_input_fail,
                                    execution_time_ms=time.time() - start_time,
                                )
                                tool_guard._record_execution(result)
                            raise ToolValidationError(
                                "Tool input validation failed", input_failures
                            )
                        # If should_continue is True (NOOP), mark as failed but continue
                        input_validation_passed = False
                except ToolValidationError:
                    raise

                bound = sig.bind_partial()
                bound.arguments.update(validated_inputs)

                try:
                    result = func(*bound.args, **bound.kwargs)
                    validated_output, output_validation_passed, output_failures = (
                        run_output_validation(result)
                    )
                    if not output_validation_passed:
                        should_continue, _ = apply_on_fail_action(
                            on_output_fail,
                            output_failures,
                            validated_output,
                            is_input=False,
                        )
                        if not should_continue:
                            trace_tool_execution(
                                func_name,
                                validated_inputs,
                                input_validation_passed,
                                output_validation_passed,
                                input_failures=input_failures,
                                output_failures=output_failures,
                            )
                            if tool_guard:
                                exec_result = ToolValidationResult(
                                    tool_id=tool_id,
                                    tool_name=func_name,
                                    input_validation_passed=input_validation_passed,
                                    output_validation_passed=False,
                                    input_failures=input_failures,
                                    output_failures=output_failures,
                                    validated_inputs=validated_inputs,
                                    input_on_fail_action=on_input_fail,
                                    output_on_fail_action=on_output_fail,
                                    execution_time_ms=time.time() - start_time,
                                )
                                tool_guard._record_execution(exec_result)
                            raise ToolValidationError(
                                "Tool output validation failed", output_failures
                            )
                        # If should_continue is True (NOOP), mark as failed but continue
                        output_validation_passed = False
                except ToolValidationError:
                    raise

                trace_tool_execution(
                    func_name,
                    validated_inputs,
                    input_validation_passed,
                    output_validation_passed,
                    validated_output=validated_output,
                )

                if tool_guard:
                    exec_result = ToolValidationResult(
                        tool_id=tool_id,
                        tool_name=func_name,
                        input_validation_passed=input_validation_passed,
                        output_validation_passed=output_validation_passed,
                        validated_inputs=validated_inputs,
                        validated_output=validated_output,
                        input_on_fail_action=on_input_fail,
                        output_on_fail_action=on_output_fail,
                        execution_time_ms=time.time() - start_time,
                    )
                    tool_guard._record_execution(exec_result)

                return validated_output

            return sync_wrapper

    return decorator
