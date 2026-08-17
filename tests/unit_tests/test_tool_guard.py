import pytest
from typing import Any, Dict

from guardrails import guard_tool, ToolGuard, ToolValidationError, ToolValidationResult
from guardrails.validator_base import Validator, register_validator
from guardrails_ai.types import FailResult, PassResult
from guardrails.types.on_fail import OnFailAction


@register_validator(name="test_is_even_fixed", data_type="integer")
class IsEvenValidator(Validator):
    def _validate(self, value: Any, metadata: Dict[str, Any]):
        if isinstance(value, int) and value % 2 == 0:
            return PassResult()
        return FailResult(error_message="Value must be even")


@register_validator(name="test_is_upper_fixed", data_type="string")
class IsUpperValidator(Validator):
    def _validate(self, value: Any, metadata: Dict[str, Any]):
        if isinstance(value, str) and value.isupper():
            return PassResult()
        return FailResult(error_message="Value must be uppercase")


# ==================== Sync Tests ====================


def test_tool_guard_sync_pass():
    @guard_tool(
        input_validators={"num": [IsEvenValidator()]},
        output_validators=[IsUpperValidator()],
        on_input_fail=OnFailAction.EXCEPTION,
        on_output_fail=OnFailAction.EXCEPTION,
    )
    def my_tool(num: int, prefix: str = "RES_"):
        return f"{prefix}{num}"

    result = my_tool(4, "RES_")
    assert result == "RES_4"


def test_tool_guard_sync_fail_input():
    @guard_tool(
        input_validators={"num": [IsEvenValidator()]},
        on_input_fail=OnFailAction.EXCEPTION,
    )
    def my_tool(num: int):
        return num

    with pytest.raises(ToolValidationError) as exc_info:
        my_tool(5)

    assert "num" in exc_info.value.failures


def test_tool_guard_sync_fail_output():
    @guard_tool(
        output_validators=[IsUpperValidator()],
        on_output_fail=OnFailAction.EXCEPTION,
    )
    def my_tool(text: str):
        return text.lower()

    with pytest.raises(ToolValidationError) as exc_info:
        my_tool("HELLO")

    assert "output" in exc_info.value.failures


def test_tool_guard_on_fail_noop():
    @guard_tool(
        input_validators={"num": [IsEvenValidator()]},
        on_input_fail=OnFailAction.NOOP,
    )
    def my_tool(num: int):
        return num * 2

    # NOOP allows execution despite validation failure
    result = my_tool(5)
    assert result == 10


# ==================== Async Tests ====================


@pytest.mark.asyncio
async def test_tool_guard_async_pass():
    @guard_tool(
        input_validators={"num": [IsEvenValidator()]},
        output_validators=[IsUpperValidator()],
        on_input_fail=OnFailAction.EXCEPTION,
        on_output_fail=OnFailAction.EXCEPTION,
    )
    async def my_async_tool(num: int):
        return f"RES_{num}"

    result = await my_async_tool(4)
    assert result == "RES_4"


@pytest.mark.asyncio
async def test_tool_guard_async_fail():
    @guard_tool(
        input_validators={"num": [IsEvenValidator()]},
        on_input_fail=OnFailAction.EXCEPTION,
    )
    async def my_async_tool(num: int):
        return num

    with pytest.raises(ToolValidationError):
        await my_async_tool(5)


# ==================== ToolGuard Manager Tests ====================


def test_tool_guard_manager_creation():
    guard = ToolGuard(name="test_guard")
    assert guard.name == "test_guard"
    assert len(guard.execution_history) == 0


def test_tool_guard_manager_register():
    guard = ToolGuard()
    tool_id = guard.register_tool(
        "test_tool",
        input_validators=[IsEvenValidator()],
        output_validators=[IsUpperValidator()],
    )
    assert isinstance(tool_id, str)
    assert tool_id in guard._tool_validators


def test_tool_guard_manager_with_decorator():
    guard = ToolGuard()

    @guard_tool(
        input_validators={"num": [IsEvenValidator()]},
        tool_guard=guard,
        on_input_fail=OnFailAction.EXCEPTION,
    )
    def my_tool(num: int):
        return num * 2

    result = my_tool(4)
    assert result == 8
    assert len(guard.execution_history) == 1

    result_obj = guard.execution_history[0]
    assert isinstance(result_obj, ToolValidationResult)
    assert result_obj.tool_name == "my_tool"
    assert result_obj.input_validation_passed is True
    assert result_obj.output_validation_passed is True


def test_tool_guard_manager_get_history_by_tool():
    guard = ToolGuard()

    @guard_tool(tool_guard=guard)
    def tool_a():
        return "a"

    @guard_tool(tool_guard=guard)
    def tool_b():
        return "b"

    tool_a()
    tool_a()
    tool_b()

    history_a = guard.get_execution_history("tool_a")
    history_b = guard.get_execution_history("tool_b")

    assert len(history_a) == 2
    assert len(history_b) == 1


def test_tool_guard_manager_on_fail_actions():
    guard = ToolGuard()

    @guard_tool(
        input_validators={"num": [IsEvenValidator()]},
        on_input_fail=OnFailAction.NOOP,
        on_output_fail=OnFailAction.EXCEPTION,
        tool_guard=guard,
    )
    def my_tool(num: int):
        return num

    my_tool(5)  # NOOP allows execution
    result_obj = guard.execution_history[0]
    assert result_obj.input_on_fail_action == OnFailAction.NOOP
    assert result_obj.output_on_fail_action == OnFailAction.EXCEPTION


def test_tool_guard_manager_to_dict():
    guard = ToolGuard()

    @guard_tool(
        input_validators={"num": [IsEvenValidator()]},
        tool_guard=guard,
        on_input_fail=OnFailAction.EXCEPTION,
    )
    def my_tool(num: int):
        return num * 2

    my_tool(4)
    result_obj = guard.execution_history[0]
    result_dict = result_obj.to_dict()

    assert "tool_id" in result_dict
    assert "tool_name" in result_dict
    assert "validation_passed" in result_dict
    assert "execution_time_ms" in result_dict
    assert result_dict["tool_name"] == "my_tool"


def test_tool_guard_no_validators():
    @guard_tool()
    def my_tool(x: int):
        return x * 2

    result = my_tool(5)
    assert result == 10
