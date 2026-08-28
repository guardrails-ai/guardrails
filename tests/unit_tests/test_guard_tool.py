import pytest
from typing import Any, Dict
from pydantic import BaseModel

from guardrails import guard_tool, register_validator, Validator
from guardrails.errors import ValidationError
from guardrails.types import OnFailAction
from guardrails_ai.types import FailResult, PassResult, ValidationResult


@register_validator("test_lowercase", data_type="string")
class LowercaseValidator(Validator):
    def _validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if value != value.lower():
            return FailResult(
                error_message="Value must be lowercase",
                fix_value=value.lower(),
            )
        return PassResult()


@register_validator("test_no_injection", data_type="string")
class NoInjectionValidator(Validator):
    def _validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        dangerous = ["<script>", "DROP TABLE", "rm -rf"]
        for pattern in dangerous:
            if pattern.lower() in value.lower():
                return FailResult(error_message=f"Potential injection: {pattern}")
        return PassResult()


class TestGuardToolBasic:
    def test_no_validators_passthrough(self):
        @guard_tool()
        def my_tool(query: str) -> str:
            return query.upper()

        assert my_tool(query="hello") == "HELLO"

    def test_input_validators_pass(self):
        @guard_tool(input_validators=[LowercaseValidator(on_fail=OnFailAction.EXCEPTION)])
        def my_tool(query: str) -> str:
            return query.upper()

        assert my_tool(query="hello") == "HELLO"

    def test_input_validators_fail_exception(self):
        @guard_tool(input_validators=[LowercaseValidator(on_fail=OnFailAction.EXCEPTION)])
        def my_tool(query: str) -> str:
            return query.upper()

        with pytest.raises(ValidationError, match="must be lowercase"):
            my_tool(query="HELLO")

    def test_input_validators_fail_fix(self):
        @guard_tool(
            input_validators=[LowercaseValidator(on_fail=OnFailAction.FIX)],
            on_fail=OnFailAction.FIX,
        )
        def my_tool(query: str) -> str:
            return query

        assert my_tool(query="HELLO") == "hello"

    def test_output_validators_pass(self):
        @guard_tool(output_validators=[LowercaseValidator(on_fail=OnFailAction.EXCEPTION)])
        def my_tool(query: str) -> str:
            return query.lower()

        assert my_tool(query="HELLO") == "hello"

    def test_output_validators_fail_exception(self):
        @guard_tool(
            output_validators=[LowercaseValidator(on_fail=OnFailAction.EXCEPTION)],
            on_fail=OnFailAction.EXCEPTION,
        )
        def my_tool(query: str) -> str:
            return query.upper()

        with pytest.raises(ValidationError, match="must be lowercase"):
            my_tool(query="hello")


class TestGuardToolSecurity:
    def test_injection_blocked(self):
        @guard_tool(
            input_validators=[NoInjectionValidator(on_fail=OnFailAction.EXCEPTION)],
            on_fail=OnFailAction.EXCEPTION,
        )
        def web_search(query: str) -> str:
            return f"Results for: {query}"

        with pytest.raises(ValidationError, match="injection"):
            web_search(query="<script>alert('xss')</script>")

    def test_output_sanitization(self):
        @guard_tool(
            output_validators=[NoInjectionValidator(on_fail=OnFailAction.EXCEPTION)],
            on_fail=OnFailAction.EXCEPTION,
        )
        def fetch_external_data(url: str) -> str:
            return "<script>malicious()</script>"

        with pytest.raises(ValidationError, match="injection"):
            fetch_external_data(url="https://example.com")


class TestGuardToolSchema:
    def test_input_schema_validation(self):
        class QuerySchema(BaseModel):
            query: str
            limit: int = 10

        @guard_tool(input_schema=QuerySchema)
        def search(query: str, limit: int = 10) -> str:
            return f"{query}:{limit}"

        assert search(query="test", limit=5) == "test:5"

    def test_input_schema_invalid(self):
        class QuerySchema(BaseModel):
            query: str
            limit: int

        @guard_tool(input_schema=QuerySchema)
        def search(query: str, limit: int) -> str:
            return f"{query}:{limit}"

        with pytest.raises(ValidationError, match="schema validation failed"):
            search(query="test", limit="not_a_number")


class TestGuardToolAsync:
    def test_async_tool_passthrough(self):
        import asyncio

        @guard_tool()
        async def async_tool(query: str) -> str:
            return query.upper()

        result = asyncio.get_event_loop().run_until_complete(async_tool(query="hello"))
        assert result == "HELLO"

    def test_async_input_validation(self):
        import asyncio

        @guard_tool(input_validators=[LowercaseValidator(on_fail=OnFailAction.EXCEPTION)])
        async def async_tool(query: str) -> str:
            return query.upper()

        with pytest.raises(ValidationError, match="must be lowercase"):
            asyncio.get_event_loop().run_until_complete(async_tool(query="HELLO"))


class TestGuardToolOnFailActions:
    def test_on_fail_noop(self):
        @guard_tool(
            input_validators=[LowercaseValidator(on_fail=OnFailAction.NOOP)],
            on_fail=OnFailAction.NOOP,
        )
        def my_tool(query: str) -> str:
            return query

        assert my_tool(query="HELLO") == "HELLO"

    def test_on_fail_refrain(self):
        @guard_tool(
            input_validators=[LowercaseValidator(on_fail=OnFailAction.REFRAIN)],
            on_fail=OnFailAction.REFRAIN,
        )
        def my_tool(query: str) -> str:
            return query

        assert my_tool(query="HELLO") is None
