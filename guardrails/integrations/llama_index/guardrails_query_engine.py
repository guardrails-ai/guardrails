from typing import Any, List, Optional, Dict, TYPE_CHECKING
import asyncio
import importlib

from guardrails import Guard
from guardrails.errors import ValidationError

if TYPE_CHECKING:
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.schema import QueryBundle, QueryType, NodeWithScore
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.base.response.schema import RESPONSE_TYPE
else:
    BaseQueryEngine = QueryBundle = NodeWithScore = CallbackManager = QueryType = (
        RESPONSE_TYPE
    ) = Any


class GuardrailsQueryEngine:
    def __init__(
        self,
        query_engine: "BaseQueryEngine",
        input_guard: Optional[Guard] = None,
        output_guard: Optional[Guard] = None,
        input_guard_kwargs: Optional[Dict[str, Any]] = None,
        output_guard_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional["CallbackManager"] = None,
    ):
        LLAMA_INDEX_CORE = importlib.util.find_spec("llama_index.core")
        if LLAMA_INDEX_CORE is not None:
            raise ImportError(
                "llama_index is not installed. Please install it with "
                "`pip install llama-index` to use GuardrailsQueryEngine."
            )
        self._query_engine = query_engine
        self._input_guard = input_guard
        self._output_guard = output_guard
        self._input_guard_kwargs = input_guard_kwargs or {}
        self._output_guard_kwargs = output_guard_kwargs or {}
        self._callback_manager = callback_manager or CallbackManager([])

    async def _validate_input(self, query: str) -> str:
        if self._input_guard:
            try:
                async with self._callback_manager.event(event_type="input_validation"):
                    validated_input = await self._input_guard.validate(
                        query, **self._input_guard_kwargs
                    )
                    return validated_input.validated_output
            except ValidationError as e:
                await self._callback_manager.on_event(
                    event_type="input_validation_error", metadata={"error": str(e)}
                )
                raise ValueError(f"Input validation failed: {str(e)}")
        return query

    async def _validate_output(self, response: "RESPONSE_TYPE") -> "RESPONSE_TYPE":
        if self._output_guard:
            try:
                async with self._callback_manager.event(event_type="output_validation"):
                    validated_output = await self._output_guard.validate(
                        response.response, **self._output_guard_kwargs
                    )
                response.metadata.update(
                    {
                        "validation_passed": validated_output.validation_passed,
                        "validated_output": validated_output.validated_output,
                        "error": validated_output.error,
                        "raw_llm_output": validated_output.raw_llm_output,
                    }
                )
                if validated_output.validation_passed:
                    response.response = validated_output.validated_output
                else:
                    raise ValidationError(
                        f"Output validation failed: {validated_output.error}"
                    )
            except ValidationError as e:
                await self._callback_manager.on_event(
                    event_type="output_validation_error", metadata={"error": str(e)}
                )
                raise ValueError(f"Output validation failed: {str(e)}")
        return response

    async def _aquery(self, query_bundle: "QueryType") -> "RESPONSE_TYPE":
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        async with self._callback_manager.event(
            event_type="query", metadata={"query": query_bundle.query_str}
        ):
            query_bundle.query_str = await self._validate_input(query_bundle.query_str)

            async with self._callback_manager.event(
                event_type="underlying_query_engine"
            ):
                response = await self._query_engine.aquery(query_bundle)

            return await self._validate_output(response)

    def _query(self, query_bundle: "QueryType") -> "RESPONSE_TYPE":
        return asyncio.run(self._aquery(query_bundle))

    def retrieve(self, query_bundle: "QueryType") -> List["NodeWithScore"]:
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        with self._callback_manager.event(
            event_type="retrieve", metadata={"query": query_bundle.query_str}
        ):
            if self._input_guard:
                try:
                    with self._callback_manager.event(event_type="input_validation"):
                        validated_input = self._input_guard.validate(
                            query_bundle.query_str, **self._input_guard_kwargs
                        )
                        query_bundle.query_str = validated_input.validated_output
                except ValidationError as e:
                    self._callback_manager.on_event(
                        event_type="input_validation_error", metadata={"error": str(e)}
                    )
                    raise ValueError(f"Input validation failed: {str(e)}")

            return self._query_engine.retrieve(query_bundle)
