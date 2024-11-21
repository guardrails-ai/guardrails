from typing import Any, Optional, Dict, List, cast
from guardrails import Guard
from guardrails.errors import ValidationError
from guardrails.classes.validation_outcome import ValidationOutcome


try:
    import llama_index  # noqa: F401
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.schema import QueryBundle
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.base.response.schema import (
        RESPONSE_TYPE,
        Response,
        StreamingResponse,
        AsyncStreamingResponse,
        PydanticResponse,
    )
    from llama_index.core.prompts.mixin import PromptMixinType
except ImportError:
    raise ImportError(
        "llama_index is not installed. Please install it with "
        "`pip install llama-index` to use GuardrailsEngine."
    )


class GuardrailsQueryEngine(BaseQueryEngine):
    _engine_response: RESPONSE_TYPE

    def __init__(
        self,
        engine: BaseQueryEngine,
        guard: Guard,
        guard_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional["CallbackManager"] = None,
    ):
        self._engine = engine
        self._guard = guard
        self._guard_kwargs = guard_kwargs or {}
        super().__init__(callback_manager)

    @property
    def guard(self) -> Guard:
        return self._guard

    def engine_api(self, *, messages: List[Dict[str, str]], **kwargs) -> str:
        query = messages[0]["content"]
        response = self._engine.query(query)
        self._engine_response = response
        return str(response)

    def _query(self, query_bundle: "QueryBundle") -> RESPONSE_TYPE:
        if not isinstance(self._engine, BaseQueryEngine):
            raise ValueError(
                "Cannot perform query with a ChatEngine. Use chat() method instead."
            )
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        try:
            messages = [
                {
                    "role": "user",
                    "content": query_bundle.query_str,
                }
            ]
            validated_output = self.guard(
                llm_api=self.engine_api,
                messages=messages,
                **self._guard_kwargs,
            )

            validated_output = cast(ValidationOutcome, validated_output)
            # self._engine_response = cast(RESPONSE_TYPE, self._engine_response)
            if not validated_output.validation_passed:
                raise ValidationError(f"Validation failed: {validated_output.error}")
            self._update_response_metadata(validated_output)
            if validated_output.validation_passed:
                if isinstance(self._engine_response, Response):
                    self._engine_response.response = validated_output.validated_output
                elif isinstance(
                    self._engine_response, (StreamingResponse, AsyncStreamingResponse)
                ):
                    self._engine_response.response_txt = (
                        validated_output.validated_output
                    )
                elif isinstance(self._engine_response, PydanticResponse):
                    if self._engine_response.response:
                        import json

                        json_str = (
                            validated_output.validated_output
                            if isinstance(validated_output.validated_output, str)
                            else json.dumps(validated_output.validated_output)
                        )
                        self._engine_response.response = self._engine_response.response.__class__.model_validate_json(  # noqa: E501
                            json_str
                        )
                else:
                    raise ValueError("Unsupported response type")
        except ValidationError as e:
            raise ValidationError(f"Validation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during query processing: {str(e)}")
        return self._engine_response

    def _update_response_metadata(self, validated_output):
        if self._engine_response is None:
            return
        self._engine_response = cast(RESPONSE_TYPE, self._engine_response)

        metadata_update = {
            "validation_passed": validated_output.validation_passed,
            "validated_output": validated_output.validated_output,
            "error": validated_output.error,
            "raw_llm_output": validated_output.raw_llm_output,
        }

        if self._engine_response.metadata is None:
            self._engine_response.metadata = {}
        self._engine_response.metadata.update(metadata_update)

    async def _aquery(self, query_bundle: "QueryBundle") -> "RESPONSE_TYPE":
        """Async version of _query."""
        return self._query(query_bundle)

    def _get_prompt_modules(self) -> "PromptMixinType":
        """Get prompt modules."""
        return self._engine._get_prompt_modules()
