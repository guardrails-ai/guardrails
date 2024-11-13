# pyright: reportMissingImports=false

from typing import Any, Optional, Dict, List, Union, TYPE_CHECKING, cast
from guardrails import Guard
from guardrails.errors import ValidationError
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.decorators.experimental import experimental
import importlib.util

LLAMA_INDEX_AVAILABLE = importlib.util.find_spec("llama_index") is not None

if TYPE_CHECKING or LLAMA_INDEX_AVAILABLE:
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.chat_engine.types import (
        BaseChatEngine,
        AGENT_CHAT_RESPONSE_TYPE,
        AgentChatResponse,
        StreamingAgentChatResponse,
    )
    from llama_index.core.schema import QueryBundle
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.base.response.schema import (
        RESPONSE_TYPE,
        Response,
        StreamingResponse,
        AsyncStreamingResponse,
        PydanticResponse,
    )
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.prompts.mixin import PromptMixinType


class GuardrailsEngine(BaseQueryEngine, BaseChatEngine):
    def __init__(
        self,
        engine: Union["BaseQueryEngine", "BaseChatEngine"],
        guard: Guard,
        guard_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional["CallbackManager"] = None,
    ):
        try:
            import llama_index  # noqa: F401
        except ImportError:
            raise ImportError(
                "llama_index is not installed. Please install it with "
                "`pip install llama-index` to use GuardrailsEngine."
            )

        self._engine = engine
        self._guard = guard
        self._guard_kwargs = guard_kwargs or {}
        self._engine_response = None
        super().__init__(callback_manager)

    @property
    def guard(self) -> Guard:
        return self._guard

    def engine_api(self, *, messages: List[Dict[str, str]], **kwargs) -> str:
        user_messages = [m for m in messages if m["role"] == "user"]
        query = user_messages[-1]["content"]
        if isinstance(self._engine, BaseQueryEngine):
            response = self._engine.query(query)
        elif isinstance(self._engine, BaseChatEngine):
            chat_history = kwargs.get("chat_history", [])
            response = self._engine.chat(query, chat_history)
        else:
            raise ValueError("Unsupported engine type")

        self._engine_response = response
        return str(response)

    @experimental
    def _query(self, query_bundle: "QueryBundle") -> "RESPONSE_TYPE":
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
            self._engine_response = cast(RESPONSE_TYPE, self._engine_response)
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

    @experimental
    def chat(
        self, message: str, chat_history: Optional[List["ChatMessage"]] = None
    ) -> "AGENT_CHAT_RESPONSE_TYPE":
        if not isinstance(self._engine, BaseChatEngine):
            raise ValueError(
                "Cannot perform chat with a QueryEngine. Use query() method instead."
            )
        if chat_history is None:
            chat_history = []
        try:
            messages = [
                {
                    "role": "user",
                    "content": message,
                }
            ]
            validated_output = self.guard(
                llm_api=self.engine_api,
                messages=messages,
                chat_history=chat_history,
                **self._guard_kwargs,
            )
            response = self._create_chat_response(validated_output)
            if response is None:
                raise ValueError("Failed to create a valid chat response")

            return cast(AGENT_CHAT_RESPONSE_TYPE, response)
        except ValidationError as e:
            raise ValidationError(f"Validation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during chat processing: {str(e)}")

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

    def _create_chat_response(
        self, validated_output
    ) -> Optional[AGENT_CHAT_RESPONSE_TYPE]:
        if validated_output.validation_passed:
            content = validated_output.validated_output
        else:
            content = "I'm sorry, but I couldn't generate a valid response."

        if self._engine_response is None:
            return None

        metadata_update = {
            "validation_passed": validated_output.validation_passed,
            "validated_output": validated_output.validated_output,
            "error": validated_output.error,
            "raw_llm_output": validated_output.raw_llm_output,
        }

        if isinstance(self._engine_response, AgentChatResponse):
            if self._engine_response.metadata is None:
                self._engine_response.metadata = {}
            self._engine_response.metadata.update(metadata_update)
        elif isinstance(self._engine_response, StreamingAgentChatResponse):
            for key, value in metadata_update.items():
                setattr(self._engine_response, key, value)

        self._engine_response.response = content
        return self._engine_response

    async def _aquery(self, query_bundle: "QueryBundle") -> "RESPONSE_TYPE":
        """Async version of _query."""
        return self._query(query_bundle)

    async def achat(
        self, message: str, chat_history: Optional[List["ChatMessage"]] = None
    ):
        """Async version of chat."""
        raise NotImplementedError(
            "Async chat is not supported in the GuardrailsQueryEngine."
        )

    def stream_chat(
        self, message: str, chat_history: Optional[List["ChatMessage"]] = None
    ):
        """Stream chat responses."""
        raise NotImplementedError(
            "Stream chat is not supported in the GuardrailsQueryEngine."
        )

    async def astream_chat(
        self, message: str, chat_history: Optional[List["ChatMessage"]] = None
    ):
        """Async stream chat responses."""
        raise NotImplementedError(
            "Async stream chat is not supported in the GuardrailsQueryEngine."
        )

    def reset(self):
        """Reset the chat history."""
        if isinstance(self._engine, BaseChatEngine):
            self._engine.reset()
        else:
            raise NotImplementedError("Reset is only available for chat engines.")

    @property
    def chat_history(self) -> List["ChatMessage"]:
        """Get the chat history."""
        if isinstance(self._engine, BaseChatEngine):
            return self._engine.chat_history
        raise NotImplementedError("Chat history is only available for chat engines.")

    def _get_prompt_modules(self) -> "PromptMixinType":
        """Get prompt modules."""
        if isinstance(self._engine, BaseQueryEngine):
            return self._engine._get_prompt_modules()
        return {}
