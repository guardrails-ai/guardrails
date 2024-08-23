from typing import Any, Optional, Dict, List, Union
from guardrails import Guard
from guardrails.errors import ValidationError

try:
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.schema import QueryBundle
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.base.response.schema import RESPONSE_TYPE
    from llama_index.core.chat_engine.types import BaseChatEngine
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.prompts.mixin import PromptMixinType

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    BaseQueryEngine = BaseChatEngine = object
    QueryBundle = CallbackManager = RESPONSE_TYPE = ChatResponse = ChatMessage = Any
    PromptMixinType = Any
    LLAMA_INDEX_AVAILABLE = False


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

    def engine_api(self, prompt: str, **kwargs) -> str:
        if isinstance(self._engine, BaseQueryEngine):
            response = self._engine.query(prompt)
        elif isinstance(self._engine, BaseChatEngine):
            chat_history = kwargs.get("chat_history", [])
            response = self._engine.chat(prompt, chat_history)
        else:
            raise ValueError("Unsupported engine type")

        self._engine_response = response
        return response.response

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        if not isinstance(self._engine, BaseQueryEngine):
            raise ValueError(
                "Cannot perform query with a ChatEngine. Use chat() method instead."
            )
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        try:
            validated_output = self.guard(
                llm_api=self.engine_api,
                prompt=query_bundle.query_str,
                **self._guard_kwargs,
            )
            self._update_response_metadata(validated_output)
            if validated_output.validation_passed:
                self._engine_response.response = validated_output.validated_output
        except ValidationError as e:
            raise ValidationError(f"Validation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during query processing: {str(e)}")
        return self._engine_response

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> ChatResponse:
        if not isinstance(self._engine, BaseChatEngine):
            raise ValueError(
                "Cannot perform chat with a QueryEngine. Use query() method instead."
            )
        if chat_history is None:
            chat_history = []
        try:
            validated_output = self.guard(
                llm_api=self.engine_api,
                prompt=message,
                chat_history=chat_history,
                **self._guard_kwargs,
            )
            response = self._create_chat_response(validated_output)
        except ValidationError as e:
            raise ValidationError(f"Validation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during chat processing: {str(e)}")
        return response

    def _update_response_metadata(self, validated_output):
        self._engine_response.metadata.update(
            {
                "validation_passed": validated_output.validation_passed,
                "validated_output": validated_output.validated_output,
                "error": validated_output.error,
                "raw_llm_output": validated_output.raw_llm_output,
            }
        )

    def _create_chat_response(self, validated_output):
        if validated_output.validation_passed:
            content = validated_output.validated_output
        else:
            content = "I'm sorry, but I couldn't generate a valid response."

        response = ChatResponse(message=ChatMessage(role="assistant", content=content))
        response.metadata = {
            "validation_passed": validated_output.validation_passed,
            "validated_output": validated_output.validated_output,
            "error": validated_output.error,
            "raw_llm_output": validated_output.raw_llm_output,
        }
        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Async version of _query."""
        return self._query(query_bundle)

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> ChatResponse:
        """Async version of chat."""
        return self.chat(message, chat_history)

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ):
        raise NotImplementedError("Stream chat is not supported with guardrails.")

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ):
        raise NotImplementedError("Async stream chat is not supported with guardrails.")

    def reset(self):
        """Reset the chat history."""
        if isinstance(self._engine, BaseChatEngine):
            self._engine.reset()
        else:
            raise NotImplementedError("Reset is only available for chat engines.")

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get the chat history."""
        if isinstance(self._engine, BaseChatEngine):
            return self._engine.chat_history
        raise NotImplementedError("Chat history is only available for chat engines.")

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return self._engine._get_prompt_modules()
