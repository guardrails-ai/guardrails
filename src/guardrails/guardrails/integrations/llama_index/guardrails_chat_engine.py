from typing import Any, Optional, Dict, List
from guardrails import Guard
from guardrails.errors import ValidationError


try:
    import llama_index  # noqa: F401
    from llama_index.core.chat_engine.types import (
        BaseChatEngine,
        AGENT_CHAT_RESPONSE_TYPE,
        AgentChatResponse,
        StreamingAgentChatResponse,
    )
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.prompts.mixin import PromptMixinType
except ImportError:
    raise ImportError(
        "llama_index is not installed. Please install it with "
        "`pip install llama-index` to use GuardrailsEngine."
    )


class GuardrailsChatEngine(BaseChatEngine):
    _engine_response: AGENT_CHAT_RESPONSE_TYPE

    def __init__(
        self,
        engine: BaseChatEngine,
        guard: Guard,
        guard_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._engine = engine
        self._guard = guard
        self._guard_kwargs = guard_kwargs or {}
        super().__init__()

    @property
    def guard(self) -> Guard:
        return self._guard

    def engine_api(self, *, messages: List[Dict[str, str]], **kwargs) -> str:
        query = messages[0]["content"]
        chat_history = kwargs.get("chat_history", [])
        response = self._engine.chat(query, chat_history)

        self._engine_response = response
        return str(response)

    def chat(
        self, message: str, chat_history: Optional[List["ChatMessage"]] = None
    ) -> AGENT_CHAT_RESPONSE_TYPE:
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

            return response
        except ValidationError as e:
            raise ValidationError(f"Validation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during chat processing: {str(e)}")

    def _create_chat_response(self, validated_output) -> AGENT_CHAT_RESPONSE_TYPE:
        if validated_output.validation_passed:
            content = validated_output.validated_output
        else:
            content = "I'm sorry, but I couldn't generate a valid response."

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

    async def achat(
        self, message: str, chat_history: Optional[List["ChatMessage"]] = None
    ):
        """Async version of chat."""
        raise NotImplementedError(
            "Async chat is not yet supported in the GuardrailsChatEngine."
        )

    def stream_chat(
        self, message: str, chat_history: Optional[List["ChatMessage"]] = None
    ):
        """Stream chat responses."""
        raise NotImplementedError(
            "Stream chat is not yet supported in the GuardrailsChatEngine."
        )

    async def astream_chat(
        self, message: str, chat_history: Optional[List["ChatMessage"]] = None
    ):
        """Async stream chat responses."""
        raise NotImplementedError(
            "Async stream chat is not yet supported in the GuardrailsChatEngine."
        )

    def reset(self):
        """Reset the chat history."""
        self._engine.reset()

    @property
    def chat_history(self) -> List["ChatMessage"]:
        """Get the chat history."""
        return self._engine.chat_history

    def _get_prompt_modules(self) -> "PromptMixinType":
        """Get prompt modules."""
        return {}
