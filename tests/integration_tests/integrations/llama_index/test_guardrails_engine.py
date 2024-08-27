import pytest
from guardrails import Guard
from guardrails.integrations.llama_index.guardrails_engine import GuardrailsEngine
from guardrails.errors import ValidationError
from typing import List, Optional
from tests.integration_tests.test_assets.validators import RegexMatch

try:
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.chat_engine.types import BaseChatEngine, AgentChatResponse
    from llama_index.core.schema import QueryBundle
    from llama_index.core.base.response.schema import Response
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.prompts.mixin import PromptMixinType
    from llama_index.core.callbacks import CallbackManager

    class MockQueryEngine(BaseQueryEngine):
        def __init__(self, callback_manager: Optional[CallbackManager] = None):
            super().__init__(callback_manager)

        def _query(self, query_bundle: QueryBundle) -> Response:
            return Response(response="Mock response")

        async def _aquery(self, query_bundle: QueryBundle) -> Response:
            return Response(response="Mock async query response")

        def _get_prompt_modules(self) -> PromptMixinType:
            return {}

    class MockChatEngine(BaseChatEngine):
        def chat(
            self, message: str, chat_history: Optional[List[ChatMessage]] = None
        ) -> AgentChatResponse:
            return AgentChatResponse(response="Mock response")

        async def achat(
            self, message: str, chat_history: Optional[List[ChatMessage]] = None
        ) -> AgentChatResponse:
            return AgentChatResponse(response="Mock async chat response")

        def stream_chat(
            self, message: str, chat_history: Optional[List[ChatMessage]] = None
        ):
            yield AgentChatResponse(response="Mock stream chat response")

        async def astream_chat(
            self, message: str, chat_history: Optional[List[ChatMessage]] = None
        ):
            yield AgentChatResponse(response="Mock async stream chat response")

        @property
        def chat_history(self) -> List[ChatMessage]:
            return []

        def reset(self):
            pass
except ImportError:
    pytest.skip("llama_index is not installed", allow_module_level=True)

pytest.importorskip("llama_index")


@pytest.fixture
def guard():
    return Guard().use(RegexMatch("Mock response", match_type="search"))


def test_guardrails_engine_init(guard):
    engine = MockQueryEngine()
    guardrails_engine = GuardrailsEngine(engine, guard)
    assert isinstance(guardrails_engine, GuardrailsEngine)
    assert guardrails_engine.guard == guard


def test_guardrails_engine_query(guard):
    engine = MockQueryEngine()
    guardrails_engine = GuardrailsEngine(engine, guard)

    result = guardrails_engine._query(QueryBundle(query_str="Mock response"))
    assert isinstance(result, Response)
    assert result.response == "Mock response"


def test_guardrails_engine_query_validation_failure(guard):
    engine = MockQueryEngine()
    guardrails_engine = GuardrailsEngine(engine, guard)

    engine._query = lambda _: Response(response="Invalid response")

    with pytest.raises(ValidationError, match="Validation failed"):
        guardrails_engine._query(QueryBundle(query_str="Invalid query"))


def test_guardrails_engine_chat(guard):
    engine = MockChatEngine()
    guardrails_engine = GuardrailsEngine(engine, guard)

    result = guardrails_engine.chat("Mock response")
    assert isinstance(result, AgentChatResponse)
    assert result.response == "Mock response"


def test_guardrails_engine_unsupported_engine(guard):
    class UnsupportedEngine:
        pass

    engine = UnsupportedEngine()
    guardrails_engine = GuardrailsEngine(engine, guard)

    with pytest.raises(ValueError, match="Unsupported engine type"):
        guardrails_engine.engine_api("Test prompt")
