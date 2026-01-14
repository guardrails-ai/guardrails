import pytest
from guardrails import Guard
from typing import List, Optional
from tests.integration_tests.test_assets.validators import RegexMatch

pytest.importorskip("llama_index")

from llama_index.core.chat_engine.types import (  # noqa
    BaseChatEngine,  # noqa
    AgentChatResponse,  # noqa
    StreamingAgentChatResponse,  # noqa
)  # noqa
from llama_index.core.base.llms.types import ChatMessage  # noqa
from guardrails.integrations.llama_index import GuardrailsChatEngine  # noqa


class MockChatEngine(BaseChatEngine):
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        return AgentChatResponse(response="Mock response")

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        return AgentChatResponse(response="Mock async chat response")

    def stream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None):
        return StreamingAgentChatResponse(response="Mock stream chat response")

    async def astream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None):
        return StreamingAgentChatResponse(response="Mock async stream chat response")

    @property
    def chat_history(self) -> List[ChatMessage]:
        return []

    def reset(self):
        pass


pytest.importorskip("llama_index")


@pytest.fixture
def guard():
    return Guard().use(RegexMatch("Mock response", match_type="search"))


class TestGuardrailsChatEngine:
    def test_guardrails_engine_init(self, guard):
        engine = MockChatEngine()
        guardrails_engine = GuardrailsChatEngine(engine, guard)
        assert isinstance(guardrails_engine, GuardrailsChatEngine)
        assert guardrails_engine.guard == guard

    def test_guardrails_engine_chat(self, guard):
        engine = MockChatEngine()
        guardrails_engine = GuardrailsChatEngine(engine, guard)

        result = guardrails_engine.chat("Mock response")
        assert isinstance(result, AgentChatResponse)
        assert result.response == "Mock response"
